//! Inference lifecycle manager.
//!
//! Monitors the node's pipeline assignment (via Supabase) and automatically
//! starts/stops llama-server when work is assigned or released.
//!
//! Flow:
//! 1. Daemon heartbeat checks node's pipeline_id in Supabase
//! 2. If assigned and not running → start llama-server with the model
//! 3. If unassigned and running → stop llama-server
//! 4. Reports inference metrics back to the daemon state

use std::process::{Child, Command, Stdio};
use tracing::{error, info, warn};

/// Status of the local inference server.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceStatus {
    /// No pipeline assigned, not running.
    Idle,
    /// Starting up llama-server.
    Starting,
    /// Running inference for a pipeline.
    Running { pipeline_id: String, model_name: String, port: u16 },
    /// Error state.
    Error(String),
}

impl std::fmt::Display for InferenceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceStatus::Idle => write!(f, "idle"),
            InferenceStatus::Starting => write!(f, "starting"),
            InferenceStatus::Running { model_name, .. } => write!(f, "running ({model_name})"),
            InferenceStatus::Error(e) => write!(f, "error: {e}"),
        }
    }
}

/// Manages the llama-server process lifecycle.
pub struct InferenceManager {
    status: InferenceStatus,
    server_process: Option<Child>,
    port: u16,
    model_path: Option<String>,
    current_pipeline_id: Option<String>,
}

impl Default for InferenceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceManager {
    pub fn new() -> Self {
        Self {
            status: InferenceStatus::Idle,
            server_process: None,
            port: 8090,
            model_path: None,
            current_pipeline_id: None,
        }
    }

    pub fn status(&self) -> &InferenceStatus {
        &self.status
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Check if we should start or stop inference based on the pipeline assignment.
    /// Called by the daemon on each heartbeat cycle.
    pub fn check_assignment(&mut self, pipeline_id: Option<&str>, model_name: Option<&str>) {
        match (&self.status, pipeline_id) {
            // Not running, got assigned → start
            (InferenceStatus::Idle | InferenceStatus::Error(_), Some(pid)) => {
                let model = model_name.unwrap_or("unknown");
                info!("Pipeline assigned: {pid} ({model}) — starting inference");
                self.current_pipeline_id = Some(pid.to_string());
                self.start_server(pid, model);
            }
            // Running a different pipeline → restart
            (InferenceStatus::Running { pipeline_id: current, .. }, Some(pid))
                if current != pid =>
            {
                let model = model_name.unwrap_or("unknown");
                info!("Pipeline changed: {current} → {pid} — restarting inference");
                self.stop_server();
                self.current_pipeline_id = Some(pid.to_string());
                self.start_server(pid, model);
            }
            // Running but assignment cleared → stop
            (InferenceStatus::Running { .. }, None) => {
                info!("Pipeline released — stopping inference");
                self.stop_server();
                self.current_pipeline_id = None;
            }
            // Already running same pipeline, or idle with no assignment → no-op
            _ => {}
        }

        // Check if server process is still alive
        if let Some(ref mut child) = self.server_process {
            match child.try_wait() {
                Ok(Some(exit)) => {
                    warn!("llama-server exited unexpectedly: {exit}");
                    self.server_process = None;
                    self.status = InferenceStatus::Error("Server crashed".into());
                }
                Ok(None) => {} // Still running
                Err(e) => {
                    error!("Failed to check server status: {e}");
                }
            }
        }
    }

    /// Start llama-server with the appropriate model.
    fn start_server(&mut self, pipeline_id: &str, model_name: &str) {
        self.status = InferenceStatus::Starting;

        // Find the model file
        let model_path = find_model_path(model_name);
        let Some(path) = model_path else {
            let msg = format!("Model not found locally: {model_name}");
            warn!("{msg}");
            self.status = InferenceStatus::Error(msg);
            return;
        };

        info!("Starting llama-server on port {} with {}", self.port, path);

        let result = Command::new("llama-server")
            .args([
                "--model",
                &path,
                "--port",
                &self.port.to_string(),
                "--ctx-size",
                "16384",
                "--n-gpu-layers",
                "999",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn();

        match result {
            Ok(child) => {
                self.server_process = Some(child);
                self.model_path = Some(path.clone());
                self.status = InferenceStatus::Running {
                    pipeline_id: pipeline_id.to_string(),
                    model_name: model_name.to_string(),
                    port: self.port,
                };
                info!("llama-server started (PID: {})", self.server_process.as_ref().unwrap().id());
            }
            Err(e) => {
                let msg = format!("Failed to start llama-server: {e}");
                error!("{msg}");
                self.status = InferenceStatus::Error(msg);
            }
        }
    }

    /// Stop the llama-server process.
    fn stop_server(&mut self) {
        if let Some(ref mut child) = self.server_process {
            info!("Stopping llama-server (PID: {})", child.id());
            let _ = child.kill();
            let _ = child.wait();
        }
        self.server_process = None;
        self.model_path = None;
        self.status = InferenceStatus::Idle;
    }

    /// Check if the llama-server is healthy (responds to /health).
    pub async fn health_check(&self) -> bool {
        if !matches!(self.status, InferenceStatus::Running { .. }) {
            return false;
        }

        let url = format!("http://127.0.0.1:{}/health", self.port);
        match reqwest::Client::new()
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Get inference metrics by polling llama-server's /slots endpoint.
    /// Fast local HTTP call (< 1ms latency).
    ///
    /// Note: /slots can return malformed JSON when prompts contain control chars,
    /// so we parse with string matching rather than JSON deserialization.
    pub async fn get_metrics(&self) -> Option<InferenceMetrics> {
        if !matches!(self.status, InferenceStatus::Running { .. }) {
            return None;
        }

        let url = format!("http://127.0.0.1:{}/slots", self.port);
        let resp = reqwest::Client::new()
            .get(&url)
            .timeout(std::time::Duration::from_millis(200))
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        // Use string matching — JSON may be malformed when prompts contain control chars
        let body = resp.text().await.ok()?;
        let slots_processing = body.matches("\"is_processing\":true").count() as u32;
        let slots_total = body.matches("\"is_processing\":").count() as u32;
        let slots_idle = slots_total.saturating_sub(slots_processing);

        Some(InferenceMetrics { slots_idle, slots_processing })
    }
}

impl Drop for InferenceManager {
    fn drop(&mut self) {
        self.stop_server();
    }
}

/// Simple inference metrics from llama-server.
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    pub slots_idle: u32,
    pub slots_processing: u32,
}

/// Find a model file in the local cache.
/// Checks ~/.compute/models/ for GGUF files matching the model name.
fn find_model_path(model_name: &str) -> Option<String> {
    let cache_dir = dirs::home_dir()?.join(".compute").join("models");

    if !cache_dir.exists() {
        return None;
    }

    // Direct filename match
    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".gguf") {
            // Match by model name substring
            let lower_name = name.to_lowercase();
            let lower_model = model_name.to_lowercase();

            // Try various matching strategies
            if lower_name.contains(&lower_model)
                || lower_model.contains(&lower_name.replace(".gguf", ""))
            {
                return Some(entry.path().to_string_lossy().to_string());
            }
        }
    }

    // Fallback: return the first GGUF file found (for testing)
    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".gguf") {
            return Some(entry.path().to_string_lossy().to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let mgr = InferenceManager::new();
        assert_eq!(*mgr.status(), InferenceStatus::Idle);
        assert_eq!(mgr.port(), 8090);
    }

    #[test]
    fn test_status_display() {
        assert_eq!(InferenceStatus::Idle.to_string(), "idle");
        assert_eq!(InferenceStatus::Starting.to_string(), "starting");
        assert_eq!(
            InferenceStatus::Running {
                pipeline_id: "p".into(),
                model_name: "Llama 70B".into(),
                port: 8090,
            }
            .to_string(),
            "running (Llama 70B)"
        );
    }
}
