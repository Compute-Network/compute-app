//! Inference lifecycle manager.
//!
//! Monitors the node's pipeline assignment (via the orchestrator) and automatically
//! starts/stops llama-server when work is assigned or released.
//!
//! Flow:
//! 1. Daemon heartbeat checks node's pipeline_id via the orchestrator
//! 2. If assigned and not running → start llama-server with the model
//! 3. If unassigned and running → stop llama-server
//! 4. Reports inference metrics back to the daemon state

use std::process::{Child, Command, Stdio};
use tracing::{error, info, warn};

/// Role this node plays in a multi-node pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineRole {
    /// Single-node inference (default, no RPC)
    Solo,
    /// Head node: runs llama-server with --rpc pointing to worker nodes
    Head { rpc_peers: Vec<String> },
    /// Worker node: runs rpc-server, head node connects to us
    Worker { rpc_port: u16 },
}

/// Status of the local inference server.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceStatus {
    /// No pipeline assigned, not running.
    Idle,
    /// Starting up llama-server or rpc-server.
    Starting,
    /// Running inference for a pipeline.
    Running { pipeline_id: String, model_name: String, port: u16 },
    /// Running as RPC worker (not serving HTTP, head node connects to us)
    RunningRpcWorker { pipeline_id: String, model_name: String, rpc_port: u16 },
    /// Error state.
    Error(String),
}

impl std::fmt::Display for InferenceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceStatus::Idle => write!(f, "idle"),
            InferenceStatus::Starting => write!(f, "starting"),
            InferenceStatus::Running { model_name, .. } => write!(f, "running ({model_name})"),
            InferenceStatus::RunningRpcWorker { model_name, .. } => {
                write!(f, "rpc-worker ({model_name})")
            }
            InferenceStatus::Error(e) => write!(f, "error: {e}"),
        }
    }
}

/// Manages the llama-server and rpc-server process lifecycle.
/// Supports three modes:
/// - Solo: single-node llama-server (current default)
/// - Head: llama-server with --rpc flag connecting to remote workers
/// - Worker: rpc-server listening for head node connections
///
/// Note:
/// The RPC path is still considered experimental. The current multi-node v1
/// direction is stage-based pipeline parallelism with explicit activation passing,
/// not RPC-offload as the strategic long-term architecture.
pub struct InferenceManager {
    status: InferenceStatus,
    server_process: Option<Child>,
    port: u16,
    rpc_port: u16,
    model_path: Option<String>,
    current_pipeline_id: Option<String>,
    current_model_name: Option<String>,
    current_role: PipelineRole,
    externally_managed: bool,
    restart_attempts: u32,
    restart_after: Option<std::time::Instant>,
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
            rpc_port: 50052,
            model_path: None,
            current_pipeline_id: None,
            current_model_name: None,
            current_role: PipelineRole::Solo,
            externally_managed: false,
            restart_attempts: 0,
            restart_after: None,
        }
    }

    pub fn status(&self) -> &InferenceStatus {
        &self.status
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn rpc_port(&self) -> u16 {
        self.rpc_port
    }

    pub fn role(&self) -> &PipelineRole {
        &self.current_role
    }

    /// Force-stop any managed inference process regardless of assignment semantics.
    /// Used when switching to a different experimental runtime path.
    pub fn shutdown_server(&mut self) {
        self.stop_server();
        self.current_pipeline_id = None;
        self.current_model_name = None;
        self.current_role = PipelineRole::Solo;
    }

    /// When true, the regular llama/rpc lifecycle manager is disabled because
    /// another runtime path (currently stage prototype mode) owns execution.
    pub fn set_externally_managed(&mut self, externally_managed: bool) {
        self.externally_managed = externally_managed;
        if externally_managed {
            self.current_pipeline_id = None;
            self.current_model_name = None;
            self.status = InferenceStatus::Idle;
        }
    }

    /// Set the pipeline role for this node. Call before check_assignment.
    /// - Solo: single-node llama-server (default)
    /// - Head { rpc_peers }: llama-server with --rpc connecting to workers
    /// - Worker { rpc_port }: rpc-server listening for head connections
    pub fn set_role(&mut self, role: PipelineRole) {
        if self.current_role != role {
            info!("[inference] Role changed: {:?} → {:?}", self.current_role, role);
            self.current_role = role;
        }
    }

    /// Check if we should start or stop inference based on the pipeline assignment.
    /// Called by the daemon on each heartbeat cycle.
    pub fn check_assignment(&mut self, pipeline_id: Option<&str>, model_name: Option<&str>) {
        if self.externally_managed {
            return;
        }
        match (&self.status, pipeline_id) {
            // Not running, got assigned → start
            (InferenceStatus::Idle | InferenceStatus::Error(_), Some(pid)) => {
                let model = model_name.unwrap_or("unknown");
                info!("Pipeline assigned: {pid} ({model}) — starting inference");
                self.current_pipeline_id = Some(pid.to_string());
                self.start_server(pid, model);
            }
            // Running a different pipeline → check if same model (avoid restart)
            (
                InferenceStatus::Running { pipeline_id: current, model_name: current_model, port },
                Some(pid),
            ) if current != pid => {
                let new_model = model_name.unwrap_or("unknown");
                if current_model == new_model {
                    // Same model, different pipeline (e.g. pre-warm → real assignment)
                    // Just update the pipeline ID without restarting llama-server
                    info!(
                        "Pipeline reassigned: {current} → {pid} (same model {new_model}, keeping server)"
                    );
                    let port = *port;
                    self.status = InferenceStatus::Running {
                        pipeline_id: pid.to_string(),
                        model_name: new_model.to_string(),
                        port,
                    };
                    self.current_pipeline_id = Some(pid.to_string());
                    self.current_model_name = Some(new_model.to_string());
                } else {
                    // Different model → must restart
                    info!(
                        "Pipeline changed: {current} → {pid} (model {current_model} → {new_model}) — restarting"
                    );
                    self.stop_server();
                    self.current_pipeline_id = Some(pid.to_string());
                    self.current_model_name = Some(new_model.to_string());
                    self.start_server(pid, new_model);
                }
            }
            // Running but assignment cleared → only stop if not pre-warming
            (InferenceStatus::Running { pipeline_id: current, .. }, None) => {
                if current == "pre-warm" {
                    // Keep pre-warmed server running — a real assignment will come soon
                } else {
                    info!("Pipeline released — stopping inference");
                    self.stop_server();
                    self.current_pipeline_id = None;
                    self.current_model_name = None;
                }
            }
            // RPC worker running but assignment cleared → stop worker
            (InferenceStatus::RunningRpcWorker { .. }, None) => {
                info!("Pipeline released — stopping RPC worker");
                self.stop_server();
                self.current_pipeline_id = None;
                self.current_model_name = None;
            }
            // Already running same pipeline, or idle with no assignment → no-op
            _ => {}
        }

        // Check if server process is still alive (also called from fast_health_check at 1s intervals)
        self.check_process_alive();
    }

    /// Start the inference process based on the current pipeline role.
    /// - Solo/Head → llama-server (Head adds --rpc flag for remote workers)
    /// - Worker → rpc-server (listens for head node to connect)
    fn start_server(&mut self, pipeline_id: &str, model_name: &str) {
        self.status = InferenceStatus::Starting;
        self.current_model_name = Some(model_name.to_string());

        // Find the model file
        let model_path = find_model_path(model_name);
        let Some(path) = model_path else {
            let msg = format!("Model not found locally: {model_name}");
            warn!("{msg}");
            self.status = InferenceStatus::Error(msg);
            return;
        };

        match &self.current_role {
            PipelineRole::Worker { rpc_port } => {
                self.start_rpc_worker(pipeline_id, model_name, &path, *rpc_port);
            }
            PipelineRole::Head { rpc_peers } => {
                self.start_llama_server_with_rpc(pipeline_id, model_name, &path, rpc_peers.clone());
            }
            PipelineRole::Solo => {
                self.start_llama_server_solo(pipeline_id, model_name, &path);
            }
        }
    }

    /// Solo mode: standard llama-server with all optimized flags.
    fn start_llama_server_solo(&mut self, pipeline_id: &str, model_name: &str, path: &str) {
        info!("Starting llama-server (solo) on port {} with {}", self.port, path);

        let result = Command::new("llama-server")
            .args([
                "--model",
                path,
                "--port",
                &self.port.to_string(),
                "--ctx-size",
                "32768",
                "--n-gpu-layers",
                "999",
                "--flash-attn",
                "on",
                "--cont-batching",
                "--parallel",
                "2",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--batch-size",
                "2048",
                "--ubatch-size",
                "512",
                "--threads",
                "6",
                "--mlock",
                "--jinja",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn();

        self.handle_server_spawn(result, pipeline_id, model_name, path);
    }

    /// Head mode: llama-server with --rpc connecting to remote worker nodes.
    /// The head node holds the model and distributes layers to workers proportionally.
    /// Workers run rpc-server, head connects to them via TCP.
    fn start_llama_server_with_rpc(
        &mut self,
        pipeline_id: &str,
        model_name: &str,
        path: &str,
        rpc_peers: Vec<String>,
    ) {
        let rpc_list = rpc_peers.join(",");
        info!(
            "Starting llama-server (head) on port {} with RPC peers: {} model: {}",
            self.port, rpc_list, path
        );

        // Build args — same as solo plus --rpc for distributed layers
        let mut args = vec![
            "--model".to_string(),
            path.to_string(),
            "--port".to_string(),
            self.port.to_string(),
            "--ctx-size".to_string(),
            "8192".to_string(),
            "--n-gpu-layers".to_string(),
            "999".to_string(),
            "--flash-attn".to_string(),
            "on".to_string(),
            "--cont-batching".to_string(),
            "--parallel".to_string(),
            "4".to_string(),
            "--cache-type-k".to_string(),
            "q8_0".to_string(),
            "--cache-type-v".to_string(),
            "q8_0".to_string(),
            "--batch-size".to_string(),
            "2048".to_string(),
            "--ubatch-size".to_string(),
            "512".to_string(),
            "--threads".to_string(),
            "6".to_string(),
            "--mlock".to_string(),
            "--jinja".to_string(),
        ];

        // --rpc tells llama-server to distribute layers to remote rpc-server instances
        if !rpc_list.is_empty() {
            args.push("--rpc".to_string());
            args.push(rpc_list);
        }

        let result = Command::new("llama-server")
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn();

        self.handle_server_spawn(result, pipeline_id, model_name, path);
    }

    /// Worker mode: run rpc-server that listens for the head node to connect.
    /// The head node's llama-server will offload layer computation to us.
    fn start_rpc_worker(
        &mut self,
        pipeline_id: &str,
        model_name: &str,
        _path: &str,
        rpc_port: u16,
    ) {
        info!("Starting rpc-server (worker) on port {} for pipeline {}", rpc_port, pipeline_id);

        // rpc-server doesn't need a model file — the head node sends weights
        let result = Command::new("rpc-server")
            .args([
                "--host",
                "0.0.0.0",
                "--port",
                &rpc_port.to_string(),
                "--mem",
                "0", // Use all available memory
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn();

        match result {
            Ok(child) => {
                self.server_process = Some(child);
                self.status = InferenceStatus::RunningRpcWorker {
                    pipeline_id: pipeline_id.to_string(),
                    model_name: model_name.to_string(),
                    rpc_port,
                };
                info!("rpc-server started (PID: {})", self.server_process.as_ref().unwrap().id());
            }
            Err(e) => {
                let msg = format!("Failed to start rpc-server: {e}");
                error!("{msg}");
                self.status = InferenceStatus::Error(msg);
            }
        }
    }

    fn handle_server_spawn(
        &mut self,
        result: std::io::Result<Child>,
        pipeline_id: &str,
        model_name: &str,
        path: &str,
    ) {
        match result {
            Ok(child) => {
                self.server_process = Some(child);
                self.model_path = Some(path.to_string());
                self.restart_attempts = 0;
                self.restart_after = None;
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

    /// Stop the llama-server process gracefully.
    /// Sends SIGTERM first (allows pending requests to drain), then SIGKILL after 5s.
    fn stop_server(&mut self) {
        if let Some(ref mut child) = self.server_process {
            let pid = child.id();
            info!("Stopping llama-server (PID: {pid}) — sending SIGTERM");

            // Send SIGTERM for graceful shutdown
            #[cfg(unix)]
            {
                unsafe {
                    libc::kill(pid as i32, libc::SIGTERM);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = child.kill();
            }

            // Wait up to 5 seconds for graceful exit
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    _ => {
                        warn!("llama-server did not exit gracefully — sending SIGKILL");
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                }
            }
        }
        self.server_process = None;
        self.model_path = None;
        self.restart_after = None;
        self.restart_attempts = 0;
        self.status = InferenceStatus::Idle;
    }

    /// Fast process liveness check — call at 1-second intervals.
    /// Detects crashes without waiting for the next heartbeat cycle.
    pub fn check_process_alive(&mut self) -> bool {
        if self.externally_managed {
            return true;
        }
        if let Some(ref mut child) = self.server_process {
            match child.try_wait() {
                Ok(Some(exit)) => {
                    warn!("llama-server exited unexpectedly: {exit}");
                    self.server_process = None;
                    self.restart_attempts = self.restart_attempts.saturating_add(1);
                    let backoff_secs = 2u64.pow(self.restart_attempts.min(4));
                    self.restart_after = Some(
                        std::time::Instant::now() + std::time::Duration::from_secs(backoff_secs),
                    );
                    self.status = InferenceStatus::Error(format!(
                        "Server crashed (restart in {backoff_secs}s)"
                    ));
                    return false;
                }
                Ok(None) => return true, // Still running
                Err(e) => {
                    error!("Failed to check server status: {e}");
                    return false;
                }
            }
        }
        // No process to check — only an issue if we think we're running
        !matches!(
            self.status,
            InferenceStatus::Running { .. } | InferenceStatus::RunningRpcWorker { .. }
        )
    }

    /// Attempt to restart the currently assigned process after a crash.
    /// Returns true when a restart attempt was triggered.
    pub fn recover_if_needed(&mut self) -> bool {
        if self.externally_managed {
            return false;
        }
        if !matches!(self.status, InferenceStatus::Error(_)) {
            return false;
        }

        let Some(pipeline_id) = self.current_pipeline_id.clone() else {
            return false;
        };
        let Some(model_name) = self.current_model_name.clone() else {
            return false;
        };

        if let Some(restart_after) = self.restart_after {
            if std::time::Instant::now() < restart_after {
                return false;
            }
        }

        warn!(
            "[inference] Attempting automatic recovery for pipeline {} model {} (attempt #{})",
            pipeline_id, model_name, self.restart_attempts
        );
        self.start_server(&pipeline_id, &model_name);
        true
    }

    /// Check if the llama-server is healthy (responds to /health).
    /// For RPC workers, just checks that the process is alive (no HTTP endpoint).
    pub async fn health_check(&self) -> bool {
        if self.externally_managed {
            return false;
        }
        if matches!(self.status, InferenceStatus::RunningRpcWorker { .. }) {
            // RPC workers don't have an HTTP /health endpoint
            return self.server_process.is_some();
        }
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
        if self.externally_managed {
            return None;
        }
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
///
/// Model ID to filename mapping:
///   gemma-4-26b-a4b-q4 → gemma-4-26B-A4B-*.gguf
///   gemma-4-e4b-q4     → gemma-4-E4B-*.gguf
fn find_model_path(model_name: &str) -> Option<String> {
    let cache_dir = dirs::home_dir()?.join(".compute").join("models");

    if !cache_dir.exists() {
        return None;
    }

    let lower_model = model_name.to_lowercase();

    // Extract key segments from model ID for fuzzy matching
    // "gemma-4-26b-a4b-q4" → ["gemma", "4", "26b", "a4b"]
    // "gemma-4-e4b-q4" → ["gemma", "4", "e4b"]
    let segments: Vec<&str> = lower_model
        .split('-')
        .filter(|s| *s != "q4" && *s != "q8" && *s != "fp16" && *s != "q2") // Skip quantization suffix
        .collect();

    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".gguf") {
            continue;
        }

        let lower_name = name.to_lowercase();

        // Check if all key segments appear in the filename
        let all_match = segments.iter().all(|seg| lower_name.contains(seg));
        if all_match && !segments.is_empty() {
            let path = entry.path();
            // Verify file is a valid GGUF and not too small (corrupt/partial)
            if let Ok(meta) = std::fs::metadata(&path) {
                if meta.len() < 100 * 1024 * 1024 {
                    warn!(
                        "Skipping model {} — too small ({:.1} MB)",
                        name,
                        meta.len() as f64 / 1048576.0
                    );
                    continue;
                }
            }
            if !verify_gguf_magic(&path) {
                warn!("Skipping model {} — invalid GGUF header", name);
                continue;
            }
            return Some(path.to_string_lossy().to_string());
        }
    }

    // Fallback: return the first valid GGUF file found (for single-model setups)
    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".gguf") {
            let path = entry.path();
            if verify_gguf_magic(&path) {
                return Some(path.to_string_lossy().to_string());
            }
        }
    }

    None
}

/// Quick check that a file starts with the GGUF magic bytes.
fn verify_gguf_magic(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }
    magic == [0x47, 0x47, 0x55, 0x46]
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
