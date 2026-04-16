//! llama.cpp inference backend.
//!
//! Manages a llama.cpp server process that handles model loading and inference.
//! Three modes:
//! - **Docker/CUDA**: runs llama-server in a container with GPU passthrough
//! - **Native/Metal**: runs llama-server as a native process on Apple Silicon
//! - **CPU**: runs llama-server in CPU-only mode

use anyhow::{Context, Result};
use std::process::{Child, Command};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, error, info, warn};

use super::engine::*;
use crate::inference::ggml_runtime::build_ggml_launch_spec_for_backend;

/// llama.cpp based inference engine.
///
/// Manages a llama-server process and communicates via HTTP API.
/// The server handles model loading, KV cache, and token sampling.
pub struct LlamaCppEngine {
    backend: InferenceBackend,
    status: EngineStatus,
    shard_config: Option<ShardConfig>,
    server_process: Option<Child>,
    server_port: u16,
    http_client: reqwest::Client,
    metrics: InferenceMetrics,
    paused: Arc<AtomicBool>,
    nice_value: i32,
}

impl LlamaCppEngine {
    /// Create a new llama.cpp engine with the specified backend.
    pub fn new(backend: InferenceBackend) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // Long timeout for large models
            .build()
            .unwrap_or_default();

        Self {
            backend,
            status: EngineStatus::Unloaded,
            shard_config: None,
            server_process: None,
            server_port: 8090, // Default port, can be overridden
            http_client,
            metrics: InferenceMetrics::default(),
            paused: Arc::new(AtomicBool::new(false)),
            nice_value: 19, // Lowest priority by default
        }
    }

    /// Build the command to start llama-server based on backend.
    fn build_server_command(&self, config: &ShardConfig) -> Result<Command> {
        let launch =
            build_ggml_launch_spec_for_backend(self.backend, config, self.server_port, num_cpus())?;
        let mut cmd = launch.into_command();

        // Set process priority
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                let nice = self.nice_value;
                cmd.pre_exec(move || {
                    libc::setpriority(libc::PRIO_PROCESS, 0, nice);
                    Ok(())
                });
            }
        }

        Ok(cmd)
    }

    /// Check if the llama-server is healthy.
    async fn health(&self) -> bool {
        let url = format!("http://127.0.0.1:{}/health", self.server_port);
        match self.http_client.get(&url).timeout(std::time::Duration::from_secs(2)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Wait for the server to become healthy, with timeout.
    async fn wait_for_ready(&self, timeout_secs: u64) -> Result<()> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(timeout_secs);

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!(
                    "llama-server failed to start within {}s on port {}",
                    timeout_secs,
                    self.server_port
                );
            }

            if self.health().await {
                return Ok(());
            }

            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    /// Send a completion request to the llama-server.
    async fn completion(&self, prompt_tokens: &[u32]) -> Result<Vec<GeneratedToken>> {
        if self.paused.load(Ordering::Relaxed) {
            anyhow::bail!("Engine is paused");
        }

        let url = format!("http://127.0.0.1:{}/completion", self.server_port);

        // llama-server accepts prompt as text or token IDs
        let body = serde_json::json!({
            "prompt": prompt_tokens,
            "n_predict": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": false,
        });

        let resp = self
            .http_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send completion request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Completion failed ({status}): {text}");
        }

        let result: serde_json::Value = resp.json().await?;

        let content = result["content"].as_str().unwrap_or("").to_string();
        let stop = result["stop"].as_bool().unwrap_or(false);

        // Extract token IDs if available
        let tokens = result["tokens"]
            .as_array()
            .map(|arr| {
                arr.iter().filter_map(|v| v.as_u64().map(|id| id as u32)).collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let token_id = tokens.first().copied().unwrap_or(0);

        Ok(vec![GeneratedToken {
            request_id: String::new(), // Filled by caller
            token_id,
            token_text: content,
            is_finished: stop,
            logprob: None,
        }])
    }

    pub async fn generate_completion_text(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<String> {
        if self.paused.load(Ordering::Relaxed) {
            anyhow::bail!("Engine is paused");
        }

        let url = format!("http://127.0.0.1:{}/completion", self.server_port);
        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": max_tokens.unwrap_or(96),
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": false,
        });

        let resp = self
            .http_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send completion request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Completion failed ({status}): {text}");
        }

        let result: serde_json::Value = resp.json().await?;
        Ok(result["content"].as_str().unwrap_or("").to_string())
    }

    /// Send an embedding/activation request for pipeline mid-stage.
    /// This uses the /embedding endpoint to get hidden states.
    async fn get_activations(&self, input_data: &[u8], shape: &[usize]) -> Result<Vec<u8>> {
        if self.paused.load(Ordering::Relaxed) {
            anyhow::bail!("Engine is paused");
        }

        // For pipeline stages, we pass raw activation data through the model layers.
        // In a real implementation, this would use a custom llama.cpp endpoint
        // that accepts hidden states as input and returns hidden states as output.
        //
        // For now, we pass through the data as a placeholder.
        // The actual implementation requires a patched llama.cpp server with
        // a /forward endpoint that accepts raw tensor input.
        debug!("Processing activation: shape={:?}, size={}B", shape, input_data.len());

        // Placeholder: in production, this calls a custom /forward endpoint
        // on llama-server that accepts and returns raw hidden states.
        Ok(input_data.to_vec())
    }
}

impl InferenceEngine for LlamaCppEngine {
    fn backend(&self) -> InferenceBackend {
        self.backend
    }

    fn status(&self) -> EngineStatus {
        self.status.clone()
    }

    async fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        info!(
            "Loading shard: {} layers {}-{} ({} backend)",
            config.model_id, config.start_layer, config.end_layer, self.backend
        );

        self.status = EngineStatus::Loading;

        // Verify shard file exists
        if !config.shard_path.exists() {
            self.status = EngineStatus::Error("Shard file not found".into());
            anyhow::bail!("Shard file not found: {}", config.shard_path.display());
        }

        // Build and start the server
        let mut cmd = self.build_server_command(config)?;
        let child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("Failed to start llama-server")?;

        self.server_process = Some(child);
        self.shard_config = Some(config.clone());

        // Wait for server to be ready (model loading can take a while)
        let timeout = match self.backend {
            InferenceBackend::DockerCuda => 120, // Docker pull + load
            InferenceBackend::NativeMetal => 60,
            InferenceBackend::Cpu => 120, // CPU loading is slow
        };

        match self.wait_for_ready(timeout).await {
            Ok(()) => {
                info!("llama-server ready on port {}", self.server_port);
                self.status = EngineStatus::Ready;
                Ok(())
            }
            Err(e) => {
                error!("llama-server failed to start: {e}");
                self.status = EngineStatus::Error(e.to_string());
                // Kill the process if it's still running
                if let Some(ref mut child) = self.server_process {
                    let _ = child.kill();
                }
                self.server_process = None;
                Err(e)
            }
        }
    }

    async fn unload(&mut self) -> Result<()> {
        info!("Unloading inference engine");

        if let Some(ref mut child) = self.server_process {
            // Graceful shutdown
            let _ = child.kill();
            let _ = child.wait();
        }

        // If using Docker, stop the container
        if self.backend == InferenceBackend::DockerCuda {
            let container_name = format!("compute-inference-{}", self.server_port);
            let _ = Command::new("docker").args(["stop", &container_name]).output();
        }

        self.server_process = None;
        self.shard_config = None;
        self.status = EngineStatus::Unloaded;
        self.metrics = InferenceMetrics::default();

        Ok(())
    }

    async fn forward(&self, input: Activation) -> Result<ForwardResult> {
        let config =
            self.shard_config.as_ref().ok_or_else(|| anyhow::anyhow!("No shard loaded"))?;

        if self.status != EngineStatus::Ready {
            anyhow::bail!("Engine not ready (status: {:?})", self.status);
        }

        if config.is_last_stage {
            // Last stage: sample tokens from the final hidden states
            // Convert activation data back to token IDs for completion
            let tokens = self.completion(&[]).await?;
            let mut result = tokens;
            for t in &mut result {
                t.request_id = input.request_id.clone();
            }
            Ok(ForwardResult::Tokens(result))
        } else {
            // First or middle stage: run layers and output activations
            let output_data = self.get_activations(&input.data, &input.shape).await?;

            Ok(ForwardResult::Activations(Activation {
                request_id: input.request_id,
                shape: input.shape,
                data: output_data,
                seq_position: input.seq_position,
                batch_index: input.batch_index,
            }))
        }
    }

    async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let url = format!("http://127.0.0.1:{}/tokenize", self.server_port);
        let body = serde_json::json!({ "content": text });

        let resp = self.http_client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("Tokenize failed: {}", resp.status());
        }

        let result: serde_json::Value = resp.json().await?;
        let tokens = result["tokens"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|id| id as u32)).collect())
            .unwrap_or_default();

        Ok(tokens)
    }

    async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        let url = format!("http://127.0.0.1:{}/detokenize", self.server_port);
        let body = serde_json::json!({ "tokens": tokens });

        let resp = self.http_client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("Detokenize failed: {}", resp.status());
        }

        let result: serde_json::Value = resp.json().await?;
        Ok(result["content"].as_str().unwrap_or("").to_string())
    }

    fn metrics(&self) -> InferenceMetrics {
        self.metrics.clone()
    }

    async fn pause(&mut self) -> Result<()> {
        info!("Pausing inference engine");
        self.paused.store(true, Ordering::SeqCst);
        self.status = EngineStatus::Paused;
        Ok(())
    }

    async fn resume(&mut self) -> Result<()> {
        info!("Resuming inference engine");
        self.paused.store(false, Ordering::SeqCst);
        if self.shard_config.is_some() && self.server_process.is_some() {
            self.status = EngineStatus::Ready;
        } else {
            self.status = EngineStatus::Unloaded;
        }
        Ok(())
    }

    fn set_priority(&mut self, nice_value: i32) {
        self.nice_value = nice_value;
        // If process is already running, try to renice it
        #[cfg(unix)]
        if let Some(ref child) = self.server_process {
            let pid = child.id();
            unsafe {
                libc::setpriority(libc::PRIO_PROCESS, pid, nice_value);
            }
            debug!("Set inference process {pid} priority to {nice_value}");
        }
    }
}

impl Drop for LlamaCppEngine {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.server_process {
            warn!("Inference engine dropped with running process — killing");
            let _ = child.kill();
        }
    }
}

/// Get the number of physical CPU cores for thread count.
fn num_cpus() -> usize {
    sysinfo::System::physical_core_count().unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = LlamaCppEngine::new(InferenceBackend::NativeMetal);
        assert_eq!(engine.backend(), InferenceBackend::NativeMetal);
        assert_eq!(engine.status(), EngineStatus::Unloaded);
        assert_eq!(engine.nice_value, 19);
    }

    #[test]
    fn test_engine_metrics_default() {
        let engine = LlamaCppEngine::new(InferenceBackend::Cpu);
        let m = engine.metrics();
        assert_eq!(m.tokens_per_sec, 0.0);
        assert_eq!(m.active_requests, 0);
    }

    #[tokio::test]
    async fn test_pause_resume() {
        let mut engine = LlamaCppEngine::new(InferenceBackend::Cpu);
        assert_eq!(engine.status(), EngineStatus::Unloaded);

        engine.pause().await.unwrap();
        assert_eq!(engine.status(), EngineStatus::Paused);

        engine.resume().await.unwrap();
        // No shard loaded, so stays Unloaded
        assert_eq!(engine.status(), EngineStatus::Unloaded);
    }
}
