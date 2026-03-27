use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

/// Client for communicating with the Compute orchestrator API.
pub struct OrchestratorClient {
    base_url: String,
    client: reqwest::Client,
    node_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeRegistration {
    pub node_name: String,
    pub wallet_address: String,
    pub gpu_name: String,
    pub gpu_vram_mb: u64,
    pub gpu_backend: String,
    pub tflops_fp16: f64,
    pub cpu_cores: usize,
    pub cpu_brand: String,
    pub memory_gb: f64,
    pub os: String,
    pub arch: String,
    pub version: String,
    pub listen_port: u16,
}

#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatPayload {
    pub node_id: String,
    pub cpu_usage: f32,
    pub memory_used_gb: f64,
    pub gpu_temp: Option<u32>,
    pub gpu_usage: Option<f32>,
    pub idle_state: String,
    pub uptime_secs: u64,
    pub pipeline_stage: Option<u32>,
    pub requests_served: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegistrationResponse {
    pub node_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeartbeatResponse {
    pub status: String,
    pub assigned_pipeline: Option<PipelineAssignment>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineAssignment {
    pub pipeline_id: String,
    pub model_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub total_layers: u32,
    pub peers: Vec<PeerInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PeerInfo {
    pub node_id: String,
    pub address: String,
    pub stage: u32,
}

impl OrchestratorClient {
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();

        Self { base_url: base_url.to_string(), client, node_id: None }
    }

    /// Get the registered node ID.
    pub fn node_id(&self) -> Option<&str> {
        self.node_id.as_deref()
    }

    /// Register this node with the orchestrator.
    pub async fn register(&mut self, registration: &NodeRegistration) -> Result<String> {
        debug!("Registering node with orchestrator at {}", self.base_url);

        let resp = self
            .client
            .post(format!("{}/v1/nodes/register", self.base_url))
            .json(registration)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Registration failed ({status}): {body}");
        }

        let body: RegistrationResponse = resp.json().await?;
        self.node_id = Some(body.node_id.clone());
        Ok(body.node_id)
    }

    /// Send a heartbeat to the orchestrator.
    pub async fn heartbeat(&self, payload: &HeartbeatPayload) -> Result<HeartbeatResponse> {
        let node_id = self.node_id.as_deref().unwrap_or(&payload.node_id);

        let resp = self
            .client
            .post(format!("{}/v1/nodes/{}/heartbeat", self.base_url, node_id))
            .json(payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            anyhow::bail!("Heartbeat failed: {status}");
        }

        let body: HeartbeatResponse = resp.json().await?;
        Ok(body)
    }

    /// Check if the orchestrator is reachable.
    pub async fn health_check(&self) -> bool {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match resp {
            Ok(r) => r.status().is_success(),
            Err(e) => {
                debug!("Health check failed: {e}");
                false
            }
        }
    }

    /// Fetch the latest pipeline assignment for this node.
    pub async fn get_assignment(&self) -> Result<Option<PipelineAssignment>> {
        let node_id = self.node_id.as_deref().ok_or_else(|| anyhow::anyhow!("Not registered"))?;

        let resp = self
            .client
            .get(format!("{}/v1/nodes/{}/assignment", self.base_url, node_id))
            .send()
            .await?;

        if resp.status().is_success() {
            let body: Option<PipelineAssignment> = resp.json().await?;
            Ok(body)
        } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(None)
        } else {
            anyhow::bail!("Failed to get assignment: {}", resp.status());
        }
    }
}
