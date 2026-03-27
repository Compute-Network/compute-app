use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Client for communicating with the Compute orchestrator API.
pub struct OrchestratorClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
pub struct NodeRegistration {
    pub node_name: String,
    pub wallet_address: String,
    pub gpu_name: String,
    pub gpu_vram_mb: u64,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub os: String,
    pub arch: String,
    pub version: String,
}

#[derive(Debug, Deserialize)]
pub struct HeartbeatResponse {
    pub status: String,
    pub assigned_pipeline: Option<String>,
}

impl OrchestratorClient {
    pub fn new(base_url: &str) -> Self {
        Self { base_url: base_url.to_string(), client: reqwest::Client::new() }
    }

    /// Register this node with the orchestrator.
    pub async fn register(&self, registration: &NodeRegistration) -> Result<String> {
        let resp = self
            .client
            .post(format!("{}/v1/nodes/register", self.base_url))
            .json(registration)
            .send()
            .await?;

        let body: serde_json::Value = resp.json().await?;
        let node_id = body["node_id"].as_str().unwrap_or("unknown").to_string();
        Ok(node_id)
    }

    /// Send a heartbeat to the orchestrator.
    pub async fn heartbeat(&self, node_id: &str) -> Result<HeartbeatResponse> {
        let resp = self
            .client
            .post(format!("{}/v1/nodes/{}/heartbeat", self.base_url, node_id))
            .send()
            .await?;

        let body: HeartbeatResponse = resp.json().await?;
        Ok(body)
    }

    /// Check if the orchestrator is reachable.
    pub async fn health_check(&self) -> Result<bool> {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await;

        Ok(resp.is_ok())
    }
}
