use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

/// Client for communicating with the Compute orchestrator API.
pub struct OrchestratorClient {
    base_url: String,
    client: reqwest::Client,
    node_id: Option<String>,
    wallet_address: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeRegistration {
    pub wallet_address: String,
    pub node_name: Option<String>,
    pub gpu_model: Option<String>,
    pub gpu_vram_mb: Option<u64>,
    pub gpu_backend: Option<String>,
    pub tflops_fp16: Option<f64>,
    pub cpu_model: Option<String>,
    pub cpu_cores: Option<usize>,
    pub memory_mb: Option<u64>,
    pub os: Option<String>,
    pub app_version: Option<String>,
    pub region: Option<String>,
    pub listen_port: Option<u16>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatPayload {
    pub status: String,
    pub cpu_usage_percent: Option<f64>,
    pub gpu_usage_percent: Option<f64>,
    pub gpu_temp_celsius: Option<f64>,
    pub memory_used_mb: Option<i64>,
    pub idle_state: Option<String>,
    pub uptime_seconds: Option<i64>,
    pub pipeline_id: Option<String>,
    pub pipeline_stage: Option<i32>,
    pub requests_served: Option<i64>,
    pub tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegistrationResponse {
    pub node_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineAssignment {
    pub pipeline_id: String,
    pub model_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub total_layers: u32,
    pub upstream_addr: Option<String>,
    pub downstream_addr: Option<String>,
    pub stage_index: u32,
    pub total_stages: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RewardInfo {
    pub total: f64,
    pub count: u64,
}

impl OrchestratorClient {
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();

        Self { base_url: base_url.to_string(), client, node_id: None, wallet_address: None }
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
        self.wallet_address = Some(registration.wallet_address.clone());
        Ok(body.node_id)
    }

    /// Send a heartbeat to the orchestrator.
    pub async fn heartbeat(&self, wallet_address: &str, payload: &HeartbeatPayload) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/v1/nodes/{}/heartbeat", self.base_url, wallet_address))
            .json(payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            anyhow::bail!("Heartbeat failed: {status}");
        }

        Ok(())
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
    pub async fn get_assignment(&self, node_id: &str) -> Result<Option<PipelineAssignment>> {
        let resp = self
            .client
            .get(format!("{}/v1/pipelines/assignment/{}", self.base_url, node_id))
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

    /// Get pending rewards for a wallet.
    pub async fn get_rewards(&self, wallet_address: &str) -> Result<RewardInfo> {
        let resp = self
            .client
            .get(format!("{}/v1/rewards/{}", self.base_url, wallet_address))
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to get rewards: {}", resp.status());
        }

        let body: RewardInfo = resp.json().await?;
        Ok(body)
    }
}
