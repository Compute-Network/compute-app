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
    node_token: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ip_address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_capable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_gbps: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatPayload {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_usage_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_usage_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_temp_celsius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_used_mb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime_seconds: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_stage: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests_served: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downloaded_models: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_download_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_slots_total: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_slots_busy: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_vram_free_mb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ip_address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_capable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_gbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_heartbeat: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_backend_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_down_mbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caffeinated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegistrationResponse {
    pub node_id: String,
    #[serde(default)]
    pub node_token: Option<String>,
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
    #[serde(default)]
    pub artifact_url: Option<String>,
    #[serde(default)]
    pub artifact_sha256: Option<String>,
    #[serde(default)]
    pub artifact_size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RewardInfo {
    pub total: f64,
    pub count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct EarningsData {
    pub today: f64,
    pub this_week: f64,
    pub this_month: f64,
    pub all_time: f64,
    pub pending: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetworkStats {
    pub total_nodes: u64,
    pub online_nodes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeListEntry {
    pub wallet_address: String,
    pub node_name: Option<String>,
    pub status: Option<String>,
    pub gpu_model: Option<String>,
    pub gpu_backend: Option<String>,
    pub tflops_fp16: Option<f64>,
    pub region: Option<String>,
    pub last_heartbeat: Option<String>,
    pub uptime_seconds: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OwnNodeInfo {
    pub id: String,
    pub pipeline_id: Option<String>,
    pub pipeline_stage: Option<i32>,
    pub pipeline_total_stages: Option<i32>,
    pub model_name: Option<String>,
    pub pending_compute: Option<f64>,
    pub total_earned_compute: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub requests_served: Option<i64>,
    pub inference_slots_total: Option<i32>,
    pub inference_slots_busy: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OnlineNode {
    pub wallet_address: String,
    pub node_name: Option<String>,
    pub region: Option<String>,
    pub gpu_model: Option<String>,
    pub gpu_backend: Option<String>,
    pub tflops_fp16: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct NodesResponse {
    pub nodes: Vec<NodeListEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct RewardsSummaryResponse {
    pub today: f64,
    pub this_week: f64,
    pub this_month: f64,
    pub all_time: f64,
    pub pending: f64,
}

impl OrchestratorClient {
    pub fn new(base_url: &str, node_token: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();

        Self {
            base_url: base_url.to_string(),
            client,
            node_id: None,
            wallet_address: None,
            node_token,
        }
    }

    /// Get the registered node ID.
    pub fn node_id(&self) -> Option<&str> {
        self.node_id.as_deref()
    }

    /// Get the current node session token. Registration may rotate this to a
    /// node-bound token that should be persisted by the caller.
    pub fn node_token(&self) -> Option<&str> {
        self.node_token.as_deref()
    }

    /// Register this node with the orchestrator.
    pub async fn register(&mut self, registration: &NodeRegistration) -> Result<String> {
        debug!("Registering node with orchestrator at {}", self.base_url);

        let resp = self
            .client
            .post(format!("{}/v1/nodes/register", self.base_url))
            .bearer_auth(
                self.node_token
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Missing node session token"))?,
            )
            .json(registration)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(orchestrator_error("Registration failed", resp).await);
        }

        let body: RegistrationResponse = resp.json().await?;
        self.node_id = Some(body.node_id.clone());
        self.wallet_address = Some(registration.wallet_address.clone());
        if let Some(node_token) = body.node_token {
            self.node_token = Some(node_token);
        }
        Ok(body.node_id)
    }

    /// Send a heartbeat to the orchestrator.
    pub async fn heartbeat(&self, wallet_address: &str, payload: &HeartbeatPayload) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/v1/nodes/{}/heartbeat", self.base_url, wallet_address))
            .bearer_auth(
                self.node_token
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Missing node session token"))?,
            )
            .json(payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(orchestrator_error("Heartbeat failed", resp).await);
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
        let mut request = self
            .client
            .get(format!("{}/v1/pipelines/assignment/{}", self.base_url, node_id));
        if let Some(token) = self.node_token.as_deref() {
            request = request.bearer_auth(token);
        }
        let resp = request.send().await?;

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
        let mut request = self
            .client
            .get(format!("{}/v1/rewards/{}", self.base_url, wallet_address));
        if let Some(token) = self.node_token.as_deref() {
            request = request.bearer_auth(token);
        }
        let resp = request.send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to get rewards: {}", resp.status());
        }

        let body: RewardInfo = resp.json().await?;
        Ok(body)
    }

    pub async fn get_node_by_wallet(&self, wallet_address: &str) -> Result<Option<OwnNodeInfo>> {
        let mut request = self
            .client
            .get(format!("{}/v1/nodes/{}", self.base_url, wallet_address));
        if let Some(token) = self.node_token.as_deref() {
            request = request.bearer_auth(token);
        }
        let resp = request.send().await?;

        if resp.status().is_success() {
            let body: OwnNodeInfo = resp.json().await?;
            Ok(Some(body))
        } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(None)
        } else {
            anyhow::bail!("Failed to get node by wallet: {}", resp.status());
        }
    }

    pub async fn get_network_stats(&self) -> Result<NetworkStats> {
        let resp = self.client.get(format!("{}/v1/nodes/stats", self.base_url)).send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to fetch network stats: {}", resp.status());
        }

        let body = resp.json::<serde_json::Value>().await?;
        Ok(NetworkStats {
            total_nodes: body.get("total").and_then(|v| v.as_u64()).unwrap_or(0),
            online_nodes: body.get("online").and_then(|v| v.as_u64()).unwrap_or(0),
        })
    }

    pub async fn list_nodes(&self, online_only: bool, limit: usize) -> Result<Vec<NodeListEntry>> {
        let url = if online_only {
            format!("{}/v1/nodes?limit={}", self.base_url, limit)
        } else {
            format!("{}/v1/nodes?status=all&limit={}", self.base_url, limit)
        };
        let resp = self.client.get(url).send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to list nodes: {}", resp.status());
        }

        let body: NodesResponse = resp.json().await?;
        Ok(body.nodes)
    }

    pub async fn get_online_nodes(&self) -> Result<Vec<OnlineNode>> {
        let resp = self.client.get(format!("{}/v1/nodes", self.base_url)).send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to fetch online nodes: {}", resp.status());
        }

        let body = resp.json::<serde_json::Value>().await?;
        let nodes: Vec<OnlineNode> =
            serde_json::from_value(body["nodes"].clone()).unwrap_or_default();
        Ok(nodes)
    }

    pub async fn get_earnings(&self, wallet_address: &str) -> Result<EarningsData> {
        let mut request = self
            .client
            .get(format!("{}/v1/rewards/{}/summary", self.base_url, wallet_address));
        if let Some(token) = self.node_token.as_deref() {
            request = request.bearer_auth(token);
        }
        let resp = request.send().await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to fetch earnings summary: {}", resp.status());
        }

        let body: RewardsSummaryResponse = resp.json().await?;
        Ok(EarningsData {
            today: body.today,
            this_week: body.this_week,
            this_month: body.this_month,
            all_time: body.all_time,
            pending: body.pending,
        })
    }

    pub async fn set_offline(&self, wallet_address: &str) -> Result<()> {
        let token = self
            .node_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Missing node session token"))?;

        let resp = self
            .client
            .post(format!("{}/v1/nodes/{}/offline", self.base_url, wallet_address))
            .bearer_auth(token)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(orchestrator_error("Failed to set node offline", resp).await);
        }
        Ok(())
    }
}

async fn orchestrator_error(prefix: &str, resp: reqwest::Response) -> anyhow::Error {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        if body.contains("node session") || body.contains("wallet mismatch") {
            return anyhow::anyhow!(
                "{prefix}: node session expired or invalid. Run `compute wallet login`."
            );
        }
    }
    anyhow::anyhow!("{prefix} ({status}): {body}")
}
