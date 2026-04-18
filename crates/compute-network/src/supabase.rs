use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

/// Supabase project configuration.
const SUPABASE_URL: &str = "https://eqkdwakvhwzsdmnklqfs.supabase.co";
const SUPABASE_ANON_KEY: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVxa2R3YWt2aHd6c2RtbmtscWZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ2NTcwNjcsImV4cCI6MjA5MDIzMzA2N30.szE28x9LMac3_u4EqukJ8r6EggYeZNHeYbg1SchaUEc";

/// A row in the `nodes` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRow {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub wallet_address: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    // Hardware
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_vram_mb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tflops_fp16: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_capable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_gbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_backend_kind: Option<String>,
}

/// Heartbeat update payload (subset of node fields).
#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatUpdate {
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
    pub downloaded_models: Option<String>, // comma-separated model IDs
    // Inference-aware heartbeat fields (for smarter scheduling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_slots_total: Option<i32>, // --parallel value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_slots_busy: Option<i32>, // currently processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_vram_free_mb: Option<i64>, // free unified/VRAM
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_capable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_gbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_backend_kind: Option<String>,
    pub last_heartbeat: String, // ISO 8601
}

/// Network-wide stats returned from Supabase.
#[derive(Debug, Clone, Deserialize)]
pub struct NetworkStats {
    pub total_nodes: u64,
    pub online_nodes: u64,
}

/// Node info for CLI listing.
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

/// Info about own node (for checking pipeline assignments).
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

/// Minimal node info for discovery/visualization.
#[derive(Debug, Clone, Deserialize)]
pub struct OnlineNode {
    pub wallet_address: String,
    pub node_name: Option<String>,
    pub region: Option<String>,
    pub gpu_model: Option<String>,
    pub gpu_backend: Option<String>,
    pub tflops_fp16: Option<f64>,
}

/// A single reward event from the reward_events table.
#[derive(Debug, Clone, Deserialize)]
struct RewardEvent {
    pub final_reward: Option<f64>,
    pub created_at: Option<String>,
    pub status: Option<String>,
}

/// Earnings breakdown by time period.
#[derive(Debug, Clone, Default)]
pub struct EarningsData {
    pub today: f64,
    pub this_week: f64,
    pub this_month: f64,
    pub all_time: f64,
    pub pending: f64,
}

/// Client for the Supabase REST API (PostgREST).
pub struct SupabaseClient {
    client: reqwest::Client,
    rest_url: String,
}

impl SupabaseClient {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .connect_timeout(Duration::from_secs(5))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert("apikey", SUPABASE_ANON_KEY.parse().unwrap());
                headers.insert(
                    "Authorization",
                    format!("Bearer {SUPABASE_ANON_KEY}").parse().unwrap(),
                );
                headers.insert("Content-Type", "application/json".parse().unwrap());
                headers
            })
            .build()
            .unwrap_or_default();

        Self { client, rest_url: format!("{SUPABASE_URL}/rest/v1") }
    }

    /// Register a new node or update existing one (upsert by wallet_address).
    /// Returns the node UUID.
    pub async fn register_node(&self, node: &NodeRow) -> Result<String> {
        debug!("Registering node with wallet {}", node.wallet_address);

        // Use PostgREST upsert: POST with Prefer: resolution=merge-duplicates
        // This will insert if new, or update if wallet_address already exists
        let resp = self
            .client
            .post(format!("{}/nodes", self.rest_url))
            .header("Prefer", "resolution=merge-duplicates,return=representation")
            .json(node)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Node registration failed ({status}): {body}");
        }

        let rows: Vec<NodeRow> = resp.json().await?;
        let id = rows
            .first()
            .and_then(|r| r.id.clone())
            .ok_or_else(|| anyhow::anyhow!("No node ID returned from registration"))?;

        debug!("Node registered with ID: {id}");
        Ok(id)
    }

    /// Send a heartbeat update for a node identified by wallet_address.
    pub async fn heartbeat(&self, wallet_address: &str, update: &HeartbeatUpdate) -> Result<()> {
        debug!("Sending heartbeat for {wallet_address}");

        let resp = self
            .client
            .patch(format!("{}/nodes?wallet_address=eq.{}", self.rest_url, wallet_address))
            .header("Prefer", "return=minimal")
            .json(update)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            warn!("Heartbeat failed ({status}): {body}");
            anyhow::bail!("Heartbeat failed ({status}): {body}");
        }

        Ok(())
    }

    /// Mark a node as offline.
    pub async fn set_offline(&self, wallet_address: &str) -> Result<()> {
        #[derive(Serialize)]
        struct StatusUpdate {
            status: String,
        }

        let resp = self
            .client
            .patch(format!("{}/nodes?wallet_address=eq.{}", self.rest_url, wallet_address))
            .header("Prefer", "return=minimal")
            .json(&StatusUpdate { status: "offline".into() })
            .send()
            .await?;

        if !resp.status().is_success() {
            warn!("Failed to set node offline: {}", resp.status());
        }

        Ok(())
    }

    /// Get network-wide stats (total nodes, online nodes).
    pub async fn get_network_stats(&self) -> Result<NetworkStats> {
        // Count total nodes
        let resp = self
            .client
            .get(format!("{}/nodes?select=id", self.rest_url))
            .header("Prefer", "count=exact")
            .header("Range-Unit", "items")
            .header("Range", "0-0")
            .send()
            .await?;

        let total_nodes = resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split('/').next_back())
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(0);

        // Count online nodes
        let resp = self
            .client
            .get(format!("{}/nodes?select=id&status=eq.online", self.rest_url))
            .header("Prefer", "count=exact")
            .header("Range-Unit", "items")
            .header("Range", "0-0")
            .send()
            .await?;

        let online_nodes = resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split('/').next_back())
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(0);

        Ok(NetworkStats { total_nodes, online_nodes })
    }

    /// Fetch nodes for CLI listing. Optionally filter to online-only.
    pub async fn list_nodes(&self, online_only: bool, limit: usize) -> Result<Vec<NodeListEntry>> {
        let status_filter = if online_only { "&status=eq.online" } else { "" };

        let resp = self
            .client
            .get(format!(
                "{}/nodes?select=wallet_address,node_name,status,gpu_model,gpu_backend,tflops_fp16,region,last_heartbeat,uptime_seconds&order=last_heartbeat.desc.nullslast&limit={limit}{status_filter}",
                self.rest_url
            ))
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to list nodes: {}", resp.status());
        }

        let nodes: Vec<NodeListEntry> = resp.json().await?;
        Ok(nodes)
    }

    /// Fetch online nodes (for globe visualization).
    /// Returns wallet_address, node_name, region, gpu_model for each online node.
    pub async fn get_online_nodes(&self) -> Result<Vec<OnlineNode>> {
        let resp = self
            .client
            .get(format!(
                "{}/nodes?select=wallet_address,node_name,region,gpu_model,gpu_backend,tflops_fp16&status=eq.online&limit=500",
                self.rest_url
            ))
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to fetch online nodes: {}", resp.status());
        }

        let nodes: Vec<OnlineNode> = resp.json().await?;
        Ok(nodes)
    }

    /// Get this node's current record (to check for pipeline assignment).
    pub async fn get_own_node(&self, wallet_address: &str) -> Result<Option<OwnNodeInfo>> {
        let resp = self
            .client
            .get(format!(
                "{}/nodes?select=id,pipeline_id,pipeline_stage,pipeline_total_stages,model_name,pending_compute,total_earned_compute,tokens_per_second,requests_served&wallet_address=eq.{}&limit=1",
                self.rest_url, wallet_address
            ))
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to fetch node: {}", resp.status());
        }

        let rows: Vec<OwnNodeInfo> = resp.json().await?;
        Ok(rows.into_iter().next())
    }

    /// Fetch earnings breakdown from reward_events table.
    pub async fn get_earnings(&self, wallet_address: &str) -> Result<EarningsData> {
        // Fetch all reward events for this wallet
        let resp = self
            .client
            .get(format!(
                "{}/reward_events?select=final_reward,created_at,status&wallet_address=eq.{}&order=created_at.desc&limit=10000",
                self.rest_url, wallet_address
            ))
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to fetch earnings: {}", resp.status());
        }

        let events: Vec<RewardEvent> = resp.json().await?;

        let now = chrono::Utc::now();
        let today_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
        let week_start = (now - chrono::Duration::days(7)).naive_utc();
        let month_start = (now - chrono::Duration::days(30)).naive_utc();

        let mut today = 0.0;
        let mut this_week = 0.0;
        let mut this_month = 0.0;
        let mut all_time = 0.0;
        let mut pending = 0.0;

        for event in &events {
            let reward = event.final_reward.unwrap_or(0.0);
            all_time += reward;

            if event.status.as_deref() == Some("pending") {
                pending += reward;
            }

            if let Some(ref ts) = event.created_at
                && let Ok(dt) = ts.parse::<chrono::DateTime<chrono::Utc>>()
            {
                if dt.naive_utc() >= today_start {
                    today += reward;
                }
                if dt.naive_utc() >= week_start {
                    this_week += reward;
                }
                if dt.naive_utc() >= month_start {
                    this_month += reward;
                }
            }
        }

        Ok(EarningsData { today, this_week, this_month, all_time, pending })
    }

    /// Check if Supabase is reachable.
    pub async fn health_check(&self) -> bool {
        let resp = self
            .client
            .get(format!("{}/nodes?select=id&limit=1", self.rest_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match resp {
            Ok(r) => r.status().is_success(),
            Err(e) => {
                debug!("Supabase health check failed: {e}");
                false
            }
        }
    }
}

impl Default for SupabaseClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_row_serialization() {
        let node = NodeRow {
            id: None,
            wallet_address: "DRpbCBMxVnDK7maPM5tGv6MvB3v1sRMC86PZ8okm21hy".into(),
            node_name: Some("test-node".into()),
            status: Some("online".into()),
            gpu_model: Some("Apple M3 Max".into()),
            gpu_vram_mb: Some(36864),
            gpu_backend: Some("metal".into()),
            cpu_model: Some("Apple M3 Max".into()),
            cpu_cores: Some(14),
            memory_mb: Some(36864),
            os: Some("macOS 15.0".into()),
            app_version: Some("0.1.0".into()),
            region: Some("auto".into()),
            tflops_fp16: Some(14.2),
            pipeline_capable: Some(true),
            memory_bandwidth_gbps: Some(400.0),
            stage_backend_kind: Some("llama_stage_gateway".into()),
        };

        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains("DRpbCBMxVnDK7maPM5tGv6MvB3v1sRMC86PZ8okm21hy"));
        assert!(!json.contains("\"id\"")); // None fields should be skipped

        let parsed: NodeRow = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.wallet_address, node.wallet_address);
        assert_eq!(parsed.gpu_model, node.gpu_model);
    }

    #[test]
    fn test_heartbeat_serialization() {
        let hb = HeartbeatUpdate {
            status: "online".into(),
            cpu_usage_percent: Some(45.2),
            gpu_usage_percent: Some(78.0),
            gpu_temp_celsius: Some(62.0),
            memory_used_mb: Some(16384),
            idle_state: Some("idle".into()),
            uptime_seconds: Some(3600),
            pipeline_id: None,
            pipeline_stage: None,
            requests_served: None,
            tokens_per_second: None,
            downloaded_models: None,
            inference_slots_total: Some(4),
            inference_slots_busy: Some(1),
            gpu_vram_free_mb: Some(20480),
            last_heartbeat: "2026-03-28T12:00:00Z".into(),
            pipeline_capable: Some(true),
            memory_bandwidth_gbps: Some(400.0),
            stage_backend_kind: Some("llama_stage_gateway".into()),
        };

        let json = serde_json::to_string(&hb).unwrap();
        assert!(json.contains("online"));
        assert!(!json.contains("pipeline_id")); // None fields skipped
    }
}
