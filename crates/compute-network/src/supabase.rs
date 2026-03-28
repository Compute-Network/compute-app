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
    pub last_heartbeat: String, // ISO 8601
}

/// Network-wide stats returned from Supabase.
#[derive(Debug, Clone, Deserialize)]
pub struct NetworkStats {
    pub total_nodes: u64,
    pub online_nodes: u64,
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

        Self {
            client,
            rest_url: format!("{SUPABASE_URL}/rest/v1"),
        }
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
            .patch(format!(
                "{}/nodes?wallet_address=eq.{}",
                self.rest_url, wallet_address
            ))
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
            .patch(format!(
                "{}/nodes?wallet_address=eq.{}",
                self.rest_url, wallet_address
            ))
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
            .get(format!(
                "{}/nodes?select=id&status=eq.online",
                self.rest_url
            ))
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
            last_heartbeat: "2026-03-28T12:00:00Z".into(),
        };

        let json = serde_json::to_string(&hb).unwrap();
        assert!(json.contains("online"));
        assert!(!json.contains("pipeline_id")); // None fields skipped
    }
}
