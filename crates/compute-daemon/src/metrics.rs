use serde::{Deserialize, Serialize};

/// Earnings data for display in the dashboard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Earnings {
    pub today: f64,
    pub this_week: f64,
    pub this_month: f64,
    pub all_time: f64,
    pub pending: f64,
    pub usd_rate: f64,
}

impl Earnings {
    /// Mock earnings for initial development.
    pub fn mock() -> Self {
        Self {
            today: 142.5,
            this_week: 891.2,
            this_month: 3_250.0,
            all_time: 12_450.0,
            pending: 23.4,
            usd_rate: 0.317,
        }
    }
}

/// Pipeline status information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub active: bool,
    pub stage: Option<u32>,
    pub total_stages: Option<u32>,
    pub model: Option<String>,
    pub requests_served: u64,
    pub avg_latency_ms: f64,
    pub tokens_per_sec: f64,
}

impl PipelineStatus {
    /// Mock pipeline status for initial development.
    pub fn mock() -> Self {
        Self {
            active: true,
            stage: Some(3),
            total_stages: Some(5),
            model: Some("Llama-3.1-70B-Q4".into()),
            requests_served: 1_247,
            avg_latency_ms: 124.0,
            tokens_per_sec: 47.2,
        }
    }
}

/// Network stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_nodes: u32,
    pub peak_petaflops: f64,
}

impl NetworkStats {
    pub fn mock() -> Self {
        Self { total_nodes: 12_847, peak_petaflops: 847.0 }
    }
}
