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

/// Pipeline status information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub active: bool,
    pub active_requests: u32,
    pub stage: Option<u32>,
    pub total_stages: Option<u32>,
    pub model: Option<String>,
    pub backend: String,
    pub requests_served: u64,
    pub avg_latency_ms: f64,
    pub tokens_per_sec: f64,
}

/// Network stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_nodes: u32,
    pub peak_petaflops: f64,
}
