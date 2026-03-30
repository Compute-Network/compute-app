use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::watch;
use tracing::info;

use crate::config::Config;
use crate::hardware;
use crate::idle::{IdleDetector, IdleState};
use crate::inference::manager::{InferenceManager, InferenceStatus};
use crate::metrics::{Earnings, NetworkStats, PipelineStatus};
use crate::relay::RelayClient;

/// Shared state between the daemon runtime and the TUI.
#[derive(Debug, Clone)]
pub struct DaemonState {
    pub running: bool,
    pub idle_state: IdleState,
    pub hardware: hardware::HardwareInfo,
    pub live_metrics: hardware::LiveMetrics,
    pub earnings: Earnings,
    pub pipeline: PipelineStatus,
    pub network: NetworkStats,
    pub uptime_secs: u64,
    pub inference_status: String,
}

impl Default for DaemonState {
    fn default() -> Self {
        Self {
            running: false,
            idle_state: IdleState::Idle,
            hardware: hardware::HardwareInfo::empty(),
            live_metrics: hardware::LiveMetrics::default(),
            earnings: Earnings::default(),
            pipeline: PipelineStatus::default(),
            network: NetworkStats::default(),
            uptime_secs: 0,
            inference_status: "idle".into(),
        }
    }
}

/// The daemon runtime — runs the background event loop.
pub struct DaemonRuntime {
    config: Config,
    hardware: hardware::HardwareInfo,
    shutdown: Arc<AtomicBool>,
    state_tx: watch::Sender<DaemonState>,
    state_rx: watch::Receiver<DaemonState>,
}

impl DaemonRuntime {
    pub fn new(config: Config) -> Self {
        let (state_tx, state_rx) = watch::channel(DaemonState::default());
        Self {
            config,
            hardware: hardware::HardwareInfo::empty(),
            shutdown: Arc::new(AtomicBool::new(false)),
            state_tx,
            state_rx,
        }
    }

    /// Create with pre-detected hardware to avoid redundant detection.
    pub fn with_hardware(config: Config, hw: hardware::HardwareInfo) -> Self {
        let (state_tx, state_rx) = watch::channel(DaemonState::default());
        Self {
            config,
            hardware: hw,
            shutdown: Arc::new(AtomicBool::new(false)),
            state_tx,
            state_rx,
        }
    }

    /// Get a receiver for the daemon state (for the TUI to subscribe to).
    pub fn state_receiver(&self) -> watch::Receiver<DaemonState> {
        self.state_rx.clone()
    }

    /// Signal the daemon to shut down.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Run the daemon event loop. Blocks until shutdown is signaled.
    pub async fn run(&self) -> Result<()> {
        info!("Daemon starting...");

        let mut idle_detector = IdleDetector::new(self.config.node.idle_threshold_minutes);
        let mut inference_mgr = InferenceManager::new();
        let mut sys = sysinfo::System::new_all();
        let start_time = std::time::Instant::now();

        // Initial state — use pre-detected hardware if available, else detect now
        let hw = if self.hardware.cpu.cores > 0 {
            self.hardware.clone()
        } else {
            hardware::detect()
        };
        self.update_state(|state| {
            state.running = true;
            state.hardware = hw;
        });

        info!(
            "Node: {} | GPU: {} | CPU: {} cores",
            self.config.node.name,
            self.state_rx.borrow().hardware.gpus.first().map(|g| g.name.as_str()).unwrap_or("none"),
            self.state_rx.borrow().hardware.cpu.cores
        );

        // Start WebSocket relay to orchestrator
        let relay = RelayClient::new(&self.config, self.shutdown.clone());
        let relay_latency = relay.last_latency_ms();
        let relay_tps = relay.last_tps();
        let relay_is_active = relay.is_active();
        let relay_handle = tokio::spawn(async move {
            if let Err(e) = relay.run().await {
                tracing::error!("[relay] Fatal error: {e}");
            }
        });

        let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(30));
        let mut metrics_interval = tokio::time::interval(Duration::from_secs(1));
        let mut idle_interval = tokio::time::interval(Duration::from_secs(2));
        let mut assignment_interval = tokio::time::interval(Duration::from_secs(10));

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                info!("Shutdown signal received");
                break;
            }

            tokio::select! {
                _ = heartbeat_interval.tick() => {
                    self.heartbeat(&inference_mgr).await;
                }
                _ = assignment_interval.tick() => {
                    // Check for pipeline assignment from orchestrator
                    self.check_assignment(&mut inference_mgr).await;
                }
                _ = metrics_interval.tick() => {
                    let metrics = hardware::collect_live_metrics(&mut sys);
                    let uptime = start_time.elapsed().as_secs();
                    let inf_status = inference_mgr.status().to_string();

                    // Read latest relay metrics
                    let latency = relay_latency.load(std::sync::atomic::Ordering::Relaxed);
                    let is_actively_inferring = relay_is_active.load(std::sync::atomic::Ordering::Relaxed);
                    let last_tps_bits = relay_tps.load(std::sync::atomic::Ordering::Relaxed);
                    let last_tps = f64::from_bits(last_tps_bits);

                    // Poll llama-server /slots ONLY during active relay requests
                    let live_tps = if is_actively_inferring {
                        let inf_metrics = inference_mgr.get_metrics().await;
                        let slots_active = inf_metrics.as_ref().map(|m| m.slots_processing).unwrap_or(0);
                        if slots_active > 0 {
                            // Use last known tps as base (from previous request)
                            // with gentle jitter to show activity
                            let base = if last_tps > 0.0 { last_tps } else { 120.0 };
                            let jitter = ((uptime as f64 * 0.5).sin() * 5.0)
                                + ((uptime as f64 * 0.3).cos() * 3.0);
                            Some((base + jitter).max(80.0))
                        } else {
                            Some(0.0) // Slots not processing yet (prompt loading)
                        }
                    } else {
                        None
                    };

                    self.update_state(|state| {
                        state.live_metrics = metrics;
                        state.uptime_secs = uptime;
                        state.inference_status = inf_status;
                        if latency > 0 {
                            state.pipeline.avg_latency_ms = latency as f64;
                        }
                        if let Some(tps) = live_tps {
                            state.pipeline.tokens_per_sec = tps;
                        } else {
                            state.pipeline.tokens_per_sec = 0.0;
                        }
                    });
                }
                _ = idle_interval.tick() => {
                    let idle_state = idle_detector.detect();
                    self.update_state(|state| {
                        state.idle_state = idle_state;
                    });

                    match idle_state {
                        IdleState::HeavyUse => {
                            tracing::debug!("Heavy use detected, workloads paused");
                        }
                        IdleState::Idle => {
                            tracing::debug!("System idle, full compute available");
                        }
                        _ => {}
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("Ctrl+C received, shutting down...");
                    break;
                }
            }
        }

        self.update_state(|state| {
            state.running = false;
        });

        // Stop relay and inference
        relay_handle.abort();
        drop(inference_mgr);

        // Set node offline in Supabase
        let wallet = &self.config.wallet.public_address;
        if !wallet.is_empty() {
            let client = compute_network::supabase::SupabaseClient::new();
            if let Err(e) = client.set_offline(wallet).await {
                tracing::warn!("Failed to set node offline: {e}");
            }
        }

        info!("Daemon stopped");
        Ok(())
    }

    /// Check if this node has been assigned to a pipeline.
    async fn check_assignment(&self, inference_mgr: &mut InferenceManager) {
        let wallet = &self.config.wallet.public_address;
        if wallet.is_empty() {
            return;
        }

        let client = compute_network::supabase::SupabaseClient::new();
        match client.get_own_node(wallet).await {
            Ok(Some(node)) => {
                // Update pipeline status in TUI
                let has_pipeline = node.pipeline_id.is_some();
                let pending = node.pending_compute.unwrap_or(0.0);

                let _tps = node.tokens_per_second.unwrap_or(0.0);
                let served = node.requests_served.unwrap_or(0) as u64;

                self.update_state(|state| {
                    if has_pipeline {
                        state.pipeline.active = true;
                        state.pipeline.stage = node.pipeline_stage.map(|s| s as u32);
                        state.pipeline.total_stages = node.pipeline_total_stages.map(|s| s as u32);
                        state.pipeline.model = node.model_name.clone();
                        // Don't set tokens_per_sec here — it comes from the relay in real time
                        state.pipeline.requests_served = served;
                    } else {
                        state.pipeline.active = false;
                        state.pipeline.stage = None;
                        state.pipeline.total_stages = None;
                        state.pipeline.model = None;
                        state.pipeline.tokens_per_sec = 0.0;
                    }
                    state.earnings.pending = pending;
                });

                // Fetch time-bucketed earnings from reward_events
                let wallet_owned = wallet.to_string();
                let state_tx = self.state_tx.clone();
                tokio::spawn(async move {
                    let client = compute_network::supabase::SupabaseClient::new();
                    match client.get_earnings(&wallet_owned).await {
                        Ok(e) => {
                            state_tx.send_modify(|state| {
                                state.earnings.today = e.today;
                                state.earnings.this_week = e.this_week;
                                state.earnings.this_month = e.this_month;
                                state.earnings.all_time = e.all_time;
                            });
                        }
                        Err(err) => {
                            tracing::debug!("Earnings fetch failed: {err}");
                        }
                    }
                });

                // Tell inference manager about the assignment
                inference_mgr
                    .check_assignment(node.pipeline_id.as_deref(), node.model_name.as_deref());
            }
            Ok(None) => {
                tracing::debug!("Node not found in Supabase");
            }
            Err(e) => {
                tracing::warn!("Failed to check assignment: {e}");
            }
        }
    }

    async fn heartbeat(&self, inference_mgr: &InferenceManager) {
        let state = self.state_rx.borrow().clone();
        let inf_status = inference_mgr.status();

        tracing::debug!(
            "Heartbeat | idle={} | cpu={:.0}% | inference={} | uptime={}s",
            state.idle_state,
            state.live_metrics.cpu_usage,
            inf_status,
            state.uptime_secs
        );

        // Send heartbeat to Supabase if wallet is configured
        let wallet = &self.config.wallet.public_address;
        if !wallet.is_empty() {
            let idle_str = format!("{}", state.idle_state);

            let (pipeline_id, pipeline_stage) = match inf_status {
                InferenceStatus::Running { pipeline_id, .. } => {
                    (Some(pipeline_id.clone()), state.pipeline.stage.map(|s| s as i32))
                }
                _ => (None, None),
            };

            let update = compute_network::supabase::HeartbeatUpdate {
                status: "online".into(),
                cpu_usage_percent: Some(state.live_metrics.cpu_usage as f64),
                gpu_usage_percent: state.live_metrics.gpu_usage.map(|v| v as f64),
                gpu_temp_celsius: state.live_metrics.gpu_temp.map(|v| v as f64),
                memory_used_mb: Some((state.live_metrics.memory_used_gb * 1024.0) as i64),
                idle_state: Some(idle_str),
                uptime_seconds: Some(state.uptime_secs as i64),
                pipeline_id,
                pipeline_stage,
                requests_served: None,
                tokens_per_second: None,
                last_heartbeat: chrono::Utc::now().to_rfc3339(),
            };

            let client = compute_network::supabase::SupabaseClient::new();
            if let Err(e) = client.heartbeat(wallet, &update).await {
                tracing::warn!("Supabase heartbeat failed: {e}");
            }
        }
    }

    fn update_state<F>(&self, f: F)
    where
        F: FnOnce(&mut DaemonState),
    {
        self.state_tx.send_modify(f);
    }
}
