use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::watch;
use tracing::info;

use crate::config::Config;
use crate::hardware;
use crate::idle::{IdleDetector, IdleState};
use crate::metrics::{Earnings, NetworkStats, PipelineStatus};

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
}

impl Default for DaemonState {
    fn default() -> Self {
        Self {
            running: false,
            idle_state: IdleState::Idle,
            hardware: hardware::detect(),
            live_metrics: hardware::LiveMetrics::default(),
            earnings: Earnings::mock(),
            pipeline: PipelineStatus::mock(),
            network: NetworkStats::mock(),
            uptime_secs: 0,
        }
    }
}

/// The daemon runtime — runs the background event loop.
pub struct DaemonRuntime {
    config: Config,
    shutdown: Arc<AtomicBool>,
    state_tx: watch::Sender<DaemonState>,
    state_rx: watch::Receiver<DaemonState>,
}

impl DaemonRuntime {
    pub fn new(config: Config) -> Self {
        let (state_tx, state_rx) = watch::channel(DaemonState::default());
        Self { config, shutdown: Arc::new(AtomicBool::new(false)), state_tx, state_rx }
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
        let mut sys = sysinfo::System::new_all();
        let start_time = std::time::Instant::now();

        // Initial state
        let hw = hardware::detect();
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

        let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(30));
        let mut metrics_interval = tokio::time::interval(Duration::from_secs(5));
        let mut idle_interval = tokio::time::interval(Duration::from_secs(2));

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                info!("Shutdown signal received");
                break;
            }

            tokio::select! {
                _ = heartbeat_interval.tick() => {
                    self.heartbeat().await;
                }
                _ = metrics_interval.tick() => {
                    let metrics = hardware::collect_live_metrics(&mut sys);
                    let uptime = start_time.elapsed().as_secs();
                    self.update_state(|state| {
                        state.live_metrics = metrics;
                        state.uptime_secs = uptime;
                    });
                }
                _ = idle_interval.tick() => {
                    let idle_state = idle_detector.detect();
                    self.update_state(|state| {
                        state.idle_state = idle_state;
                    });

                    match idle_state {
                        IdleState::HeavyUse => {
                            // Would pause workloads here
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

        info!("Daemon stopped");
        Ok(())
    }

    async fn heartbeat(&self) {
        // In the future, this will send a heartbeat to the orchestrator.
        // For now, just log.
        tracing::debug!(
            "Heartbeat | idle={} | cpu={:.0}% | uptime={}s",
            self.state_rx.borrow().idle_state,
            self.state_rx.borrow().live_metrics.cpu_usage,
            self.state_rx.borrow().uptime_secs
        );
    }

    fn update_state<F>(&self, f: F)
    where
        F: FnOnce(&mut DaemonState),
    {
        self.state_tx.send_modify(f);
    }
}
