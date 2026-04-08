use anyhow::Result;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::config::Config;
use crate::hardware;
use crate::idle::{IdleDetector, IdleState};
use crate::inference::manager::{InferenceManager, InferenceStatus};
use crate::inference::stage_backend::StageBackendKind;
use crate::metrics::{Earnings, NetworkStats, PipelineStatus};
use crate::relay::{AssignmentPush, RelayClient};
use crate::stage_runtime::{StagePrototypeClient, StagePrototypeHandle, StagePrototypeSpec, start_stage_prototype};

fn spawn_ready_probe(
    port: u16,
    model_name: Option<String>,
    ws_tx: tokio::sync::mpsc::Sender<String>,
) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        for attempt in 0..120u32 {
            match client
                .get(format!("http://127.0.0.1:{port}/health"))
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    info!("[health] llama-server ready after {:.1}s", attempt as f64 * 0.5);
                    let msg = serde_json::json!({
                        "type": "node_ready",
                        "model_name": model_name,
                    })
                    .to_string();
                    let _ = ws_tx.send(msg).await;
                    return;
                }
                _ => {}
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        tracing::error!("[health] llama-server did not become ready after 60s");
    });
}

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
        let mut stage_runtime: Option<StagePrototypeHandle> = None;
        let stage_runtime_client: Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let mut sys = sysinfo::System::new_all();
        let start_time = std::time::Instant::now();
        let mut held_tps = 0.0f64;
        let mut held_tps_until: Option<std::time::Instant> = None;

        let stage_backend_kind = StageBackendKind::parse(&self.config.experimental.stage_backend);
        let prototype_stage_mode =
            self.config.experimental.stage_mode_enabled && stage_backend_kind == StageBackendKind::Prototype;

        // Pre-warm: start llama-server during splash so the model is ready for the first request.
        // Skip entirely for prototype stage mode, which does not use local llama-server.
        if !prototype_stage_mode {
            let active = &self.config.models.active_model;
            let model_name = if active == "auto" {
                // Prefer the small Gemma test model, then larger options, then first available.
                let downloaded = detect_downloaded_models();
                if downloaded.contains("gemma-4-e4b-q4") {
                    "gemma-4-e4b-q4".to_string()
                } else if downloaded.contains("gemma-4-26b-a4b-q4") {
                    "gemma-4-26b-a4b-q4".to_string()
                } else if downloaded.contains("qwen3.5-27b-q4") {
                    "qwen3.5-27b-q4".to_string()
                } else {
                    downloaded.split(',').next().unwrap_or("").to_string()
                }
            } else {
                active.clone()
            };
            if !model_name.is_empty() {
                info!("Pre-warming llama-server with model: {model_name}");
                inference_mgr.check_assignment(Some("pre-warm"), Some(&model_name));
            }
        } else {
            info!("[stage] Prototype backend enabled — skipping local llama-server pre-warm");
        }

        // Initial state — use pre-detected hardware if available, else detect now
        let hw = if self.hardware.cpu.cores > 0 {
            self.hardware.clone()
        } else {
            hardware::detect()
        };
        self.update_state(|state| {
            state.running = true;
            state.hardware = hw.clone();
        });

        info!(
            "Node: {} | GPU: {} | CPU: {} cores",
            self.config.node.name,
            self.state_rx.borrow().hardware.gpus.first().map(|g| g.name.as_str()).unwrap_or("none"),
            self.state_rx.borrow().hardware.cpu.cores
        );

        // Channel for assignment pushes from orchestrator via WebSocket
        let (assignment_tx, mut assignment_rx) = tokio::sync::mpsc::channel::<AssignmentPush>(8);
        // Channel for sending messages back through the WS (e.g. node_ready)
        let (ws_outbound_tx, ws_outbound_rx) = tokio::sync::mpsc::channel::<String>(16);

        if let InferenceStatus::Running { model_name, .. } = inference_mgr.status() {
            spawn_ready_probe(
                inference_mgr.port(),
                Some(model_name.to_string()),
                ws_outbound_tx.clone(),
            );
        }

        // Start WebSocket relay to orchestrator
        let relay = RelayClient::new(
            &self.config,
            self.shutdown.clone(),
            assignment_tx,
            ws_outbound_rx,
            stage_runtime_client.clone(),
        );
        let relay_latency = relay.last_latency_ms();
        let relay_tps = relay.last_tps();
        let relay_completed_requests = relay.completed_requests();
        let relay_is_connected = relay.is_connected();
        let relay_active_requests = relay.active_requests();
        let relay_handle = tokio::spawn(async move {
            if let Err(e) = relay.run().await {
                tracing::error!("[relay] Fatal error: {e}");
            }
        });

        let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(10)); // 10s for faster failure detection (3 missed = 30s)
        let mut metrics_interval = tokio::time::interval(Duration::from_secs(1));
        let mut idle_interval = tokio::time::interval(Duration::from_secs(2));
        let mut assignment_interval = tokio::time::interval(Duration::from_secs(30)); // Fallback only — primary is WS push
        let mut baseline_tps = 0.0f64;
        let mut served_baseline = 0u64;
        let mut last_busy_slots = 0i32;

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                info!("Shutdown signal received");
                break;
            }

            tokio::select! {
                _ = heartbeat_interval.tick() => {
                    let relay_busy_slots = relay_active_requests.load(std::sync::atomic::Ordering::Relaxed) as i32;
                    self.heartbeat(&inference_mgr, stage_runtime.as_ref(), relay_busy_slots).await;
                }
                Some(assignment) = assignment_rx.recv() => {
                    // Instant assignment push from orchestrator via WebSocket
                    info!("Assignment push: pipeline={} model={} stage={}/{}{}",
                        assignment.pipeline_id, assignment.model_name,
                        assignment.stage, assignment.total_stages,
                        assignment.shard_id.as_ref().map(|s| format!(" shard={s}")).unwrap_or_default());

                    let assignment_mode = assignment.assignment_mode.as_deref().unwrap_or(
                        if assignment.total_stages > 1 { "rpc" } else { "solo" }
                    );

                    // Set pipeline role for multi-node RPC inference
                    use crate::inference::manager::PipelineRole;
                    match assignment_mode {
                        "stage" => {
                            if !self.config.experimental.stage_mode_enabled {
                                warn!(
                                    "[stage] Ignoring stage-based assignment for shard {:?} because experimental.stage_mode_enabled=false",
                                    assignment.shard_id
                                );
                                continue;
                            }

                            inference_mgr.shutdown_server();
                            inference_mgr.set_externally_managed(true);

                            let Some(shard_id) = assignment.shard_id.clone() else {
                                warn!("[stage] Ignoring stage assignment without shard_id");
                                continue;
                            };
                            let (Some(start_layer), Some(end_layer)) = (assignment.start_layer, assignment.end_layer) else {
                                warn!("[stage] Ignoring stage assignment without explicit layer bounds");
                                continue;
                            };

                            let spec = StagePrototypeSpec {
                                pipeline_id: assignment.pipeline_id.clone(),
                                model_name: assignment.model_name.clone(),
                                shard_id,
                                start_layer: start_layer as u32,
                                end_layer: end_layer as u32,
                                stage_index: assignment.stage as u32,
                                total_stages: assignment.total_stages as u32,
                                upstream_addr: assignment.upstream_addr.clone(),
                                downstream_addr: assignment.downstream_addr.clone(),
                            };

                            let restart_required = match stage_runtime.as_ref() {
                                Some(handle) => !handle.matches(&spec),
                                None => true,
                            };

                            if restart_required {
                                if let Some(handle) = stage_runtime.take() {
                                    handle.stop().await;
                                }
                                *stage_runtime_client.lock().await = None;

                                match start_stage_prototype(&self.config, &hw, spec.clone()).await {
                                    Ok(handle) => {
                                        info!("[stage] Prototype runtime listening on {}", handle.listen_addr());
                                        *stage_runtime_client.lock().await = Some(handle.client());
                                        stage_runtime = Some(handle);
                                        let msg = serde_json::json!({
                                            "type": "node_ready",
                                            "model_name": assignment.model_name,
                                        }).to_string();
                                        let _ = ws_outbound_tx.send(msg).await;
                                    }
                                    Err(err) => {
                                        warn!("[stage] Failed to start prototype runtime: {err}");
                                        let msg = serde_json::json!({
                                            "type": "node_error",
                                            "model_name": assignment.model_name,
                                            "error": format!("stage prototype startup failed: {err}"),
                                        }).to_string();
                                        let _ = ws_outbound_tx.send(msg).await;
                                        continue;
                                    }
                                }
                            }

                            self.update_state(|state| {
                                state.pipeline.active = true;
                                state.pipeline.stage = Some(assignment.stage as u32);
                                state.pipeline.total_stages = Some(assignment.total_stages as u32);
                                state.pipeline.model = Some(assignment.model_name.clone());
                                state.inference_status = stage_runtime.as_ref().map(|h| h.status_label()).unwrap_or_else(|| "stage-prototype".into());
                            });
                            continue;
                        }
                        "rpc" if assignment.total_stages > 1 => {
                            inference_mgr.set_externally_managed(false);
                            if let Some(handle) = stage_runtime.take() {
                                handle.stop().await;
                            }
                            *stage_runtime_client.lock().await = None;
                            if assignment.stage == 0 {
                                // Head node: runs llama-server with --rpc to workers
                                let peers = assignment.rpc_peers.clone().unwrap_or_default();
                                info!("[rpc] Head node with {} RPC peers: {:?}", peers.len(), peers);
                                inference_mgr.set_role(PipelineRole::Head { rpc_peers: peers });
                            } else {
                                // Worker node: runs rpc-server for head to connect to
                                let rpc_port = assignment.rpc_port.unwrap_or(50052);
                                info!("[rpc] Worker node on RPC port {rpc_port}");
                                inference_mgr.set_role(PipelineRole::Worker { rpc_port });
                            }
                        }
                        _ => {
                            inference_mgr.set_externally_managed(false);
                            if let Some(handle) = stage_runtime.take() {
                                handle.stop().await;
                            }
                            *stage_runtime_client.lock().await = None;
                            inference_mgr.set_role(PipelineRole::Solo);
                        }
                    }

                    inference_mgr.check_assignment(
                        Some(&assignment.pipeline_id),
                        Some(&assignment.model_name),
                    );

                    // Update TUI state
                    self.update_state(|state| {
                        state.pipeline.active = true;
                        state.pipeline.stage = Some(assignment.stage as u32);
                        state.pipeline.total_stages = Some(assignment.total_stages as u32);
                        state.pipeline.model = Some(assignment.model_name.clone());
                    });

                    // Health-check loop: wait for llama-server to be ready, then notify orchestrator via WS
                    spawn_ready_probe(
                        inference_mgr.port(),
                        Some(assignment.model_name.clone()),
                        ws_outbound_tx.clone(),
                    );
                }
                _ = assignment_interval.tick() => {
                    // Fallback: only poll orchestrator when WS relay is not actively connected
                    // When WS is connected, assignments come via push (no polling needed)
                    if !relay_is_connected.load(std::sync::atomic::Ordering::Relaxed) {
                        self.check_assignment(
                            &mut inference_mgr,
                            &mut stage_runtime,
                            &stage_runtime_client,
                            &ws_outbound_tx,
                            &mut baseline_tps,
                            &mut served_baseline,
                        ).await;
                    }
                }
                _ = metrics_interval.tick() => {
                    // Fast crash detection: check process alive every 1s (not 30s)
                    if stage_runtime.is_some() {
                        inference_mgr.set_externally_managed(true);
                        if !matches!(inference_mgr.status(), InferenceStatus::Idle) {
                            warn!("[stage] Forcing shutdown of legacy inference manager while stage runtime is active");
                            inference_mgr.shutdown_server();
                            inference_mgr.set_externally_managed(true);
                        }
                        crate::relay::mark_llama_unhealthy();
                    } else {
                        if !inference_mgr.check_process_alive() {
                            crate::relay::mark_llama_unhealthy();
                            if matches!(inference_mgr.status(), InferenceStatus::Error(_)) {
                                // Notify orchestrator immediately via WS
                                let msg = serde_json::json!({"type": "node_error", "error": "llama-server crashed"}).to_string();
                                let _ = ws_outbound_tx.try_send(msg);
                                self.update_state(|state| {
                                    state.inference_status = "error: server crashed".into();
                                });
                            }
                        }

                        if inference_mgr.recover_if_needed() {
                            crate::relay::mark_llama_unhealthy();
                            let active_model = match inference_mgr.status() {
                                InferenceStatus::Running { model_name, .. } => Some(model_name.clone()),
                                InferenceStatus::RunningRpcWorker { model_name, .. } => Some(model_name.clone()),
                                _ => None,
                            };
                            spawn_ready_probe(inference_mgr.port(), active_model, ws_outbound_tx.clone());
                        }
                    }

                    let metrics = hardware::collect_live_metrics(&mut sys);
                    let uptime = start_time.elapsed().as_secs();
                    let inf_status = stage_runtime
                        .as_ref()
                        .map(|handle| handle.status_label())
                        .unwrap_or_else(|| inference_mgr.status().to_string());

                    // Read latest relay metrics
                    let latency = relay_latency.load(std::sync::atomic::Ordering::Relaxed);
                    let active_requests = relay_active_requests.load(std::sync::atomic::Ordering::Relaxed);
                    let completed_requests = relay_completed_requests.load(std::sync::atomic::Ordering::Relaxed);
                    let is_actively_inferring = active_requests > 0;
                    let last_tps_bits = relay_tps.load(std::sync::atomic::Ordering::Relaxed);
                    let last_tps = f64::from_bits(last_tps_bits);

                    // Hold the most recent measured TPS for a few seconds so the
                    // dashboard chart and numeric label can actually render it.
                    if is_actively_inferring && last_tps > 0.0 {
                        held_tps = last_tps;
                        held_tps_until = Some(std::time::Instant::now() + Duration::from_secs(5));
                        baseline_tps = last_tps;
                    }

                    let live_tps = if let Some(until) = held_tps_until {
                        if std::time::Instant::now() <= until {
                            Some(held_tps)
                        } else {
                            held_tps = 0.0;
                            held_tps_until = None;
                            Some(0.0)
                        }
                    } else if is_actively_inferring {
                        Some(baseline_tps.max(0.0))
                    } else {
                        None
                    };

                    let current_busy_slots = active_requests as i32;
                    if current_busy_slots != last_busy_slots {
                        last_busy_slots = current_busy_slots;
                        self.heartbeat(&inference_mgr, stage_runtime.as_ref(), current_busy_slots).await;
                    }

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
                        state.pipeline.active_requests = active_requests as u32;
                        state.pipeline.requests_served = served_baseline.saturating_add(completed_requests);
                    });
                }
                _ = idle_interval.tick() => {
                    // Wrap blocking subprocess calls (ioreg, pmset) in spawn_blocking
                    // to avoid stalling the async executor
                    let idle_state = {
                        let mut detector = std::mem::replace(&mut idle_detector, IdleDetector::new(self.config.node.idle_threshold_minutes));
                        let result = tokio::task::spawn_blocking(move || {
                            let state = detector.detect();
                            (detector, state)
                        }).await;
                        match result {
                            Ok((d, state)) => {
                                idle_detector = d;
                                state
                            }
                            Err(_) => IdleState::LightUse, // Default on panic
                        }
                    };
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
        if let Some(handle) = stage_runtime.take() {
            handle.stop().await;
        }
        *stage_runtime_client.lock().await = None;
        drop(inference_mgr);

        // Set node offline in the orchestrator
        let wallet = &self.config.wallet.public_address;
        if !wallet.is_empty() && !self.config.wallet.node_token.is_empty() {
            let client = compute_network::client::OrchestratorClient::new(
                &self.config.network.orchestrator_url,
                Some(self.config.wallet.node_token.clone()),
            );
            if let Err(e) = client.set_offline(wallet).await {
                tracing::warn!("Failed to set node offline: {e}");
            }
        }

        info!("Daemon stopped");
        Ok(())
    }

    /// Check if this node has been assigned to a pipeline.
    async fn check_assignment(
        &self,
        inference_mgr: &mut InferenceManager,
        stage_runtime: &mut Option<StagePrototypeHandle>,
        stage_runtime_client: &Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>>,
        ws_outbound_tx: &tokio::sync::mpsc::Sender<String>,
        baseline_tps: &mut f64,
        served_baseline: &mut u64,
    ) {
        let wallet = &self.config.wallet.public_address;
        if wallet.is_empty() {
            return;
        }

        let client = compute_network::client::OrchestratorClient::new(
            &self.config.network.orchestrator_url,
            if self.config.wallet.node_token.is_empty() {
                None
            } else {
                Some(self.config.wallet.node_token.clone())
            },
        );
        match client.get_node_by_wallet(wallet).await {
            Ok(Some(node)) => {
                // Update pipeline status in TUI
                let has_pipeline = node.pipeline_id.is_some();
                let pending = node.pending_compute.unwrap_or(0.0);

                let tps = node.tokens_per_second.unwrap_or(0.0);
                let served = node.requests_served.unwrap_or(0) as u64;
                if tps > 0.0 {
                    *baseline_tps = tps;
                }
                if served > *served_baseline {
                    *served_baseline = served;
                }
                let active_requests = node.inference_slots_busy.unwrap_or(0).max(0) as u32;

                self.update_state(|state| {
                    if has_pipeline {
                        state.pipeline.active = true;
                        state.pipeline.active_requests = active_requests;
                        state.pipeline.stage = node.pipeline_stage.map(|s| s as u32);
                        state.pipeline.total_stages = node.pipeline_total_stages.map(|s| s as u32);
                        state.pipeline.model = node.model_name.clone();
                        // Don't set tokens_per_sec here — it comes from the relay in real time.
                        state.pipeline.requests_served = *served_baseline;
                    } else {
                        state.pipeline.active = false;
                        state.pipeline.active_requests = 0;
                        state.pipeline.stage = None;
                        state.pipeline.total_stages = None;
                        state.pipeline.model = None;
                        state.pipeline.tokens_per_sec = 0.0;
                        state.pipeline.requests_served = *served_baseline;
                    }
                    state.earnings.pending = pending;
                });

                // Fetch time-bucketed earnings from reward_events
                let wallet_owned = wallet.to_string();
                let orchestrator_url = self.config.network.orchestrator_url.clone();
                let state_tx = self.state_tx.clone();
                tokio::spawn(async move {
                    let client = compute_network::client::OrchestratorClient::new(
                        &orchestrator_url,
                        None,
                    );
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
                let stage_pipeline_match = stage_runtime
                    .as_ref()
                    .map(|handle| {
                        node.pipeline_id.as_deref() == Some(handle.pipeline_id())
                            && node.model_name.as_deref() == Some(handle.model_name())
                            && node.pipeline_stage.map(|s| s as u32) == Some(handle.stage_index())
                            && node.pipeline_total_stages.map(|s| s as u32) == Some(handle.total_stages())
                    })
                    .unwrap_or(false);

                let looks_like_stage_assignment =
                    self.config.experimental.stage_mode_enabled
                        && node.model_name.as_deref() == Some("gemma-4-e4b-q4")
                        && node.pipeline_id.is_some()
                        && node.pipeline_stage.is_some()
                        && node.pipeline_total_stages == Some(2);

                let previous_pipeline_id = match inference_mgr.status() {
                    InferenceStatus::Running { pipeline_id, .. } => Some(pipeline_id.clone()),
                    InferenceStatus::RunningRpcWorker { pipeline_id, .. } => Some(pipeline_id.clone()),
                    _ => None,
                };
                let previous_model_name = match inference_mgr.status() {
                    InferenceStatus::Running { model_name, .. } => Some(model_name.clone()),
                    InferenceStatus::RunningRpcWorker { model_name, .. } => Some(model_name.clone()),
                    _ => None,
                };

                if stage_runtime.is_some() && node.pipeline_id.is_some() {
                    if !stage_pipeline_match {
                        tracing::debug!(
                            "[stage] Stage runtime active for pipeline {:?}; skipping poll-based inference manager sync",
                            node.pipeline_id
                        );
                    }
                } else if stage_pipeline_match || looks_like_stage_assignment {
                    if looks_like_stage_assignment && !stage_pipeline_match {
                        tracing::debug!(
                            "[stage] Poll observed stage assignment for pipeline {:?}; waiting for push/runtime instead of falling back to solo",
                            node.pipeline_id
                        );
                    }
                } else {
                    if stage_runtime.is_some() && node.pipeline_id.is_none() {
                        if let Some(handle) = stage_runtime.take() {
                            handle.stop().await;
                        }
                        *stage_runtime_client.lock().await = None;
                        inference_mgr.set_externally_managed(false);
                    }

                    inference_mgr.check_assignment(
                        node.pipeline_id.as_deref(),
                        node.model_name.as_deref(),
                    );
                }

                let pipeline_changed = previous_pipeline_id.as_deref() != node.pipeline_id.as_deref();
                let model_changed = previous_model_name.as_deref() != node.model_name.as_deref();
                if !looks_like_stage_assignment && node.pipeline_id.is_some() && (pipeline_changed || model_changed) {
                    // Poll-based assignment detection runs before the WS push path on startup/reconnect.
                    // Emit the same ready probe here so the orchestrator doesn't wait 120s for a
                    // node_ready message that only the push path would have produced.
                    let active_model = match inference_mgr.status() {
                        InferenceStatus::Running { model_name, .. } => Some(model_name.clone()),
                        InferenceStatus::RunningRpcWorker { model_name, .. } => Some(model_name.clone()),
                        _ => None,
                    };
                    spawn_ready_probe(inference_mgr.port(), active_model, ws_outbound_tx.clone());
                }
            }
            Ok(None) => {
                tracing::debug!("Node not found in orchestrator");
            }
            Err(e) => {
                tracing::warn!("Failed to check assignment: {e}");
            }
        }
    }

    async fn heartbeat(
        &self,
        inference_mgr: &InferenceManager,
        stage_runtime: Option<&StagePrototypeHandle>,
        relay_busy_slots: i32,
    ) {
        let state = self.state_rx.borrow().clone();
        let inf_status = inference_mgr.status();

        tracing::debug!(
            "Heartbeat | idle={} | cpu={:.0}% | inference={} | uptime={}s",
            state.idle_state,
            state.live_metrics.cpu_usage,
            inf_status,
            state.uptime_secs
        );

        // Send heartbeat to orchestrator if wallet is configured
        let wallet = &self.config.wallet.public_address;
        if !wallet.is_empty() {
            let idle_str = format!("{}", state.idle_state);

            let (pipeline_id, pipeline_stage) = if let Some(handle) = stage_runtime {
                (
                    Some(handle.pipeline_id().to_string()),
                    Some(handle.stage_index() as i32),
                )
            } else {
                match inf_status {
                    InferenceStatus::Running { pipeline_id, .. } => {
                        if pipeline_id == "pre-warm" {
                            (None, None)
                        } else {
                            (Some(pipeline_id.clone()), state.pipeline.stage.map(|s| s as i32))
                        }
                    }
                    _ => (None, None),
                }
            };

            // Get inference slot metrics for smarter orchestrator scheduling
            let inf_metrics = inference_mgr.get_metrics().await;
            let (slots_total, slots_busy) = match &inf_metrics {
                Some(m) => {
                    let total = (m.slots_idle + m.slots_processing) as i32;
                    let busy = (m.slots_processing as i32).max(relay_busy_slots);
                    (Some(total), Some(busy))
                }
                None => {
                    let total = if stage_runtime.is_some() {
                        Some(1)
                    } else if relay_busy_slots > 0 {
                        Some(2)
                    } else {
                        None
                    };
                    let busy = if stage_runtime.is_some() {
                        Some(relay_busy_slots.max(0))
                    } else if relay_busy_slots > 0 {
                        Some(relay_busy_slots)
                    } else {
                        None
                    };
                    (total, busy)
                }
            };

            // Calculate free VRAM (total - used)
            let total_mem_mb = state.hardware.gpus.first().map(|g| g.vram_mb as i64);
            let used_mem_mb = Some((state.live_metrics.memory_used_gb * 1024.0) as i64);
            let vram_free = match (total_mem_mb, used_mem_mb) {
                (Some(total), Some(used)) => Some((total - used).max(0)),
                _ => None,
            };

            let update = compute_network::client::HeartbeatPayload {
                status: "online".into(),
                cpu_usage_percent: Some(state.live_metrics.cpu_usage as f64),
                gpu_usage_percent: state.live_metrics.gpu_usage.map(|v| v as f64),
                gpu_temp_celsius: state.live_metrics.gpu_temp.map(|v| v as f64),
                memory_used_mb: used_mem_mb,
                idle_state: Some(idle_str),
                uptime_seconds: Some(state.uptime_secs as i64),
                pipeline_id,
                pipeline_stage,
                requests_served: None,
                tokens_per_second: None,
                downloaded_models: Some(detect_downloaded_models()),
                inference_slots_total: slots_total,
                inference_slots_busy: slots_busy,
                gpu_vram_free_mb: vram_free,
                ip_address: detect_advertise_ip(),
                last_heartbeat: Some(chrono::Utc::now().to_rfc3339()),
            };

            let client = compute_network::client::OrchestratorClient::new(
                &self.config.network.orchestrator_url,
                if self.config.wallet.node_token.is_empty() {
                    None
                } else {
                    Some(self.config.wallet.node_token.clone())
                },
            );
            if let Err(e) = client.heartbeat(wallet, &update).await {
                tracing::warn!("Orchestrator heartbeat failed: {e}");
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

fn detect_advertise_ip() -> Option<String> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((Ipv4Addr::new(1, 1, 1, 1), 80)).ok()?;
    let addr = socket.local_addr().ok()?;
    match addr.ip() {
        std::net::IpAddr::V4(ip) if !ip.is_loopback() && !ip.is_unspecified() => Some(ip.to_string()),
        _ => None,
    }
}

/// Check if a file is a valid GGUF by reading the magic header.
/// GGUF files start with the bytes "GGUF" (0x47475546).
fn is_valid_gguf(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }
    // GGUF magic: "GGUF" in ASCII = [0x47, 0x47, 0x55, 0x46]
    magic == [0x47, 0x47, 0x55, 0x46]
}

fn detect_downloaded_models() -> String {
    let cache_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".compute")
        .join("models");

    // Clean up stale .tmp files from interrupted downloads
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".tmp") {
                // If .tmp file is older than 1 hour, delete it
                if let Ok(meta) = entry.metadata() {
                    if let Ok(modified) = meta.modified() {
                        if modified.elapsed().map_or(false, |d| d.as_secs() > 3600) {
                            let _ = std::fs::remove_file(entry.path());
                            tracing::info!("Cleaned up stale download: {name}");
                        }
                    }
                }
            }
        }
    }

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_lowercase();

            if !name.ends_with(".gguf") {
                continue;
            }

            // Skip files that are too small to be valid models (< 100MB)
            if let Ok(meta) = entry.metadata() {
                if meta.len() < 100 * 1024 * 1024 {
                    tracing::warn!("Skipping suspiciously small GGUF: {} ({:.1} MB)", name, meta.len() as f64 / 1048576.0);
                    continue;
                }
            }

            // Verify GGUF magic header
            if !is_valid_gguf(&path) {
                tracing::warn!("Skipping corrupt/invalid GGUF (bad magic): {}", name);
                continue;
            }

            if name.contains("gemma-4-26b") {
                if !models.contains(&"gemma-4-26b-a4b-q4".to_string()) {
                    models.push("gemma-4-26b-a4b-q4".to_string());
                }
            } else if name.contains("gemma-4-e4b") {
                if !models.contains(&"gemma-4-e4b-q4".to_string()) {
                    models.push("gemma-4-e4b-q4".to_string());
                }
            } else if name.contains("qwen3.5-27b") || name.contains("qwen3_5-27b") {
                if !models.contains(&"qwen3.5-27b-q4".to_string()) {
                    models.push("qwen3.5-27b-q4".to_string());
                }
            }
        }
    }
    models.join(",")
}
