use anyhow::Result;
use std::net::{Ipv4Addr, UdpSocket};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::config::Config;
use crate::hardware;
use crate::idle::{IdleDetector, IdleState};
use crate::inference::llama_stage_gateway::LlamaStageGatewayRelayClient;
use crate::inference::manager::{InferenceManager, InferenceStatus};
use crate::inference::stage_backend::StageBackendKind;
use crate::metrics::{Earnings, NetworkStats, PipelineStatus};
use crate::relay::{AssignmentPush, RelayClient};
use crate::stage_runtime::{
    StagePrototypeClient, StagePrototypeHandle, StagePrototypeSpec, start_stage_prototype,
};
use llama_stage_backend::{
    ManagedGatewayLaunchSpec, ManagedGatewayStack, ManagedHeadGatewayStack, ManagedTailNode,
    default_gemma_model_path,
};

/// Default bind for the gateway-tail worker on a 2-machine split.
/// The orchestrator can override via `gateway_tail_bind` on the assignment.
const DEFAULT_GATEWAY_TAIL_BIND: &str = "0.0.0.0:9182";

/// Captures whether the currently-running split-gateway role/topology already
/// matches a fresh assignment so we don't re-spawn workers on benign repeats.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitGatewaySignature {
    role: String,
    pipeline_id: String,
    model_name: String,
    start_layer: i32,
    end_layer: i32,
    total_stages: i32,
    tail_addr: Option<String>,
    tail_bind: Option<String>,
    shard_url: Option<String>,
}

fn classify_gateway_error_message(message: &str) -> (&'static str, String) {
    if message.contains("protocol mismatch") {
        ("protocol_mismatch", format!("stage gateway protocol/version mismatch: {message}"))
    } else if message.contains("model mismatch") {
        ("model_mismatch", format!("stage gateway model mismatch: {message}"))
    } else if message.contains("tcp gateway error:")
        || message.contains("expected info response")
        || message.contains("head endpoint")
        || message.contains("tail endpoint")
    {
        ("gateway_unusable", format!("stage gateway is reachable but unusable: {message}"))
    } else if message.contains("resolving ")
        || message.contains("no socket addresses resolved")
        || message.contains("connecting to ")
    {
        ("connect_failure", format!("stage gateway connect failed: {message}"))
    } else {
        ("gateway_error", format!("stage gateway error: {message}"))
    }
}

fn classify_gateway_error(err: &anyhow::Error) -> (&'static str, String) {
    let chain = err.chain().map(ToString::to_string).collect::<Vec<_>>().join(": ");
    classify_gateway_error_message(&chain)
}

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

async fn send_gateway_ready_or_error(
    gateway_result: anyhow::Result<LlamaStageGatewayRelayClient>,
    expected_model: &str,
    ws_tx: &tokio::sync::mpsc::Sender<String>,
) {
    let msg = match gateway_result {
        Ok(client) => match client.model_name() {
            Ok(model_name) if model_name == expected_model => serde_json::json!({
                "type": "node_ready",
                "model_name": model_name,
            })
            .to_string(),
            Ok(model_name) => serde_json::json!({
                "type": "node_error",
                "model_name": expected_model,
                "error": format!("stage gateway model mismatch: expected {expected_model}, got {model_name}"),
            })
            .to_string(),
            Err(err) => {
                let (kind, detail) = classify_gateway_error(&err);
                warn!("[gateway] {kind}: {detail}");
                serde_json::json!({
                "type": "node_error",
                "model_name": expected_model,
                "error": detail,
                })
                .to_string()
            }
        },
        Err(err) => {
            let (kind, detail) = classify_gateway_error(&err);
            warn!("[gateway] {kind}: {detail}");
            serde_json::json!({
            "type": "node_error",
            "model_name": expected_model,
            "error": detail,
            })
            .to_string()
        }
    };
    let _ = ws_tx.send(msg).await;
}

fn gateway_assignment_supported(
    stage: i32,
    total_stages: i32,
    assignment_mode: Option<&str>,
) -> bool {
    // Solo (single-stage) gateway path
    if stage == 0 && total_stages == 1 && !matches!(assignment_mode, Some("stage") | Some("rpc")) {
        return true;
    }
    // 2-machine split: orchestrator picks one head and one tail by mode tag.
    matches!(assignment_mode, Some("gateway-head") | Some("gateway-tail"))
}

fn gateway_model_path(config: &Config) -> PathBuf {
    if config.experimental.stage_gateway_model_path.trim().is_empty() {
        default_gemma_model_path()
    } else {
        config.experimental.stage_gateway_model_path.clone().into()
    }
}

/// Resolves the (model_path, local_start_layer, local_end_layer) that the
/// managed gateway stack should load for a split assignment.
///
/// When `shard_url` is set, the per-stage GGUF has renumbered layers 0..(N-1),
/// so the bounds we pass to llama-stage-backend must be LOCAL indices even
/// though the assignment carries original (shard-global) indices.
async fn resolve_gateway_stage_inputs(
    config: &Config,
    model_name: &str,
    role: &str,
    shard_url: Option<&str>,
    start_layer: i32,
    end_layer: i32,
) -> anyhow::Result<(PathBuf, u32, u32)> {
    match shard_url {
        Some(url) if !url.is_empty() => {
            let path = crate::inference::stage_artifacts::ensure_gguf_shard(
                model_name,
                role,
                start_layer as u32,
                end_layer as u32,
                url,
            )
            .await?;
            let local_end = (end_layer - start_layer) as u32;
            Ok((path, 0, local_end))
        }
        _ => Ok((gateway_model_path(config), start_layer as u32, end_layer as u32)),
    }
}

fn gateway_launch_spec(config: &Config) -> ManagedGatewayLaunchSpec {
    ManagedGatewayLaunchSpec {
        stage_node_bin: (!config.experimental.stage_gateway_stage_node_bin.trim().is_empty())
            .then(|| config.experimental.stage_gateway_stage_node_bin.clone().into()),
        gateway_bin: (!config.experimental.stage_gateway_gateway_bin.trim().is_empty())
            .then(|| config.experimental.stage_gateway_gateway_bin.clone().into()),
        ..ManagedGatewayLaunchSpec::default()
    }
}

fn configured_gateway_addr(
    config: &Config,
    managed_gateway_stack: Option<&ManagedGatewayStack>,
) -> Option<String> {
    let configured = config.experimental.stage_gateway_addr.trim();
    if !configured.is_empty() {
        Some(configured.to_string())
    } else {
        managed_gateway_stack.map(|stack| stack.gateway_addr().to_string())
    }
}

fn gateway_connect_timeout(config: &Config) -> Duration {
    Duration::from_millis(config.experimental.stage_gateway_connect_timeout_ms.max(1))
}

fn gateway_retry_window(config: &Config) -> Duration {
    Duration::from_millis(config.experimental.stage_gateway_retry_window_ms.max(1))
}

fn gateway_retry_interval(config: &Config) -> Duration {
    Duration::from_millis(config.experimental.stage_gateway_retry_interval_ms.max(1))
}

fn gateway_startup_grace(config: &Config) -> Duration {
    Duration::from_millis(config.experimental.stage_gateway_startup_grace_ms)
}

fn gateway_addr_from_config(config: &Config) -> Option<String> {
    let configured = config.experimental.stage_gateway_addr.trim();
    (!configured.is_empty()).then(|| configured.to_string())
}

async fn attempt_gateway_client_once(
    config: &Config,
    gateway_client: &Arc<tokio::sync::Mutex<Option<LlamaStageGatewayRelayClient>>>,
    managed_gateway_stack: &mut Option<ManagedGatewayStack>,
) -> anyhow::Result<LlamaStageGatewayRelayClient> {
    if let Some(client) = gateway_client.lock().await.clone() {
        return Ok(client);
    }

    let mut gateway_addr = configured_gateway_addr(config, managed_gateway_stack.as_ref());
    if gateway_addr.is_none() && config.experimental.stage_gateway_autostart {
        let model_path = gateway_model_path(config);
        let launch_spec = gateway_launch_spec(config);
        let stack =
            ManagedGatewayStack::spawn_local_with_spec(model_path.clone(), false, &launch_spec)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "failed to auto-start local gateway stack with model {}: {err}",
                        model_path.display()
                    )
                })?;
        let addr = stack.gateway_addr().to_string();
        info!(
            "[gateway] Auto-started local gateway stack at {} with model {}",
            addr,
            model_path.display()
        );
        *managed_gateway_stack = Some(stack);
        gateway_addr = Some(addr);
    }

    let Some(gateway_addr) = gateway_addr else {
        anyhow::bail!(
            "neither experimental.stage_gateway_addr nor experimental.stage_gateway_autostart is configured"
        );
    };

    let client = LlamaStageGatewayRelayClient::connect_with_timeout(
        &gateway_addr,
        Some(gateway_connect_timeout(config)),
    )?;
    info!("[gateway] Connected relay client to {}", client.addr());
    *gateway_client.lock().await = Some(client.clone());
    Ok(client)
}

fn spawn_gateway_startup_grace_connect(
    config: Config,
    gateway_client: Arc<tokio::sync::Mutex<Option<LlamaStageGatewayRelayClient>>>,
) {
    let Some(gateway_addr) = gateway_addr_from_config(&config) else {
        return;
    };
    let startup_grace = gateway_startup_grace(&config);
    if startup_grace.is_zero() || config.experimental.stage_gateway_autostart {
        return;
    }

    tokio::spawn(async move {
        let connect_timeout = gateway_connect_timeout(&config);
        let retry_interval = gateway_retry_interval(&config);
        let deadline = tokio::time::Instant::now() + startup_grace;

        loop {
            if gateway_client.lock().await.is_some() {
                return;
            }

            match LlamaStageGatewayRelayClient::connect_with_timeout(
                &gateway_addr,
                Some(connect_timeout),
            ) {
                Ok(client) => {
                    info!(
                        "[gateway] Connected relay client to {} during startup grace",
                        client.addr()
                    );
                    *gateway_client.lock().await = Some(client);
                    return;
                }
                Err(err) if tokio::time::Instant::now() >= deadline => {
                    let (kind, detail) = classify_gateway_error(&err);
                    warn!(
                        "[gateway] {kind} after startup grace {}ms: {detail}",
                        startup_grace.as_millis()
                    );
                    return;
                }
                Err(_) => {
                    tokio::time::sleep(retry_interval).await;
                }
            }
        }
    });
}

async fn ensure_gateway_client(
    config: &Config,
    gateway_client: &Arc<tokio::sync::Mutex<Option<LlamaStageGatewayRelayClient>>>,
    managed_gateway_stack: &mut Option<ManagedGatewayStack>,
) -> anyhow::Result<LlamaStageGatewayRelayClient> {
    if let Some(client) = gateway_client.lock().await.clone() {
        return Ok(client);
    }

    let mut gateway_addr = configured_gateway_addr(config, managed_gateway_stack.as_ref());
    if gateway_addr.is_none() && config.experimental.stage_gateway_autostart {
        let model_path = gateway_model_path(config);
        let launch_spec = gateway_launch_spec(config);
        let stack =
            ManagedGatewayStack::spawn_local_with_spec(model_path.clone(), false, &launch_spec)
                .map_err(|err| {
                    anyhow::anyhow!(
                        "failed to auto-start local gateway stack with model {}: {err}",
                        model_path.display()
                    )
                })?;
        let addr = stack.gateway_addr().to_string();
        info!(
            "[gateway] Auto-started local gateway stack at {} with model {}",
            addr,
            model_path.display()
        );
        *managed_gateway_stack = Some(stack);
        gateway_addr = Some(addr);
    }

    let Some(gateway_addr) = gateway_addr else {
        anyhow::bail!(
            "neither experimental.stage_gateway_addr nor experimental.stage_gateway_autostart is configured"
        );
    };

    let connect_timeout = gateway_connect_timeout(config);
    let retry_window = gateway_retry_window(config);
    let retry_interval = gateway_retry_interval(config);
    let deadline = tokio::time::Instant::now() + retry_window;
    loop {
        match LlamaStageGatewayRelayClient::connect_with_timeout(
            &gateway_addr,
            Some(connect_timeout),
        ) {
            Ok(client) => {
                info!("[gateway] Connected relay client to {}", client.addr());
                *gateway_client.lock().await = Some(client.clone());
                return Ok(client);
            }
            Err(err) if tokio::time::Instant::now() >= deadline => {
                *gateway_client.lock().await = None;
                let (_, detail) = classify_gateway_error(&err);
                anyhow::bail!(
                    "{} after retrying {} for {}ms (attempt timeout {}ms, retry interval {}ms)",
                    detail,
                    gateway_addr,
                    retry_window.as_millis(),
                    connect_timeout.as_millis(),
                    retry_interval.as_millis()
                );
            }
            Err(_) => {
                tokio::time::sleep(retry_interval).await;
            }
        }
    }
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

        // Sweep any orphaned stage-node / gateway children left behind by a
        // previous daemon crash. These processes only ever run as our children,
        // so it's safe to reap them unconditionally at startup — otherwise
        // they hold ports (e.g. 9182) and the next spawn fails.
        sweep_orphan_stage_children();

        let mut idle_detector = IdleDetector::new(self.config.node.idle_threshold_minutes);
        let mut inference_mgr = InferenceManager::new();
        let mut stage_runtime: Option<StagePrototypeHandle> = None;
        let mut managed_gateway_stack: Option<ManagedGatewayStack> = None;
        // Split-gateway state: tail-only worker on one machine, head+gateway on the other.
        let mut managed_tail_node: Option<ManagedTailNode> = None;
        let mut managed_head_gateway_stack: Option<ManagedHeadGatewayStack> = None;
        let mut current_split_signature: Option<SplitGatewaySignature> = None;
        let stage_runtime_client: Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let gateway_client: Arc<tokio::sync::Mutex<Option<LlamaStageGatewayRelayClient>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let mut sys = sysinfo::System::new_all();
        let start_time = std::time::Instant::now();
        let mut held_tps = 0.0f64;
        let mut held_tps_until: Option<std::time::Instant> = None;

        let stage_backend_kind = StageBackendKind::parse(&self.config.experimental.stage_backend);
        let prototype_stage_mode = self.config.experimental.stage_mode_enabled
            && matches!(
                stage_backend_kind,
                StageBackendKind::Prototype
                    | StageBackendKind::TailLlama
                    | StageBackendKind::LlamaStageGateway
                    | StageBackendKind::RealForward
            );

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
        let hw =
            if self.hardware.cpu.cores > 0 { self.hardware.clone() } else { hardware::detect() };
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

        if matches!(stage_backend_kind, StageBackendKind::LlamaStageGateway) {
            inference_mgr.shutdown_server();
            inference_mgr.set_externally_managed(true);
            if let Err(err) = attempt_gateway_client_once(
                &self.config,
                &gateway_client,
                &mut managed_gateway_stack,
            )
            .await
            {
                let startup_grace = gateway_startup_grace(&self.config);
                let (_, detail) = classify_gateway_error(&err);
                if startup_grace.is_zero() || gateway_addr_from_config(&self.config).is_none() {
                    warn!("[gateway] Initial gateway connection unavailable: {detail}");
                } else {
                    info!(
                        "[gateway] Initial gateway unavailable; entering startup grace {}ms: {}",
                        startup_grace.as_millis(),
                        detail
                    );
                    spawn_gateway_startup_grace_connect(
                        self.config.clone(),
                        gateway_client.clone(),
                    );
                }
            }
        }

        // Start WebSocket relay to orchestrator
        let relay = RelayClient::new(
            &self.config,
            self.shutdown.clone(),
            assignment_tx,
            ws_outbound_rx,
            stage_runtime_client.clone(),
            gateway_client.clone(),
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

                    if matches!(stage_backend_kind, StageBackendKind::LlamaStageGateway)
                        && gateway_assignment_supported(
                            assignment.stage,
                            assignment.total_stages,
                            assignment.assignment_mode.as_deref(),
                        )
                    {
                        if let Some(handle) = stage_runtime.take() {
                            handle.stop().await;
                        }
                        *stage_runtime_client.lock().await = None;
                        inference_mgr.shutdown_server();
                        inference_mgr.set_externally_managed(true);

                        let mode = assignment.assignment_mode.as_deref().unwrap_or("solo");

                        match mode {
                            "gateway-tail" => {
                                let bind = assignment
                                    .gateway_tail_bind
                                    .clone()
                                    .unwrap_or_else(|| DEFAULT_GATEWAY_TAIL_BIND.to_string());
                                let (Some(start), Some(end)) =
                                    (assignment.start_layer, assignment.end_layer)
                                else {
                                    warn!("[gateway-tail] missing layer bounds; skipping assignment");
                                    continue;
                                };

                                let signature = SplitGatewaySignature {
                                    role: "tail".to_string(),
                                    pipeline_id: assignment.pipeline_id.clone(),
                                    model_name: assignment.model_name.clone(),
                                    start_layer: start,
                                    end_layer: end,
                                    total_stages: assignment.total_stages,
                                    tail_addr: None,
                                    tail_bind: Some(bind.clone()),
                                    shard_url: assignment.gateway_shard_url.clone(),
                                };

                                let already_matches = managed_tail_node.is_some()
                                    && current_split_signature
                                        .as_ref()
                                        .map(|sig| sig == &signature)
                                        .unwrap_or(false);

                                if !already_matches {
                                    // Drop any prior gateway state — both the
                                    // split topology and the local solo stack.
                                    managed_head_gateway_stack = None;
                                    managed_tail_node = None;
                                    managed_gateway_stack = None;
                                    *gateway_client.lock().await = None;

                                    let (model_path, local_start, local_end) =
                                        match resolve_gateway_stage_inputs(
                                            &self.config,
                                            &assignment.model_name,
                                            "tail",
                                            assignment.gateway_shard_url.as_deref(),
                                            start,
                                            end,
                                        )
                                        .await
                                        {
                                            Ok(tuple) => tuple,
                                            Err(err) => {
                                                warn!("[gateway-tail] shard prep failed: {err}");
                                                let msg = serde_json::json!({
                                                    "type": "node_error",
                                                    "model_name": assignment.model_name,
                                                    "error": format!("gateway-tail shard prep failed: {err}"),
                                                })
                                                .to_string();
                                                let _ = ws_outbound_tx.send(msg).await;
                                                continue;
                                            }
                                        };
                                    let launch_spec = gateway_launch_spec(&self.config);
                                    info!(
                                        "[gateway-tail] Spawning tail worker on {bind} orig_layers={start}-{end} local_layers={local_start}-{local_end} model={}",
                                        model_path.display()
                                    );
                                    match ManagedTailNode::spawn(
                                        model_path.clone(),
                                        bind.clone(),
                                        local_start,
                                        local_end,
                                        &launch_spec,
                                    ) {
                                        Ok(node) => {
                                            info!(
                                                "[gateway-tail] Tail listening on {} (announced)",
                                                node.addr()
                                            );
                                            managed_tail_node = Some(node);
                                            current_split_signature = Some(signature);
                                            let msg = serde_json::json!({
                                                "type": "node_ready",
                                                "model_name": assignment.model_name,
                                            })
                                            .to_string();
                                            let _ = ws_outbound_tx.send(msg).await;
                                        }
                                        Err(err) => {
                                            warn!("[gateway-tail] spawn failed: {err}");
                                            let msg = serde_json::json!({
                                                "type": "node_error",
                                                "model_name": assignment.model_name,
                                                "error": format!("gateway-tail spawn failed: {err}"),
                                            })
                                            .to_string();
                                            let _ = ws_outbound_tx.send(msg).await;
                                            continue;
                                        }
                                    }
                                } else {
                                    let msg = serde_json::json!({
                                        "type": "node_ready",
                                        "model_name": assignment.model_name,
                                    })
                                    .to_string();
                                    let _ = ws_outbound_tx.send(msg).await;
                                }

                                self.update_state(|state| {
                                    state.pipeline.active = true;
                                    state.pipeline.stage = Some(assignment.stage as u32);
                                    state.pipeline.total_stages =
                                        Some(assignment.total_stages as u32);
                                    state.pipeline.model = Some(assignment.model_name.clone());
                                    state.inference_status = "llama-stage-gateway-tail".into();
                                });
                                continue;
                            }
                            "gateway-head" => {
                                let Some(tail_addr) = assignment.gateway_tail_addr.clone() else {
                                    warn!("[gateway-head] assignment missing gateway_tail_addr");
                                    let msg = serde_json::json!({
                                        "type": "node_error",
                                        "model_name": assignment.model_name,
                                        "error": "gateway-head assignment missing gateway_tail_addr",
                                    })
                                    .to_string();
                                    let _ = ws_outbound_tx.send(msg).await;
                                    continue;
                                };
                                let (Some(start), Some(end)) =
                                    (assignment.start_layer, assignment.end_layer)
                                else {
                                    warn!("[gateway-head] missing layer bounds; skipping assignment");
                                    continue;
                                };

                                let signature = SplitGatewaySignature {
                                    role: "head".to_string(),
                                    pipeline_id: assignment.pipeline_id.clone(),
                                    model_name: assignment.model_name.clone(),
                                    start_layer: start,
                                    end_layer: end,
                                    total_stages: assignment.total_stages,
                                    tail_addr: Some(tail_addr.clone()),
                                    tail_bind: None,
                                    shard_url: assignment.gateway_shard_url.clone(),
                                };

                                let already_matches = managed_head_gateway_stack.is_some()
                                    && current_split_signature
                                        .as_ref()
                                        .map(|sig| sig == &signature)
                                        .unwrap_or(false)
                                    && gateway_client.lock().await.is_some();

                                if !already_matches {
                                    // Drop any prior split-gateway state and the local gateway stack.
                                    managed_tail_node = None;
                                    managed_head_gateway_stack = None;
                                    managed_gateway_stack = None;
                                    *gateway_client.lock().await = None;

                                    let (model_path, local_start, local_end) =
                                        match resolve_gateway_stage_inputs(
                                            &self.config,
                                            &assignment.model_name,
                                            "head",
                                            assignment.gateway_shard_url.as_deref(),
                                            start,
                                            end,
                                        )
                                        .await
                                        {
                                            Ok(tuple) => tuple,
                                            Err(err) => {
                                                warn!("[gateway-head] shard prep failed: {err}");
                                                let msg = serde_json::json!({
                                                    "type": "node_error",
                                                    "model_name": assignment.model_name,
                                                    "error": format!("gateway-head shard prep failed: {err}"),
                                                })
                                                .to_string();
                                                let _ = ws_outbound_tx.send(msg).await;
                                                continue;
                                            }
                                        };
                                    let launch_spec = gateway_launch_spec(&self.config);
                                    info!(
                                        "[gateway-head] Spawning head + remote-tail gateway orig_layers={start}-{end} local_layers={local_start}-{local_end} tail={tail_addr} model={}",
                                        model_path.display()
                                    );
                                    match ManagedHeadGatewayStack::spawn_with_remote_tail(
                                        model_path.clone(),
                                        local_start,
                                        local_end,
                                        tail_addr.clone(),
                                        false,
                                        &launch_spec,
                                    ) {
                                        Ok(stack) => {
                                            let addr = stack.gateway_addr().to_string();
                                            info!(
                                                "[gateway-head] gateway listening on {addr}; connecting relay client"
                                            );
                                            managed_head_gateway_stack = Some(stack);
                                            current_split_signature = Some(signature);

                                            match LlamaStageGatewayRelayClient::connect_with_timeout(
                                                &addr,
                                                Some(gateway_connect_timeout(&self.config)),
                                            ) {
                                                Ok(client) => {
                                                    *gateway_client.lock().await = Some(client.clone());
                                                    send_gateway_ready_or_error(
                                                        Ok(client),
                                                        &assignment.model_name,
                                                        &ws_outbound_tx,
                                                    )
                                                    .await;
                                                }
                                                Err(err) => {
                                                    warn!("[gateway-head] relay client connect failed: {err}");
                                                    let msg = serde_json::json!({
                                                        "type": "node_error",
                                                        "model_name": assignment.model_name,
                                                        "error": format!("gateway-head relay client failed: {err}"),
                                                    })
                                                    .to_string();
                                                    let _ = ws_outbound_tx.send(msg).await;
                                                    continue;
                                                }
                                            }
                                        }
                                        Err(err) => {
                                            warn!("[gateway-head] spawn failed: {err}");
                                            let msg = serde_json::json!({
                                                "type": "node_error",
                                                "model_name": assignment.model_name,
                                                "error": format!("gateway-head spawn failed: {err}"),
                                            })
                                            .to_string();
                                            let _ = ws_outbound_tx.send(msg).await;
                                            continue;
                                        }
                                    }
                                } else if let Some(client) = gateway_client.lock().await.clone() {
                                    send_gateway_ready_or_error(
                                        Ok(client),
                                        &assignment.model_name,
                                        &ws_outbound_tx,
                                    )
                                    .await;
                                }

                                self.update_state(|state| {
                                    state.pipeline.active = true;
                                    state.pipeline.stage = Some(assignment.stage as u32);
                                    state.pipeline.total_stages =
                                        Some(assignment.total_stages as u32);
                                    state.pipeline.model = Some(assignment.model_name.clone());
                                    state.inference_status = "llama-stage-gateway-head".into();
                                });
                                continue;
                            }
                            _ => {}
                        }

                        // Solo path: shut down any split-gateway state and use the
                        // local 3-process gateway stack.
                        managed_tail_node = None;
                        managed_head_gateway_stack = None;
                        current_split_signature = None;

                        self.update_state(|state| {
                            state.pipeline.active = true;
                            state.pipeline.stage = Some(assignment.stage as u32);
                            state.pipeline.total_stages = Some(assignment.total_stages as u32);
                            state.pipeline.model = Some(assignment.model_name.clone());
                            state.inference_status = "llama-stage-gateway".into();
                        });

                        let gateway_result = ensure_gateway_client(
                            &self.config,
                            &gateway_client,
                            &mut managed_gateway_stack,
                        )
                        .await;
                        send_gateway_ready_or_error(gateway_result, &assignment.model_name, &ws_outbound_tx)
                            .await;
                        continue;
                    }

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

                            if !matches!(stage_backend_kind, StageBackendKind::LlamaStageGateway) {
                                if let Some(ref url) = assignment.artifact_url {
                                if !crate::inference::stage_artifacts::is_stage_cached(
                                    &assignment.model_name,
                                    start_layer as u32,
                                    end_layer as u32,
                                ) {
                                    info!("[stage] Downloading packed stage artifacts...");
                                    match crate::inference::stage_artifacts::ensure_stage_artifacts(
                                        &assignment.model_name,
                                        start_layer as u32,
                                        end_layer as u32,
                                        url,
                                        assignment.artifact_sha256.as_deref(),
                                        assignment.artifact_size_bytes,
                                    ).await {
                                        Ok(path) => info!("[stage] Artifacts ready at {}", path.display()),
                                        Err(e) => {
                                            warn!("[stage] Failed to download stage artifacts: {e}");
                                            let msg = serde_json::json!({
                                                "type": "node_error",
                                                "model_name": assignment.model_name,
                                                "error": format!("stage artifact download failed: {e}"),
                                            }).to_string();
                                            let _ = ws_outbound_tx.send(msg).await;
                                            continue;
                                        }
                                    }
                                }
                            }
                            }

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
                            &gateway_client,
                            &mut managed_gateway_stack,
                            stage_backend_kind,
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
                    let inf_status = if matches!(stage_backend_kind, StageBackendKind::LlamaStageGateway) {
                        if gateway_client.lock().await.is_some() {
                            "llama-stage-gateway".to_string()
                        } else {
                            "llama-stage-gateway (disconnected)".to_string()
                        }
                    } else {
                        stage_runtime
                            .as_ref()
                            .map(|handle| handle.status_label())
                            .unwrap_or_else(|| inference_mgr.status().to_string())
                    };

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
        drop(managed_gateway_stack);
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
        gateway_client: &Arc<tokio::sync::Mutex<Option<LlamaStageGatewayRelayClient>>>,
        managed_gateway_stack: &mut Option<ManagedGatewayStack>,
        stage_backend_kind: StageBackendKind,
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
                let previous_pipeline = {
                    let state = self.state_rx.borrow();
                    (
                        state.pipeline.active,
                        state.pipeline.model.clone(),
                        state.pipeline.stage,
                        state.pipeline.total_stages,
                    )
                };
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
                    let client =
                        compute_network::client::OrchestratorClient::new(&orchestrator_url, None);
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

                if matches!(stage_backend_kind, StageBackendKind::LlamaStageGateway)
                    && has_pipeline
                    && node.pipeline_stage.unwrap_or(0) == 0
                    && node.pipeline_total_stages.unwrap_or(1) == 1
                {
                    if let Some(handle) = stage_runtime.take() {
                        handle.stop().await;
                    }
                    *stage_runtime_client.lock().await = None;
                    inference_mgr.shutdown_server();
                    inference_mgr.set_externally_managed(true);

                    let supported = if has_pipeline {
                        gateway_assignment_supported(
                            node.pipeline_stage.unwrap_or(0),
                            node.pipeline_total_stages.unwrap_or(1),
                            None,
                        )
                    } else {
                        true
                    };
                    let gateway_assignment_changed = has_pipeline != previous_pipeline.0
                        || node.model_name != previous_pipeline.1
                        || node.pipeline_stage.map(|s| s as u32) != previous_pipeline.2
                        || node.pipeline_total_stages.map(|s| s as u32) != previous_pipeline.3;

                    if has_pipeline && !supported {
                        if gateway_assignment_changed {
                            let msg = serde_json::json!({
                                "type": "node_error",
                                "model_name": node.model_name,
                                "error": format!(
                                    "llama-stage-gateway only supports single-stage assignments; got stage={:?}/{:?}",
                                    node.pipeline_stage,
                                    node.pipeline_total_stages
                                ),
                            }).to_string();
                            let _ = ws_outbound_tx.send(msg).await;
                        }
                        return;
                    }

                    let gateway_needs_connect = gateway_client.lock().await.is_none();
                    if has_pipeline && (gateway_assignment_changed || gateway_needs_connect) {
                        if let Some(model_name) = node.model_name.as_deref() {
                            let gateway_result = ensure_gateway_client(
                                &self.config,
                                gateway_client,
                                managed_gateway_stack,
                            )
                            .await;
                            send_gateway_ready_or_error(gateway_result, model_name, ws_outbound_tx)
                                .await;
                        }
                    }
                    return;
                }

                // Tell inference manager about the assignment
                let stage_pipeline_match = stage_runtime
                    .as_ref()
                    .map(|handle| {
                        node.pipeline_id.as_deref() == Some(handle.pipeline_id())
                            && node.model_name.as_deref() == Some(handle.model_name())
                            && node.pipeline_stage.map(|s| s as u32) == Some(handle.stage_index())
                            && node.pipeline_total_stages.map(|s| s as u32)
                                == Some(handle.total_stages())
                    })
                    .unwrap_or(false);

                let looks_like_stage_assignment = self.config.experimental.stage_mode_enabled
                    && node.model_name.as_deref() == Some("gemma-4-e4b-q4")
                    && node.pipeline_id.is_some()
                    && node.pipeline_stage.is_some()
                    && node.pipeline_total_stages == Some(2);

                let previous_pipeline_id = match inference_mgr.status() {
                    InferenceStatus::Running { pipeline_id, .. } => Some(pipeline_id.clone()),
                    InferenceStatus::RunningRpcWorker { pipeline_id, .. } => {
                        Some(pipeline_id.clone())
                    }
                    _ => None,
                };
                let previous_model_name = match inference_mgr.status() {
                    InferenceStatus::Running { model_name, .. } => Some(model_name.clone()),
                    InferenceStatus::RunningRpcWorker { model_name, .. } => {
                        Some(model_name.clone())
                    }
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

                    inference_mgr
                        .check_assignment(node.pipeline_id.as_deref(), node.model_name.as_deref());
                }

                let pipeline_changed =
                    previous_pipeline_id.as_deref() != node.pipeline_id.as_deref();
                let model_changed = previous_model_name.as_deref() != node.model_name.as_deref();
                if !looks_like_stage_assignment
                    && node.pipeline_id.is_some()
                    && (pipeline_changed || model_changed)
                {
                    // Poll-based assignment detection runs before the WS push path on startup/reconnect.
                    // Emit the same ready probe here so the orchestrator doesn't wait 120s for a
                    // node_ready message that only the push path would have produced.
                    let active_model = match inference_mgr.status() {
                        InferenceStatus::Running { model_name, .. } => Some(model_name.clone()),
                        InferenceStatus::RunningRpcWorker { model_name, .. } => {
                            Some(model_name.clone())
                        }
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
                (Some(handle.pipeline_id().to_string()), Some(handle.stage_index() as i32))
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
                    let gateway_mode = matches!(
                        StageBackendKind::parse(&self.config.experimental.stage_backend),
                        StageBackendKind::LlamaStageGateway
                    );
                    let total = if stage_runtime.is_some() {
                        Some(1)
                    } else if gateway_mode {
                        Some(1)
                    } else if relay_busy_slots > 0 {
                        Some(2)
                    } else {
                        None
                    };
                    let busy = if stage_runtime.is_some() || gateway_mode {
                        Some(relay_busy_slots.max(0))
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
                ip_address: advertised_host(&self.config),
                pipeline_capable: Some(self.config.experimental.stage_mode_enabled),
                memory_bandwidth_gbps: None,
                last_heartbeat: Some(chrono::Utc::now().to_rfc3339()),
                stage_backend_kind: if self.config.experimental.stage_mode_enabled {
                    Some(
                        StageBackendKind::parse(&self.config.experimental.stage_backend)
                            .as_str()
                            .to_string(),
                    )
                } else {
                    None
                },
                app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
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
        std::net::IpAddr::V4(ip) if !ip.is_loopback() && !ip.is_unspecified() => {
            Some(ip.to_string())
        }
        _ => None,
    }
}

fn advertised_host(config: &Config) -> Option<String> {
    let configured = config.network.advertise_host.trim();
    if !configured.is_empty() {
        return Some(configured.to_string());
    }
    detect_advertise_ip()
}

/// Kill any leftover stage-node / gateway child processes from a prior daemon
/// run so they don't keep their bind ports held (otherwise the next spawn
/// gets "Address already in use" on e.g. 9182).
fn sweep_orphan_stage_children() {
    const TARGETS: &[&str] = &[
        "llama_stage_tcp_node",
        "llama_stage_gateway_tcp_node",
    ];
    #[cfg(unix)]
    {
        for target in TARGETS {
            let _ = std::process::Command::new("pkill")
                .arg("-f")
                .arg(target)
                .output();
        }
    }
    let _ = TARGETS;
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
    let cache_dir = dirs::home_dir().unwrap_or_default().join(".compute").join("models");

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
                    tracing::warn!(
                        "Skipping suspiciously small GGUF: {} ({:.1} MB)",
                        name,
                        meta.len() as f64 / 1048576.0
                    );
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

    // Also advertise per-stage shards. The scheduler uses these synthetic IDs
    // (e.g. "gemma-4-e4b-q4-stage-head-0-20") to pick head/tail assignments
    // that match what each node already has cached, avoiding redundant
    // 2.5 GB downloads for the opposite role.
    for shard in detect_cached_stage_shards() {
        if !models.contains(&shard) {
            models.push(shard);
        }
    }

    models.join(",")
}

/// Scan ~/.compute/stages/<model_id>/<role>-<start>-<end>.gguf and return
/// synthetic IDs like "gemma-4-e4b-q4-stage-head-0-20" for each valid shard.
fn detect_cached_stage_shards() -> Vec<String> {
    let stages_dir = dirs::home_dir().unwrap_or_default().join(".compute").join("stages");
    let mut shards = Vec::new();

    let Ok(model_dirs) = std::fs::read_dir(&stages_dir) else {
        return shards;
    };

    for model_entry in model_dirs.flatten() {
        if !model_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let model_id = model_entry.file_name().to_string_lossy().to_string();
        let Ok(files) = std::fs::read_dir(model_entry.path()) else { continue };

        for file in files.flatten() {
            let name = file.file_name().to_string_lossy().to_string();
            if !name.ends_with(".gguf") {
                continue;
            }
            // Expect "<role>-<start>-<end>.gguf".
            let stem = name.trim_end_matches(".gguf");
            let parts: Vec<&str> = stem.splitn(3, '-').collect();
            if parts.len() != 3 {
                continue;
            }
            let role = parts[0];
            if role != "head" && role != "tail" {
                continue;
            }
            let (Ok(start), Ok(end)) = (parts[1].parse::<u32>(), parts[2].parse::<u32>()) else {
                continue;
            };

            // Basic size + magic sanity (shards are 2-3 GB).
            if let Ok(meta) = file.metadata() {
                if meta.len() < 100 * 1024 * 1024 {
                    continue;
                }
            }
            if !is_valid_gguf(&file.path()) {
                continue;
            }

            shards.push(format!("{}-stage-{}-{}-{}", model_id, role, start, end));
        }
    }

    shards
}
