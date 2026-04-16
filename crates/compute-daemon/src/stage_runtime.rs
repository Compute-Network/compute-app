use anyhow::{Context, Result};
use compute_network::models::ModelCatalog;
use compute_network::transport::node::TransportNode;
use compute_network::transport::protocol::{
    ActivationPayload, ControlMessage, PipelineMessage, PipelineStageTiming, PipelineTimingProfile,
    PongMessage, TokenPayload,
};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::config::Config;
use crate::hardware::HardwareInfo;
use crate::inference::engine::{
    Activation, ForwardResult, GeneratedToken, ShardConfig as EngineShardConfig,
};
use crate::inference::stage_backend::{StageBackendKind, StageExecutionBackend};

const STAGE_RUNTIME_LISTEN_PORT: u16 = 9090;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePrototypeSpec {
    pub pipeline_id: String,
    pub model_name: String,
    pub shard_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub stage_index: u32,
    pub total_stages: u32,
    pub upstream_addr: Option<String>,
    pub downstream_addr: Option<String>,
}

pub struct StagePrototypeHandle {
    client: StagePrototypeClient,
    spec: StagePrototypeSpec,
    listen_addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub struct StagePrototypeClient {
    spec: StagePrototypeSpec,
    request_tx: mpsc::Sender<StagePrototypeCommand>,
}

#[derive(Debug)]
pub struct StagePrototypeResponse {
    pub model_name: String,
    pub content: String,
    pub finish_reason: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub completion_token_ids: Vec<u32>,
    pub total_tokens: u32,
    pub ttft_ms: u128,
    pub total_ms: u128,
    pub runtime_profile: StagePrototypeRuntimeProfile,
}

#[derive(Debug, Clone, Default)]
pub struct StagePrototypeRuntimeProfile {
    pub tokenize_ms: u128,
    pub connect_ms: u128,
    pub assign_ms: u128,
    pub head_engine_ms: u128,
    pub downstream_wait_ms: u128,
    pub downstream_stage_engine_ms: u128,
    pub downstream_stage_total_ms: u128,
    pub detokenize_ms: u128,
    pub clear_decode_session_ms: u128,
    pub steps: u32,
    pub first_head_engine_ms: u128,
    pub first_downstream_wait_ms: u128,
    pub first_downstream_stage_engine_ms: u128,
    pub first_downstream_stage_total_ms: u128,
}

#[derive(Debug)]
enum StagePrototypeCommand {
    Complete {
        request_id: String,
        prompt: String,
        max_tokens: Option<u32>,
        stop_sequences: Vec<String>,
        response_tx: oneshot::Sender<Result<StagePrototypeResponse>>,
    },
}

impl StagePrototypeHandle {
    pub fn matches(&self, other: &StagePrototypeSpec) -> bool {
        &self.spec == other
    }

    pub fn pipeline_id(&self) -> &str {
        &self.spec.pipeline_id
    }

    pub fn model_name(&self) -> &str {
        &self.spec.model_name
    }

    pub fn stage_index(&self) -> u32 {
        self.spec.stage_index
    }

    pub fn total_stages(&self) -> u32 {
        self.spec.total_stages
    }

    pub fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    pub fn client(&self) -> StagePrototypeClient {
        self.client.clone()
    }

    pub fn status_label(&self) -> String {
        format!(
            "stage-prototype ({}, shard {}, {}/{})",
            self.spec.model_name,
            self.spec.shard_id,
            self.spec.stage_index + 1,
            self.spec.total_stages
        )
    }

    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }
}

impl StagePrototypeClient {
    pub fn is_head_stage(&self) -> bool {
        self.spec.stage_index == 0
    }

    pub async fn complete_prompt(
        &self,
        request_id: String,
        prompt: String,
        max_tokens: Option<u32>,
        stop_sequences: Vec<String>,
    ) -> Result<StagePrototypeResponse> {
        info!(
            "[stage] Queueing head request {} stage={}/{} prompt_len={} max_tokens={:?} stop_count={}",
            request_id,
            self.spec.stage_index + 1,
            self.spec.total_stages,
            prompt.len(),
            max_tokens,
            stop_sequences.len()
        );
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(StagePrototypeCommand::Complete {
                request_id,
                prompt,
                max_tokens,
                stop_sequences,
                response_tx,
            })
            .await
            .context("Stage prototype runtime is unavailable")?;
        response_rx.await.context("Stage prototype runtime dropped completion response")?
    }
}

pub async fn start_stage_prototype(
    config: &Config,
    hw: &HardwareInfo,
    spec: StagePrototypeSpec,
) -> Result<StagePrototypeHandle> {
    start_stage_prototype_with_bind_addr(
        config,
        hw,
        spec,
        SocketAddr::from((Ipv4Addr::UNSPECIFIED, STAGE_RUNTIME_LISTEN_PORT)),
    )
    .await
}

pub async fn start_stage_prototype_chain(
    config: &Config,
    hw: &HardwareInfo,
    pipeline_id: &str,
    model_name: &str,
    stage_ranges: &[(u32, u32)],
) -> Result<Vec<StagePrototypeHandle>> {
    let bind_addrs = vec![SocketAddr::from((Ipv4Addr::LOCALHOST, 0)); stage_ranges.len()];
    start_stage_prototype_chain_with_bind_addrs(
        config,
        hw,
        pipeline_id,
        model_name,
        stage_ranges,
        &bind_addrs,
    )
    .await
}

pub async fn start_stage_prototype_chain_with_bind_addrs(
    config: &Config,
    hw: &HardwareInfo,
    pipeline_id: &str,
    model_name: &str,
    stage_ranges: &[(u32, u32)],
    bind_addrs: &[SocketAddr],
) -> Result<Vec<StagePrototypeHandle>> {
    if stage_ranges.is_empty() {
        anyhow::bail!("Stage prototype chain requires at least one stage");
    }
    if bind_addrs.len() != stage_ranges.len() {
        anyhow::bail!(
            "Stage prototype chain bind address count {} did not match stage count {}",
            bind_addrs.len(),
            stage_ranges.len()
        );
    }

    let total_stages = stage_ranges.len() as u32;
    let mut handles = Vec::with_capacity(stage_ranges.len());
    let mut downstream_addr = None;

    for stage_index in (0..stage_ranges.len()).rev() {
        let (start_layer, end_layer) = stage_ranges[stage_index];
        let handle = start_stage_prototype_with_bind_addr(
            config,
            hw,
            StagePrototypeSpec {
                pipeline_id: pipeline_id.to_string(),
                model_name: model_name.to_string(),
                shard_id: format!("stage-{stage_index}"),
                start_layer,
                end_layer,
                stage_index: stage_index as u32,
                total_stages,
                upstream_addr: None,
                downstream_addr: downstream_addr.clone(),
            },
            bind_addrs[stage_index],
        )
        .await?;
        downstream_addr = Some(handle.listen_addr().to_string());
        handles.push(handle);
    }

    handles.reverse();
    Ok(handles)
}

pub async fn start_stage_prototype_with_bind_addr(
    config: &Config,
    hw: &HardwareInfo,
    spec: StagePrototypeSpec,
    bind_addr: SocketAddr,
) -> Result<StagePrototypeHandle> {
    let total_layers = resolve_total_layers(&spec.model_name)?;

    let stage_backend = StageBackendKind::parse(&config.experimental.stage_backend);
    if matches!(stage_backend, StageBackendKind::LlamaCpp) {
        anyhow::bail!(
            "Stage backend `{}` cannot back the local stage prototype runtime; use `prototype`, `tail-llama`, `llama-stage-gateway`, or `real_forward`",
            stage_backend.as_str()
        );
    }
    let mut engine = StageExecutionBackend::new_for_hardware(
        hw,
        stage_backend,
        &config.experimental.stage_acceleration,
        &config.experimental.stage_acceleration_provider,
    );
    let shard_path = match stage_backend {
        StageBackendKind::Prototype => std::path::PathBuf::from(format!(
            "prototype://{}:{}-{}",
            spec.model_name, spec.start_layer, spec.end_layer
        )),
        StageBackendKind::TailLlama => {
            if spec.stage_index + 1 == spec.total_stages {
                resolve_model_path(&spec.model_name).with_context(|| {
                    format!("No local GGUF found for stage prototype model {}", spec.model_name)
                })?
            } else {
                std::path::PathBuf::from(format!(
                    "prototype://{}:{}-{}",
                    spec.model_name, spec.start_layer, spec.end_layer
                ))
            }
        }
        StageBackendKind::LlamaCpp => resolve_model_path(&spec.model_name).with_context(|| {
            format!("No local GGUF found for stage prototype model {}", spec.model_name)
        })?,
        StageBackendKind::LlamaStageGateway => {
            resolve_model_path(&spec.model_name).with_context(|| {
                format!(
                    "No local GGUF found for llama-stage runtime model {}",
                    spec.model_name
                )
            })?
        }
        StageBackendKind::RealForward => {
            let stages_dir = dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".compute")
                .join("stages")
                .join(&spec.model_name)
                .join(format!("packed-stage-{}-{}", spec.start_layer, spec.end_layer));
            stages_dir
        }
    };
    let shard_config = EngineShardConfig {
        model_id: spec.model_name.clone(),
        shard_path,
        start_layer: spec.start_layer,
        end_layer: spec.end_layer,
        total_layers,
        is_first_stage: spec.stage_index == 0,
        is_last_stage: spec.stage_index + 1 == spec.total_stages,
        max_batch_size: 2048,
        context_length: 8192,
    };

    engine.load_shard(&shard_config).await?;

    let transport = TransportNode::bind(bind_addr).await?;
    let listen_addr = transport.listen_addr();

    info!(
        "[stage] Prototype runtime ready: pipeline={} model={} shard={} layers {}-{} listen={} backend={} upstream={:?} downstream={:?}",
        spec.pipeline_id,
        spec.model_name,
        spec.shard_id,
        spec.start_layer,
        spec.end_layer,
        listen_addr,
        engine.backend_label(),
        spec.upstream_addr,
        spec.downstream_addr
    );

    let downstream_addr =
        spec.downstream_addr.as_deref().and_then(|addr| addr.parse::<SocketAddr>().ok());

    let engine = Arc::new(Mutex::new(engine));
    let (request_tx, mut request_rx) = mpsc::channel::<StagePrototypeCommand>(8);
    let client = StagePrototypeClient { spec: spec.clone(), request_tx };
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
    let spec_for_task = spec.clone();
    let task = tokio::spawn(async move {
        let downstream =
            Arc::new(Mutex::new(None::<compute_network::transport::node::PeerConnection>));

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    break;
                }
                Some(command) = request_rx.recv() => {
                    match command {
                        StagePrototypeCommand::Complete {
                            request_id,
                            prompt,
                            max_tokens,
                            stop_sequences,
                            response_tx,
                        } => {
                            info!("[stage] Executing queued head request {}", request_id);
                            let result = handle_local_completion_command(
                                &transport,
                                &engine,
                                &spec_for_task,
                                &downstream,
                                downstream_addr,
                                request_id,
                                prompt,
                                max_tokens,
                                stop_sequences,
                            ).await;
                            let _ = response_tx.send(result);
                        }
                    }
                }
                accept_result = transport.accept() => {
                    match accept_result {
                        Ok(peer) => {
                            info!("[stage] Accepted prototype peer connection from {}", peer.remote_addr());

                            loop {
                                let msg = match peer.recv_activations().await {
                                    Ok(msg) => msg,
                                    Err(err) => {
                                        warn!("[stage] Peer {} disconnected or read failed: {err}", peer.remote_addr());
                                        break;
                                    }
                                };

                                match msg {
                                    PipelineMessage::Activations(activation) => {
                                        info!(
                                            "[stage] Received activations for request {} from {}",
                                            activation.request_id,
                                            peer.remote_addr()
                                        );
                                        let input = Activation {
                                            request_id: activation.request_id.clone(),
                                            shape: activation.shape.clone(),
                                            data: activation.data.clone(),
                                            seq_position: activation.seq_position,
                                            batch_index: activation.batch_index,
                                        };

                                        let stage_total_start = Instant::now();
                                        let stage_engine_start = Instant::now();
                                        let result = {
                                            let engine = engine.lock().await;
                                            engine.continue_forward(input).await
                                        };
                                        let stage_engine_ms = stage_engine_start.elapsed().as_millis();

                                        match result {
                                            Ok(ForwardResult::Activations(output)) => {
                                                let payload = ActivationPayload {
                                                    request_id: output.request_id,
                                                    seq_position: output.seq_position,
                                                    batch_index: output.batch_index,
                                                    shape: output.shape,
                                                    data: output.data,
                                                    dtype: compute_network::transport::protocol::TensorDtype::Float16,
                                                };
                                                match forward_activation_via_downstream(
                                                    &transport,
                                                    &spec_for_task,
                                                    &downstream,
                                                    downstream_addr,
                                                    payload,
                                                ).await {
                                                    Ok(mut tokens) => {
                                                        append_stage_timing(
                                                            &mut tokens,
                                                            &spec_for_task,
                                                            stage_engine_ms,
                                                            stage_total_start.elapsed().as_millis(),
                                                        );
                                                        if let Err(err) = peer
                                                            .send_activations(&PipelineMessage::Tokens(tokens))
                                                            .await
                                                        {
                                                            warn!("[stage] Failed to send relayed tokens upstream: {err}");
                                                        }
                                                    }
                                                    Err(err) => {
                                                        let payload = PipelineMessage::Control(ControlMessage::Error {
                                                            node_id: transport.node_id().to_string(),
                                                            message: err.to_string(),
                                                        });
                                                        let _ = peer.send_activations(&payload).await;
                                                    }
                                                }
                                            }
                                            Ok(ForwardResult::Tokens(tokens)) => {
                                                info!(
                                                    "[stage] Returning {} token(s) upstream for request {}",
                                                    tokens.len(),
                                                    activation.request_id
                                                );
                                                let payload = PipelineMessage::Tokens(
                                                    upstream_token_payload(
                                                        activation.request_id,
                                                        &tokens,
                                                        Some(stage_timing_profile(
                                                            &spec_for_task,
                                                            stage_engine_ms,
                                                            stage_total_start.elapsed().as_millis(),
                                                        )),
                                                    ),
                                                );
                                                if let Err(err) = peer.send_activations(&payload).await {
                                                    warn!("[stage] Failed to send tokens upstream: {err}");
                                                }
                                            }
                                            Err(err) => {
                                                let payload = PipelineMessage::Control(ControlMessage::Error {
                                                    node_id: transport.node_id().to_string(),
                                                    message: err.to_string(),
                                                });
                                                let _ = peer.send_activations(&payload).await;
                                            }
                                        }
                                    }
                                    PipelineMessage::Ping(ping) => {
                                        let pong = PipelineMessage::Pong(PongMessage {
                                            node_id: transport.node_id().to_string(),
                                            timestamp_ms: ping.timestamp_ms,
                                            latency_ms: None,
                                        });
                                        let _ = peer.send_activations(&pong).await;
                                    }
                                    PipelineMessage::Control(ControlMessage::AssignLayers { .. }) => {
                                        let ready = PipelineMessage::Control(ControlMessage::Ready {
                                            node_id: transport.node_id().to_string(),
                                        });
                                        let _ = peer.send_activations(&ready).await;
                                    }
                                    PipelineMessage::Control(ControlMessage::Release { reason }) => {
                                        info!("[stage] Received release signal: {reason}");
                                        break;
                                    }
                                    other => {
                                        warn!("[stage] Ignoring unsupported prototype message: {:?}", other);
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            warn!("[stage] Transport accept error: {err}");
                            break;
                        }
                    }
                }
            }
        }

        if let Some(peer) = downstream.lock().await.take() {
            peer.close();
        }

        let mut engine = engine.lock().await;
        if let Err(err) = engine.unload().await {
            warn!("[stage] Failed to unload prototype shard cleanly: {err}");
        }
        transport.close();
        info!("[stage] Prototype runtime stopped");
    });

    Ok(StagePrototypeHandle { client, spec, listen_addr, shutdown_tx: Some(shutdown_tx), task })
}

async fn handle_local_completion_command(
    transport: &TransportNode,
    engine: &Arc<Mutex<StageExecutionBackend>>,
    spec: &StagePrototypeSpec,
    downstream: &Arc<Mutex<Option<compute_network::transport::node::PeerConnection>>>,
    downstream_addr: Option<SocketAddr>,
    request_id: String,
    prompt: String,
    max_tokens: Option<u32>,
    stop_sequences: Vec<String>,
) -> Result<StagePrototypeResponse> {
    let total_start = Instant::now();
    let mut runtime_profile = StagePrototypeRuntimeProfile::default();
    if spec.stage_index != 0 {
        anyhow::bail!("Only the head stage can accept direct completion commands");
    }

    info!(
        "[stage] Head begin request {} downstream_present={} prompt_len={} max_tokens={:?}",
        request_id,
        downstream.lock().await.is_some(),
        prompt.len(),
        max_tokens
    );

    let tokenize_start = Instant::now();
    let prompt_tokens = {
        let engine = engine.lock().await;
        engine.tokenize_generation_prompt(&prompt).await?
    };
    runtime_profile.tokenize_ms = tokenize_start.elapsed().as_millis();
    let token_limit = max_tokens.unwrap_or(1).max(1) as usize;
    let use_token_id_prompt_ingress = {
        let engine = engine.lock().await;
        engine.supports_token_id_prompt_ingress()
    };

    let connect_start = Instant::now();
    let mut downstream_guard =
        ensure_downstream_connection(transport, downstream, downstream_addr).await?;
    runtime_profile.connect_ms = connect_start.elapsed().as_millis();
    let downstream_peer = downstream_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No downstream peer configured for head stage"))?;
    info!(
        "[stage] Sending assign/control message for request {} to downstream {}",
        request_id,
        downstream_peer.remote_addr()
    );
    let assign_start = Instant::now();
    if let Err(err) = downstream_peer
        .send_activations(&PipelineMessage::Control(ControlMessage::AssignLayers {
            model_id: spec.model_name.clone(),
            start_layer: spec.start_layer,
            end_layer: spec.end_layer,
            total_layers: resolve_total_layers(&spec.model_name)?,
        }))
        .await
    {
        warn!("[stage] Failed to send assign/control to downstream: {err}");
        *downstream_guard = None;
        anyhow::bail!("Failed to send assign/control to downstream: {err}");
    }
    wait_for_downstream_ready(downstream_peer, &request_id).await?;
    runtime_profile.assign_ms = assign_start.elapsed().as_millis();
    let (content, finish_reason, completion_tokens, completion_token_ids, ttft_ms) =
        if use_token_id_prompt_ingress {
            let eos_token_id = {
                let engine = engine.lock().await;
                engine.eos_token_id()
            };
            let mut current_prompt_tokens = prompt_tokens.clone();
            let mut generated_tokens = Vec::with_capacity(token_limit);
            let mut content = String::new();
            let mut finish_reason = "length".to_string();
            let mut ttft_ms = 0u128;

            for step in 0..token_limit {
                let step_start = Instant::now();
                let step_token_ids: Vec<u32> = if step == 0 {
                    current_prompt_tokens.clone()
                } else {
                    generated_tokens.last().copied().into_iter().collect()
                };
                let head_engine_start = Instant::now();
                let activation = {
                    let engine = engine.lock().await;
                    engine
                        .begin_token_ids(request_id.clone(), &step_token_ids, Some(1), 2048)
                        .await?
                };
                let head_engine_ms = head_engine_start.elapsed().as_millis();
                runtime_profile.head_engine_ms += head_engine_ms;
                runtime_profile.steps += 1;
                if step == 0 {
                    runtime_profile.first_head_engine_ms = head_engine_ms;
                }
                let activation = ActivationPayload {
                    request_id: activation.request_id,
                    seq_position: activation.seq_position,
                    batch_index: activation.batch_index,
                    shape: activation.shape,
                    data: activation.data,
                    dtype: compute_network::transport::protocol::TensorDtype::Float16,
                };

                let downstream_wait_start = Instant::now();
                let tokens = match request_downstream_tokens(
                    transport,
                    downstream_peer,
                    &request_id,
                    activation,
                )
                .await
                {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        *downstream_guard = None;
                        engine.lock().await.clear_decode_session(&request_id);
                        return Err(err);
                    }
                };
                let downstream_wait_ms = downstream_wait_start.elapsed().as_millis();
                runtime_profile.downstream_wait_ms += downstream_wait_ms;
                if step == 0 {
                    runtime_profile.first_downstream_wait_ms = downstream_wait_ms;
                }
                if let Some(timings) = &tokens.timings {
                    let stage_engine_ms = timings.total_engine_ms();
                    let stage_total_ms = timings.total_stage_ms();
                    runtime_profile.downstream_stage_engine_ms += stage_engine_ms;
                    runtime_profile.downstream_stage_total_ms += stage_total_ms;
                    if step == 0 {
                        runtime_profile.first_downstream_stage_engine_ms = stage_engine_ms;
                        runtime_profile.first_downstream_stage_total_ms = stage_total_ms;
                    }
                }

                if tokens.tokens.is_empty() {
                    break;
                }
                if ttft_ms == 0 {
                    ttft_ms = step_start.elapsed().as_millis();
                }

                let remaining = token_limit.saturating_sub(generated_tokens.len());
                let step_tokens = &tokens.tokens[..tokens.tokens.len().min(remaining)];
                let mut hit_eos = false;
                let mut accepted_step_tokens = Vec::with_capacity(step_tokens.len());
                for &token in step_tokens {
                    if eos_token_id == Some(token) {
                        hit_eos = true;
                        break;
                    }
                    accepted_step_tokens.push(token);
                }
                current_prompt_tokens.extend(accepted_step_tokens.iter().copied());
                generated_tokens.extend(accepted_step_tokens.iter().copied());

                let detokenize_start = Instant::now();
                content = {
                    let engine = engine.lock().await;
                    engine.detokenize(&generated_tokens).await.unwrap_or_else(|_| {
                        generated_tokens
                            .iter()
                            .map(|tok| tok.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                };
                runtime_profile.detokenize_ms += detokenize_start.elapsed().as_millis();
                if let Some(trimmed) = trim_at_stop_sequence(&content, &stop_sequences) {
                    content = trimmed;
                    finish_reason = "stop".to_string();
                    break;
                }
                if hit_eos {
                    finish_reason = "stop".to_string();
                    break;
                }
            }

            drop(downstream_guard);

            let completion_tokens = generated_tokens.len() as u32;
            if content.is_empty() && !generated_tokens.is_empty() {
                let detokenize_start = Instant::now();
                content = {
                    let engine = engine.lock().await;
                    engine.detokenize(&generated_tokens).await.unwrap_or_else(|_| {
                        generated_tokens
                            .iter()
                            .map(|tok| tok.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                };
                runtime_profile.detokenize_ms += detokenize_start.elapsed().as_millis();
            }
            (content, finish_reason, completion_tokens, generated_tokens, ttft_ms)
        } else {
            let step_start = Instant::now();
            let head_engine_start = Instant::now();
            let activation = {
                let engine = engine.lock().await;
                engine.begin_prompt(request_id.clone(), &prompt, max_tokens, 2048).await?
            };
            let head_engine_ms = head_engine_start.elapsed().as_millis();
            runtime_profile.head_engine_ms = head_engine_ms;
            runtime_profile.first_head_engine_ms = head_engine_ms;
            runtime_profile.steps = 1;
            let activation = ActivationPayload {
                request_id: activation.request_id,
                seq_position: activation.seq_position,
                batch_index: activation.batch_index,
                shape: activation.shape,
                data: activation.data,
                dtype: compute_network::transport::protocol::TensorDtype::Float16,
            };
            let downstream_wait_start = Instant::now();
            let tokens = match request_downstream_tokens(
                transport,
                downstream_peer,
                &request_id,
                activation,
            )
            .await
            {
                Ok(tokens) => tokens,
                Err(err) => {
                    *downstream_guard = None;
                    engine.lock().await.clear_decode_session(&request_id);
                    return Err(err);
                }
            };
            let downstream_wait_ms = downstream_wait_start.elapsed().as_millis();
            runtime_profile.downstream_wait_ms = downstream_wait_ms;
            runtime_profile.first_downstream_wait_ms = downstream_wait_ms;
            if let Some(timings) = &tokens.timings {
                runtime_profile.downstream_stage_engine_ms = timings.total_engine_ms();
                runtime_profile.downstream_stage_total_ms = timings.total_stage_ms();
                runtime_profile.first_downstream_stage_engine_ms =
                    runtime_profile.downstream_stage_engine_ms;
                runtime_profile.first_downstream_stage_total_ms =
                    runtime_profile.downstream_stage_total_ms;
            }
            drop(downstream_guard);

            let raw_completion_tokens = tokens.tokens.len() as u32;
            let completion_tokens = max_tokens
                .map(|limit| raw_completion_tokens.min(limit))
                .unwrap_or(raw_completion_tokens);
            let content = if let Some(text) = tokens.text.clone() {
                normalize_completion_text(&text, raw_completion_tokens, completion_tokens)
            } else {
                let token_slice = &tokens.tokens[..completion_tokens as usize];
                let detokenize_start = Instant::now();
                let content = {
                    let engine = engine.lock().await;
                    engine.detokenize(token_slice).await.unwrap_or_else(|_| {
                        token_slice.iter().map(|tok| tok.to_string()).collect::<Vec<_>>().join(" ")
                    })
                };
                runtime_profile.detokenize_ms += detokenize_start.elapsed().as_millis();
                content
            };
            let (content, finish_reason) =
                if let Some(trimmed) = trim_at_stop_sequence(&content, &stop_sequences) {
                    (trimmed, "stop".to_string())
                } else if max_tokens.is_some_and(|limit| completion_tokens >= limit) {
                    (content, "length".to_string())
                } else {
                    (content, "stop".to_string())
                };
            let completion_token_ids = tokens.tokens[..completion_tokens as usize].to_vec();
            (
                content,
                finish_reason,
                completion_tokens,
                completion_token_ids,
                step_start.elapsed().as_millis(),
            )
        };
    let clear_start = Instant::now();
    engine.lock().await.clear_decode_session(&request_id);
    runtime_profile.clear_decode_session_ms = clear_start.elapsed().as_millis();

    Ok(StagePrototypeResponse {
        model_name: spec.model_name.clone(),
        content,
        finish_reason,
        prompt_tokens: prompt_tokens.len() as u32,
        completion_tokens,
        completion_token_ids,
        total_tokens: prompt_tokens.len() as u32 + completion_tokens,
        ttft_ms,
        total_ms: total_start.elapsed().as_millis(),
        runtime_profile,
    })
}

async fn request_downstream_tokens(
    transport: &TransportNode,
    downstream_peer: &compute_network::transport::node::PeerConnection,
    request_id: &str,
    activation: ActivationPayload,
) -> Result<TokenPayload> {
    info!(
        "[stage] Sending activation payload for request {} to downstream {}",
        request_id,
        downstream_peer.remote_addr()
    );
    downstream_peer.send_activations(&PipelineMessage::Activations(activation)).await?;
    info!("[stage] Waiting for downstream tokens for request {}", request_id);

    loop {
        match downstream_peer.recv_activations().await {
            Ok(PipelineMessage::Tokens(tokens)) if tokens.request_id == request_id => {
                info!(
                    "[stage] Received downstream tokens for request {} count={}",
                    request_id,
                    tokens.tokens.len()
                );
                return Ok(tokens);
            }
            Ok(PipelineMessage::Control(ControlMessage::Error { message, .. })) => {
                anyhow::bail!("Downstream stage error: {message}");
            }
            Ok(PipelineMessage::Control(ControlMessage::Ready { node_id })) => {
                info!(
                    "[stage] Downstream {} reported ready while request {} was waiting for tokens",
                    node_id, request_id
                );
            }
            Ok(other) => {
                warn!(
                    "[stage] Ignoring unexpected downstream response while handling {} on {}: {:?}",
                    request_id,
                    transport.node_id(),
                    other
                );
            }
            Err(err) => {
                anyhow::bail!("Downstream receive failed: {err}");
            }
        }
    }
}

async fn forward_activation_via_downstream(
    transport: &TransportNode,
    spec: &StagePrototypeSpec,
    downstream: &Arc<Mutex<Option<compute_network::transport::node::PeerConnection>>>,
    downstream_addr: Option<SocketAddr>,
    activation: ActivationPayload,
) -> Result<TokenPayload> {
    let request_id = activation.request_id.clone();
    let mut downstream_guard =
        ensure_downstream_connection(transport, downstream, downstream_addr).await?;
    let downstream_peer = downstream_guard.as_ref().ok_or_else(|| {
        anyhow::anyhow!("No downstream peer configured for stage {}", spec.stage_index)
    })?;

    info!(
        "[stage] Sending assign/control message for request {} to downstream {}",
        request_id,
        downstream_peer.remote_addr()
    );
    if let Err(err) = downstream_peer
        .send_activations(&PipelineMessage::Control(ControlMessage::AssignLayers {
            model_id: spec.model_name.clone(),
            start_layer: spec.start_layer,
            end_layer: spec.end_layer,
            total_layers: resolve_total_layers(&spec.model_name)?,
        }))
        .await
    {
        warn!("[stage] Failed to send assign/control to downstream: {err}");
        *downstream_guard = None;
        anyhow::bail!("Failed to send assign/control to downstream: {err}");
    }
    wait_for_downstream_ready(downstream_peer, &request_id).await?;

    let tokens = match request_downstream_tokens(
        transport,
        downstream_peer,
        &request_id,
        activation,
    )
    .await
    {
        Ok(tokens) => tokens,
        Err(err) => {
            *downstream_guard = None;
            return Err(err);
        }
    };
    drop(downstream_guard);
    Ok(tokens)
}

async fn wait_for_downstream_ready(
    downstream_peer: &compute_network::transport::node::PeerConnection,
    request_id: &str,
) -> Result<()> {
    match timeout(Duration::from_millis(750), downstream_peer.recv_activations()).await {
        Ok(Ok(PipelineMessage::Control(ControlMessage::Ready { node_id }))) => {
            info!("[stage] Downstream {} ready for request {}", node_id, request_id);
            Ok(())
        }
        Ok(Ok(PipelineMessage::Control(ControlMessage::Error { message, .. }))) => {
            anyhow::bail!("Downstream stage rejected assignment: {message}")
        }
        Ok(Ok(other)) => {
            warn!(
                "[stage] Expected downstream Ready after assignment for request {}, got {:?}; continuing",
                request_id, other
            );
            Ok(())
        }
        Ok(Err(err)) => {
            anyhow::bail!("Downstream receive failed while waiting for Ready: {err}")
        }
        Err(_) => {
            warn!(
                "[stage] Timed out waiting for downstream Ready after assignment for request {}; continuing",
                request_id
            );
            Ok(())
        }
    }
}

fn normalize_completion_text(
    text: &str,
    raw_completion_tokens: u32,
    completion_tokens: u32,
) -> String {
    let char_count = text.chars().count() as u32;
    if completion_tokens < raw_completion_tokens && char_count <= raw_completion_tokens {
        text.chars().take(completion_tokens as usize).collect()
    } else {
        text.to_string()
    }
}

fn trim_at_stop_sequence(text: &str, stop_sequences: &[String]) -> Option<String> {
    let stop_at = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()?;
    Some(text[..stop_at].to_string())
}

fn upstream_token_payload(
    request_id: String,
    tokens: &[GeneratedToken],
    timings: Option<PipelineTimingProfile>,
) -> TokenPayload {
    TokenPayload {
        request_id,
        tokens: tokens.iter().map(|t| t.token_id).collect(),
        is_finished: tokens.last().map(|t| t.is_finished).unwrap_or(false),
        text: None,
        timings,
    }
}

fn stage_timing_profile(
    spec: &StagePrototypeSpec,
    engine_ms: u128,
    total_ms: u128,
) -> PipelineTimingProfile {
    PipelineTimingProfile {
        stages: vec![PipelineStageTiming {
            stage_index: spec.stage_index,
            role: stage_role(spec).to_string(),
            engine_ms: engine_ms.min(u128::from(u64::MAX)) as u64,
            total_ms: total_ms.min(u128::from(u64::MAX)) as u64,
        }],
    }
}

fn append_stage_timing(
    tokens: &mut TokenPayload,
    spec: &StagePrototypeSpec,
    engine_ms: u128,
    total_ms: u128,
) {
    let mut timings = tokens.timings.take().unwrap_or_default();
    timings.stages.push(PipelineStageTiming {
        stage_index: spec.stage_index,
        role: stage_role(spec).to_string(),
        engine_ms: engine_ms.min(u128::from(u64::MAX)) as u64,
        total_ms: total_ms.min(u128::from(u64::MAX)) as u64,
    });
    timings.stages.sort_by_key(|stage| stage.stage_index);
    tokens.timings = Some(timings);
}

fn stage_role(spec: &StagePrototypeSpec) -> &'static str {
    if spec.stage_index == 0 {
        "head"
    } else if spec.stage_index + 1 == spec.total_stages {
        "tail"
    } else {
        "middle"
    }
}

async fn ensure_downstream_connection<'a>(
    transport: &TransportNode,
    downstream: &'a Arc<Mutex<Option<compute_network::transport::node::PeerConnection>>>,
    downstream_addr: Option<SocketAddr>,
) -> Result<tokio::sync::MutexGuard<'a, Option<compute_network::transport::node::PeerConnection>>> {
    let mut downstream_guard = downstream.lock().await;
    if downstream_guard.is_some() {
        return Ok(downstream_guard);
    }

    let addr = downstream_addr
        .ok_or_else(|| anyhow::anyhow!("No downstream address configured for stage runtime"))?;
    info!("[stage] Connecting to downstream peer {}", addr);
    let peer = transport
        .connect(addr)
        .await
        .with_context(|| format!("Failed to connect to downstream peer {addr}"))?;
    info!("[stage] Connected to downstream peer {}", peer.remote_addr());
    *downstream_guard = Some(peer);
    Ok(downstream_guard)
}

fn resolve_total_layers(model_name: &str) -> Result<u32> {
    let catalog = ModelCatalog::default_catalog();
    catalog
        .find(model_name)
        .map(|m| m.total_layers)
        .ok_or_else(|| anyhow::anyhow!("Unknown model in stage prototype: {model_name}"))
}

fn resolve_model_path(model_name: &str) -> Option<PathBuf> {
    let cache_dir = dirs::home_dir()?.join(".compute").join("models");

    if !cache_dir.exists() {
        return None;
    }

    let lower_model = model_name.to_lowercase();
    let segments: Vec<&str> = lower_model
        .split('-')
        .filter(|s| *s != "q4" && *s != "q8" && *s != "fp16" && *s != "q2")
        .collect();

    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".gguf") {
            continue;
        }

        let lower_name = name.to_lowercase();
        let all_match = segments.iter().all(|seg| lower_name.contains(seg));
        if all_match && !segments.is_empty() {
            return Some(entry.path());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{normalize_completion_text, trim_at_stop_sequence, upstream_token_payload};
    use crate::inference::engine::GeneratedToken;

    #[test]
    fn normalize_completion_text_truncates_char_per_token_output() {
        assert_eq!(normalize_completion_text("abcdef", 6, 3), "abc");
    }

    #[test]
    fn normalize_completion_text_handles_byte_counted_unicode_output() {
        assert_eq!(normalize_completion_text("できてできてでき", 24, 8), "できてできてでき");
        assert_eq!(normalize_completion_text("できてできてでき", 24, 4), "できてで");
    }

    #[test]
    fn normalize_completion_text_preserves_bpe_text_when_chars_exceed_tokens() {
        assert_eq!(
            normalize_completion_text("longer-than-token-count", 3, 2),
            "longer-than-token-count"
        );
    }

    #[test]
    fn upstream_token_payload_omits_plaintext_text() {
        let tokens = vec![
            GeneratedToken {
                request_id: "req".into(),
                token_id: 41,
                token_text: "a".into(),
                is_finished: false,
                logprob: None,
            },
            GeneratedToken {
                request_id: "req".into(),
                token_id: 42,
                token_text: "secret".into(),
                is_finished: true,
                logprob: None,
            },
        ];

        let payload = upstream_token_payload("req".into(), &tokens, None);
        assert_eq!(payload.request_id, "req");
        assert_eq!(payload.tokens, vec![41, 42]);
        assert!(payload.is_finished);
        assert_eq!(payload.text, None);
    }

    #[test]
    fn trim_at_stop_sequence_uses_earliest_match() {
        assert_eq!(
            trim_at_stop_sequence("hello<END>world", &[String::from("<END>")]),
            Some("hello".to_string())
        );
        assert_eq!(
            trim_at_stop_sequence("abcSTOPdefEND", &[String::from("END"), String::from("STOP")]),
            Some("abc".to_string())
        );
        assert_eq!(trim_at_stop_sequence("hello", &[String::from("")]), None);
    }
}
