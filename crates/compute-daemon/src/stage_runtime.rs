use anyhow::{Context, Result};
use compute_network::models::ModelCatalog;
use compute_network::transport::protocol::{ActivationPayload, ControlMessage, PipelineMessage, PongMessage};
use compute_network::transport::node::TransportNode;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::config::Config;
use crate::hardware::HardwareInfo;
use crate::inference::engine::{Activation, ForwardResult, ShardConfig as EngineShardConfig};
use crate::inference::stage_backend::{StageBackendKind, StageExecutionBackend, encode_stage_prompt};

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
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug)]
enum StagePrototypeCommand {
    Complete {
        request_id: String,
        prompt: String,
        max_tokens: Option<u32>,
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
    ) -> Result<StagePrototypeResponse> {
        info!(
            "[stage] Queueing head request {} stage={}/{} prompt_len={} max_tokens={:?}",
            request_id,
            self.spec.stage_index + 1,
            self.spec.total_stages,
            prompt.len(),
            max_tokens
        );
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(StagePrototypeCommand::Complete {
                request_id,
                prompt,
                max_tokens,
                response_tx,
            })
            .await
            .context("Stage prototype runtime is unavailable")?;
        response_rx
            .await
            .context("Stage prototype runtime dropped completion response")?
    }
}

pub async fn start_stage_prototype(
    config: &Config,
    hw: &HardwareInfo,
    spec: StagePrototypeSpec,
) -> Result<StagePrototypeHandle> {
    let total_layers = resolve_total_layers(&spec.model_name)?;

    let stage_backend = StageBackendKind::parse(&config.experimental.stage_backend);
    let mut engine = StageExecutionBackend::new_for_hardware(hw, stage_backend);
    let shard_path = match stage_backend {
        StageBackendKind::Prototype => std::path::PathBuf::from(format!(
            "prototype://{}:{}-{}",
            spec.model_name, spec.start_layer, spec.end_layer
        )),
        StageBackendKind::TailLlama => {
            if spec.stage_index + 1 == spec.total_stages {
                resolve_model_path(&spec.model_name)
                    .with_context(|| format!("No local GGUF found for stage prototype model {}", spec.model_name))?
            } else {
                std::path::PathBuf::from(format!(
                    "prototype://{}:{}-{}",
                    spec.model_name, spec.start_layer, spec.end_layer
                ))
            }
        }
        StageBackendKind::LlamaCpp => resolve_model_path(&spec.model_name)
            .with_context(|| format!("No local GGUF found for stage prototype model {}", spec.model_name))?,
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

    let bind_addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, STAGE_RUNTIME_LISTEN_PORT));
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
        engine.backend_name(),
        spec.upstream_addr,
        spec.downstream_addr
    );

    let downstream_addr = spec
        .downstream_addr
        .as_deref()
        .and_then(|addr| addr.parse::<SocketAddr>().ok());

    let engine = Arc::new(Mutex::new(engine));
    let (request_tx, mut request_rx) = mpsc::channel::<StagePrototypeCommand>(8);
    let client = StagePrototypeClient {
        spec: spec.clone(),
        request_tx,
    };
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
    let spec_for_task = spec.clone();
    let task = tokio::spawn(async move {
        let downstream = Arc::new(Mutex::new(None::<compute_network::transport::node::PeerConnection>));

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    break;
                }
                Some(command) = request_rx.recv() => {
                    match command {
                        StagePrototypeCommand::Complete { request_id, prompt, max_tokens, response_tx } => {
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

                                        let result = {
                                            let engine = engine.lock().await;
                                            engine.forward(input).await
                                        };

                                        match result {
                                            Ok(ForwardResult::Activations(output)) => {
                                                let downstream_guard = downstream.lock().await;
                                                if let Some(ref downstream_peer) = *downstream_guard {
                                                    info!(
                                                        "[stage] Forwarding activations for request {} downstream to {}",
                                                        output.request_id,
                                                        downstream_peer.remote_addr()
                                                    );
                                                    let payload = PipelineMessage::Activations(ActivationPayload {
                                                        request_id: output.request_id,
                                                        seq_position: output.seq_position,
                                                        batch_index: output.batch_index,
                                                        shape: output.shape,
                                                        data: output.data,
                                                        dtype: compute_network::transport::protocol::TensorDtype::Float16,
                                                    });
                                                    if let Err(err) = downstream_peer.send_activations(&payload).await {
                                                        warn!("[stage] Failed to forward activations downstream: {err}");
                                                    }
                                                } else {
                                                    warn!("[stage] Activation output produced with no downstream peer configured");
                                                }
                                            }
                                            Ok(ForwardResult::Tokens(tokens)) => {
                                                info!(
                                                    "[stage] Returning {} token(s) upstream for request {}",
                                                    tokens.len(),
                                                    activation.request_id
                                                );
                                                let payload = PipelineMessage::Tokens(compute_network::transport::protocol::TokenPayload {
                                                    request_id: activation.request_id,
                                                    tokens: tokens.iter().map(|t| t.token_id).collect(),
                                                    is_finished: tokens.last().map(|t| t.is_finished).unwrap_or(false),
                                                    text: Some(tokens.iter().map(|t| t.token_text.as_str()).collect::<String>()),
                                                });
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

    Ok(StagePrototypeHandle {
        client,
        spec,
        listen_addr,
        shutdown_tx: Some(shutdown_tx),
        task,
    })
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
) -> Result<StagePrototypeResponse> {
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

    let prompt_tokens = {
        let engine = engine.lock().await;
        engine.tokenize(&prompt).await?
    };

    let initial_input = Activation {
        request_id: request_id.clone(),
        shape: vec![1, prompt_tokens.len().max(1), 2048],
        data: encode_stage_prompt(&prompt, max_tokens)?,
        seq_position: 0,
        batch_index: 0,
    };

    let head_result = {
        let engine = engine.lock().await;
        engine.forward(initial_input).await?
    };

    let activation = match head_result {
        ForwardResult::Activations(activation) => ActivationPayload {
            request_id: activation.request_id,
            seq_position: activation.seq_position,
            batch_index: activation.batch_index,
            shape: activation.shape,
            data: activation.data,
            dtype: compute_network::transport::protocol::TensorDtype::Float16,
        },
        ForwardResult::Tokens(tokens) => {
            let raw_completion_tokens = tokens.len() as u32;
            let completion_tokens = max_tokens
                .map(|limit| raw_completion_tokens.min(limit))
                .unwrap_or(raw_completion_tokens);
            let token_slice = &tokens[..completion_tokens as usize];
            let content = {
                let engine = engine.lock().await;
                engine
                    .detokenize(&token_slice.iter().map(|t| t.token_id).collect::<Vec<_>>())
                    .await
                    .unwrap_or_else(|_| token_slice.iter().map(|t| t.token_text.clone()).collect::<Vec<_>>().join(""))
            };

            return Ok(StagePrototypeResponse {
                model_name: spec.model_name.clone(),
                content,
                prompt_tokens: prompt_tokens.len() as u32,
                completion_tokens,
                total_tokens: prompt_tokens.len() as u32 + completion_tokens,
            });
        }
    };

    let mut downstream_guard = ensure_downstream_connection(transport, downstream, downstream_addr).await?;
    let downstream_peer = downstream_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No downstream peer configured for head stage"))?;
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
    info!(
        "[stage] Sending activation payload for request {} to downstream {}",
        request_id,
        downstream_peer.remote_addr()
    );
    if let Err(err) = downstream_peer
        .send_activations(&PipelineMessage::Activations(activation))
        .await
    {
        warn!("[stage] Failed to send activations to downstream: {err}");
        *downstream_guard = None;
        anyhow::bail!("Failed to send activations to downstream: {err}");
    }
    info!("[stage] Waiting for downstream tokens for request {}", request_id);

    let tokens = loop {
        match downstream_peer.recv_activations().await {
            Ok(PipelineMessage::Tokens(tokens)) if tokens.request_id == request_id => {
                info!(
                    "[stage] Received downstream tokens for request {} count={}",
                    request_id,
                    tokens.tokens.len()
                );
                break tokens
            }
            Ok(PipelineMessage::Control(ControlMessage::Error { message, .. })) => {
                anyhow::bail!("Downstream stage error: {message}");
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
                warn!("[stage] Downstream receive failed for request {}: {err}", request_id);
                *downstream_guard = None;
                anyhow::bail!("Downstream receive failed: {err}");
            }
        }
    };
    drop(downstream_guard);

    let raw_completion_tokens = tokens.tokens.len() as u32;
    let completion_tokens = max_tokens
        .map(|limit| raw_completion_tokens.min(limit))
        .unwrap_or(raw_completion_tokens);
    let content = if let Some(text) = tokens.text.clone() {
        text.chars().take(completion_tokens as usize).collect::<String>()
    } else {
        let token_slice = &tokens.tokens[..completion_tokens as usize];
        let engine = engine.lock().await;
        engine
            .detokenize(token_slice)
            .await
            .unwrap_or_else(|_| token_slice.iter().map(|tok| tok.to_string()).collect::<Vec<_>>().join(" "))
    };

    Ok(StagePrototypeResponse {
        model_name: spec.model_name.clone(),
        content,
        prompt_tokens: prompt_tokens.len() as u32,
        completion_tokens,
        total_tokens: prompt_tokens.len() as u32 + completion_tokens,
    })
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

    let addr = downstream_addr.ok_or_else(|| anyhow::anyhow!("No downstream address configured for head stage"))?;
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
