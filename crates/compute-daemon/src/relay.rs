//! WebSocket relay client — maintains a persistent connection to the orchestrator
//! so it can forward inference requests to the local llama-server.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time::sleep;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::config::Config;
use crate::stage_runtime::StagePrototypeClient;
use std::net::{Ipv4Addr, UdpSocket};

#[derive(Serialize)]
struct IdentifyMessage {
    r#type: String,
    node_id: String,
    wallet_address: String,
    token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    advertise_host: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct RelayRequest {
    id: String,
    r#type: String,
    method: String,
    path: String,
    body: serde_json::Value,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct RelayCancelMessage {
    id: String,
    r#type: String,
}

#[derive(Serialize)]
struct RelayResponse {
    id: String,
    r#type: String,
    status: u16,
    body: serde_json::Value,
}

/// Assignment pushed from orchestrator via WebSocket.
/// For multi-node pipelines using llama.cpp RPC:
/// - Stage 0 (head): receives `rpc_peers` list of "host:port" for worker nodes
/// - Stage 1+ (workers): receives `rpc_port` to listen on
/// - Single-stage: neither field is set (solo mode)
#[derive(Deserialize, Debug, Clone)]
pub struct AssignmentPush {
    pub pipeline_id: String,
    pub model_name: String,
    pub stage: i32,
    pub total_stages: i32,
    #[serde(default)]
    pub assignment_mode: Option<String>,
    #[serde(default)]
    pub shard_id: Option<String>,
    #[serde(default)]
    pub start_layer: Option<i32>,
    #[serde(default)]
    pub end_layer: Option<i32>,
    #[serde(default)]
    pub upstream_addr: Option<String>,
    #[serde(default)]
    pub downstream_addr: Option<String>,
    /// For head node (stage 0): list of "host:port" for worker rpc-servers
    #[serde(default)]
    pub rpc_peers: Option<Vec<String>>,
    /// For worker nodes (stage 1+): port to run rpc-server on
    #[serde(default)]
    pub rpc_port: Option<u16>,
}

/// Channel for sending messages back through the WebSocket from other tasks
pub type WsSender = tokio::sync::mpsc::Sender<String>;

fn detect_advertise_ip() -> Option<String> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((Ipv4Addr::new(1, 1, 1, 1), 80)).ok()?;
    let addr = socket.local_addr().ok()?;
    match addr.ip() {
        std::net::IpAddr::V4(ip) if !ip.is_loopback() && !ip.is_unspecified() => Some(ip.to_string()),
        _ => None,
    }
}

fn advertised_host_from_config(config: &Config) -> Option<String> {
    let configured = config.network.advertise_host.trim();
    if !configured.is_empty() {
        return Some(configured.to_string());
    }
    detect_advertise_ip()
}

pub struct RelayClient {
    ws_url: String,
    node_id: String,
    wallet_address: String,
    node_token: String,
    advertise_host: Option<String>,
    inference_port: u16,
    shutdown: Arc<AtomicBool>,
    last_latency_ms: Arc<AtomicU64>,
    /// Real tok/s from the last inference response (stored as f64 bits)
    last_tps: Arc<AtomicU64>,
    /// Number of successfully completed relay requests observed by this process
    completed_requests: Arc<AtomicU64>,
    /// Whether a relay request is actively being processed
    is_active: Arc<AtomicBool>,
    /// Whether the websocket relay is currently connected and identified.
    is_connected: Arc<AtomicBool>,
    /// Number of in-flight relay requests currently being processed
    active_requests: Arc<AtomicU32>,
    /// Channel to send assignment pushes to the runtime
    assignment_tx: tokio::sync::mpsc::Sender<AssignmentPush>,
    /// Channel for runtime to send messages through the WS (e.g. node_ready)
    outbound_rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<String>>>,
    /// Experimental stage runtime client for stage-based prototype requests.
    stage_client: Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>>,
}

impl RelayClient {
    pub fn last_latency_ms(&self) -> Arc<AtomicU64> {
        self.last_latency_ms.clone()
    }

    pub fn last_tps(&self) -> Arc<AtomicU64> {
        self.last_tps.clone()
    }

    pub fn completed_requests(&self) -> Arc<AtomicU64> {
        self.completed_requests.clone()
    }

    pub fn is_active(&self) -> Arc<AtomicBool> {
        self.is_active.clone()
    }

    pub fn is_connected(&self) -> Arc<AtomicBool> {
        self.is_connected.clone()
    }

    pub fn active_requests(&self) -> Arc<AtomicU32> {
        self.active_requests.clone()
    }
}

impl RelayClient {
    pub fn new(
        config: &Config,
        shutdown: Arc<AtomicBool>,
        assignment_tx: tokio::sync::mpsc::Sender<AssignmentPush>,
        outbound_rx: tokio::sync::mpsc::Receiver<String>,
        stage_client: Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>>,
    ) -> Self {
        // Convert https:// to wss:// (or http:// to ws://)
        let ws_url = config
            .network
            .orchestrator_url
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        let ws_url = format!("{ws_url}/v1/ws");

        Self {
            ws_url,
            node_id: config.wallet.node_id.clone(),
            wallet_address: config.wallet.public_address.clone(),
            node_token: config.wallet.node_token.clone(),
            advertise_host: advertised_host_from_config(config),
            inference_port: 8090,
            shutdown,
            last_latency_ms: Arc::new(AtomicU64::new(0)),
            last_tps: Arc::new(AtomicU64::new(0)),
            completed_requests: Arc::new(AtomicU64::new(0)),
            is_active: Arc::new(AtomicBool::new(false)),
            is_connected: Arc::new(AtomicBool::new(false)),
            active_requests: Arc::new(AtomicU32::new(0)),
            assignment_tx,
            outbound_rx: Arc::new(tokio::sync::Mutex::new(outbound_rx)),
            stage_client,
        }
    }

    /// Run the relay loop with automatic reconnection.
    pub async fn run(&self) -> Result<()> {
        if self.node_id.is_empty() || self.wallet_address.is_empty() || self.node_token.is_empty() {
            warn!("[relay] No node_id, wallet, or node token configured, relay disabled");
            return Ok(());
        }

        let mut backoff = Duration::from_secs(1);
        let max_backoff = Duration::from_secs(30);

        loop {
            if self.shutdown.load(Ordering::Relaxed) {
                info!("[relay] Shutting down");
                return Ok(());
            }

            match self.connect_and_run().await {
                Ok(()) => {
                    info!("[relay] Connection closed cleanly");
                    backoff = Duration::from_secs(1);
                }
                Err(e) => {
                    warn!("[relay] Connection error: {e}");
                }
            }

            if self.shutdown.load(Ordering::Relaxed) {
                return Ok(());
            }

            // Jitter: ±25% to prevent thundering herd when multiple nodes reconnect
            // Use cheap pseudo-random from system time nanos
            let jitter_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64;
            let jitter_ms = (jitter_nanos % (backoff.as_millis() as u64 / 2 + 1)) as i64
                - (backoff.as_millis() as i64 / 4);
            let actual_backoff = Duration::from_millis(
                (backoff.as_millis() as i64 + jitter_ms).max(500) as u64
            );
            info!("[relay] Reconnecting in {}ms...", actual_backoff.as_millis());
            sleep(actual_backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    async fn connect_and_run(&self) -> Result<()> {
        info!("[relay] Connecting to {}", self.ws_url);
        self.is_connected.store(false, Ordering::Relaxed);

        // Connection timeout prevents hanging on unreachable orchestrator
        let (ws_stream, _) = tokio::time::timeout(
            Duration::from_secs(10),
            connect_async(&self.ws_url),
        )
        .await
        .map_err(|_| anyhow::anyhow!("WebSocket connection timed out after 10s"))??;
        let (mut write, mut read) = ws_stream.split();

        // Send identify message
        let identify = IdentifyMessage {
            r#type: "identify".into(),
            node_id: self.node_id.clone(),
            wallet_address: self.wallet_address.clone(),
            token: self.node_token.clone(),
            advertise_host: self.advertise_host.clone(),
        };
        write.send(Message::Text(serde_json::to_string(&identify)?)).await?;

        info!("[relay] Connected and identified as {}", self.node_id);

        let http_client = reqwest::Client::new();
        let inference_port = self.inference_port;
        let shutdown = self.shutdown.clone();
        let outbound_rx = self.outbound_rx.clone();
        let mut outbound = outbound_rx.lock().await;
        let (response_tx, mut response_rx) = mpsc::channel::<String>(32);
        let active_cancels = Arc::new(Mutex::new(std::collections::HashMap::<String, oneshot::Sender<()>>::new()));

        // Process messages
        loop {
            if shutdown.load(Ordering::Relaxed) {
                let _ = write.send(Message::Close(None)).await;
                return Ok(());
            }

            let msg = tokio::select! {
                msg = read.next() => msg,
                Some(out_msg) = outbound.recv() => {
                    // Send outbound message (e.g. node_ready) through WS
                    let _ = write.send(Message::Text(out_msg)).await;
                    continue;
                }
                Some(relay_msg) = response_rx.recv() => {
                    let _ = write.send(Message::Text(relay_msg)).await;
                    continue;
                }
                _ = sleep(Duration::from_secs(15)) => {
                    // Send ping to keep connection alive
                    let _ = write.send(Message::Ping(vec![])).await;
                    continue;
                }
            };

            let msg = match msg {
                Some(Ok(msg)) => msg,
                Some(Err(e)) => return Err(e.into()),
                None => return Ok(()), // Stream ended
            };

            match msg {
                Message::Text(text) => {
                    // Parse as generic JSON to check type before deserializing specific structs
                    let msg_type = serde_json::from_str::<serde_json::Value>(&text)
                        .ok()
                        .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(String::from));

                    match msg_type.as_deref() {
                        Some("identified") => {
                            self.is_connected.store(true, Ordering::Relaxed);
                            info!("[relay] Orchestrator confirmed identification");
                        }
                        Some("assignment") => {
                            match serde_json::from_str::<AssignmentPush>(&text) {
                                Ok(assignment) => {
                                    info!("[relay] Assignment push: pipeline={} model={}", assignment.pipeline_id, assignment.model_name);
                                    // Reset health flag — new assignment may load a different model
                                    LLAMA_HEALTHY.store(false, Ordering::Relaxed);
                                    let _ = self.assignment_tx.try_send(assignment);
                                }
                                Err(e) => warn!("[relay] Failed to parse assignment: {e}"),
                            }
                        }
                        Some("inference_request") => {
                            let req: RelayRequest = match serde_json::from_str(&text) {
                                Ok(r) => r,
                                Err(e) => {
                                    warn!("[relay] Failed to parse inference request: {e}");
                                    continue;
                                }
                            };

                            self.is_active.store(true, Ordering::Relaxed);
                            self.active_requests.fetch_add(1, Ordering::Relaxed);
                            let is_stream = req.body.get("stream")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let start = std::time::Instant::now();
                            let response_tx = response_tx.clone();
                            let http_client = http_client.clone();
                            let stage_client = self.stage_client.clone();
                            let is_active = self.is_active.clone();
                            let active_requests = self.active_requests.clone();
                            let last_latency_ms = self.last_latency_ms.clone();
                            let last_tps = self.last_tps.clone();
                            let completed_requests = self.completed_requests.clone();
                            let active_cancels = active_cancels.clone();
                            let (cancel_tx, cancel_rx) = oneshot::channel();
                            active_cancels.lock().await.insert(req.id.clone(), cancel_tx);

                            tokio::spawn(async move {
                                if is_stream {
                                    let final_response = handle_streaming_request(
                                        &http_client, inference_port, &req, response_tx.clone(), cancel_rx, stage_client,
                                    ).await;
                                    let total_ms = start.elapsed().as_millis() as u64;
                                    let remaining = active_requests.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                                    is_active.store(remaining > 0, Ordering::Relaxed);
                                    let network_latency = total_ms.min(200);
                                    last_latency_ms.store(network_latency, Ordering::Relaxed);

                                    if let Some(usage) = final_response.body.get("usage") {
                                        let comp_tokens = usage.get("completion_tokens")
                                            .and_then(|v| v.as_u64()).unwrap_or(0);
                                        if comp_tokens > 0 && total_ms > 0 {
                                            let tps = (comp_tokens as f64 / total_ms as f64) * 1000.0;
                                            last_tps.store(tps.to_bits(), Ordering::Relaxed);
                                        }
                                    }
                                    if (200..400).contains(&final_response.status) {
                                        completed_requests.fetch_add(1, Ordering::Relaxed);
                                    }

                                    if let Ok(response_json) = serde_json::to_string(&final_response) {
                                        let _ = response_tx.send(response_json).await;
                                    }
                                } else {
                                    let response =
                                        handle_inference_request(&http_client, inference_port, &req, cancel_rx, stage_client).await;
                                    let total_ms = start.elapsed().as_millis() as u64;
                                    let remaining = active_requests.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
                                    is_active.store(remaining > 0, Ordering::Relaxed);

                                    let inference_ms = response.body.get("timings").map(|t| {
                                        let prompt = t.get("prompt_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                        let predicted = t.get("predicted_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                        (prompt + predicted) as u64
                                    }).unwrap_or(0);
                                    last_latency_ms.store(total_ms.saturating_sub(inference_ms), Ordering::Relaxed);

                                    if let Some(tps) = response.body.get("timings")
                                        .and_then(|t| t.get("predicted_per_second"))
                                        .and_then(|v| v.as_f64())
                                    {
                                        last_tps.store(tps.to_bits(), Ordering::Relaxed);
                                    }
                                    if (200..400).contains(&response.status) {
                                        completed_requests.fetch_add(1, Ordering::Relaxed);
                                    }

                                    if let Ok(response_json) = serde_json::to_string(&response) {
                                        let _ = response_tx.send(response_json).await;
                                    }
                                }

                                active_cancels.lock().await.remove(&req.id);
                            });
                        }
                        Some("inference_cancel") => {
                            let cancel: RelayCancelMessage = match serde_json::from_str(&text) {
                                Ok(c) => c,
                                Err(e) => {
                                    warn!("[relay] Failed to parse inference cancel: {e}");
                                    continue;
                                }
                            };
                            if let Some(cancel_tx) = active_cancels.lock().await.remove(&cancel.id) {
                                let _ = cancel_tx.send(());
                                info!("[relay] Cancelled request {}", cancel.id);
                            }
                        }
                        Some(other) => {
                            warn!("[relay] Unknown message type: {other}");
                        }
                        None => {
                            warn!("[relay] Message missing type field");
                        }
                    }
                }
                Message::Ping(data) => {
                    let _ = write.send(Message::Pong(data)).await;
                }
                Message::Close(frame) => {
                    self.is_connected.store(false, Ordering::Relaxed);
                    if let Some(frame) = frame {
                        match frame.code {
                            tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Library(4004) => {
                                warn!("[relay] Node session expired or invalid. Run `compute wallet login`.");
                            }
                            tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Library(4005) => {
                                warn!("[relay] Relay wallet mismatch. Re-authenticate with `compute wallet login`.");
                            }
                            _ => {}
                        }
                    }
                    return Ok(());
                }
                _ => {}
            }
        }
    }
}

/// Tracks whether llama-server has been confirmed healthy at least once.
/// Reset when a new model is loaded (assignment push with different model).
static LLAMA_HEALTHY: AtomicBool = AtomicBool::new(false);

pub fn mark_llama_unhealthy() {
    LLAMA_HEALTHY.store(false, Ordering::Relaxed);
}

/// Wait for llama-server to be healthy before forwarding a request.
/// Skips entirely if already confirmed healthy. Polls /health every 500ms, up to 60s.
async fn wait_for_healthy(client: &reqwest::Client, port: u16) {
    if LLAMA_HEALTHY.load(Ordering::Relaxed) {
        return;
    }
    for attempt in 0..120u32 {
        match client
            .get(format!("http://127.0.0.1:{port}/health"))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                if attempt > 0 {
                    info!("[relay] llama-server became healthy after {:.1}s", attempt as f64 * 0.5);
                }
                LLAMA_HEALTHY.store(true, Ordering::Relaxed);
                return;
            }
            _ => {}
        }
        sleep(Duration::from_millis(500)).await;
    }
    warn!("[relay] llama-server did not become healthy after 60s, proceeding anyway");
    LLAMA_HEALTHY.store(true, Ordering::Relaxed); // Don't block future requests
}

/// Handle a streaming inference request: forward SSE chunks from llama-server
/// over the WebSocket as individual `inference_stream_chunk` messages, then
/// send a final `inference_response` with aggregated usage.
async fn handle_streaming_request(
    client: &reqwest::Client,
    port: u16,
    request: &RelayRequest,
    response_tx: mpsc::Sender<String>,
    mut cancel_rx: oneshot::Receiver<()>,
    stage_client: Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>>,
) -> RelayResponse {
    if stage_client.lock().await.is_some() {
        return RelayResponse {
            id: request.id.clone(),
            r#type: "inference_response".into(),
            status: 501,
            body: serde_json::json!({"error": "Streaming is not implemented for experimental stage mode yet"}),
        };
    }

    // Wait for llama-server to be ready before forwarding
    wait_for_healthy(client, port).await;

    let url = format!("http://127.0.0.1:{port}{}", request.path);
    info!("[relay] Streaming {} {}", request.method, request.path);

    let resp = match tokio::select! {
        _ = &mut cancel_rx => {
            return RelayResponse {
                id: request.id.clone(),
                r#type: "inference_response".into(),
                status: 499,
                body: serde_json::json!({"error": "Request cancelled by client"}),
            };
        }
        result = client
            .post(&url)
            .json(&request.body)
            .timeout(Duration::from_secs(120))
            .send() => result
    } {
        Ok(r) => r,
        Err(e) => {
            return RelayResponse {
                id: request.id.clone(),
                r#type: "inference_response".into(),
                status: 502,
                body: serde_json::json!({"error": format!("Failed to reach llama-server: {e}")}),
            };
        }
    };

    let status = resp.status().as_u16();
    if status != 200 {
        let body = resp
            .text()
            .await
            .unwrap_or_default();
        return RelayResponse {
            id: request.id.clone(),
            r#type: "inference_response".into(),
            status,
            body: serde_json::json!({"error": body}),
        };
    }

    // Read SSE stream — optimized buffer management to minimize allocations
    let mut byte_stream = resp.bytes_stream();
    let mut buffer = String::with_capacity(4096); // Pre-allocate reasonable buffer
    let mut last_usage: Option<serde_json::Value> = None;
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;

    loop {
        let chunk_result = tokio::select! {
            _ = &mut cancel_rx => {
                return RelayResponse {
                    id: request.id.clone(),
                    r#type: "inference_response".into(),
                    status: 499,
                    body: serde_json::json!({"error": "Request cancelled by client"}),
                };
            }
            next = byte_stream.next() => next
        };

        let Some(chunk_result) = chunk_result else {
            break;
        };

        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                warn!("[relay] Stream read error: {e}");
                break;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE lines — drain from front without re-allocating
        let mut search_start = 0;
        while let Some(rel_pos) = buffer[search_start..].find('\n') {
            let pos = search_start + rel_pos;
            let line = buffer[search_start..pos].trim_end();

            if line.starts_with("data: ") {
                let data = &line[6..];

                // Forward SSE chunk to orchestrator via WebSocket
                // Build JSON manually to avoid struct serialization overhead
                let chunk_json = format!(
                    r#"{{"id":"{}","type":"inference_stream_chunk","chunk":"{}\\n\\n"}}"#,
                    request.id, line.replace('\\', "\\\\").replace('"', "\\\"")
                );
                let _ = response_tx.send(chunk_json).await;

                if data == "[DONE]" {
                    search_start = pos + 1;
                    break;
                }

                // Parse only for usage tracking (only chunks with "usage" key matter)
                if data.contains("usage") {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(usage) = parsed.get("usage") {
                            last_usage = Some(usage.clone());
                            prompt_tokens = usage.get("prompt_tokens")
                                .and_then(|v| v.as_u64()).unwrap_or(prompt_tokens);
                            completion_tokens = usage.get("completion_tokens")
                                .and_then(|v| v.as_u64()).unwrap_or(completion_tokens);
                        }
                    }
                }
            }

            search_start = pos + 1;
        }

        // Drain processed data from buffer (single allocation instead of per-line)
        if search_start > 0 {
            buffer.drain(..search_start);
        }
    }

    // Return aggregated final response
    RelayResponse {
        id: request.id.clone(),
        r#type: "inference_response".into(),
        status: 200,
        body: serde_json::json!({
            "stream_complete": true,
            "usage": last_usage.unwrap_or(serde_json::json!({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })),
        }),
    }
}

async fn handle_inference_request(
    client: &reqwest::Client,
    port: u16,
    request: &RelayRequest,
    mut cancel_rx: oneshot::Receiver<()>,
    stage_client: Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>>,
) -> RelayResponse {
    info!(
        "[relay] Received inference request {} path={} stream=false",
        request.id, request.path
    );
    if let Some(stage_client) = wait_for_stage_head_client(&stage_client, Duration::from_secs(2)).await {
        info!("[relay] Routing request {} to stage prototype head", request.id);
        return tokio::select! {
            _ = &mut cancel_rx => {
                RelayResponse {
                    id: request.id.clone(),
                    r#type: "inference_response".into(),
                    status: 499,
                    body: serde_json::json!({"error": "Request cancelled by client"}),
                }
            }
            result = handle_stage_prototype_request(&stage_client, request) => result
        };
    }

    if let Some(stage_client) = stage_client.lock().await.clone() {
        if stage_client.is_head_stage() {
            info!("[relay] Routing request {} to already-ready stage prototype head", request.id);
            return tokio::select! {
                _ = &mut cancel_rx => {
                    RelayResponse {
                        id: request.id.clone(),
                        r#type: "inference_response".into(),
                        status: 499,
                        body: serde_json::json!({"error": "Request cancelled by client"}),
                    }
                }
                result = handle_stage_prototype_request(&stage_client, request) => result
            };
        }
    }

    // Wait for llama-server to be ready before forwarding
    info!("[relay] Routing request {} to local llama-server fallback", request.id);
    wait_for_healthy(client, port).await;

    let url = format!("http://127.0.0.1:{port}{}", request.path);
    tracing::debug!("[relay] Proxying {} {}", request.method, request.path);

    // Retry on 503 (server busy) — llama-server single slot may still be cleaning up
    let max_retries = 5;
    for attempt in 0..=max_retries {
        match tokio::select! {
            _ = &mut cancel_rx => {
                return RelayResponse {
                    id: request.id.clone(),
                    r#type: "inference_response".into(),
                    status: 499,
                    body: serde_json::json!({"error": "Request cancelled by client"}),
                };
            }
            result = client.post(&url).json(&request.body).timeout(Duration::from_secs(120)).send() => result
        } {
            Ok(resp) => {
                let status = resp.status().as_u16();
                // Read body as text first so we can log errors
                let body_text = resp.text().await.unwrap_or_default();

                if status != 200 {
                    warn!("[relay] llama-server responded {status}: {}", &body_text[..body_text.len().min(1000)]);
                }

                if status == 503 && attempt < max_retries {
                    info!("[relay] llama-server busy (503), retry {}/{} in {}ms", attempt + 1, max_retries, 500 * (attempt + 1));
                    tokio::time::sleep(Duration::from_millis(500 * (attempt as u64 + 1))).await;
                    continue;
                }

                return match serde_json::from_str::<serde_json::Value>(&body_text) {
                    Ok(body) => RelayResponse {
                        id: request.id.clone(),
                        r#type: "inference_response".into(),
                        status,
                        body,
                    },
                    Err(e) => RelayResponse {
                        id: request.id.clone(),
                        r#type: "inference_response".into(),
                        status: 502,
                        body: serde_json::json!({"error": format!("Failed to parse response: {e}")}),
                    },
                };
            }
            Err(e) => {
                if attempt < max_retries {
                    info!("[relay] llama-server unreachable, retry {}/{}", attempt + 1, max_retries);
                    tokio::time::sleep(Duration::from_millis(500 * (attempt as u64 + 1))).await;
                    continue;
                }
                return RelayResponse {
                    id: request.id.clone(),
                    r#type: "inference_response".into(),
                    status: 502,
                    body: serde_json::json!({"error": format!("Failed to reach llama-server: {e}")}),
                };
            }
        }
    }
    unreachable!()
}

async fn wait_for_stage_head_client(
    stage_client: &Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>>,
    timeout: Duration,
) -> Option<StagePrototypeClient> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(client) = stage_client.lock().await.clone() {
            if client.is_head_stage() {
                return Some(client);
            }
        }

        if std::time::Instant::now() >= deadline {
            return None;
        }

        sleep(Duration::from_millis(100)).await;
    }
}

async fn handle_stage_prototype_request(
    stage_client: &StagePrototypeClient,
    request: &RelayRequest,
) -> RelayResponse {
    let prompt = extract_request_prompt(&request.body);
    let max_tokens = request
        .body
        .get("max_tokens")
        .or_else(|| request.body.get("max_completion_tokens"))
        .and_then(|value| value.as_u64())
        .map(|value| value as u32);

    info!(
        "[stage] Head handling request {} prompt_len={} max_tokens={:?}",
        request.id,
        prompt.len(),
        max_tokens
    );

    match stage_client
        .complete_prompt(request.id.clone(), prompt, max_tokens)
        .await
    {
        Ok(result) => {
            info!(
                "[stage] Head completed request {} completion_tokens={}",
                request.id,
                result.completion_tokens
            );
            RelayResponse {
            id: request.id.clone(),
            r#type: "inference_response".into(),
            status: 200,
            body: serde_json::json!({
                "id": request.id,
                "object": "chat.completion",
                "model": result.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.content,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                },
                "prototype_stage_mode": true,
            }),
        }
        },
        Err(err) => RelayResponse {
            id: request.id.clone(),
            r#type: "inference_response".into(),
            status: 502,
            body: serde_json::json!({"error": format!("Stage prototype request failed: {err}")}),
        },
    }
}

fn extract_request_prompt(body: &serde_json::Value) -> String {
    if let Some(prompt) = body.get("prompt").and_then(|value| value.as_str()) {
        return prompt.to_string();
    }

    if let Some(messages) = body.get("messages").and_then(|value| value.as_array()) {
        let combined = messages
            .iter()
            .filter_map(|message| {
                let role = message.get("role").and_then(|value| value.as_str()).unwrap_or("user");
                let content = message.get("content")?;
                if let Some(text) = content.as_str() {
                    Some(format!("{role}: {text}"))
                } else if let Some(parts) = content.as_array() {
                    let text = parts
                        .iter()
                        .filter_map(|part| {
                            if part.get("type").and_then(|value| value.as_str()) == Some("text") {
                                part.get("text").and_then(|value| value.as_str()).map(str::to_string)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    if text.is_empty() {
                        None
                    } else {
                        Some(format!("{role}: {text}"))
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !combined.is_empty() {
            return combined;
        }
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::extract_request_prompt;

    #[test]
    fn extract_request_prompt_prefers_prompt_field() {
        let body = serde_json::json!({
            "prompt": "hello from prompt",
            "messages": [
                {"role": "user", "content": "ignored"}
            ]
        });
        assert_eq!(extract_request_prompt(&body), "hello from prompt");
    }

    #[test]
    fn extract_request_prompt_flattens_chat_messages() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Say hi"}
            ]
        });
        assert_eq!(
            extract_request_prompt(&body),
            "system: You are terse.\nuser: Say hi"
        );
    }

    #[test]
    fn extract_request_prompt_flattens_text_parts() {
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part one"},
                        {"type": "image_url", "image_url": {"url": "ignored"}},
                        {"type": "text", "text": "Part two"}
                    ]
                }
            ]
        });
        assert_eq!(extract_request_prompt(&body), "user: Part one Part two");
    }
}
