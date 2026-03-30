//! WebSocket relay client — maintains a persistent connection to the orchestrator
//! so it can forward inference requests to the local llama-server.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::config::Config;

#[derive(Serialize)]
struct IdentifyMessage {
    r#type: String,
    node_id: String,
    wallet_address: String,
}

#[derive(Deserialize)]
struct RelayRequest {
    id: String,
    r#type: String,
    method: String,
    path: String,
    body: serde_json::Value,
}

#[derive(Serialize)]
struct RelayResponse {
    id: String,
    r#type: String,
    status: u16,
    body: serde_json::Value,
}

pub struct RelayClient {
    ws_url: String,
    node_id: String,
    wallet_address: String,
    inference_port: u16,
    shutdown: Arc<AtomicBool>,
    last_latency_ms: Arc<AtomicU64>,
    /// Real tok/s from the last inference response (stored as f64 bits)
    last_tps: Arc<AtomicU64>,
    /// Whether a relay request is actively being processed
    is_active: Arc<AtomicBool>,
}

impl RelayClient {
    pub fn last_latency_ms(&self) -> Arc<AtomicU64> {
        self.last_latency_ms.clone()
    }

    pub fn last_tps(&self) -> Arc<AtomicU64> {
        self.last_tps.clone()
    }

    pub fn is_active(&self) -> Arc<AtomicBool> {
        self.is_active.clone()
    }
}

impl RelayClient {
    pub fn new(config: &Config, shutdown: Arc<AtomicBool>) -> Self {
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
            inference_port: 8090,
            shutdown,
            last_latency_ms: Arc::new(AtomicU64::new(0)),
            last_tps: Arc::new(AtomicU64::new(0)),
            is_active: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run the relay loop with automatic reconnection.
    pub async fn run(&self) -> Result<()> {
        if self.node_id.is_empty() || self.wallet_address.is_empty() {
            warn!("[relay] No node_id or wallet configured, relay disabled");
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

            info!("[relay] Reconnecting in {}s...", backoff.as_secs());
            sleep(backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    async fn connect_and_run(&self) -> Result<()> {
        info!("[relay] Connecting to {}", self.ws_url);

        let (ws_stream, _) = connect_async(&self.ws_url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Send identify message
        let identify = IdentifyMessage {
            r#type: "identify".into(),
            node_id: self.node_id.clone(),
            wallet_address: self.wallet_address.clone(),
        };
        write.send(Message::Text(serde_json::to_string(&identify)?)).await?;

        info!("[relay] Connected and identified as {}", self.node_id);

        let http_client = reqwest::Client::new();
        let inference_port = self.inference_port;
        let shutdown = self.shutdown.clone();

        // Process messages
        loop {
            if shutdown.load(Ordering::Relaxed) {
                let _ = write.send(Message::Close(None)).await;
                return Ok(());
            }

            let msg = tokio::select! {
                msg = read.next() => msg,
                _ = sleep(Duration::from_secs(30)) => {
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
                    let request: Result<RelayRequest, _> = serde_json::from_str(&text);
                    match request {
                        Ok(req) if req.r#type == "inference_request" => {
                            self.is_active.store(true, Ordering::Relaxed);

                            let start = std::time::Instant::now();
                            let response =
                                handle_inference_request(&http_client, inference_port, &req).await;
                            let total_ms = start.elapsed().as_millis() as u64;

                            self.is_active.store(false, Ordering::Relaxed);

                            // Latency = total time - inference time
                            let inference_ms = response
                                .body
                                .get("timings")
                                .map(|t| {
                                    let prompt =
                                        t.get("prompt_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    let predicted = t
                                        .get("predicted_ms")
                                        .and_then(|v| v.as_f64())
                                        .unwrap_or(0.0);
                                    (prompt + predicted) as u64
                                })
                                .unwrap_or(0);
                            let network_latency = total_ms.saturating_sub(inference_ms);
                            self.last_latency_ms.store(network_latency, Ordering::Relaxed);

                            // Store real tps from response
                            if let Some(tps) = response
                                .body
                                .get("timings")
                                .and_then(|t| t.get("predicted_per_second"))
                                .and_then(|v| v.as_f64())
                            {
                                self.last_tps.store(tps.to_bits(), Ordering::Relaxed);
                            }

                            let response_json = serde_json::to_string(&response)?;
                            write.send(Message::Text(response_json)).await?;
                        }
                        Ok(msg) => {
                            // identified, pong, etc — ignore
                            if msg.r#type != "identified" {
                                warn!("[relay] Unknown message type: {}", msg.r#type);
                            }
                        }
                        Err(e) => {
                            warn!("[relay] Failed to parse message: {e}");
                        }
                    }
                }
                Message::Ping(data) => {
                    let _ = write.send(Message::Pong(data)).await;
                }
                Message::Close(_) => return Ok(()),
                _ => {}
            }
        }
    }
}

async fn handle_inference_request(
    client: &reqwest::Client,
    port: u16,
    request: &RelayRequest,
) -> RelayResponse {
    let url = format!("http://127.0.0.1:{port}{}", request.path);
    info!("[relay] Proxying {} {}", request.method, request.path);

    match client.post(&url).json(&request.body).timeout(Duration::from_secs(120)).send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.json::<serde_json::Value>().await {
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
                    body: serde_json::json!({"error": format!("Failed to read response: {e}")}),
                },
            }
        }
        Err(e) => RelayResponse {
            id: request.id.clone(),
            r#type: "inference_response".into(),
            status: 502,
            body: serde_json::json!({"error": format!("Failed to reach llama-server: {e}")}),
        },
    }
}
