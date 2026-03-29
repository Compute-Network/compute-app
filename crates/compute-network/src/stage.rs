//! Pipeline stage runner.
//!
//! Connects the inference engine to the QUIC transport layer.
//! Each node runs one stage of the pipeline:
//!
//! 1. Receives activations (or initial prompt) from upstream
//! 2. Runs them through local model layers via the inference engine
//! 3. Forwards the result to the downstream node
//!
//! Handles the full lifecycle: setup, running, pausing, and teardown.

use anyhow::{Context, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, error, info};

use crate::transport::node::{PeerConnection, TransportNode};
use crate::transport::protocol::{
    ActivationPayload, ControlMessage, PipelineMessage, TensorDtype, TokenPayload,
};

/// Configuration for a pipeline stage.
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Pipeline ID this stage belongs to.
    pub pipeline_id: String,
    /// Model being served.
    pub model_id: String,
    /// This stage's index (0-based).
    pub stage_index: u32,
    /// Total stages in the pipeline.
    pub total_stages: u32,
    /// Layer range assigned to this stage (inclusive).
    pub start_layer: u32,
    pub end_layer: u32,
    /// Total layers in the model.
    pub total_layers: u32,
    /// Address of the upstream peer (None if first stage).
    pub upstream_addr: Option<SocketAddr>,
    /// Address of the downstream peer (None if last stage).
    pub downstream_addr: Option<SocketAddr>,
}

impl StageConfig {
    pub fn is_first_stage(&self) -> bool {
        self.stage_index == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.stage_index == self.total_stages - 1
    }

    pub fn num_layers(&self) -> u32 {
        self.end_layer - self.start_layer + 1
    }
}

/// Runtime stats for the stage.
#[derive(Debug, Clone, Default)]
pub struct StageStats {
    pub requests_processed: u64,
    pub tokens_generated: u64,
    pub activations_forwarded: u64,
    pub errors: u64,
    pub avg_forward_ms: f64,
}

/// Pipeline stage runner.
///
/// Manages the QUIC connections and routes activations
/// through the local inference engine.
pub struct StageRunner {
    config: StageConfig,
    transport: TransportNode,
    shutdown: Arc<AtomicBool>,
    stats: StageStats,
}

impl StageRunner {
    /// Create a new stage runner bound to the given address.
    pub async fn new(config: StageConfig, bind_addr: SocketAddr) -> Result<Self> {
        let transport = TransportNode::bind(bind_addr)
            .await
            .context("Failed to bind transport for stage runner")?;

        info!(
            "Stage {} bound on {} (layers {}-{}, pipeline {})",
            config.stage_index,
            transport.listen_addr(),
            config.start_layer,
            config.end_layer,
            config.pipeline_id
        );

        Ok(Self {
            config,
            transport,
            shutdown: Arc::new(AtomicBool::new(false)),
            stats: StageStats::default(),
        })
    }

    /// Get the local listening address.
    pub fn listen_addr(&self) -> SocketAddr {
        self.transport.listen_addr()
    }

    /// Get current stage stats.
    pub fn stats(&self) -> &StageStats {
        &self.stats
    }

    /// Signal the stage to shut down.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Run the stage — listen for upstream activations and process them.
    ///
    /// This is the main event loop for a pipeline stage. It:
    /// 1. Accepts connections from upstream (or orchestrator)
    /// 2. Receives activation tensors
    /// 3. Processes them through local layers (via callback)
    /// 4. Forwards results downstream
    ///
    /// The `process_fn` callback represents the inference engine's forward pass.
    /// It receives an activation and returns the processed result.
    pub async fn run<F>(&mut self, process_fn: F) -> Result<()>
    where
        F: Fn(ActivationPayload) -> Result<ProcessResult> + Send + Sync,
    {
        info!(
            "Stage {} running (layers {}-{})",
            self.config.stage_index, self.config.start_layer, self.config.end_layer
        );

        // Connect to downstream if not the last stage
        let downstream: Option<PeerConnection> = if let Some(addr) = self.config.downstream_addr {
            let conn =
                self.transport.connect(addr).await.context("Failed to connect to downstream")?;
            info!("Connected to downstream stage at {addr}");
            Some(conn)
        } else {
            None
        };

        // Accept and process incoming activations
        loop {
            if self.shutdown.load(Ordering::Relaxed) {
                info!("Stage {} shutting down", self.config.stage_index);
                break;
            }

            // Accept a connection from upstream (or orchestrator sending work)
            let peer = tokio::select! {
                result = self.transport.accept() => {
                    match result {
                        Ok(peer) => peer,
                        Err(e) => {
                            if self.shutdown.load(Ordering::Relaxed) {
                                break;
                            }
                            error!("Failed to accept connection: {e}");
                            self.stats.errors += 1;
                            continue;
                        }
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("Ctrl+C received, shutting down stage");
                    break;
                }
            };

            debug!("Accepted connection from {}", peer.remote_addr());

            // Process messages from this peer
            loop {
                let msg = match peer.recv_activations().await {
                    Ok(msg) => msg,
                    Err(_) => break, // Peer disconnected
                };

                match msg {
                    PipelineMessage::Activations(activation) => {
                        let start = std::time::Instant::now();

                        match process_fn(activation) {
                            Ok(ProcessResult::Activations(output)) => {
                                // Forward to downstream
                                if let Some(ref downstream_conn) = downstream {
                                    let msg = PipelineMessage::Activations(output);
                                    if let Err(e) = downstream_conn.send_activations(&msg).await {
                                        error!("Failed to forward activations: {e}");
                                        self.stats.errors += 1;
                                    } else {
                                        self.stats.activations_forwarded += 1;
                                    }
                                }
                            }
                            Ok(ProcessResult::Tokens(tokens)) => {
                                // Last stage: send tokens back via the peer connection
                                let msg = PipelineMessage::Tokens(tokens.clone());
                                if let Err(e) = peer.send_activations(&msg).await {
                                    error!("Failed to send tokens: {e}");
                                    self.stats.errors += 1;
                                } else {
                                    self.stats.tokens_generated += tokens.tokens.len() as u64;
                                }
                            }
                            Err(e) => {
                                error!("Forward pass failed: {e}");
                                self.stats.errors += 1;

                                // Send error back
                                let err_msg = PipelineMessage::Control(ControlMessage::Error {
                                    node_id: self.transport.node_id().to_string(),
                                    message: e.to_string(),
                                });
                                let _ = peer.send_activations(&err_msg).await;
                            }
                        }

                        self.stats.requests_processed += 1;
                        let elapsed = start.elapsed().as_millis() as f64;
                        // Running average
                        let n = self.stats.requests_processed as f64;
                        self.stats.avg_forward_ms =
                            self.stats.avg_forward_ms * ((n - 1.0) / n) + elapsed / n;
                    }
                    PipelineMessage::Control(ControlMessage::Release { reason }) => {
                        info!("Pipeline release: {reason}");
                        break;
                    }
                    PipelineMessage::Control(ControlMessage::AssignLayers {
                        start_layer,
                        end_layer,
                        ..
                    }) => {
                        info!("Layer assignment received: {start_layer}-{end_layer}");
                        // Could dynamically reassign layers here
                    }
                    PipelineMessage::Ping(ping) => {
                        let pong = PipelineMessage::Pong(crate::transport::protocol::PongMessage {
                            node_id: self.transport.node_id().to_string(),
                            timestamp_ms: ping.timestamp_ms,
                            latency_ms: None,
                        });
                        let _ = peer.send_activations(&pong).await;
                    }
                    other => {
                        debug!("Unexpected message: {other:?}");
                    }
                }
            }
        }

        // Cleanup
        if let Some(downstream_conn) = downstream {
            downstream_conn.close();
        }
        self.transport.close();

        info!(
            "Stage {} stopped. Processed {} requests, {} tokens, {} errors",
            self.config.stage_index,
            self.stats.requests_processed,
            self.stats.tokens_generated,
            self.stats.errors
        );

        Ok(())
    }
}

/// Result of processing an activation through local layers.
pub enum ProcessResult {
    /// Output activations to forward to the next stage.
    Activations(ActivationPayload),
    /// Generated tokens (from the final stage).
    Tokens(TokenPayload),
}

/// Helper to create a mock activation for testing.
pub fn mock_activation(request_id: &str, hidden_dim: usize) -> ActivationPayload {
    ActivationPayload {
        request_id: request_id.to_string(),
        seq_position: 0,
        batch_index: 0,
        shape: vec![1, 1, hidden_dim],
        data: vec![0u8; hidden_dim * 2], // f16 = 2 bytes per element
        dtype: TensorDtype::Float16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config() {
        let config = StageConfig {
            pipeline_id: "test-pipe".into(),
            model_id: "llama-70b".into(),
            stage_index: 1,
            total_stages: 3,
            start_layer: 27,
            end_layer: 53,
            total_layers: 80,
            upstream_addr: Some("127.0.0.1:9000".parse().unwrap()),
            downstream_addr: Some("127.0.0.1:9002".parse().unwrap()),
        };

        assert!(!config.is_first_stage());
        assert!(!config.is_last_stage());
        assert_eq!(config.num_layers(), 27);
    }

    #[test]
    fn test_stage_config_first_last() {
        let first = StageConfig {
            pipeline_id: "p".into(),
            model_id: "m".into(),
            stage_index: 0,
            total_stages: 3,
            start_layer: 0,
            end_layer: 26,
            total_layers: 80,
            upstream_addr: None,
            downstream_addr: Some("127.0.0.1:9001".parse().unwrap()),
        };
        assert!(first.is_first_stage());
        assert!(!first.is_last_stage());

        let last = StageConfig {
            pipeline_id: "p".into(),
            model_id: "m".into(),
            stage_index: 2,
            total_stages: 3,
            start_layer: 54,
            end_layer: 79,
            total_layers: 80,
            upstream_addr: Some("127.0.0.1:9001".parse().unwrap()),
            downstream_addr: None,
        };
        assert!(!last.is_first_stage());
        assert!(last.is_last_stage());
    }

    #[test]
    fn test_mock_activation() {
        let act = mock_activation("req-1", 4096);
        assert_eq!(act.request_id, "req-1");
        assert_eq!(act.shape, vec![1, 1, 4096]);
        assert_eq!(act.data.len(), 8192); // 4096 * 2 bytes
    }

    #[tokio::test]
    async fn test_stage_runner_bind() {
        let config = StageConfig {
            pipeline_id: "test".into(),
            model_id: "test-model".into(),
            stage_index: 0,
            total_stages: 1,
            start_layer: 0,
            end_layer: 31,
            total_layers: 32,
            upstream_addr: None,
            downstream_addr: None,
        };

        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let runner = StageRunner::new(config, addr).await.unwrap();
        assert_ne!(runner.listen_addr().port(), 0);
        assert_eq!(runner.stats().requests_processed, 0);
        runner.shutdown();
    }
}
