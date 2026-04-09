use serde::{Deserialize, Serialize};

/// Message types exchanged between pipeline stages.
///
/// Wire protocol: all messages are length-prefixed (4-byte LE length + serialized payload).
/// Uses JSON serialization (future: investigate FP8 quantized transfer for activations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineMessage {
    /// Forward pass activations from one stage to the next.
    Activations(ActivationPayload),

    /// Sampled tokens flowing backward from the last stage to the first.
    Tokens(TokenPayload),

    /// Pipeline control messages.
    Control(ControlMessage),

    /// Health/status ping between pipeline peers.
    Ping(PingMessage),

    /// Response to a ping.
    Pong(PongMessage),
}

/// Activations (hidden states) flowing forward through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationPayload {
    /// Unique request ID.
    pub request_id: String,
    /// Sequence position in the generation.
    pub seq_position: u32,
    /// Micro-batch index.
    pub batch_index: u32,
    /// Shape: [batch_size, seq_len, hidden_dim].
    pub shape: Vec<usize>,
    /// Serialized tensor data (f16/f32 bytes).
    pub data: Vec<u8>,
    /// Data type (f16, f32, bf16).
    pub dtype: TensorDtype,
}

/// Data type of the serialized tensor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TensorDtype {
    Float16,
    Float32,
    BFloat16,
    Int8, // For quantized activation compression
}

/// Sampled tokens flowing backward to rank 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPayload {
    pub request_id: String,
    pub tokens: Vec<u32>,
    pub is_finished: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Pipeline control messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Pipeline setup: assign layers to this node.
    AssignLayers { model_id: String, start_layer: u32, end_layer: u32, total_layers: u32 },

    /// Pipeline ready: this node has loaded its layers.
    Ready { node_id: String },

    /// Pipeline teardown: release resources.
    Release { reason: String },

    /// Node is leaving the pipeline (graceful).
    Leaving { node_id: String },

    /// Error during pipeline execution.
    Error { node_id: String, message: String },
}

/// Latency ping message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingMessage {
    pub node_id: String,
    pub timestamp_ms: u64,
}

/// Pong response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PongMessage {
    pub node_id: String,
    pub timestamp_ms: u64,
    pub latency_ms: Option<u64>,
}

/// Encode a message as length-prefixed bytes.
pub fn encode_message(msg: &PipelineMessage) -> anyhow::Result<Vec<u8>> {
    let payload = serde_json::to_vec(msg)?;
    let len = (payload.len() as u32).to_le_bytes();
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len);
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decode a message from bytes (after length prefix has been read).
pub fn decode_message(data: &[u8]) -> anyhow::Result<PipelineMessage> {
    let msg: PipelineMessage = serde_json::from_slice(data)?;
    Ok(msg)
}

/// Read the 4-byte length prefix from a stream.
pub async fn read_length_prefix(recv: &mut quinn::RecvStream) -> anyhow::Result<u32> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await?;
    Ok(u32::from_le_bytes(len_buf))
}

/// Read a complete message from a QUIC stream.
pub async fn read_message(recv: &mut quinn::RecvStream) -> anyhow::Result<PipelineMessage> {
    let len = read_length_prefix(recv).await?;
    let mut buf = vec![0u8; len as usize];
    recv.read_exact(&mut buf).await?;
    decode_message(&buf)
}

/// Write a complete message to a QUIC stream.
pub async fn write_message(
    send: &mut quinn::SendStream,
    msg: &PipelineMessage,
) -> anyhow::Result<()> {
    let encoded = encode_message(msg)?;
    send.write_all(&encoded).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let msg = PipelineMessage::Ping(PingMessage {
            node_id: "test-node".into(),
            timestamp_ms: 1234567890,
        });

        let encoded = encode_message(&msg).unwrap();
        assert!(encoded.len() > 4);

        // Skip length prefix
        let payload = &encoded[4..];
        let decoded = decode_message(payload).unwrap();

        match decoded {
            PipelineMessage::Ping(ping) => {
                assert_eq!(ping.node_id, "test-node");
                assert_eq!(ping.timestamp_ms, 1234567890);
            }
            _ => panic!("Expected Ping message"),
        }
    }

    #[test]
    fn test_encode_activations() {
        let msg = PipelineMessage::Activations(ActivationPayload {
            request_id: "req-1".into(),
            seq_position: 0,
            batch_index: 0,
            shape: vec![1, 128, 4096],
            data: vec![0u8; 100],
            dtype: TensorDtype::Float16,
        });

        let encoded = encode_message(&msg).unwrap();
        let payload = &encoded[4..];
        let decoded = decode_message(payload).unwrap();

        match decoded {
            PipelineMessage::Activations(act) => {
                assert_eq!(act.request_id, "req-1");
                assert_eq!(act.shape, vec![1, 128, 4096]);
            }
            _ => panic!("Expected Activations message"),
        }
    }

    #[test]
    fn test_encode_control_assign_layers() {
        let msg = PipelineMessage::Control(ControlMessage::AssignLayers {
            model_id: "llama-70b".into(),
            start_layer: 16,
            end_layer: 31,
            total_layers: 80,
        });

        let encoded = encode_message(&msg).unwrap();
        let payload = &encoded[4..];
        let decoded = decode_message(payload).unwrap();

        match decoded {
            PipelineMessage::Control(ControlMessage::AssignLayers {
                model_id,
                start_layer,
                end_layer,
                total_layers,
            }) => {
                assert_eq!(model_id, "llama-70b");
                assert_eq!(start_layer, 16);
                assert_eq!(end_layer, 31);
                assert_eq!(total_layers, 80);
            }
            _ => panic!("Expected AssignLayers control message"),
        }
    }
}
