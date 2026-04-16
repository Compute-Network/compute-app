use crate::inference::stage_backend::encode_stage_prompt;
use anyhow::Result;
use compute_network::transport::node::TransportNode;
use compute_network::transport::protocol::{
    ActivationPayload, ControlMessage, PipelineMessage, TensorDtype,
};
use std::net::{Ipv4Addr, SocketAddr};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct PrototypeHostedRun {
    pub elapsed_ms: u128,
    pub prompt_tokens: usize,
    pub completion_token_ids: Vec<u32>,
    pub finish_reason: String,
    pub content: String,
}

pub async fn request_hosted_prototype_head(
    head_addr: SocketAddr,
    prompt: &str,
    max_tokens: Option<u32>,
    stop_sequences: &[String],
    hidden_dim: usize,
) -> Result<PrototypeHostedRun> {
    let request_id =
        format!("prototype-client-{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis());
    let payload = encode_stage_prompt(prompt, max_tokens)?;
    let prompt_tokens = prompt.as_bytes().len().max(1);

    let node = TransportNode::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).await?;
    let peer = node.connect(head_addr).await?;
    let request = PipelineMessage::Activations(ActivationPayload {
        request_id,
        seq_position: 0,
        batch_index: 0,
        shape: vec![1, prompt_tokens, hidden_dim.max(1)],
        data: payload,
        dtype: TensorDtype::Float16,
    });

    let start = Instant::now();
    peer.send_activations(&request).await?;
    let reply = peer.recv_activations().await?;
    let elapsed_ms = start.elapsed().as_millis();

    let result = match reply {
        PipelineMessage::Tokens(tokens) => {
            let token_limit = max_tokens.map(|value| value as usize).unwrap_or(tokens.tokens.len());
            let completion_token_ids =
                tokens.tokens.iter().copied().take(token_limit).collect::<Vec<_>>();
            let raw_text =
                tokens.text.unwrap_or_else(|| detokenize_prototype(&completion_token_ids));
            let (content, stopped) = trim_at_stop_sequence(&raw_text, stop_sequences);
            let was_trimmed_to_limit =
                max_tokens.is_some() && completion_token_ids.len() < tokens.tokens.len();
            let finish_reason = if was_trimmed_to_limit {
                "length".to_string()
            } else if stopped || tokens.is_finished {
                "stop".to_string()
            } else {
                "unknown".to_string()
            };
            PrototypeHostedRun {
                elapsed_ms,
                prompt_tokens,
                completion_token_ids,
                finish_reason,
                content,
            }
        }
        PipelineMessage::Control(ControlMessage::Error { message, .. }) => {
            anyhow::bail!("Stage runtime returned error: {message}");
        }
        other => {
            anyhow::bail!("Expected token payload from head stage, got {other:?}");
        }
    };

    peer.close();
    node.close();
    Ok(result)
}

pub fn detokenize_prototype(token_ids: &[u32]) -> String {
    let bytes = token_ids.iter().map(|token| (*token).min(255) as u8).collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).to_string()
}

pub fn trim_at_stop_sequence(text: &str, stop_sequences: &[String]) -> (String, bool) {
    let stop_at = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min();
    match stop_at {
        Some(index) => (text[..index].to_string(), true),
        None => (text.to_string(), false),
    }
}

#[cfg(test)]
mod tests {
    use super::trim_at_stop_sequence;

    #[test]
    fn trim_uses_earliest_stop_match() {
        let stop_sequences = vec!["END".to_string(), "STOP".to_string()];
        let (trimmed, stopped) = trim_at_stop_sequence("abcSTOPxyzEND", &stop_sequences);
        assert!(stopped);
        assert_eq!(trimmed, "abc");
    }
}
