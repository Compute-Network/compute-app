use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::hardware::HardwareInfo;
use crate::inference::engine::{
    detect_backend, Activation, ForwardResult, GeneratedToken, InferenceEngine, ShardConfig,
};
use crate::inference::llamacpp::LlamaCppEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageBackendKind {
    Prototype,
    LlamaCpp,
}

impl StageBackendKind {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "llamacpp" | "llama.cpp" | "llama" => Self::LlamaCpp,
            _ => Self::Prototype,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Prototype => "prototype",
            Self::LlamaCpp => "llamacpp",
        }
    }
}

/// Execution backend for the experimental stage runtime.
///
/// This isolates stage transport/runtime code from any specific serving engine so
/// the prototype can swap away from llama.cpp without disturbing the rest of the
/// multi-node plumbing.
pub enum StageExecutionBackend {
    Prototype(PrototypeStageEngine),
    LlamaCpp(LlamaCppEngine),
}

impl StageExecutionBackend {
    pub fn new_for_hardware(hw: &HardwareInfo, kind: StageBackendKind) -> Self {
        match kind {
            StageBackendKind::Prototype => Self::Prototype(PrototypeStageEngine::default()),
            StageBackendKind::LlamaCpp => Self::LlamaCpp(LlamaCppEngine::new(detect_backend(hw))),
        }
    }

    pub async fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        match self {
            Self::Prototype(engine) => engine.load_shard(config),
            Self::LlamaCpp(engine) => engine.load_shard(config).await,
        }
    }

    pub async fn unload(&mut self) -> Result<()> {
        match self {
            Self::Prototype(engine) => engine.unload(),
            Self::LlamaCpp(engine) => engine.unload().await,
        }
    }

    pub async fn forward(&self, input: Activation) -> Result<ForwardResult> {
        match self {
            Self::Prototype(engine) => engine.forward(input),
            Self::LlamaCpp(engine) => engine.forward(input).await,
        }
    }

    pub async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Prototype(engine) => engine.tokenize(text),
            Self::LlamaCpp(engine) => engine.tokenize(text).await,
        }
    }

    pub async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::Prototype(engine) => engine.detokenize(tokens),
            Self::LlamaCpp(engine) => engine.detokenize(tokens).await,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Prototype(_) => "prototype",
            Self::LlamaCpp(_) => "llamacpp",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrototypePayload {
    prompt: String,
    stages: Vec<String>,
}

#[derive(Default)]
pub struct PrototypeStageEngine {
    shard_config: Option<ShardConfig>,
}

impl PrototypeStageEngine {
    fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        self.shard_config = Some(config.clone());
        Ok(())
    }

    fn unload(&mut self) -> Result<()> {
        self.shard_config = None;
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.as_bytes().iter().map(|byte| *byte as u32).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        let bytes = tokens
            .iter()
            .map(|token| (*token).min(255) as u8)
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    fn forward(&self, input: Activation) -> Result<ForwardResult> {
        let config = self
            .shard_config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Prototype stage backend has no shard loaded"))?;

        if config.is_last_stage {
            let payload = parse_payload(&input.data)?;
            let stage_summary = if payload.stages.is_empty() {
                "none".to_string()
            } else {
                payload.stages.join(" -> ")
            };
            let content = format!(
                "Prototype stage completion for {}: {} [stages: {}]",
                config.model_id, payload.prompt, stage_summary
            );
            let tokens = content
                .as_bytes()
                .iter()
                .enumerate()
                .map(|(idx, byte)| GeneratedToken {
                    request_id: input.request_id.clone(),
                    token_id: *byte as u32,
                    token_text: (*byte as char).to_string(),
                    is_finished: idx + 1 == content.len(),
                    logprob: None,
                })
                .collect::<Vec<_>>();
            return Ok(ForwardResult::Tokens(tokens));
        }

        let mut payload = if input.data.is_empty() {
            PrototypePayload {
                prompt: String::new(),
                stages: Vec::new(),
            }
        } else if let Ok(existing) = parse_payload(&input.data) {
            existing
        } else {
            PrototypePayload {
                prompt: String::from_utf8_lossy(&input.data).to_string(),
                stages: Vec::new(),
            }
        };

        payload
            .stages
            .push(format!("{}-{}", config.start_layer, config.end_layer));

        Ok(ForwardResult::Activations(Activation {
            request_id: input.request_id,
            shape: input.shape,
            data: serde_json::to_vec(&payload)?,
            seq_position: input.seq_position,
            batch_index: input.batch_index,
        }))
    }
}

fn parse_payload(data: &[u8]) -> Result<PrototypePayload> {
    serde_json::from_slice(data).or_else(|_| {
        Ok(PrototypePayload {
            prompt: String::from_utf8_lossy(data).to_string(),
            stages: Vec::new(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::engine::Activation;

    #[tokio::test]
    async fn prototype_backend_builds_stage_trace() {
        let mut head = PrototypeStageEngine::default();
        head.load_shard(&ShardConfig {
            model_id: "gemma-4-e4b-q4".into(),
            shard_path: "ignored".into(),
            start_layer: 0,
            end_layer: 13,
            total_layers: 28,
            is_first_stage: true,
            is_last_stage: false,
            max_batch_size: 2048,
            context_length: 8192,
        })
        .unwrap();

        let activation = Activation {
            request_id: "req".into(),
            shape: vec![1, 1, 2048],
            data: b"hello".to_vec(),
            seq_position: 0,
            batch_index: 0,
        };

        let forwarded = match head.forward(activation).unwrap() {
            ForwardResult::Activations(a) => a,
            ForwardResult::Tokens(_) => panic!("expected forwarded activations"),
        };

        let payload = parse_payload(&forwarded.data).unwrap();
        assert_eq!(payload.prompt, "hello");
        assert_eq!(payload.stages, vec!["0-13"]);
    }

    #[tokio::test]
    async fn prototype_backend_returns_tail_tokens() {
        let mut tail = PrototypeStageEngine::default();
        tail.load_shard(&ShardConfig {
            model_id: "gemma-4-e4b-q4".into(),
            shard_path: "ignored".into(),
            start_layer: 14,
            end_layer: 27,
            total_layers: 28,
            is_first_stage: false,
            is_last_stage: true,
            max_batch_size: 2048,
            context_length: 8192,
        })
        .unwrap();

        let payload = PrototypePayload {
            prompt: "hello".into(),
            stages: vec!["0-13".into(), "14-27".into()],
        };
        let activation = Activation {
            request_id: "req".into(),
            shape: vec![1, 1, 2048],
            data: serde_json::to_vec(&payload).unwrap(),
            seq_position: 0,
            batch_index: 0,
        };

        let tokens = match tail.forward(activation).unwrap() {
            ForwardResult::Tokens(tokens) => tokens,
            ForwardResult::Activations(_) => panic!("expected tokens"),
        };

        let text = tail
            .detokenize(&tokens.iter().map(|t| t.token_id).collect::<Vec<_>>())
            .unwrap();
        assert!(text.contains("Prototype stage completion"));
        assert!(text.contains("hello"));
        assert!(text.contains("0-13 -> 14-27"));
    }
}
