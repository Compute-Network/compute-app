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
    TailLlama,
    LlamaCpp,
}

impl StageBackendKind {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "tail-llama" | "tailllama" | "tail_llama" | "hybrid" => Self::TailLlama,
            "llamacpp" | "llama.cpp" | "llama" => Self::LlamaCpp,
            _ => Self::Prototype,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Prototype => "prototype",
            Self::TailLlama => "tail-llama",
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
    TailLlama(TailLlamaStageEngine),
    LlamaCpp(LlamaCppEngine),
}

impl StageExecutionBackend {
    pub fn new_for_hardware(hw: &HardwareInfo, kind: StageBackendKind) -> Self {
        match kind {
            StageBackendKind::Prototype => Self::Prototype(PrototypeStageEngine::default()),
            StageBackendKind::TailLlama => Self::TailLlama(TailLlamaStageEngine::new(hw)),
            StageBackendKind::LlamaCpp => Self::LlamaCpp(LlamaCppEngine::new(detect_backend(hw))),
        }
    }

    pub async fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        match self {
            Self::Prototype(engine) => engine.load_shard(config),
            Self::TailLlama(engine) => engine.load_shard(config).await,
            Self::LlamaCpp(engine) => engine.load_shard(config).await,
        }
    }

    pub async fn unload(&mut self) -> Result<()> {
        match self {
            Self::Prototype(engine) => engine.unload(),
            Self::TailLlama(engine) => engine.unload().await,
            Self::LlamaCpp(engine) => engine.unload().await,
        }
    }

    pub async fn forward(&self, input: Activation) -> Result<ForwardResult> {
        match self {
            Self::Prototype(engine) => engine.forward(input),
            Self::TailLlama(engine) => engine.forward(input).await,
            Self::LlamaCpp(engine) => engine.forward(input).await,
        }
    }

    pub async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Prototype(engine) => engine.tokenize(text),
            Self::TailLlama(engine) => engine.tokenize(text),
            Self::LlamaCpp(engine) => engine.tokenize(text).await,
        }
    }

    pub async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::Prototype(engine) => engine.detokenize(tokens),
            Self::TailLlama(engine) => engine.detokenize(tokens),
            Self::LlamaCpp(engine) => engine.detokenize(tokens).await,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Prototype(_) => "prototype",
            Self::TailLlama(_) => "tail-llama",
            Self::LlamaCpp(_) => "llamacpp",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StagePromptPayload {
    prompt: String,
    stages: Vec<String>,
    #[serde(default)]
    max_tokens: Option<u32>,
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
            StagePromptPayload {
                prompt: String::new(),
                stages: Vec::new(),
                max_tokens: None,
            }
        } else if let Ok(existing) = parse_payload(&input.data) {
            existing
        } else {
            StagePromptPayload {
                prompt: String::from_utf8_lossy(&input.data).to_string(),
                stages: Vec::new(),
                max_tokens: None,
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

pub fn encode_stage_prompt(prompt: &str, max_tokens: Option<u32>) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&StagePromptPayload {
        prompt: prompt.to_string(),
        stages: Vec::new(),
        max_tokens,
    })?)
}

fn parse_payload(data: &[u8]) -> Result<StagePromptPayload> {
    serde_json::from_slice(data).or_else(|_| {
        Ok(StagePromptPayload {
            prompt: String::from_utf8_lossy(data).to_string(),
            stages: Vec::new(),
            max_tokens: None,
        })
    })
}

pub struct TailLlamaStageEngine {
    passthrough: PrototypeStageEngine,
    tail: LlamaCppEngine,
    shard_config: Option<ShardConfig>,
}

impl TailLlamaStageEngine {
    pub fn new(hw: &HardwareInfo) -> Self {
        Self {
            passthrough: PrototypeStageEngine::default(),
            tail: LlamaCppEngine::new(detect_backend(hw)),
            shard_config: None,
        }
    }

    async fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        self.shard_config = Some(config.clone());
        self.passthrough.load_shard(config)?;
        if config.is_last_stage {
            self.tail.load_shard(config).await?;
        }
        Ok(())
    }

    async fn unload(&mut self) -> Result<()> {
        if self
            .shard_config
            .as_ref()
            .map(|config| config.is_last_stage)
            .unwrap_or(false)
        {
            self.tail.unload().await?;
        }
        self.shard_config = None;
        self.passthrough.unload()?;
        Ok(())
    }

    async fn forward(&self, input: Activation) -> Result<ForwardResult> {
        let config = self
            .shard_config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TailLlama stage backend has no shard loaded"))?;

        if !config.is_last_stage {
            return self.passthrough.forward(input);
        }

        let payload = parse_payload(&input.data)?;
        let content = self
            .tail
            .generate_completion_text(&payload.prompt, payload.max_tokens)
            .await?;
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
        Ok(ForwardResult::Tokens(tokens))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.passthrough.tokenize(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.passthrough.detokenize(tokens)
    }
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

        let payload = StagePromptPayload {
            prompt: "hello".into(),
            stages: vec!["0-13".into(), "14-27".into()],
            max_tokens: Some(32),
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
