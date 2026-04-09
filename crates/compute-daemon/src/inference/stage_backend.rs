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

    pub async fn begin_prompt(
        &self,
        request_id: String,
        prompt: &str,
        max_tokens: Option<u32>,
        hidden_dim_hint: usize,
    ) -> Result<ForwardResult> {
        let token_count = self.tokenize(prompt).await?.len().max(1);
        let ingress = Activation {
            request_id,
            shape: vec![1, token_count, hidden_dim_hint.max(1)],
            data: encode_stage_prompt(prompt, max_tokens)?,
            seq_position: 0,
            batch_index: 0,
        };
        self.forward(ingress).await
    }

    pub async fn continue_forward(&self, input: Activation) -> Result<ForwardResult> {
        self.forward(input).await
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StagePayloadKind {
    PromptV1,
    HiddenStatesV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StagePayloadEnvelope {
    kind: StagePayloadKind,
    #[serde(default)]
    prompt: Option<String>,
    stages: Vec<String>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    hidden_state_bytes: Option<usize>,
    #[serde(default)]
    hidden_dim: Option<usize>,
    #[serde(default)]
    hidden_state: Vec<u8>,
}

impl StagePayloadEnvelope {
    fn hidden_state_len(&self) -> usize {
        self.hidden_state_bytes.unwrap_or(self.hidden_state.len())
    }
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

        let expected_prompt_ingress = config.is_first_stage;
        let mut payload = if input.data.is_empty() {
            StagePayloadEnvelope {
                kind: StagePayloadKind::PromptV1,
                prompt: Some(String::new()),
                stages: Vec::new(),
                max_tokens: None,
                hidden_state_bytes: None,
                hidden_dim: None,
                hidden_state: Vec::new(),
            }
        } else if let Ok(existing) = parse_payload(&input.data) {
            existing
        } else {
            StagePayloadEnvelope {
                kind: StagePayloadKind::PromptV1,
                prompt: Some(String::from_utf8_lossy(&input.data).to_string()),
                stages: Vec::new(),
                max_tokens: None,
                hidden_state_bytes: None,
                hidden_dim: None,
                hidden_state: Vec::new(),
            }
        };

        if expected_prompt_ingress && payload.kind != StagePayloadKind::PromptV1 {
            anyhow::bail!(
                "First stage expected PromptV1 ingress payload, got {:?}",
                payload.kind
            );
        }
        if !expected_prompt_ingress && payload.kind != StagePayloadKind::HiddenStatesV1 {
            anyhow::bail!(
                "Non-head stage expected HiddenStatesV1 payload, got {:?}",
                payload.kind
            );
        }

        if config.is_last_stage {
            let stage_summary = if payload.stages.is_empty() {
                "none".to_string()
            } else {
                payload.stages.join(" -> ")
            };
            let hidden_state_summary = format!(" hidden={}B", payload.hidden_state_len());
            let content = format!(
                "Prototype stage completion for {}: {} [stages: {}{}]",
                config.model_id,
                payload.prompt.clone().unwrap_or_default(),
                stage_summary,
                hidden_state_summary
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

        let stage_span = format!("{}-{}", config.start_layer, config.end_layer);
        payload.stages.push(stage_span.clone());
        let prompt_seed = payload.prompt.clone().unwrap_or_default();
        let hidden_dim = payload.hidden_dim.unwrap_or_else(|| input.shape.last().copied().unwrap_or(256));
        let previous_hidden = std::mem::take(&mut payload.hidden_state);
        let hidden_state = synthesize_hidden_state_bytes(&prompt_seed, &stage_span, &previous_hidden, hidden_dim);
        payload.kind = StagePayloadKind::HiddenStatesV1;
        payload.hidden_state_bytes = Some(hidden_state.len());
        payload.hidden_dim = Some(hidden_dim);
        payload.hidden_state = hidden_state;

        Ok(ForwardResult::Activations(Activation {
            request_id: input.request_id,
            shape: vec![1, 1, hidden_dim],
            data: serde_json::to_vec(&payload)?,
            seq_position: input.seq_position,
            batch_index: input.batch_index,
        }))
    }
}

pub fn encode_stage_prompt(prompt: &str, max_tokens: Option<u32>) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&StagePayloadEnvelope {
        kind: StagePayloadKind::PromptV1,
        prompt: Some(prompt.to_string()),
        stages: Vec::new(),
        max_tokens,
        hidden_state_bytes: None,
        hidden_dim: None,
        hidden_state: Vec::new(),
    })?)
}

pub fn encode_stage_hidden_state_stub(
    stages: Vec<String>,
    max_tokens: Option<u32>,
    hidden_state_bytes: usize,
    hidden_dim: usize,
) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&StagePayloadEnvelope {
        kind: StagePayloadKind::HiddenStatesV1,
        prompt: None,
        stages,
        max_tokens,
        hidden_state_bytes: Some(hidden_state_bytes),
        hidden_dim: Some(hidden_dim),
        hidden_state: vec![0; hidden_state_bytes],
    })?)
}

fn parse_payload(data: &[u8]) -> Result<StagePayloadEnvelope> {
    serde_json::from_slice(data).or_else(|_| {
        Ok(StagePayloadEnvelope {
            kind: StagePayloadKind::PromptV1,
            prompt: Some(String::from_utf8_lossy(data).to_string()),
            stages: Vec::new(),
            max_tokens: None,
            hidden_state_bytes: None,
            hidden_dim: None,
            hidden_state: Vec::new(),
        })
    })
}

fn synthesize_hidden_state_bytes(
    prompt: &str,
    stage_span: &str,
    previous_hidden: &[u8],
    hidden_dim: usize,
) -> Vec<u8> {
    let seed = if previous_hidden.is_empty() {
        let mut seed = prompt.as_bytes().to_vec();
        seed.extend_from_slice(stage_span.as_bytes());
        if seed.is_empty() {
            seed.extend_from_slice(b"compute-stage-forward");
        }
        seed
    } else {
        previous_hidden.to_vec()
    };

    let width = hidden_dim.max(64).min(4096);
    let mut output = Vec::with_capacity(width);
    for idx in 0..width {
        let base = seed[idx % seed.len()];
        let stage = stage_span.as_bytes()[idx % stage_span.len()];
        output.push(base.wrapping_add(stage).wrapping_add((idx % 251) as u8));
    }
    output
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
        if payload.kind != StagePayloadKind::HiddenStatesV1 {
            anyhow::bail!(
                "TailLlama tail stage expected HiddenStatesV1 payload, got {:?}",
                payload.kind
            );
        }
        let prompt = payload
            .prompt
            .clone()
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| {
                format!(
                    "Hidden-state stage payload for {} ({} bytes across {})",
                    config.model_id,
                    payload.hidden_state_len(),
                    payload.stages.join(" -> ")
                )
            });
        let content = self
            .tail
            .generate_completion_text(&prompt, payload.max_tokens)
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
        assert_eq!(payload.prompt.as_deref(), Some("hello"));
        assert_eq!(payload.stages, vec!["0-13"]);
        assert_eq!(payload.kind, StagePayloadKind::HiddenStatesV1);
        assert_eq!(payload.hidden_dim, Some(2048));
        assert_eq!(payload.hidden_state_bytes, Some(2048));
        assert_eq!(payload.hidden_state.len(), 2048);
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

        let payload = StagePayloadEnvelope {
            kind: StagePayloadKind::HiddenStatesV1,
            prompt: Some("hello".into()),
            stages: vec!["0-13".into(), "14-27".into()],
            max_tokens: Some(32),
            hidden_state_bytes: Some(2048),
            hidden_dim: Some(2048),
            hidden_state: vec![7; 2048],
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

    #[test]
    fn hidden_state_stub_records_size_and_dim() {
        let payload = parse_payload(
            &encode_stage_hidden_state_stub(vec!["0-13".into()], Some(16), 512, 256).unwrap(),
        )
        .unwrap();
        assert_eq!(payload.kind, StagePayloadKind::HiddenStatesV1);
        assert_eq!(payload.hidden_state_bytes, Some(512));
        assert_eq!(payload.hidden_dim, Some(256));
        assert_eq!(payload.hidden_state.len(), 512);
    }

    #[tokio::test]
    async fn begin_prompt_creates_hidden_state_payload() {
        let hw = crate::hardware::detect();
        let mut backend = StageExecutionBackend::new_for_hardware(&hw, StageBackendKind::Prototype);
        backend
            .load_shard(&ShardConfig {
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
            .await
            .unwrap();

        let activation = match backend
            .begin_prompt("req".into(), "hello", Some(16), 2048)
            .await
            .unwrap()
        {
            ForwardResult::Activations(activation) => activation,
            ForwardResult::Tokens(_) => panic!("expected activation output"),
        };

        let payload = parse_payload(&activation.data).unwrap();
        assert_eq!(payload.kind, StagePayloadKind::HiddenStatesV1);
        assert_eq!(payload.prompt.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn non_head_stage_rejects_prompt_payloads() {
        let mut middle = PrototypeStageEngine::default();
        middle.load_shard(&ShardConfig {
            model_id: "gemma-4-e4b-q4".into(),
            shard_path: "ignored".into(),
            start_layer: 14,
            end_layer: 20,
            total_layers: 28,
            is_first_stage: false,
            is_last_stage: false,
            max_batch_size: 2048,
            context_length: 8192,
        })
        .unwrap();

        let activation = Activation {
            request_id: "req".into(),
            shape: vec![1, 1, 2048],
            data: encode_stage_prompt("hello", Some(16)).unwrap(),
            seq_position: 0,
            batch_index: 0,
        };

        let err = middle.forward(activation).unwrap_err().to_string();
        assert!(err.contains("expected HiddenStatesV1"));
    }
}
