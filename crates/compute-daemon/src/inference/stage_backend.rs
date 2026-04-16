use anyhow::{Context, Result};
use llama_stage_backend::{LlamaStageBackend, StageNodeConfig, build_stage_backend};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Mutex;

use crate::hardware::HardwareInfo;
use crate::inference::engine::{
    Activation, ForwardResult, GeneratedToken, InferenceEngine, ShardConfig, detect_backend,
};
use crate::inference::llamacpp::LlamaCppEngine;
use crate::inference::real_forward_provider::{
    RealForwardStageProvider, build_real_forward_provider,
};
use crate::inference::stage_acceleration::{
    StageAccelerationPlan, StageAccelerationProviderPreference, StageAccelerationTargetPreference,
};
use stage_forward_lab::{PayloadKind, StageForwardBackend, StageSample, StageTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageBackendKind {
    Prototype,
    TailLlama,
    LlamaCpp,
    LlamaStageGateway,
    RealForward,
}

impl StageBackendKind {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "tail-llama" | "tailllama" | "tail_llama" | "hybrid" => Self::TailLlama,
            "llamacpp" | "llama.cpp" | "llama" => Self::LlamaCpp,
            "llama-stage-gateway" | "llama_stage_gateway" | "gateway" => Self::LlamaStageGateway,
            "real_forward" | "realforward" | "real-forward" | "real" => Self::RealForward,
            _ => Self::Prototype,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Prototype => "prototype",
            Self::TailLlama => "tail-llama",
            Self::LlamaCpp => "llamacpp",
            Self::LlamaStageGateway => "llama-stage-gateway",
            Self::RealForward => "real_forward",
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
    LlamaStage(LlamaStageEngine),
    RealForward(RealForwardEngine),
}

impl StageExecutionBackend {
    pub fn new_for_hardware(
        hw: &HardwareInfo,
        kind: StageBackendKind,
        stage_acceleration: &str,
        stage_acceleration_provider: &str,
    ) -> Self {
        match kind {
            StageBackendKind::Prototype => Self::Prototype(PrototypeStageEngine::default()),
            StageBackendKind::TailLlama => Self::TailLlama(TailLlamaStageEngine::new(hw)),
            StageBackendKind::LlamaCpp => Self::LlamaCpp(LlamaCppEngine::new(detect_backend(hw))),
            StageBackendKind::LlamaStageGateway => Self::LlamaStage(LlamaStageEngine::default()),
            StageBackendKind::RealForward => {
                Self::RealForward(RealForwardEngine::new(StageAccelerationPlan::for_real_forward(
                    hw,
                    StageAccelerationTargetPreference::parse(stage_acceleration),
                    StageAccelerationProviderPreference::parse(stage_acceleration_provider),
                )))
            }
        }
    }

    pub async fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        match self {
            Self::Prototype(engine) => engine.load_shard(config),
            Self::TailLlama(engine) => engine.load_shard(config).await,
            Self::LlamaCpp(engine) => engine.load_shard(config).await,
            Self::LlamaStage(engine) => engine.load_shard(config),
            Self::RealForward(engine) => engine.load_shard(config),
        }
    }

    pub async fn unload(&mut self) -> Result<()> {
        match self {
            Self::Prototype(engine) => engine.unload(),
            Self::TailLlama(engine) => engine.unload().await,
            Self::LlamaCpp(engine) => engine.unload().await,
            Self::LlamaStage(engine) => engine.unload(),
            Self::RealForward(engine) => engine.unload(),
        }
    }

    pub async fn forward(&self, input: Activation) -> Result<ForwardResult> {
        match self {
            Self::Prototype(engine) => engine.forward(input),
            Self::TailLlama(engine) => engine.forward(input).await,
            Self::LlamaCpp(engine) => engine.forward(input).await,
            Self::LlamaStage(engine) => engine.forward(input),
            Self::RealForward(engine) => engine.forward(input),
        }
    }

    pub async fn begin_prompt(
        &self,
        request_id: String,
        prompt: &str,
        max_tokens: Option<u32>,
        hidden_dim_hint: usize,
    ) -> Result<Activation> {
        if let Self::RealForward(engine) = self {
            return engine.begin_prompt(request_id, prompt, max_tokens, hidden_dim_hint);
        }
        let token_count = self.tokenize(prompt).await?.len().max(1);
        let ingress = Activation {
            request_id,
            shape: vec![1, token_count, hidden_dim_hint.max(1)],
            data: encode_stage_prompt(prompt, max_tokens)?,
            seq_position: 0,
            batch_index: 0,
        };
        match self.forward(ingress).await? {
            ForwardResult::Activations(activation) => Ok(activation),
            ForwardResult::Tokens(_) => anyhow::bail!(
                "Prompt ingress unexpectedly produced tokens instead of stage activations"
            ),
        }
    }

    pub async fn begin_token_ids(
        &self,
        request_id: String,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        hidden_dim_hint: usize,
    ) -> Result<Activation> {
        match self {
            Self::RealForward(engine) => {
                engine.begin_token_ids(request_id, token_ids, max_tokens, hidden_dim_hint)
            }
            Self::LlamaStage(engine) => {
                engine.begin_token_ids(request_id, token_ids, max_tokens, hidden_dim_hint)
            }
            _ => anyhow::bail!(
                "Token-id prompt ingress is only implemented for the real_forward and llama-stage backends"
            ),
        }
    }

    pub async fn continue_forward(&self, input: Activation) -> Result<ForwardResult> {
        self.forward(input).await
    }

    pub async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Prototype(engine) => engine.tokenize(text),
            Self::TailLlama(engine) => engine.tokenize(text),
            Self::LlamaCpp(engine) => engine.tokenize(text).await,
            Self::LlamaStage(engine) => engine.tokenize(text),
            Self::RealForward(engine) => engine.tokenize(text),
        }
    }

    pub async fn tokenize_generation_prompt(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::RealForward(engine) => engine.tokenize_generation_prompt(text),
            Self::LlamaStage(engine) => engine.tokenize(text),
            _ => self.tokenize(text).await,
        }
    }

    pub async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::Prototype(engine) => engine.detokenize(tokens),
            Self::TailLlama(engine) => engine.detokenize(tokens),
            Self::LlamaCpp(engine) => engine.detokenize(tokens).await,
            Self::LlamaStage(engine) => engine.detokenize(tokens),
            Self::RealForward(engine) => engine.detokenize(tokens),
        }
    }

    pub fn clear_decode_session(&self, request_id: &str) {
        match self {
            Self::LlamaStage(engine) => engine.clear_decode_session(request_id),
            Self::RealForward(engine) => engine.clear_decode_session(request_id),
            _ => {}
        }
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        match self {
            Self::LlamaStage(engine) => engine.eos_token_id(),
            Self::RealForward(engine) => engine.eos_token_id(),
            _ => None,
        }
    }

    pub fn supports_token_id_prompt_ingress(&self) -> bool {
        matches!(self, Self::RealForward(_) | Self::LlamaStage(_))
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Prototype(_) => "prototype",
            Self::TailLlama(_) => "tail-llama",
            Self::LlamaCpp(_) => "llamacpp",
            Self::LlamaStage(_) => "llama-stage-gateway",
            Self::RealForward(_) => "real_forward",
        }
    }

    pub fn backend_label(&self) -> String {
        match self {
            Self::LlamaStage(engine) => engine.backend_label(),
            Self::RealForward(engine) => {
                format!(
                    "real_forward (provider={} target={} plan={})",
                    engine.provider_name(),
                    engine.acceleration_plan().desired_target_or_cpu().as_str(),
                    engine.acceleration_plan().summary_label()
                )
            }
            _ => self.backend_name().to_string(),
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
    #[serde(default)]
    token_ids: Vec<u32>,
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
        let bytes = tokens.iter().map(|token| (*token).min(255) as u8).collect::<Vec<_>>();
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
                token_ids: Vec::new(),
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
                token_ids: Vec::new(),
            }
        };

        if expected_prompt_ingress && payload.kind != StagePayloadKind::PromptV1 {
            anyhow::bail!("First stage expected PromptV1 ingress payload, got {:?}", payload.kind);
        }
        if !expected_prompt_ingress && payload.kind != StagePayloadKind::HiddenStatesV1 {
            anyhow::bail!("Non-head stage expected HiddenStatesV1 payload, got {:?}", payload.kind);
        }

        if config.is_last_stage {
            let stage_summary = if payload.stages.is_empty() {
                "none".to_string()
            } else {
                payload.stages.join(" -> ")
            };
            let hidden_state_summary = format!(" hidden={}B", payload.hidden_state_len());
            let content = format!(
                "Prototype stage completion for {} [stages: {}{}]",
                config.model_id, stage_summary, hidden_state_summary
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
        let hidden_dim =
            payload.hidden_dim.unwrap_or_else(|| input.shape.last().copied().unwrap_or(256));
        let previous_hidden = std::mem::take(&mut payload.hidden_state);
        let hidden_state =
            synthesize_hidden_state_bytes(&prompt_seed, &stage_span, &previous_hidden, hidden_dim);
        payload.kind = StagePayloadKind::HiddenStatesV1;
        payload.prompt = None;
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
        token_ids: Vec::new(),
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
        token_ids: Vec::new(),
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
            token_ids: Vec::new(),
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
        synthesize_seed_bytes(&[prompt.as_bytes(), stage_span.as_bytes()])
    } else {
        synthesize_seed_bytes(&[previous_hidden, stage_span.as_bytes()])
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

fn synthesize_seed_bytes(parts: &[&[u8]]) -> Vec<u8> {
    const SALTS: [u64; 4] =
        [0xcbf29ce484222325, 0x9e3779b97f4a7c15, 0xd6e8feb86659fd93, 0xa0761d6478bd642f];

    let mut seed = Vec::with_capacity(SALTS.len() * 8);
    for salt in SALTS {
        let mut hash = salt;
        let mut saw_any = false;
        for part in parts {
            for &byte in *part {
                saw_any = true;
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash ^= 0xff;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        if !saw_any {
            hash ^= b'!' as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        seed.extend_from_slice(&hash.to_le_bytes());
    }
    seed
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
        if self.shard_config.as_ref().map(|config| config.is_last_stage).unwrap_or(false) {
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
        let prompt =
            payload.prompt.clone().filter(|text| !text.trim().is_empty()).unwrap_or_else(|| {
                format!(
                    "Hidden-state stage payload for {} ({} bytes across {})",
                    config.model_id,
                    payload.hidden_state_len(),
                    payload.stages.join(" -> ")
                )
            });
        let content = self.tail.generate_completion_text(&prompt, payload.max_tokens).await?;
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

pub struct RealForwardEngine {
    provider: Box<dyn RealForwardStageProvider>,
    shard_config: Option<ShardConfig>,
    acceleration_plan: StageAccelerationPlan,
}

impl RealForwardEngine {
    pub fn new(acceleration_plan: StageAccelerationPlan) -> Self {
        let provider = build_real_forward_provider(&acceleration_plan);
        Self { provider, shard_config: None, acceleration_plan }
    }

    fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        self.provider.load_shard(config)?;
        self.shard_config = Some(config.clone());
        Ok(())
    }

    fn unload(&mut self) -> Result<()> {
        self.provider.unload()?;
        self.shard_config = None;
        Ok(())
    }

    fn config(&self) -> Result<&ShardConfig> {
        self.shard_config.as_ref().ok_or_else(|| anyhow::anyhow!("No shard config"))
    }

    fn hidden_state_seq_len(hidden_dim: usize, bytes_len: usize) -> usize {
        let frame = hidden_dim.saturating_mul(4);
        if frame == 0 { 1 } else { (bytes_len / frame).max(1) }
    }

    fn stage_tensor_to_activation(
        request_id: String,
        max_tokens: Option<u32>,
        stage_tensor: StageTensor,
    ) -> Result<Activation> {
        let seq_len =
            Self::hidden_state_seq_len(stage_tensor.hidden_dim, stage_tensor.hidden_state_len());
        let envelope = StagePayloadEnvelope {
            kind: StagePayloadKind::HiddenStatesV1,
            prompt: stage_tensor.prompt_text.clone(),
            stages: stage_tensor.stage_trace.clone(),
            max_tokens,
            hidden_state_bytes: Some(stage_tensor.hidden_state_len()),
            hidden_dim: Some(stage_tensor.hidden_dim),
            hidden_state: stage_tensor.bytes,
            token_ids: Vec::new(),
        };

        Ok(Activation {
            request_id,
            shape: vec![1, seq_len, stage_tensor.hidden_dim],
            data: serde_json::to_vec(&envelope)?,
            seq_position: 0,
            batch_index: 0,
        })
    }

    fn begin_prompt(
        &self,
        request_id: String,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<Activation> {
        let stage_tensor = self.provider.begin_prompt(&request_id, prompt, max_tokens)?;
        Self::stage_tensor_to_activation(request_id, max_tokens, stage_tensor)
    }

    fn begin_token_ids(
        &self,
        request_id: String,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<Activation> {
        let stage_tensor = self.provider.begin_token_ids(&request_id, token_ids, max_tokens)?;
        Self::stage_tensor_to_activation(request_id, max_tokens, stage_tensor)
    }

    fn forward(&self, input: Activation) -> Result<ForwardResult> {
        let config = self.config()?;
        let payload = parse_payload(&input.data)?;

        if config.is_first_stage && payload.kind == StagePayloadKind::PromptV1 {
            let prompt = payload.prompt.as_deref().unwrap_or("");
            let stage_tensor =
                self.provider.begin_prompt(&input.request_id, prompt, payload.max_tokens)?;
            let seq_len = Self::hidden_state_seq_len(
                stage_tensor.hidden_dim,
                stage_tensor.hidden_state_len(),
            );
            let envelope = StagePayloadEnvelope {
                kind: StagePayloadKind::HiddenStatesV1,
                prompt: stage_tensor.prompt_text.clone(),
                stages: stage_tensor.stage_trace.clone(),
                max_tokens: payload.max_tokens,
                hidden_state_bytes: Some(stage_tensor.hidden_state_len()),
                hidden_dim: Some(stage_tensor.hidden_dim),
                hidden_state: stage_tensor.bytes,
                token_ids: payload.token_ids,
            };
            return Ok(ForwardResult::Activations(Activation {
                request_id: input.request_id,
                shape: vec![1, seq_len, stage_tensor.hidden_dim],
                data: serde_json::to_vec(&envelope)?,
                seq_position: input.seq_position,
                batch_index: input.batch_index,
            }));
        }

        let hidden_dim = payload.hidden_dim.unwrap_or(2560);
        let stage_tensor = StageTensor {
            request_id: input.request_id.clone(),
            kind: PayloadKind::HiddenState,
            stage_trace: payload.stages.clone(),
            hidden_dim,
            bytes: payload.hidden_state,
            prompt_text: payload.prompt.clone(),
            max_tokens: payload.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        };

        if config.is_last_stage {
            let result = self.provider.continue_forward(stage_tensor)?;
            let sample = self.provider.sample_tail(result)?;
            let tokens = generated_tokens_from_stage_sample(&input.request_id, sample);
            if tokens.is_empty() {
                return Ok(ForwardResult::Tokens(vec![GeneratedToken {
                    request_id: input.request_id,
                    token_id: 0,
                    token_text: String::new(),
                    is_finished: true,
                    logprob: None,
                }]));
            }
            return Ok(ForwardResult::Tokens(tokens));
        }

        let result = self.provider.continue_forward(stage_tensor)?;
        let seq_len = Self::hidden_state_seq_len(result.hidden_dim, result.hidden_state_len());
        let envelope = StagePayloadEnvelope {
            kind: StagePayloadKind::HiddenStatesV1,
            prompt: payload.prompt,
            stages: result.stage_trace.clone(),
            max_tokens: result.max_tokens,
            hidden_state_bytes: Some(result.hidden_state_len()),
            hidden_dim: Some(result.hidden_dim),
            hidden_state: result.bytes,
            token_ids: payload.token_ids,
        };
        Ok(ForwardResult::Activations(Activation {
            request_id: input.request_id,
            shape: vec![1, seq_len, result.hidden_dim],
            data: serde_json::to_vec(&envelope)?,
            seq_position: input.seq_position,
            batch_index: input.batch_index,
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.provider.tokenize_text(text)
    }

    fn tokenize_generation_prompt(&self, text: &str) -> Result<Vec<u32>> {
        self.provider.tokenize_generation_prompt(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.provider.decode_token_ids(tokens)
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.provider.eos_token_id()
    }

    fn clear_decode_session(&self, request_id: &str) {
        self.provider.clear_decode_session(request_id);
    }

    pub fn acceleration_plan(&self) -> &StageAccelerationPlan {
        &self.acceleration_plan
    }

    pub fn provider_name(&self) -> &'static str {
        self.provider.provider_name()
    }
}

#[derive(Default)]
pub struct LlamaStageEngine {
    backend: Option<LlamaStageBackend>,
    shard_config: Option<ShardConfig>,
    active_requests: Mutex<HashSet<String>>,
}

impl LlamaStageEngine {
    fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        let backend = build_stage_backend(&StageNodeConfig {
            model_path: config.shard_path.clone(),
            stage_id: format!("stage-{}-{}", config.start_layer, config.end_layer),
            start_layer: config.start_layer,
            end_layer: config.end_layer,
            is_head: config.is_first_stage,
            is_tail: config.is_last_stage,
        })?;
        self.backend = Some(backend);
        self.shard_config = Some(config.clone());
        self.active_requests.lock().expect("llama-stage request set poisoned").clear();
        Ok(())
    }

    fn unload(&mut self) -> Result<()> {
        self.backend = None;
        self.shard_config = None;
        self.active_requests.lock().expect("llama-stage request set poisoned").clear();
        Ok(())
    }

    fn config(&self) -> Result<&ShardConfig> {
        self.shard_config.as_ref().ok_or_else(|| anyhow::anyhow!("No shard config"))
    }

    fn backend(&self) -> Result<&LlamaStageBackend> {
        self.backend.as_ref().ok_or_else(|| anyhow::anyhow!("No llama stage backend loaded"))
    }

    fn hidden_state_seq_len(hidden_dim: usize, bytes_len: usize) -> usize {
        let frame = hidden_dim.saturating_mul(4);
        if frame == 0 { 1 } else { (bytes_len / frame).max(1) }
    }

    fn stage_tensor_to_activation(
        request_id: String,
        stage_tensor: StageTensor,
        token_ids: Vec<u32>,
    ) -> Result<Activation> {
        let seq_len =
            Self::hidden_state_seq_len(stage_tensor.hidden_dim, stage_tensor.hidden_state_len());
        let envelope = StagePayloadEnvelope {
            kind: StagePayloadKind::HiddenStatesV1,
            prompt: stage_tensor.prompt_text.clone(),
            stages: stage_tensor.stage_trace.clone(),
            max_tokens: stage_tensor.max_tokens,
            hidden_state_bytes: Some(stage_tensor.hidden_state_len()),
            hidden_dim: Some(stage_tensor.hidden_dim),
            hidden_state: stage_tensor.bytes,
            token_ids,
        };
        Ok(Activation {
            request_id,
            shape: vec![1, seq_len, stage_tensor.hidden_dim],
            data: serde_json::to_vec(&envelope)?,
            seq_position: 0,
            batch_index: 0,
        })
    }

    fn token_ids_to_i32(token_ids: &[u32]) -> Result<Vec<i32>> {
        token_ids
            .iter()
            .copied()
            .map(|token| i32::try_from(token).context("token id exceeded i32 range"))
            .collect()
    }

    fn request_is_new(&self, request_id: &str) -> bool {
        let active = self.active_requests.lock().expect("llama-stage request set poisoned");
        !active.contains(request_id)
    }

    fn mark_request_active(&self, request_id: &str) {
        self.active_requests
            .lock()
            .expect("llama-stage request set poisoned")
            .insert(request_id.to_string());
    }

    fn begin_prompt(
        &self,
        request_id: String,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<Activation> {
        let backend = self.backend()?;
        let token_ids = backend
            .tokenize(prompt)?
            .into_iter()
            .map(|token| token as u32)
            .collect::<Vec<_>>();
        let stage_tensor = backend.begin_prompt_session(&request_id, prompt, max_tokens)?;
        self.mark_request_active(&request_id);
        Self::stage_tensor_to_activation(request_id, stage_tensor, token_ids)
    }

    fn begin_token_ids(
        &self,
        request_id: String,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<Activation> {
        let config = self.config()?;
        if !config.is_first_stage {
            anyhow::bail!("token-id prompt ingress is only valid on the head stage");
        }
        let backend = self.backend()?;
        let token_ids_i32 = Self::token_ids_to_i32(token_ids)?;
        if self.request_is_new(&request_id) {
            backend.clear_decode_session(&request_id)?;
        }
        let stage_tensor = backend.continue_head_tokens(&request_id, token_ids_i32, max_tokens)?;
        self.mark_request_active(&request_id);
        Self::stage_tensor_to_activation(request_id, stage_tensor, token_ids.to_vec())
    }

    fn forward(&self, input: Activation) -> Result<ForwardResult> {
        let config = self.config()?;
        let payload = parse_payload(&input.data)?;

        if config.is_first_stage && payload.kind == StagePayloadKind::PromptV1 {
            let prompt = payload.prompt.as_deref().unwrap_or("");
            let activation = self.begin_prompt(input.request_id, prompt, payload.max_tokens, 2048)?;
            return Ok(ForwardResult::Activations(activation));
        }

        let backend = self.backend()?;
        let stage_tensor = StageTensor {
            request_id: input.request_id.clone(),
            kind: PayloadKind::HiddenState,
            stage_trace: payload.stages.clone(),
            hidden_dim: payload.hidden_dim.unwrap_or(2560),
            bytes: payload.hidden_state,
            prompt_text: payload.prompt.clone(),
            max_tokens: payload.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        };
        let token_ids = Self::token_ids_to_i32(&payload.token_ids)?;
        let is_new_request = self.request_is_new(&input.request_id);
        let result_tensor = if token_ids.is_empty() {
            backend.continue_forward(stage_tensor)?
        } else {
            backend.continue_forward_with_tokens(stage_tensor, token_ids, is_new_request)?
        };
        self.mark_request_active(&input.request_id);

        if config.is_last_stage {
            let sample = backend.sample_tail_token(result_tensor)?;
            return Ok(ForwardResult::Tokens(vec![GeneratedToken {
                request_id: input.request_id,
                token_id: sample.token_id as u32,
                token_text: sample.piece,
                is_finished: sample.is_eog,
                logprob: None,
            }]));
        }

        Ok(ForwardResult::Activations(Self::stage_tensor_to_activation(
            input.request_id,
            result_tensor,
            payload.token_ids,
        )?))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .backend()?
            .tokenize(text)?
            .into_iter()
            .map(|token| token as u32)
            .collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.backend()?.decode_token_ids(tokens)
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.backend().ok().and_then(|backend| backend.eos_token_id().ok().flatten())
    }

    fn clear_decode_session(&self, request_id: &str) {
        self.active_requests
            .lock()
            .expect("llama-stage request set poisoned")
            .remove(request_id);
        if let Ok(backend) = self.backend() {
            let _ = backend.clear_decode_session(request_id);
        }
    }

    fn backend_label(&self) -> String {
        match self.shard_config.as_ref() {
            Some(config) => format!(
                "llama-stage-gateway (layers {}-{} head={} tail={})",
                config.start_layer, config.end_layer, config.is_first_stage, config.is_last_stage
            ),
            None => "llama-stage-gateway".to_string(),
        }
    }
}

fn generated_tokens_from_stage_sample(
    request_id: &str,
    sample: StageSample,
) -> Vec<GeneratedToken> {
    sample
        .token_ids
        .into_iter()
        .enumerate()
        .map(|(i, token_id)| GeneratedToken {
            request_id: request_id.to_string(),
            token_id,
            token_text: String::new(),
            is_finished: i + 1 == sample.completion_tokens as usize,
            logprob: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::engine::Activation;
    use stage_forward_lab::{PackedStageIndex, PackedTensorEntry, quants};
    use tempfile::tempdir;

    fn pin_test_worker_host() {
        unsafe {
            std::env::set_var("COMPUTE_GGML_STAGE_WORKER_HOST", "/bin/sh");
        }
    }

    fn write_f32_bytes(data: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn add_f32_tensor(
        pack_data: &mut Vec<u8>,
        tensors: &mut Vec<PackedTensorEntry>,
        offset: &mut u64,
        name: &str,
        dims: Vec<u64>,
        data: &[f32],
    ) {
        let bytes = write_f32_bytes(data);
        tensors.push(PackedTensorEntry {
            name: name.to_string(),
            pack_offset: *offset,
            byte_len: bytes.len() as u64,
            source_file_offset: 0,
            dimensions: dims,
            ggml_type: quants::GGML_TYPE_F32,
        });
        pack_data.extend_from_slice(&bytes);
        *offset += bytes.len() as u64;
    }

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
        assert_eq!(payload.prompt, None);
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
            prompt: None,
            stages: vec!["0-13".into(), "14-27".into()],
            max_tokens: Some(32),
            hidden_state_bytes: Some(2048),
            hidden_dim: Some(2048),
            hidden_state: vec![7; 2048],
            token_ids: Vec::new(),
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

        let text = tail.detokenize(&tokens.iter().map(|t| t.token_id).collect::<Vec<_>>()).unwrap();
        assert!(text.contains("Prototype stage completion"));
        assert!(text.contains("0-13 -> 14-27"));
    }

    #[test]
    fn real_forward_sample_tokens_preserve_sample_ids_without_plaintext() {
        let sample = StageSample {
            request_id: "tail-req".into(),
            model_id: "test-model".into(),
            text: "できて".into(),
            token_ids: vec![41, 42, 43],
            completion_tokens: 3,
        };

        let tokens = generated_tokens_from_stage_sample("head-req", sample);
        assert_eq!(tokens.iter().map(|token| token.token_id).collect::<Vec<_>>(), vec![41, 42, 43]);
        assert_eq!(tokens.iter().filter(|token| token.is_finished).count(), 1);
        assert_eq!(tokens.last().unwrap().token_text, "");
        assert!(tokens.last().unwrap().is_finished);
        assert!(tokens[..2].iter().all(|token| token.token_text.is_empty()));
        assert!(tokens.iter().all(|token| token.request_id == "head-req"));
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

    #[test]
    fn prototype_hidden_state_seed_does_not_directly_encode_prompt_bytes() {
        let prompt = "sk-live-secret";
        let stage_span = "0-13";
        let output = synthesize_hidden_state_bytes(prompt, stage_span, &[], 256);
        let stage_bytes = stage_span.as_bytes();

        let directly_derived = prompt.as_bytes().iter().enumerate().all(|(idx, byte)| {
            output[idx]
                == byte
                    .wrapping_add(stage_bytes[idx % stage_bytes.len()])
                    .wrapping_add((idx % 251) as u8)
        });

        assert!(!directly_derived);
    }

    #[tokio::test]
    async fn begin_prompt_creates_hidden_state_payload() {
        let hw = crate::hardware::detect();
        let mut backend = StageExecutionBackend::new_for_hardware(
            &hw,
            StageBackendKind::Prototype,
            "auto",
            "auto",
        );
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

        let activation = backend.begin_prompt("req".into(), "hello", Some(16), 2048).await.unwrap();

        let payload = parse_payload(&activation.data).unwrap();
        assert_eq!(payload.kind, StagePayloadKind::HiddenStatesV1);
        assert_eq!(payload.prompt, None);
    }

    #[test]
    fn real_forward_begin_prompt_omits_plaintext_prompt_from_forwarded_payload() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-0-required.index.json");
        let pack_path = temp.path().join("stage-0-required.pack");
        let hidden_dim = 2usize;
        let ple_dim = 2usize;
        let vocab_size = 256usize;
        let mut pack_data = Vec::new();
        let mut tensors = Vec::new();
        let mut offset = 0u64;

        let token_embd: Vec<f32> =
            (0..vocab_size).flat_map(|token| [token as f32 * 0.01, 1.0]).collect();
        add_f32_tensor(
            &mut pack_data,
            &mut tensors,
            &mut offset,
            "token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &token_embd,
        );
        add_f32_tensor(
            &mut pack_data,
            &mut tensors,
            &mut offset,
            "per_layer_token_embd.weight",
            vec![ple_dim as u64, vocab_size as u64],
            &vec![0.5; ple_dim * vocab_size],
        );
        add_f32_tensor(
            &mut pack_data,
            &mut tensors,
            &mut offset,
            "per_layer_model_proj.weight",
            vec![hidden_dim as u64, ple_dim as u64],
            &[0.0, 0.0, 0.0, 0.0],
        );
        add_f32_tensor(
            &mut pack_data,
            &mut tensors,
            &mut offset,
            "per_layer_proj_norm.weight",
            vec![ple_dim as u64],
            &[0.0, 0.0],
        );
        add_f32_tensor(
            &mut pack_data,
            &mut tensors,
            &mut offset,
            "rope_freqs.weight",
            vec![1],
            &[1.0],
        );
        add_f32_tensor(
            &mut pack_data,
            &mut tensors,
            &mut offset,
            "blk.0.attn_q.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0, 0.0, 0.0, 0.0],
        );
        add_f32_tensor(
            &mut pack_data,
            &mut tensors,
            &mut offset,
            "blk.0.attn_k.weight",
            vec![hidden_dim as u64, hidden_dim as u64],
            &[0.0, 0.0, 0.0, 0.0],
        );

        std::fs::write(&pack_path, pack_data).unwrap();
        std::fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "test-gemma".into(),
                architecture: "gemma4".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: offset,
                tensor_count: tensors.len(),
                tensors,
            })
            .unwrap(),
        )
        .unwrap();

        let mut engine = RealForwardEngine::new(StageAccelerationPlan::for_real_forward(
            &crate::hardware::HardwareInfo::empty(),
            crate::inference::stage_acceleration::StageAccelerationTargetPreference::Auto,
            crate::inference::stage_acceleration::StageAccelerationProviderPreference::Auto,
        ));
        engine
            .load_shard(&ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: index_path.clone(),
                start_layer: 0,
                end_layer: 0,
                total_layers: 1,
                is_first_stage: true,
                is_last_stage: false,
                max_batch_size: 16,
                context_length: 128,
            })
            .unwrap();

        let prompt = "sk-live-secret";
        let activation = engine.begin_prompt("req".into(), prompt, Some(1), hidden_dim).unwrap();
        let payload = parse_payload(&activation.data).unwrap();
        let hidden_state_bytes = prompt.len() * hidden_dim * 4;

        assert_eq!(payload.kind, StagePayloadKind::HiddenStatesV1);
        assert_eq!(payload.prompt, None);
        assert_eq!(payload.hidden_dim, Some(hidden_dim));
        assert_eq!(payload.hidden_state_bytes, Some(hidden_state_bytes));
        assert_eq!(payload.hidden_state_len(), hidden_state_bytes);
        assert!(payload.hidden_state.len() > hidden_state_bytes);
        assert_eq!(activation.shape, vec![1, prompt.len(), hidden_dim]);
    }

    #[tokio::test]
    async fn non_head_stage_rejects_prompt_payloads() {
        let mut middle = PrototypeStageEngine::default();
        middle
            .load_shard(&ShardConfig {
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

    #[test]
    fn explicit_real_forward_acceleration_request_uses_bootstrap_provider_until_full_backend_exists()
     {
        pin_test_worker_host();
        let temp = tempdir().unwrap();
        let stage_dir = temp.path().join("packed-stage-0-20");
        std::fs::create_dir_all(&stage_dir).unwrap();
        std::fs::write(stage_dir.join("stage-1-required.pack"), vec![0u8; 4096]).unwrap();
        std::fs::write(
            stage_dir.join("stage-1-required.index.json"),
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "gemma-4-e4b-q4".into(),
                architecture: "gemma4".into(),
                stage_index: 0,
                role: "head".into(),
                total_bytes: 4096,
                tensor_count: 8,
                tensors: vec![
                    PackedTensorEntry {
                        name: "token_embd.weight".into(),
                        pack_offset: 0,
                        byte_len: 512,
                        source_file_offset: 0,
                        dimensions: vec![2560, 256000],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.0.attn_q.weight".into(),
                        pack_offset: 512,
                        byte_len: 512,
                        source_file_offset: 512,
                        dimensions: vec![128, 2560],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.0.attn_k.weight".into(),
                        pack_offset: 1024,
                        byte_len: 512,
                        source_file_offset: 1024,
                        dimensions: vec![128, 2560],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.0.attn_v.weight".into(),
                        pack_offset: 1536,
                        byte_len: 512,
                        source_file_offset: 1536,
                        dimensions: vec![128, 2560],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.0.attn_output.weight".into(),
                        pack_offset: 2048,
                        byte_len: 512,
                        source_file_offset: 2048,
                        dimensions: vec![2560, 128],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.0.ffn_gate.weight".into(),
                        pack_offset: 2560,
                        byte_len: 512,
                        source_file_offset: 2560,
                        dimensions: vec![256, 2560],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.0.ffn_up.weight".into(),
                        pack_offset: 3072,
                        byte_len: 512,
                        source_file_offset: 3072,
                        dimensions: vec![256, 2560],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.0.ffn_down.weight".into(),
                        pack_offset: 3584,
                        byte_len: 512,
                        source_file_offset: 3584,
                        dimensions: vec![2560, 256],
                        ggml_type: quants::GGML_TYPE_F32,
                    },
                ],
            })
            .unwrap(),
        )
        .unwrap();

        let mut engine = RealForwardEngine::new(StageAccelerationPlan::for_real_forward(
            &crate::hardware::HardwareInfo::empty(),
            crate::inference::stage_acceleration::StageAccelerationTargetPreference::Metal,
            crate::inference::stage_acceleration::StageAccelerationProviderPreference::Auto,
        ));
        assert_eq!(engine.provider_name(), "ggml-bootstrap");
        engine
            .load_shard(&ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: stage_dir,
                start_layer: 0,
                end_layer: 20,
                total_layers: 42,
                is_first_stage: true,
                is_last_stage: false,
                max_batch_size: 16,
                context_length: 128,
            })
            .unwrap();
    }
}
