use anyhow::Result;

use crate::hardware::HardwareInfo;
use crate::inference::engine::{
    detect_backend, Activation, ForwardResult, InferenceEngine, ShardConfig,
};
use crate::inference::llamacpp::LlamaCppEngine;

/// Execution backend for the experimental stage runtime.
///
/// This isolates stage transport/runtime code from any specific serving engine so
/// the prototype can swap away from llama.cpp without disturbing the rest of the
/// multi-node plumbing.
pub enum StageExecutionBackend {
    LlamaCpp(LlamaCppEngine),
}

impl StageExecutionBackend {
    pub fn new_for_hardware(hw: &HardwareInfo) -> Self {
        Self::LlamaCpp(LlamaCppEngine::new(detect_backend(hw)))
    }

    pub async fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        match self {
            Self::LlamaCpp(engine) => engine.load_shard(config).await,
        }
    }

    pub async fn unload(&mut self) -> Result<()> {
        match self {
            Self::LlamaCpp(engine) => engine.unload().await,
        }
    }

    pub async fn forward(&self, input: Activation) -> Result<ForwardResult> {
        match self {
            Self::LlamaCpp(engine) => engine.forward(input).await,
        }
    }

    pub async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::LlamaCpp(engine) => engine.tokenize(text).await,
        }
    }

    pub async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::LlamaCpp(engine) => engine.detokenize(tokens).await,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::LlamaCpp(_) => "llamacpp",
        }
    }
}
