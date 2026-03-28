use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Activation tensor passed between pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activation {
    /// Unique request identifier.
    pub request_id: String,
    /// Shape: typically [batch_size, seq_len, hidden_dim].
    pub shape: Vec<usize>,
    /// Raw tensor data (f16 bytes).
    pub data: Vec<u8>,
    /// Current sequence position in autoregressive generation.
    pub seq_position: u32,
    /// Micro-batch index for pipelining multiple requests.
    pub batch_index: u32,
}

/// A generated token from the final pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedToken {
    pub request_id: String,
    pub token_id: u32,
    pub token_text: String,
    pub is_finished: bool,
    pub logprob: Option<f32>,
}

/// Configuration for loading a model shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Model identifier (e.g. "llama-3.1-70b-q4").
    pub model_id: String,
    /// Path to the model shard file on disk.
    pub shard_path: PathBuf,
    /// First layer in this shard (inclusive).
    pub start_layer: u32,
    /// Last layer in this shard (inclusive).
    pub end_layer: u32,
    /// Total layers in the full model.
    pub total_layers: u32,
    /// Whether this is the first stage (handles tokenization).
    pub is_first_stage: bool,
    /// Whether this is the last stage (handles token sampling).
    pub is_last_stage: bool,
    /// Maximum batch size.
    pub max_batch_size: u32,
    /// Context length.
    pub context_length: u32,
}

/// Backend type for inference execution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InferenceBackend {
    /// NVIDIA GPU via Docker container with CUDA.
    DockerCuda,
    /// Apple Silicon via native Metal binary.
    NativeMetal,
    /// CPU-only fallback.
    Cpu,
}

impl std::fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceBackend::DockerCuda => write!(f, "Docker/CUDA"),
            InferenceBackend::NativeMetal => write!(f, "Native/Metal"),
            InferenceBackend::Cpu => write!(f, "CPU"),
        }
    }
}

/// Resource usage reported by the inference engine.
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    /// Tokens generated per second.
    pub tokens_per_sec: f64,
    /// VRAM currently used in MB.
    pub vram_used_mb: u64,
    /// GPU utilization percentage.
    pub gpu_utilization: f32,
    /// Number of requests currently being processed.
    pub active_requests: u32,
    /// Total tokens generated since engine start.
    pub total_tokens_generated: u64,
}

/// Status of the inference engine.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineStatus {
    /// Engine is not initialized.
    Unloaded,
    /// Model shard is being loaded.
    Loading,
    /// Ready to process activations.
    Ready,
    /// Currently processing a forward pass.
    Running,
    /// Paused due to resource constraints (user activity).
    Paused,
    /// Error state.
    Error(String),
    /// Shutting down.
    ShuttingDown,
}

/// The core inference engine trait.
///
/// Each backend (Docker/CUDA, Metal, CPU) implements this trait.
/// The pipeline stage runner calls these methods to execute inference.
#[allow(async_fn_in_trait)]
pub trait InferenceEngine: Send + Sync {
    /// Get the backend type.
    fn backend(&self) -> InferenceBackend;

    /// Get the current engine status.
    fn status(&self) -> EngineStatus;

    /// Load a model shard into memory/VRAM.
    ///
    /// This may take significant time for large shards.
    /// The engine transitions from Unloaded → Loading → Ready.
    async fn load_shard(&mut self, config: &ShardConfig) -> Result<()>;

    /// Unload the current model shard and free resources.
    async fn unload(&mut self) -> Result<()>;

    /// Run a forward pass through this node's layers.
    ///
    /// - For the **first stage**: `input` contains tokenized input.
    ///   Returns activations (hidden states after this node's layers).
    /// - For **middle stages**: `input` contains activations from the previous stage.
    ///   Returns activations for the next stage.
    /// - For the **last stage**: `input` contains activations from the previous stage.
    ///   Returns generated tokens (via sampling).
    async fn forward(&self, input: Activation) -> Result<ForwardResult>;

    /// Tokenize a text prompt (only meaningful for the first stage).
    async fn tokenize(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs to text (only meaningful for the last stage).
    async fn detokenize(&self, tokens: &[u32]) -> Result<String>;

    /// Get current resource usage metrics.
    fn metrics(&self) -> InferenceMetrics;

    /// Pause inference (yield resources to user).
    /// The engine should finish its current forward pass and then stop accepting new work.
    async fn pause(&mut self) -> Result<()>;

    /// Resume inference after a pause.
    async fn resume(&mut self) -> Result<()>;

    /// Set process/thread priority (lower = more yielding to user apps).
    fn set_priority(&mut self, nice_value: i32);
}

/// Result of a forward pass — either activations for the next stage, or generated tokens.
#[derive(Debug, Clone)]
pub enum ForwardResult {
    /// Activations to send to the next pipeline stage.
    Activations(Activation),
    /// Generated tokens (from the final stage).
    Tokens(Vec<GeneratedToken>),
}

/// Select the best inference backend for the current hardware.
pub fn detect_backend(hw: &crate::hardware::HardwareInfo) -> InferenceBackend {
    // Check for Apple Silicon (Metal)
    for gpu in &hw.gpus {
        if matches!(gpu.backend, crate::hardware::GpuBackend::Metal) {
            return InferenceBackend::NativeMetal;
        }
    }

    // Check for NVIDIA GPU + Docker
    let has_cuda = hw
        .gpus
        .iter()
        .any(|g| matches!(g.backend, crate::hardware::GpuBackend::Cuda));
    if has_cuda && hw.docker.available {
        return InferenceBackend::DockerCuda;
    }

    // Fallback to CPU
    InferenceBackend::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware;

    #[test]
    fn test_detect_backend_metal() {
        let mut hw = hardware::detect();
        // If we're on a Mac with Metal, it should detect Metal
        if hw.gpus.iter().any(|g| matches!(g.backend, hardware::GpuBackend::Metal)) {
            assert_eq!(detect_backend(&hw), InferenceBackend::NativeMetal);
        }
        // Force Metal GPU for test
        hw.gpus = vec![hardware::GpuInfo {
            name: "Apple M3 Pro".into(),
            vram_mb: 36864,
            backend: hardware::GpuBackend::Metal,
        }];
        assert_eq!(detect_backend(&hw), InferenceBackend::NativeMetal);
    }

    #[test]
    fn test_detect_backend_cuda() {
        let hw = hardware::HardwareInfo {
            cpu: hardware::CpuInfo {
                brand: "Test".into(),
                cores: 8,
                threads: 16,
                frequency_mhz: 3600,
            },
            memory: hardware::MemoryInfo { total_gb: 32.0, available_gb: 16.0 },
            gpus: vec![hardware::GpuInfo {
                name: "RTX 4090".into(),
                vram_mb: 24576,
                backend: hardware::GpuBackend::Cuda,
            }],
            disk: hardware::DiskInfo { total_gb: 1000.0, available_gb: 500.0 },
            os: hardware::OsInfo {
                name: "Linux".into(),
                version: "6.0".into(),
                arch: "x86_64".into(),
            },
            docker: hardware::DockerStatus {
                available: true,
                version: Some("24.0".into()),
            },
        };
        assert_eq!(detect_backend(&hw), InferenceBackend::DockerCuda);
    }

    #[test]
    fn test_detect_backend_cpu_fallback() {
        let hw = hardware::HardwareInfo {
            cpu: hardware::CpuInfo {
                brand: "Test".into(),
                cores: 4,
                threads: 8,
                frequency_mhz: 2400,
            },
            memory: hardware::MemoryInfo { total_gb: 8.0, available_gb: 4.0 },
            gpus: vec![],
            disk: hardware::DiskInfo { total_gb: 256.0, available_gb: 100.0 },
            os: hardware::OsInfo {
                name: "Linux".into(),
                version: "6.0".into(),
                arch: "x86_64".into(),
            },
            docker: hardware::DockerStatus {
                available: false,
                version: None,
            },
        };
        assert_eq!(detect_backend(&hw), InferenceBackend::Cpu);
    }

    #[test]
    fn test_shard_config() {
        let config = ShardConfig {
            model_id: "llama-3.1-70b-q4".into(),
            shard_path: PathBuf::from("/tmp/shard.gguf"),
            start_layer: 16,
            end_layer: 31,
            total_layers: 80,
            is_first_stage: false,
            is_last_stage: false,
            max_batch_size: 8,
            context_length: 4096,
        };
        assert_eq!(config.end_layer - config.start_layer + 1, 16);
        assert!(!config.is_first_stage);
        assert!(!config.is_last_stage);
    }

    #[test]
    fn test_engine_status_equality() {
        assert_eq!(EngineStatus::Ready, EngineStatus::Ready);
        assert_ne!(EngineStatus::Ready, EngineStatus::Paused);
        assert_ne!(
            EngineStatus::Error("a".into()),
            EngineStatus::Error("b".into())
        );
    }
}
