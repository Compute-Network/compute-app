//! Inference engine abstraction for running LLM layers.
//!
//! The inference engine provides a uniform interface over different backends:
//! - **Docker/CUDA**: llama.cpp in a container with NVIDIA GPU passthrough
//! - **Metal**: llama.cpp native binary on Apple Silicon
//! - **CPU**: llama.cpp CPU-only mode as fallback
//!
//! Each backend implements the `InferenceEngine` trait which handles:
//! - Loading a model shard (subset of layers)
//! - Running a forward pass on activation tensors
//! - Resource management and graceful shutdown

pub mod engine;
pub mod llamacpp;
pub mod manager;
pub mod shard;
pub mod stage_artifacts;
pub mod stage_backend;
pub mod throttle;
