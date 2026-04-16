use anyhow::Result;
use serde::{Deserialize, Serialize};
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageSample, StageTensor};

use crate::inference::engine::ShardConfig;
use crate::inference::ggml_stage_executor::GgmlStageExecutorKind;
use crate::inference::real_forward_artifact::RealForwardStageLoadSpec;
use crate::inference::real_forward_provider_ggml::GgmlRealForwardProvider;
use crate::inference::stage_acceleration::{StageAccelerationPlan, StageExecutionProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealForwardProviderCapabilities {
    pub hidden_state_ingress: bool,
    pub hidden_state_egress: bool,
    pub token_id_prompt_ingress: bool,
    pub tail_sampling: bool,
    pub per_stage_decode_sessions: bool,
}

impl RealForwardProviderCapabilities {
    pub fn full_stage_contract() -> Self {
        Self {
            hidden_state_ingress: true,
            hidden_state_egress: true,
            token_id_prompt_ingress: true,
            tail_sampling: true,
            per_stage_decode_sessions: true,
        }
    }
}

impl Default for RealForwardProviderCapabilities {
    fn default() -> Self {
        Self {
            hidden_state_ingress: false,
            hidden_state_egress: false,
            token_id_prompt_ingress: false,
            tail_sampling: false,
            per_stage_decode_sessions: false,
        }
    }
}

pub trait RealForwardStageProvider: Send + Sync {
    fn provider_name(&self) -> &'static str;
    fn capabilities(&self) -> RealForwardProviderCapabilities;
    fn load_shard(&mut self, config: &ShardConfig) -> Result<()>;
    fn unload(&mut self) -> Result<()>;
    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor>;
    fn begin_token_ids(
        &self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor>;
    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor>;
    fn sample_tail(&self, input: StageTensor) -> Result<StageSample>;
    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>>;
    fn tokenize_generation_prompt(&self, text: &str) -> Result<Vec<u32>>;
    fn decode_token_ids(&self, tokens: &[u32]) -> Result<String>;
    fn eos_token_id(&self) -> Option<u32>;
    fn clear_decode_session(&self, request_id: &str);
}

#[derive(Default)]
pub struct CpuReferenceRealForwardProvider {
    backend: Option<RealGemmaBackend>,
    shard_config: Option<ShardConfig>,
}

impl CpuReferenceRealForwardProvider {
    fn backend(&self) -> Result<&RealGemmaBackend> {
        self.backend.as_ref().ok_or_else(|| anyhow::anyhow!("RealForward backend not loaded"))
    }
}

impl RealForwardStageProvider for CpuReferenceRealForwardProvider {
    fn provider_name(&self) -> &'static str {
        "cpu-ref"
    }

    fn capabilities(&self) -> RealForwardProviderCapabilities {
        RealForwardProviderCapabilities::full_stage_contract()
    }

    fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        let spec = RealForwardStageLoadSpec::from_shard_config(config)?;

        let mut backend = RealGemmaBackend::new(&spec.index_path);
        if let Some(vocab_path) = spec.vocab_path.as_deref() {
            backend.load_tokenizer(vocab_path, spec.vocab_scores_path.as_deref())?;
        }
        backend.load_layout(spec.layout.clone())?;

        self.backend = Some(backend);
        self.shard_config = Some(config.clone());
        Ok(())
    }

    fn unload(&mut self) -> Result<()> {
        self.backend = None;
        self.shard_config = None;
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.backend()?.begin_prompt(request_id, prompt, max_tokens, 0)
    }

    fn begin_token_ids(
        &self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.backend()?.begin_token_ids(request_id, token_ids, max_tokens, 0)
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        self.backend()?.continue_forward(input)
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        self.backend()?.sample_tail(input)
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.backend()?.tokenize_text(text))
    }

    fn tokenize_generation_prompt(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.backend()?.tokenize_generation_prompt(text))
    }

    fn decode_token_ids(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.backend()?.decode_token_ids(tokens))
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.backend.as_ref().and_then(RealGemmaBackend::eos_token_id)
    }

    fn clear_decode_session(&self, request_id: &str) {
        if let Some(backend) = &self.backend {
            backend.clear_decode_session(request_id);
        }
    }
}

pub fn build_real_forward_provider(
    plan: &StageAccelerationPlan,
) -> Box<dyn RealForwardStageProvider> {
    match plan.desired_provider {
        StageExecutionProvider::Ggml if !plan.allows_cpu_fallback() => {
            Box::new(GgmlRealForwardProvider::new(
                plan.desired_target_or_cpu(),
                GgmlStageExecutorKind::Ggml,
                GgmlStageExecutorKind::Ggml,
                GgmlStageExecutorKind::Ggml,
            ))
        }
        StageExecutionProvider::ReferenceCpu | StageExecutionProvider::Ggml => {
            Box::new(CpuReferenceRealForwardProvider::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::HardwareInfo;
    use crate::inference::engine::ShardConfig;
    use crate::inference::stage_acceleration::{
        StageAccelerationProviderPreference, StageAccelerationTargetPreference,
    };
    use stage_forward_lab::{PackedStageIndex, PackedTensorEntry, quants};
    use tempfile::tempdir;

    fn pin_test_worker_host() {
        unsafe {
            std::env::set_var("COMPUTE_GGML_STAGE_WORKER_HOST", "/bin/sh");
        }
    }

    #[test]
    fn cpu_reference_provider_exposes_full_stage_contract() {
        let provider = CpuReferenceRealForwardProvider::default();
        let capabilities = provider.capabilities();
        assert!(capabilities.hidden_state_ingress);
        assert!(capabilities.hidden_state_egress);
        assert!(capabilities.token_id_prompt_ingress);
        assert!(capabilities.tail_sampling);
        assert!(capabilities.per_stage_decode_sessions);
    }

    #[test]
    fn provider_factory_keeps_cpu_reference_until_accelerated_provider_exists() {
        let plan = StageAccelerationPlan::for_real_forward(
            &HardwareInfo::empty(),
            StageAccelerationTargetPreference::Auto,
            StageAccelerationProviderPreference::Auto,
        );
        let provider = build_real_forward_provider(&plan);
        assert_eq!(provider.provider_name(), "cpu-ref");
    }

    #[test]
    fn explicit_acceleration_request_yields_bootstrap_provider_until_backend_exists() {
        pin_test_worker_host();
        let plan = StageAccelerationPlan::for_real_forward(
            &HardwareInfo::empty(),
            StageAccelerationTargetPreference::Metal,
            StageAccelerationProviderPreference::Auto,
        );
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

        let mut provider = build_real_forward_provider(&plan);
        assert_eq!(provider.provider_name(), "ggml-bootstrap");
        provider
            .load_shard(&ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: stage_dir,
                start_layer: 0,
                end_layer: 20,
                total_layers: 42,
                is_first_stage: true,
                is_last_stage: false,
                max_batch_size: 16,
                context_length: 8192,
            })
            .unwrap();
    }

    #[test]
    fn explicit_provider_request_yields_bootstrap_provider_until_backend_exists() {
        let plan = StageAccelerationPlan::for_real_forward(
            &HardwareInfo::empty(),
            StageAccelerationTargetPreference::Auto,
            StageAccelerationProviderPreference::Ggml,
        );
        let provider = build_real_forward_provider(&plan);
        assert_eq!(provider.provider_name(), "ggml-bootstrap");
    }
}
