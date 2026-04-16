use anyhow::{Result, bail};
use stage_forward_lab::{LayerOperatorView, PackedTensorEntry, StageModelView, StageTensorStore};

use crate::inference::ggml_stage_worker::GgmlStageWorkerInitSpec;
use crate::inference::real_forward_artifact::RealForwardStageLoadSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlStageBindingManifest {
    pub role: String,
    pub token_embd: Option<PackedTensorEntry>,
    pub rope_freqs: Option<PackedTensorEntry>,
    pub per_layer_model_proj: Option<PackedTensorEntry>,
    pub per_layer_proj_norm: Option<PackedTensorEntry>,
    pub per_layer_token_embd: Option<PackedTensorEntry>,
    pub layers: Vec<LayerOperatorView>,
    pub output_norm: Option<PackedTensorEntry>,
    pub logits: Option<PackedTensorEntry>,
    pub shared_extras: Vec<PackedTensorEntry>,
    pub tail_extras: Vec<PackedTensorEntry>,
}

impl GgmlStageBindingManifest {
    pub fn from_load_spec(load_spec: &RealForwardStageLoadSpec) -> Result<Self> {
        let store = StageTensorStore::load(&load_spec.index_path)?;
        store.validate_offsets()?;
        let model_view = store.model_view();
        Self::from_model_view(load_spec, &model_view)
    }

    pub fn from_worker_init(init: &GgmlStageWorkerInitSpec) -> Result<Self> {
        let store = StageTensorStore::load(&init.index_path)?;
        store.validate_offsets()?;
        let model_view = store.model_view();
        let role_matches = matches!(
            (init.role.as_str(), model_view.role.as_str()),
            ("head", "head") | ("tail", "tail") | ("single", "single") | ("middle", "middle")
        );
        if !role_matches {
            bail!(
                "ggml stage manifest role mismatch for {}: worker init role={} packed role={}",
                init.stage_id,
                init.role,
                model_view.role
            );
        }

        let load_spec = RealForwardStageLoadSpec {
            config: crate::inference::engine::ShardConfig {
                model_id: init.model_id.clone(),
                shard_path: init.index_path.clone(),
                start_layer: init.start_layer,
                end_layer: init.end_layer,
                total_layers: init.end_layer.saturating_add(1),
                is_first_stage: matches!(init.role.as_str(), "head" | "single"),
                is_last_stage: matches!(init.role.as_str(), "tail" | "single"),
                max_batch_size: 0,
                context_length: 0,
            },
            stage_dir: init.stage_dir.clone(),
            index_path: init.index_path.clone(),
            vocab_path: init.vocab_path.clone(),
            vocab_scores_path: init.vocab_scores_path.clone(),
            layout: stage_forward_lab::StageLayout {
                model_id: init.model_id.clone(),
                stage_id: init.stage_id.clone(),
                start_layer: init.start_layer,
                end_layer: init.end_layer,
                is_head: matches!(init.role.as_str(), "head" | "single"),
                is_tail: matches!(init.role.as_str(), "tail" | "single"),
            },
        };

        Self::from_model_view(&load_spec, &model_view)
    }

    fn from_model_view(
        load_spec: &RealForwardStageLoadSpec,
        model_view: &StageModelView,
    ) -> Result<Self> {
        let token_embd = model_view
            .prompt_ingress
            .iter()
            .find(|entry| entry.name == "token_embd.weight")
            .cloned();
        let rope_freqs =
            model_view.positional.iter().find(|entry| entry.name == "rope_freqs.weight").cloned();
        let per_layer_model_proj = model_view
            .shared_auxiliary
            .iter()
            .find(|entry| entry.name == "per_layer_model_proj.weight")
            .cloned();
        let per_layer_proj_norm = model_view
            .shared_auxiliary
            .iter()
            .find(|entry| entry.name == "per_layer_proj_norm.weight")
            .cloned();
        let per_layer_token_embd = model_view
            .shared_auxiliary
            .iter()
            .find(|entry| entry.name == "per_layer_token_embd.weight")
            .cloned();

        let shared_extras = model_view
            .shared_auxiliary
            .iter()
            .filter(|entry| {
                !matches!(
                    entry.name.as_str(),
                    "per_layer_model_proj.weight"
                        | "per_layer_proj_norm.weight"
                        | "per_layer_token_embd.weight"
                )
            })
            .cloned()
            .collect::<Vec<_>>();

        let output_norm =
            model_view.tail_only.iter().find(|entry| entry.name == "output_norm.weight").cloned();
        let logits = model_view
            .tail_only
            .iter()
            .find(|entry| entry.name == "output.weight")
            .cloned()
            .or_else(|| token_embd.clone());
        let tail_extras = model_view
            .tail_only
            .iter()
            .filter(|entry| !matches!(entry.name.as_str(), "output.weight" | "output_norm.weight"))
            .cloned()
            .collect::<Vec<_>>();

        if load_spec.layout.is_head && token_embd.is_none() {
            bail!(
                "ggml stage manifest for {} is missing token_embd.weight on head stage",
                load_spec.layout.stage_id
            );
        }
        if load_spec.layout.is_tail && logits.is_none() {
            bail!(
                "ggml stage manifest for {} is missing a logits tensor (output.weight or token_embd.weight) on tail stage",
                load_spec.layout.stage_id
            );
        }

        for layer in &model_view.operator_layers {
            let mut missing = Vec::new();
            if layer.attn_q.is_none() {
                missing.push("attn_q");
            }
            if layer.attn_k.is_none() {
                missing.push("attn_k");
            }
            if layer.attn_v.is_none() {
                missing.push("attn_v");
            }
            if layer.attn_output.is_none() {
                missing.push("attn_output");
            }
            if layer.ffn_gate.is_none() {
                missing.push("ffn_gate");
            }
            if layer.ffn_up.is_none() {
                missing.push("ffn_up");
            }
            if layer.ffn_down.is_none() {
                missing.push("ffn_down");
            }
            if !missing.is_empty() {
                bail!(
                    "ggml stage manifest for {} is missing core tensors on layer {}: {}",
                    load_spec.layout.stage_id,
                    layer.layer_index,
                    missing.join(",")
                );
            }
        }

        Ok(Self {
            role: model_view.role.clone(),
            token_embd,
            rope_freqs,
            per_layer_model_proj,
            per_layer_proj_norm,
            per_layer_token_embd,
            layers: model_view.operator_layers.clone(),
            output_norm,
            logits,
            shared_extras,
            tail_extras,
        })
    }

    pub fn summary_label(&self) -> String {
        let layer_total = self.layers.len();
        let attn_norms = self.layers.iter().filter(|layer| layer.attn_norm.is_some()).count();
        let q_norms = self.layers.iter().filter(|layer| layer.attn_q_norm.is_some()).count();
        let k_norms = self.layers.iter().filter(|layer| layer.attn_k_norm.is_some()).count();
        let ffn_norms = self.layers.iter().filter(|layer| layer.ffn_norm.is_some()).count();
        let post_attn =
            self.layers.iter().filter(|layer| layer.post_attention_norm.is_some()).count();
        let post_ffn = self.layers.iter().filter(|layer| layer.post_ffw_norm.is_some()).count();
        let post_norm = self.layers.iter().filter(|layer| layer.post_norm.is_some()).count();
        let proj = self.layers.iter().filter(|layer| layer.proj.is_some()).count();
        let inp_gate = self.layers.iter().filter(|layer| layer.inp_gate.is_some()).count();
        let layer_scale =
            self.layers.iter().filter(|layer| layer.layer_output_scale.is_some()).count();
        let unknown = self.layers.iter().map(|layer| layer.unknown.len()).sum::<usize>();
        format!(
            "role={} token_embd={} rope={} ple=[model_proj:{} proj_norm:{} token_embd:{}] layers={} norms=[attn:{}/{} q:{}/{} k:{}/{} ffn:{}/{} post_attn:{}/{} post_ffn:{}/{} post:{}/{}] proj_path=[proj:{}/{} inp_gate:{}/{} scale:{}/{}] logits={} output_norm={} shared_extras={} tail_extras={} unknown_layer_tensors={}",
            self.role,
            self.token_embd.is_some(),
            self.rope_freqs.is_some(),
            self.per_layer_model_proj.is_some(),
            self.per_layer_proj_norm.is_some(),
            self.per_layer_token_embd.is_some(),
            layer_total,
            attn_norms,
            layer_total,
            q_norms,
            layer_total,
            k_norms,
            layer_total,
            ffn_norms,
            layer_total,
            post_attn,
            layer_total,
            post_ffn,
            layer_total,
            post_norm,
            layer_total,
            proj,
            layer_total,
            inp_gate,
            layer_total,
            layer_scale,
            layer_total,
            self.logits.is_some(),
            self.output_norm.is_some(),
            self.shared_extras.len(),
            self.tail_extras.len(),
            unknown
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use stage_forward_lab::{PackedStageIndex, PackedTensorEntry, StageLayout};
    use tempfile::tempdir;

    use super::*;
    use crate::inference::engine::ShardConfig;

    fn write_stage(index: &PackedStageIndex, dir: &std::path::Path) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let index_path = dir.join("stage-required.index.json");
        let pack_path = dir.join("stage-required.pack");
        fs::write(&index_path, serde_json::to_vec_pretty(index).unwrap()).unwrap();
        fs::write(&pack_path, vec![0u8; index.total_bytes as usize]).unwrap();
        index_path
    }

    fn test_load_spec(
        index_path: PathBuf,
        is_head: bool,
        is_tail: bool,
    ) -> RealForwardStageLoadSpec {
        RealForwardStageLoadSpec {
            config: ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: index_path.clone(),
                start_layer: if is_head { 0 } else { 21 },
                end_layer: if is_head { 20 } else { 41 },
                total_layers: 42,
                is_first_stage: is_head,
                is_last_stage: is_tail,
                max_batch_size: 16,
                context_length: 8192,
            },
            stage_dir: index_path.parent().unwrap().to_path_buf(),
            index_path,
            vocab_path: Some(PathBuf::from("vocab.json")),
            vocab_scores_path: Some(PathBuf::from("vocab_scores.json")),
            layout: StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: if is_head { "stage-0-20".into() } else { "stage-21-41".into() },
                start_layer: if is_head { 0 } else { 21 },
                end_layer: if is_head { 20 } else { 41 },
                is_head,
                is_tail,
            },
        }
    }

    #[test]
    fn head_manifest_binds_core_tensors() {
        let temp = tempdir().unwrap();
        let index = PackedStageIndex {
            model_name: "gemma-4-e4b-q4".into(),
            architecture: "gemma4".into(),
            stage_index: 0,
            role: "head".into(),
            total_bytes: 4096,
            tensor_count: 12,
            tensors: vec![
                PackedTensorEntry {
                    name: "token_embd.weight".into(),
                    pack_offset: 0,
                    byte_len: 256,
                    source_file_offset: 0,
                    dimensions: vec![2560, 256000],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "rope_freqs.weight".into(),
                    pack_offset: 256,
                    byte_len: 256,
                    source_file_offset: 256,
                    dimensions: vec![128],
                    ggml_type: 0,
                },
                PackedTensorEntry {
                    name: "per_layer_model_proj.weight".into(),
                    pack_offset: 512,
                    byte_len: 256,
                    source_file_offset: 512,
                    dimensions: vec![128, 128],
                    ggml_type: 30,
                },
                PackedTensorEntry {
                    name: "per_layer_proj_norm.weight".into(),
                    pack_offset: 768,
                    byte_len: 256,
                    source_file_offset: 768,
                    dimensions: vec![128],
                    ggml_type: 0,
                },
                PackedTensorEntry {
                    name: "per_layer_token_embd.weight".into(),
                    pack_offset: 1024,
                    byte_len: 256,
                    source_file_offset: 1024,
                    dimensions: vec![128, 128],
                    ggml_type: 13,
                },
                PackedTensorEntry {
                    name: "blk.0.attn_q.weight".into(),
                    pack_offset: 1280,
                    byte_len: 256,
                    source_file_offset: 1280,
                    dimensions: vec![128, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.0.attn_k.weight".into(),
                    pack_offset: 1536,
                    byte_len: 256,
                    source_file_offset: 1536,
                    dimensions: vec![128, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.0.attn_v.weight".into(),
                    pack_offset: 1792,
                    byte_len: 256,
                    source_file_offset: 1792,
                    dimensions: vec![128, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.0.attn_output.weight".into(),
                    pack_offset: 2048,
                    byte_len: 256,
                    source_file_offset: 2048,
                    dimensions: vec![2560, 128],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.0.ffn_gate.weight".into(),
                    pack_offset: 2304,
                    byte_len: 256,
                    source_file_offset: 2304,
                    dimensions: vec![256, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.0.ffn_up.weight".into(),
                    pack_offset: 2560,
                    byte_len: 256,
                    source_file_offset: 2560,
                    dimensions: vec![256, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.0.ffn_down.weight".into(),
                    pack_offset: 2816,
                    byte_len: 1280,
                    source_file_offset: 2816,
                    dimensions: vec![2560, 256],
                    ggml_type: 14,
                },
            ],
        };
        let index_path = write_stage(&index, temp.path());
        let manifest =
            GgmlStageBindingManifest::from_load_spec(&test_load_spec(index_path, true, false))
                .unwrap();
        assert!(manifest.token_embd.is_some());
        assert!(manifest.rope_freqs.is_some());
        assert!(manifest.logits.is_some());
        assert_eq!(manifest.layers.len(), 1);
        assert!(manifest.summary_label().contains("token_embd=true"));
    }

    #[test]
    fn tail_manifest_requires_logits_tensor() {
        let temp = tempdir().unwrap();
        let index = PackedStageIndex {
            model_name: "gemma-4-e4b-q4".into(),
            architecture: "gemma4".into(),
            stage_index: 1,
            role: "tail".into(),
            total_bytes: 2048,
            tensor_count: 7,
            tensors: vec![
                PackedTensorEntry {
                    name: "blk.21.attn_q.weight".into(),
                    pack_offset: 0,
                    byte_len: 256,
                    source_file_offset: 0,
                    dimensions: vec![128, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.21.attn_k.weight".into(),
                    pack_offset: 256,
                    byte_len: 256,
                    source_file_offset: 256,
                    dimensions: vec![128, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.21.attn_v.weight".into(),
                    pack_offset: 512,
                    byte_len: 256,
                    source_file_offset: 512,
                    dimensions: vec![128, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.21.attn_output.weight".into(),
                    pack_offset: 768,
                    byte_len: 256,
                    source_file_offset: 768,
                    dimensions: vec![2560, 128],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.21.ffn_gate.weight".into(),
                    pack_offset: 1024,
                    byte_len: 256,
                    source_file_offset: 1024,
                    dimensions: vec![256, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.21.ffn_up.weight".into(),
                    pack_offset: 1280,
                    byte_len: 256,
                    source_file_offset: 1280,
                    dimensions: vec![256, 2560],
                    ggml_type: 12,
                },
                PackedTensorEntry {
                    name: "blk.21.ffn_down.weight".into(),
                    pack_offset: 1536,
                    byte_len: 512,
                    source_file_offset: 1536,
                    dimensions: vec![2560, 256],
                    ggml_type: 14,
                },
            ],
        };
        let index_path = write_stage(&index, temp.path());
        let err =
            GgmlStageBindingManifest::from_load_spec(&test_load_spec(index_path, false, true))
                .unwrap_err()
                .to_string();
        assert!(err.contains("missing a logits tensor"));
    }
}
