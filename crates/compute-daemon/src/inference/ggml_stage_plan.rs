use anyhow::{Result, anyhow};
use stage_forward_lab::{
    ExecutionBinding, ExecutionOpKind, LayerExecutionProgram, LayerOperatorView, PackedTensorEntry,
    PayloadKind, StageTensor, StageTensorStore, quants, stage_tensor_byte_sections,
};

use crate::inference::ggml_stage_manifest::GgmlStageBindingManifest;
use crate::inference::ggml_stage_worker::GgmlStageWorkerInitSpec;
use crate::inference::real_forward_artifact::RealForwardStageLoadSpec;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlSharedStageBindings {
    pub token_embd: Option<PackedTensorEntry>,
    pub rope_freqs: Option<PackedTensorEntry>,
    pub per_layer_model_proj: Option<PackedTensorEntry>,
    pub per_layer_proj_norm: Option<PackedTensorEntry>,
    pub per_layer_token_embd: Option<PackedTensorEntry>,
    pub output_norm: Option<PackedTensorEntry>,
    pub logits: Option<PackedTensorEntry>,
    pub shared_extras: Vec<PackedTensorEntry>,
    pub tail_extras: Vec<PackedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlLayerBindings {
    pub layer_index: u32,
    pub attn_q: PackedTensorEntry,
    pub attn_k: PackedTensorEntry,
    pub attn_v: PackedTensorEntry,
    pub attn_output: PackedTensorEntry,
    pub ffn_gate: PackedTensorEntry,
    pub ffn_up: PackedTensorEntry,
    pub ffn_down: PackedTensorEntry,
    pub attn_norm: Option<PackedTensorEntry>,
    pub attn_q_norm: Option<PackedTensorEntry>,
    pub attn_k_norm: Option<PackedTensorEntry>,
    pub ffn_norm: Option<PackedTensorEntry>,
    pub proj: Option<PackedTensorEntry>,
    pub inp_gate: Option<PackedTensorEntry>,
    pub post_attention_norm: Option<PackedTensorEntry>,
    pub post_ffw_norm: Option<PackedTensorEntry>,
    pub post_norm: Option<PackedTensorEntry>,
    pub layer_output_scale: Option<PackedTensorEntry>,
    pub unknown: Vec<PackedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlStageOperatorPlan {
    pub role: String,
    pub shared: GgmlSharedStageBindings,
    pub layers: Vec<GgmlLayerBindings>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlBoundExecutionOp {
    pub kind: ExecutionOpKind,
    pub binding: ExecutionBinding,
    pub binding_reason: &'static str,
    pub tensors: Vec<PackedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlLayerExecutionRecipe {
    pub layer_index: u32,
    pub ops: Vec<GgmlBoundExecutionOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlStageExecutionRecipe {
    pub role: String,
    pub layers: Vec<GgmlLayerExecutionRecipe>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlMaterializedTensor {
    pub entry: PackedTensorEntry,
    pub byte_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlMaterializedExecutionOp {
    pub kind: ExecutionOpKind,
    pub binding: ExecutionBinding,
    pub binding_reason: &'static str,
    pub tensors: Vec<GgmlMaterializedTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlMaterializedLayerExecutionRecipe {
    pub layer_index: u32,
    pub ops: Vec<GgmlMaterializedExecutionOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlMaterializedStageExecutionRecipe {
    pub role: String,
    pub unique_tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub layers: Vec<GgmlMaterializedLayerExecutionRecipe>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlBeginTokenIdsPlan {
    pub role: String,
    pub hidden_dim: usize,
    pub token_count: usize,
    pub max_tokens: Option<u32>,
    pub token_embd: PackedTensorEntry,
    pub rope_freqs: PackedTensorEntry,
    pub per_layer_model_proj: Option<PackedTensorEntry>,
    pub per_layer_proj_norm: Option<PackedTensorEntry>,
    pub per_layer_token_embd: Option<PackedTensorEntry>,
    pub layers: Vec<GgmlLayerBindings>,
    pub tail_output: Option<GgmlTailOutputBindings>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlTailOutputBindings {
    pub output_norm: Option<PackedTensorEntry>,
    pub logits: PackedTensorEntry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlTensorIngressPlan {
    pub kind: PayloadKind,
    pub hidden_dim: usize,
    pub hidden_state_bytes: usize,
    pub aux_bytes: usize,
    pub stage_trace_depth: usize,
    pub prompt_text_present: bool,
    pub max_tokens: Option<u32>,
    pub continuation_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlContinueForwardPlan {
    pub role: String,
    pub input: GgmlTensorIngressPlan,
    pub per_layer_model_proj: Option<PackedTensorEntry>,
    pub per_layer_proj_norm: Option<PackedTensorEntry>,
    pub per_layer_token_embd: Option<PackedTensorEntry>,
    pub layers: Vec<GgmlLayerBindings>,
    pub tail_output: Option<GgmlTailOutputBindings>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlSampleTailPlan {
    pub role: String,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub input: GgmlTensorIngressPlan,
    pub output_norm: Option<PackedTensorEntry>,
    pub logits: PackedTensorEntry,
}

impl GgmlStageOperatorPlan {
    pub fn hidden_dim(&self) -> Result<usize> {
        derive_hidden_dim(&self.layers, self.shared.token_embd.as_ref())
    }

    pub fn execution_recipe_from_load_spec(
        load_spec: &RealForwardStageLoadSpec,
    ) -> Result<GgmlStageExecutionRecipe> {
        let store = StageTensorStore::load(&load_spec.index_path)?;
        store.validate_offsets()?;
        let model_view = store.model_view();
        GgmlStageExecutionRecipe::from_parts(
            model_view.role,
            &store,
            &model_view.execution_programs,
        )
    }

    pub fn execution_recipe_from_worker_init(
        init: &GgmlStageWorkerInitSpec,
    ) -> Result<GgmlStageExecutionRecipe> {
        let store = StageTensorStore::load(&init.index_path)?;
        store.validate_offsets()?;
        let model_view = store.model_view();
        GgmlStageExecutionRecipe::from_parts(
            model_view.role,
            &store,
            &model_view.execution_programs,
        )
    }

    pub fn materialized_execution_recipe_from_load_spec(
        load_spec: &RealForwardStageLoadSpec,
    ) -> Result<GgmlMaterializedStageExecutionRecipe> {
        let store = StageTensorStore::load(&load_spec.index_path)?;
        store.validate_offsets()?;
        let model_view = store.model_view();
        GgmlMaterializedStageExecutionRecipe::from_parts(
            model_view.role,
            &store,
            &model_view.execution_programs,
        )
    }

    pub fn materialized_execution_recipe_from_worker_init(
        init: &GgmlStageWorkerInitSpec,
    ) -> Result<GgmlMaterializedStageExecutionRecipe> {
        let store = StageTensorStore::load(&init.index_path)?;
        store.validate_offsets()?;
        let model_view = store.model_view();
        GgmlMaterializedStageExecutionRecipe::from_parts(
            model_view.role,
            &store,
            &model_view.execution_programs,
        )
    }

    pub fn from_manifest(manifest: &GgmlStageBindingManifest) -> Result<Self> {
        let layers = manifest
            .layers
            .iter()
            .map(GgmlLayerBindings::from_layer_view)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            role: manifest.role.clone(),
            shared: GgmlSharedStageBindings {
                token_embd: manifest.token_embd.clone(),
                rope_freqs: manifest.rope_freqs.clone(),
                per_layer_model_proj: manifest.per_layer_model_proj.clone(),
                per_layer_proj_norm: manifest.per_layer_proj_norm.clone(),
                per_layer_token_embd: manifest.per_layer_token_embd.clone(),
                output_norm: manifest.output_norm.clone(),
                logits: manifest.logits.clone(),
                shared_extras: manifest.shared_extras.clone(),
                tail_extras: manifest.tail_extras.clone(),
            },
            layers,
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
            "role={} head_ingress={} rope={} ple=[model_proj:{} proj_norm:{} token_embd:{}] layers={} core=[qkv:{}/{} out:{}/{} ffn:{}/{}] optional=[attn_norm:{}/{} q_norm:{}/{} k_norm:{}/{} ffn_norm:{}/{} post_attn:{}/{} post_ffn:{}/{} post:{}/{} proj:{}/{} inp_gate:{}/{} scale:{}/{}] tail=[logits:{} output_norm:{}] extras=[shared:{} tail:{} unknown:{}]",
            self.role,
            self.shared.token_embd.is_some(),
            self.shared.rope_freqs.is_some(),
            self.shared.per_layer_model_proj.is_some(),
            self.shared.per_layer_proj_norm.is_some(),
            self.shared.per_layer_token_embd.is_some(),
            layer_total,
            self.layers.len(),
            layer_total,
            self.layers.len(),
            layer_total,
            self.layers.len(),
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
            self.shared.logits.is_some(),
            self.shared.output_norm.is_some(),
            self.shared.shared_extras.len(),
            self.shared.tail_extras.len(),
            unknown
        )
    }

    pub fn begin_token_ids_plan(
        &self,
        token_count: usize,
        max_tokens: Option<u32>,
    ) -> Result<GgmlBeginTokenIdsPlan> {
        if !matches!(self.role.as_str(), "head" | "single") {
            return Err(anyhow!(
                "ggml begin_token_ids plan requires a head-capable stage, got role={}",
                self.role
            ));
        }
        let token_embd = self.shared.token_embd.clone().ok_or_else(|| {
            anyhow!("ggml begin_token_ids plan requires token_embd on role={}", self.role)
        })?;
        let rope_freqs = self.shared.rope_freqs.clone().ok_or_else(|| {
            anyhow!("ggml begin_token_ids plan requires rope_freqs on role={}", self.role)
        })?;
        let tail_output = if self.role == "single" {
            Some(GgmlTailOutputBindings {
                output_norm: self.shared.output_norm.clone(),
                logits: self.shared.logits.clone().ok_or_else(|| {
                    anyhow!("ggml begin_token_ids plan requires logits on single-stage role")
                })?,
            })
        } else {
            None
        };
        let hidden_dim = derive_hidden_dim(&self.layers, Some(&token_embd))?;

        Ok(GgmlBeginTokenIdsPlan {
            role: self.role.clone(),
            hidden_dim,
            token_count,
            max_tokens,
            token_embd,
            rope_freqs,
            per_layer_model_proj: self.shared.per_layer_model_proj.clone(),
            per_layer_proj_norm: self.shared.per_layer_proj_norm.clone(),
            per_layer_token_embd: self.shared.per_layer_token_embd.clone(),
            layers: self.layers.clone(),
            tail_output,
        })
    }

    pub fn continue_forward_plan(&self, input: &StageTensor) -> Result<GgmlContinueForwardPlan> {
        if self.role == "head" {
            return Err(anyhow!(
                "ggml continue_forward plan requires a downstream-capable stage, got role={}",
                self.role
            ));
        }
        if input.kind != PayloadKind::HiddenState {
            return Err(anyhow!(
                "ggml continue_forward plan requires hidden-state input, got kind={:?}",
                input.kind
            ));
        }
        let hidden_dim = derive_hidden_dim(&self.layers, self.shared.token_embd.as_ref())?;
        validate_hidden_input(input, hidden_dim, "continue_forward")?;
        let tail_output = if matches!(self.role.as_str(), "tail" | "single") {
            Some(GgmlTailOutputBindings {
                output_norm: self.shared.output_norm.clone(),
                logits: self.shared.logits.clone().ok_or_else(|| {
                    anyhow!("ggml continue_forward plan requires logits on role={}", self.role)
                })?,
            })
        } else {
            None
        };
        Ok(GgmlContinueForwardPlan {
            role: self.role.clone(),
            input: GgmlTensorIngressPlan::from_stage_tensor(input),
            per_layer_model_proj: self.shared.per_layer_model_proj.clone(),
            per_layer_proj_norm: self.shared.per_layer_proj_norm.clone(),
            per_layer_token_embd: self.shared.per_layer_token_embd.clone(),
            layers: self.layers.clone(),
            tail_output,
        })
    }

    pub fn continue_forward_plan_with_layer_cap(
        &self,
        input: &StageTensor,
        layer_cap: usize,
    ) -> Result<GgmlContinueForwardPlan> {
        let mut plan = self.continue_forward_plan(input)?;
        plan.layers.truncate(layer_cap.min(plan.layers.len()));
        Ok(plan)
    }

    pub fn sample_tail_plan(&self, input: &StageTensor) -> Result<GgmlSampleTailPlan> {
        if !matches!(self.role.as_str(), "tail" | "single") {
            return Err(anyhow!(
                "ggml sample_tail plan requires a tail-capable stage, got role={}",
                self.role
            ));
        }
        if input.kind != PayloadKind::HiddenState {
            return Err(anyhow!(
                "ggml sample_tail plan requires hidden-state input, got kind={:?}",
                input.kind
            ));
        }
        let hidden_dim = derive_hidden_dim(&self.layers, self.shared.token_embd.as_ref())?;
        validate_hidden_input(input, hidden_dim, "sample_tail")?;
        let logits = self.shared.logits.clone().ok_or_else(|| {
            anyhow!("ggml sample_tail plan requires logits on role={}", self.role)
        })?;
        let vocab_size = derive_vocab_size(&logits, hidden_dim)?;
        Ok(GgmlSampleTailPlan {
            role: self.role.clone(),
            hidden_dim,
            vocab_size,
            input: GgmlTensorIngressPlan::from_stage_tensor(input),
            output_norm: self.shared.output_norm.clone(),
            logits,
        })
    }

    pub fn sample_tail_static_plan(&self) -> Result<GgmlSampleTailPlan> {
        if !matches!(self.role.as_str(), "tail" | "single") {
            return Err(anyhow!(
                "ggml sample_tail plan requires a tail-capable stage, got role={}",
                self.role
            ));
        }
        let hidden_dim = derive_hidden_dim(&self.layers, self.shared.token_embd.as_ref())?;
        let logits = self.shared.logits.clone().ok_or_else(|| {
            anyhow!("ggml sample_tail plan requires logits on role={}", self.role)
        })?;
        let vocab_size = derive_vocab_size(&logits, hidden_dim)?;
        Ok(GgmlSampleTailPlan {
            role: self.role.clone(),
            hidden_dim,
            vocab_size,
            input: GgmlTensorIngressPlan {
                kind: PayloadKind::HiddenState,
                hidden_dim,
                hidden_state_bytes: hidden_dim * 4,
                aux_bytes: 0,
                stage_trace_depth: 0,
                prompt_text_present: false,
                max_tokens: Some(1),
                continuation_present: false,
            },
            output_norm: self.shared.output_norm.clone(),
            logits,
        })
    }
}

impl GgmlBeginTokenIdsPlan {
    pub fn summary_label(&self) -> String {
        let layer_total = self.layers.len();
        let attn_norms = self.layers.iter().filter(|layer| layer.attn_norm.is_some()).count();
        let q_norms = self.layers.iter().filter(|layer| layer.attn_q_norm.is_some()).count();
        let k_norms = self.layers.iter().filter(|layer| layer.attn_k_norm.is_some()).count();
        let ffn_norms = self.layers.iter().filter(|layer| layer.ffn_norm.is_some()).count();
        let proj = self.layers.iter().filter(|layer| layer.proj.is_some()).count();
        let inp_gate = self.layers.iter().filter(|layer| layer.inp_gate.is_some()).count();
        let layer_scale =
            self.layers.iter().filter(|layer| layer.layer_output_scale.is_some()).count();
        format!(
            "role={} hidden_dim={} tokens={} max_tokens={:?} ingress=[token_embd:{} rope:{}] ple=[model_proj:{} proj_norm:{} token_embd:{}] layers={} core=[qkv:{}/{} out:{}/{} ffn:{}/{}] optional=[attn_norm:{}/{} q_norm:{}/{} k_norm:{}/{} ffn_norm:{}/{} proj:{}/{} inp_gate:{}/{} scale:{}/{}] tail=[logits:{} output_norm:{}]",
            self.role,
            self.hidden_dim,
            self.token_count,
            self.max_tokens,
            self.token_embd.name,
            self.rope_freqs.name,
            self.per_layer_model_proj.is_some(),
            self.per_layer_proj_norm.is_some(),
            self.per_layer_token_embd.is_some(),
            layer_total,
            self.layers.len(),
            layer_total,
            self.layers.len(),
            layer_total,
            self.layers.len(),
            layer_total,
            attn_norms,
            layer_total,
            q_norms,
            layer_total,
            k_norms,
            layer_total,
            ffn_norms,
            layer_total,
            proj,
            layer_total,
            inp_gate,
            layer_total,
            layer_scale,
            layer_total,
            self.tail_output.is_some(),
            self.tail_output.as_ref().and_then(|tail| tail.output_norm.as_ref()).is_some(),
        )
    }
}

impl GgmlBoundExecutionOp {
    pub fn summary_label(&self) -> String {
        let tensors = self
            .tensors
            .iter()
            .map(|entry| {
                format!(
                    "{}:{}:{:?}",
                    entry.name,
                    quants::ggml_type_name(entry.ggml_type),
                    entry.dimensions
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{:?}({:?}; reason={}; tensors=[{}])",
            self.kind, self.binding, self.binding_reason, tensors
        )
    }
}

impl GgmlLayerExecutionRecipe {
    pub fn summary_label(&self) -> String {
        let ops = self
            .ops
            .iter()
            .map(GgmlBoundExecutionOp::summary_label)
            .collect::<Vec<_>>()
            .join(" -> ");
        format!("layer={} ops=[{}]", self.layer_index, ops)
    }
}

impl GgmlStageExecutionRecipe {
    fn from_parts(
        role: String,
        store: &StageTensorStore,
        programs: &[LayerExecutionProgram],
    ) -> Result<Self> {
        let layers = programs
            .iter()
            .map(|program| {
                let ops = program
                    .ops
                    .iter()
                    .map(|op| {
                        let tensors = op
                            .tensor_names
                            .iter()
                            .map(|tensor_name| {
                                store.entry(tensor_name).cloned().ok_or_else(|| {
                                    anyhow!(
                                        "ggml stage execution recipe missing tensor `{}` for layer {} op {:?}",
                                        tensor_name,
                                        program.layer_index,
                                        op.kind
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(GgmlBoundExecutionOp {
                            kind: op.kind.clone(),
                            binding: op.binding.clone(),
                            binding_reason: op.binding_reason,
                            tensors,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(GgmlLayerExecutionRecipe { layer_index: program.layer_index, ops })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { role, layers })
    }

    pub fn summary_label(&self) -> String {
        let total_ops = self.layers.iter().map(|layer| layer.ops.len()).sum::<usize>();
        let first = self
            .layers
            .first()
            .map(GgmlLayerExecutionRecipe::summary_label)
            .unwrap_or_else(|| "layer=none".into());
        let last = self
            .layers
            .last()
            .map(GgmlLayerExecutionRecipe::summary_label)
            .unwrap_or_else(|| "layer=none".into());
        format!(
            "role={} layers={} total_ops={} first=[{}] last=[{}]",
            self.role,
            self.layers.len(),
            total_ops,
            first,
            last
        )
    }
}

impl GgmlMaterializedExecutionOp {
    pub fn summary_label(&self) -> String {
        let tensors = self
            .tensors
            .iter()
            .map(|tensor| {
                format!(
                    "{}:{}:{:?}:hash={}",
                    tensor.entry.name,
                    quants::ggml_type_name(tensor.entry.ggml_type),
                    tensor.entry.dimensions,
                    tensor.byte_hash
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{:?}({:?}; reason={}; tensors=[{}])",
            self.kind, self.binding, self.binding_reason, tensors
        )
    }
}

impl GgmlMaterializedLayerExecutionRecipe {
    pub fn summary_label(&self) -> String {
        let ops = self
            .ops
            .iter()
            .map(GgmlMaterializedExecutionOp::summary_label)
            .collect::<Vec<_>>()
            .join(" -> ");
        format!("layer={} ops=[{}]", self.layer_index, ops)
    }
}

impl GgmlMaterializedStageExecutionRecipe {
    fn from_parts(
        role: String,
        store: &StageTensorStore,
        programs: &[LayerExecutionProgram],
    ) -> Result<Self> {
        let mut cache = BTreeMap::<String, GgmlMaterializedTensor>::new();
        let mut total_tensor_bytes = 0u64;
        let layers = programs
            .iter()
            .map(|program| {
                let ops = program
                    .ops
                    .iter()
                    .map(|op| {
                        let tensors = op
                            .tensor_names
                            .iter()
                            .map(|tensor_name| {
                                if let Some(existing) = cache.get(tensor_name) {
                                    return Ok(existing.clone());
                                }
                                let entry = store.entry(tensor_name).cloned().ok_or_else(|| {
                                    anyhow!(
                                        "ggml materialized execution recipe missing tensor `{}` for layer {} op {:?}",
                                        tensor_name,
                                        program.layer_index,
                                        op.kind
                                    )
                                })?;
                                let bytes = store.read(tensor_name).map_err(|err| {
                                    anyhow!(
                                        "ggml materialized execution recipe failed to read tensor `{}` for layer {} op {:?}: {err}",
                                        tensor_name,
                                        program.layer_index,
                                        op.kind
                                    )
                                })?;
                                let materialized = GgmlMaterializedTensor {
                                    entry: entry.clone(),
                                    byte_hash: stable_byte_hash(&bytes),
                                };
                                total_tensor_bytes =
                                    total_tensor_bytes.saturating_add(entry.byte_len);
                                cache.insert(tensor_name.clone(), materialized.clone());
                                Ok(materialized)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(GgmlMaterializedExecutionOp {
                            kind: op.kind.clone(),
                            binding: op.binding.clone(),
                            binding_reason: op.binding_reason,
                            tensors,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(GgmlMaterializedLayerExecutionRecipe { layer_index: program.layer_index, ops })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { role, unique_tensor_count: cache.len(), total_tensor_bytes, layers })
    }

    pub fn summary_label(&self) -> String {
        let total_ops = self.layers.iter().map(|layer| layer.ops.len()).sum::<usize>();
        let first = self
            .layers
            .first()
            .map(GgmlMaterializedLayerExecutionRecipe::summary_label)
            .unwrap_or_else(|| "layer=none".into());
        let last = self
            .layers
            .last()
            .map(GgmlMaterializedLayerExecutionRecipe::summary_label)
            .unwrap_or_else(|| "layer=none".into());
        format!(
            "role={} layers={} total_ops={} unique_tensors={} total_tensor_bytes={} first=[{}] last=[{}]",
            self.role,
            self.layers.len(),
            total_ops,
            self.unique_tensor_count,
            self.total_tensor_bytes,
            first,
            last
        )
    }
}

fn stable_byte_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl GgmlTensorIngressPlan {
    pub fn from_stage_tensor(input: &StageTensor) -> Self {
        let sections = stage_tensor_byte_sections(&input.bytes);
        let hidden_state_bytes =
            sections.map(|sections| sections.hidden_bytes.len()).unwrap_or(input.bytes.len());
        let aux_bytes =
            sections.and_then(|sections| sections.aux_bytes.map(|bytes| bytes.len())).unwrap_or(0);
        Self {
            kind: input.kind,
            hidden_dim: input.hidden_dim,
            hidden_state_bytes,
            aux_bytes,
            stage_trace_depth: input.stage_trace.len(),
            prompt_text_present: input.prompt_text.is_some(),
            max_tokens: input.max_tokens,
            continuation_present: input.continuation.is_some(),
        }
    }

    pub fn summary_label(&self) -> String {
        format!(
            "kind={:?} hidden_dim={} hidden_bytes={} aux_bytes={} trace_depth={} prompt_text={} max_tokens={:?} continuation={}",
            self.kind,
            self.hidden_dim,
            self.hidden_state_bytes,
            self.aux_bytes,
            self.stage_trace_depth,
            self.prompt_text_present,
            self.max_tokens,
            self.continuation_present
        )
    }
}

impl GgmlContinueForwardPlan {
    pub fn summary_label(&self) -> String {
        let layer_total = self.layers.len();
        let attn_norms = self.layers.iter().filter(|layer| layer.attn_norm.is_some()).count();
        let q_norms = self.layers.iter().filter(|layer| layer.attn_q_norm.is_some()).count();
        let k_norms = self.layers.iter().filter(|layer| layer.attn_k_norm.is_some()).count();
        let ffn_norms = self.layers.iter().filter(|layer| layer.ffn_norm.is_some()).count();
        let proj = self.layers.iter().filter(|layer| layer.proj.is_some()).count();
        let inp_gate = self.layers.iter().filter(|layer| layer.inp_gate.is_some()).count();
        let layer_scale =
            self.layers.iter().filter(|layer| layer.layer_output_scale.is_some()).count();
        format!(
            "role={} input=[{}] ple=[model_proj:{} proj_norm:{} token_embd:{}] layers={} core=[qkv:{}/{} out:{}/{} ffn:{}/{}] optional=[attn_norm:{}/{} q_norm:{}/{} k_norm:{}/{} ffn_norm:{}/{} proj:{}/{} inp_gate:{}/{} scale:{}/{}] tail=[logits:{} output_norm:{}]",
            self.role,
            self.input.summary_label(),
            self.per_layer_model_proj.is_some(),
            self.per_layer_proj_norm.is_some(),
            self.per_layer_token_embd.is_some(),
            layer_total,
            self.layers.len(),
            layer_total,
            self.layers.len(),
            layer_total,
            self.layers.len(),
            layer_total,
            attn_norms,
            layer_total,
            q_norms,
            layer_total,
            k_norms,
            layer_total,
            ffn_norms,
            layer_total,
            proj,
            layer_total,
            inp_gate,
            layer_total,
            layer_scale,
            layer_total,
            self.tail_output.is_some(),
            self.tail_output.as_ref().and_then(|tail| tail.output_norm.as_ref()).is_some(),
        )
    }
}

impl GgmlSampleTailPlan {
    pub fn summary_label(&self) -> String {
        format!(
            "role={} hidden_dim={} vocab_size={} input=[{}] output_norm={} logits={}",
            self.role,
            self.hidden_dim,
            self.vocab_size,
            self.input.summary_label(),
            self.output_norm.is_some(),
            self.logits.name
        )
    }
}

impl GgmlLayerBindings {
    fn from_layer_view(layer: &LayerOperatorView) -> Result<Self> {
        Ok(Self {
            layer_index: layer.layer_index,
            attn_q: required_tensor(layer.layer_index, "attn_q", layer.attn_q.clone())?,
            attn_k: required_tensor(layer.layer_index, "attn_k", layer.attn_k.clone())?,
            attn_v: required_tensor(layer.layer_index, "attn_v", layer.attn_v.clone())?,
            attn_output: required_tensor(
                layer.layer_index,
                "attn_output",
                layer.attn_output.clone(),
            )?,
            ffn_gate: required_tensor(layer.layer_index, "ffn_gate", layer.ffn_gate.clone())?,
            ffn_up: required_tensor(layer.layer_index, "ffn_up", layer.ffn_up.clone())?,
            ffn_down: required_tensor(layer.layer_index, "ffn_down", layer.ffn_down.clone())?,
            attn_norm: layer.attn_norm.clone(),
            attn_q_norm: layer.attn_q_norm.clone(),
            attn_k_norm: layer.attn_k_norm.clone(),
            ffn_norm: layer.ffn_norm.clone(),
            proj: layer.proj.clone(),
            inp_gate: layer.inp_gate.clone(),
            post_attention_norm: layer.post_attention_norm.clone(),
            post_ffw_norm: layer.post_ffw_norm.clone(),
            post_norm: layer.post_norm.clone(),
            layer_output_scale: layer.layer_output_scale.clone(),
            unknown: layer.unknown.clone(),
        })
    }
}

fn required_tensor(
    layer_index: u32,
    field: &'static str,
    entry: Option<PackedTensorEntry>,
) -> Result<PackedTensorEntry> {
    entry
        .ok_or_else(|| anyhow!("ggml stage operator plan missing `{field}` on layer {layer_index}"))
}

fn derive_hidden_dim(
    layers: &[GgmlLayerBindings],
    token_embd: Option<&PackedTensorEntry>,
) -> Result<usize> {
    let mut hidden_dim =
        token_embd.and_then(|entry| entry.dimensions.first().copied()).unwrap_or(0) as usize;
    for layer in layers {
        let candidates = [
            layer.attn_q.dimensions.first().copied(),
            layer.attn_k.dimensions.first().copied(),
            layer.attn_v.dimensions.first().copied(),
            layer.attn_output.dimensions.get(1).copied(),
            layer.ffn_gate.dimensions.first().copied(),
            layer.ffn_up.dimensions.first().copied(),
            layer.ffn_down.dimensions.get(1).copied(),
            layer.attn_norm.as_ref().and_then(|entry| entry.dimensions.first().copied()),
            layer.ffn_norm.as_ref().and_then(|entry| entry.dimensions.first().copied()),
        ];
        for candidate in candidates.into_iter().flatten() {
            let candidate = candidate as usize;
            if hidden_dim == 0 {
                hidden_dim = candidate;
            } else if hidden_dim != candidate {
                return Err(anyhow!(
                    "ggml stage hidden_dim mismatch: expected {} but saw {} on layer {}",
                    hidden_dim,
                    candidate,
                    layer.layer_index
                ));
            }
        }
    }
    if hidden_dim == 0 {
        return Err(anyhow!("ggml stage hidden_dim could not be derived from bound tensors"));
    }
    Ok(hidden_dim)
}

fn derive_vocab_size(entry: &PackedTensorEntry, hidden_dim: usize) -> Result<usize> {
    if entry.dimensions.len() != 2 {
        return Err(anyhow!(
            "ggml logits tensor {} must be 2D, got {:?}",
            entry.name,
            entry.dimensions
        ));
    }
    let a = entry.dimensions[0] as usize;
    let b = entry.dimensions[1] as usize;
    if a == hidden_dim {
        Ok(b)
    } else if b == hidden_dim {
        Ok(a)
    } else {
        Err(anyhow!(
            "ggml logits tensor {} does not contain hidden_dim {} in {:?}",
            entry.name,
            hidden_dim,
            entry.dimensions
        ))
    }
}

fn validate_hidden_input(input: &StageTensor, hidden_dim: usize, op: &str) -> Result<()> {
    if input.hidden_dim != hidden_dim {
        return Err(anyhow!(
            "ggml {} input hidden_dim mismatch: input={} expected={}",
            op,
            input.hidden_dim,
            hidden_dim
        ));
    }
    let hidden_bytes = input.hidden_state_len();
    let token_stride = hidden_dim
        .checked_mul(4)
        .ok_or_else(|| anyhow!("ggml {} hidden_dim overflow for input stride", op))?;
    if hidden_bytes % token_stride != 0 {
        return Err(anyhow!(
            "ggml {} input hidden bytes {} not divisible by hidden token stride {}",
            op,
            hidden_bytes,
            token_stride
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stage_forward_lab::{
        LayerOperatorView, PackedStageIndex, PackedTensorEntry, PayloadKind, StageLayout,
        StageTensor, encode_stage_tensor_bytes,
    };
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    use super::*;
    use crate::inference::engine::ShardConfig;
    use crate::inference::ggml_stage_manifest::GgmlStageBindingManifest;

    fn tensor(name: &str) -> PackedTensorEntry {
        PackedTensorEntry {
            name: name.to_string(),
            pack_offset: 0,
            byte_len: 16,
            source_file_offset: 0,
            dimensions: vec![4, 4],
            ggml_type: 0,
        }
    }

    fn layer(idx: u32) -> LayerOperatorView {
        LayerOperatorView {
            layer_index: idx,
            attn_q: Some(tensor(&format!("blk.{idx}.attn_q.weight"))),
            attn_k: Some(tensor(&format!("blk.{idx}.attn_k.weight"))),
            attn_v: Some(tensor(&format!("blk.{idx}.attn_v.weight"))),
            attn_output: Some(tensor(&format!("blk.{idx}.attn_output.weight"))),
            attn_norm: Some(tensor(&format!("blk.{idx}.attn_norm.weight"))),
            attn_q_norm: Some(tensor(&format!("blk.{idx}.attn_q_norm.weight"))),
            attn_k_norm: Some(tensor(&format!("blk.{idx}.attn_k_norm.weight"))),
            ffn_up: Some(tensor(&format!("blk.{idx}.ffn_up.weight"))),
            ffn_down: Some(tensor(&format!("blk.{idx}.ffn_down.weight"))),
            ffn_gate: Some(tensor(&format!("blk.{idx}.ffn_gate.weight"))),
            ffn_norm: Some(tensor(&format!("blk.{idx}.ffn_norm.weight"))),
            proj: Some(tensor(&format!("blk.{idx}.proj.weight"))),
            inp_gate: Some(tensor(&format!("blk.{idx}.inp_gate.weight"))),
            post_attention_norm: Some(tensor(&format!("blk.{idx}.post_attention_norm.weight"))),
            post_ffw_norm: Some(tensor(&format!("blk.{idx}.post_ffw_norm.weight"))),
            post_norm: Some(tensor(&format!("blk.{idx}.post_norm.weight"))),
            layer_output_scale: Some(tensor(&format!("blk.{idx}.layer_output_scale.weight"))),
            unknown: vec![],
        }
    }

    fn hidden_input() -> StageTensor {
        StageTensor {
            request_id: "req".into(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec!["stage-0-20".into()],
            hidden_dim: 2560,
            bytes: encode_stage_tensor_bytes(&vec![0_u8; 32], Some(&vec![1_u8; 8])),
            prompt_text: None,
            max_tokens: Some(1),
            continuation: None,
            transient: None,
            carry: None,
        }
    }

    fn write_stage(index: &PackedStageIndex, dir: &std::path::Path) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let index_path = dir.join("stage-required.index.json");
        let pack_path = dir.join("stage-required.pack");
        fs::write(&index_path, serde_json::to_vec_pretty(index).unwrap()).unwrap();
        fs::write(&pack_path, vec![0u8; index.total_bytes as usize]).unwrap();
        index_path
    }

    #[test]
    fn operator_plan_binds_core_layers() {
        let manifest = GgmlStageBindingManifest {
            role: "head".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: Some(tensor("per_layer_model_proj.weight")),
            per_layer_proj_norm: Some(tensor("per_layer_proj_norm.weight")),
            per_layer_token_embd: Some(tensor("per_layer_token_embd.weight")),
            layers: vec![layer(0), layer(1)],
            output_norm: None,
            logits: None,
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        assert_eq!(plan.layers.len(), 2);
        assert!(plan.summary_label().contains("core=[qkv:2/2 out:2/2 ffn:2/2]"));
        assert!(
            plan.summary_label().contains("ple=[model_proj:true proj_norm:true token_embd:true]")
        );
    }

    #[test]
    fn operator_plan_rejects_missing_core_tensor() {
        let mut bad = layer(0);
        bad.attn_q = None;
        let manifest = GgmlStageBindingManifest {
            role: "tail".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            per_layer_token_embd: None,
            layers: vec![bad],
            output_norm: Some(tensor("output_norm.weight")),
            logits: Some(tensor("output.weight")),
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let err = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap_err().to_string();
        assert!(err.contains("missing `attn_q` on layer 0"));
    }

    #[test]
    fn begin_token_ids_plan_requires_head_role() {
        let manifest = GgmlStageBindingManifest {
            role: "tail".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            per_layer_token_embd: None,
            layers: vec![layer(0)],
            output_norm: Some(tensor("output_norm.weight")),
            logits: Some(tensor("output.weight")),
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let err = plan.begin_token_ids_plan(8, Some(1)).unwrap_err().to_string();
        assert!(err.contains("head-capable stage"));
    }

    #[test]
    fn begin_token_ids_plan_summarizes_head_execution_surface() {
        let manifest = GgmlStageBindingManifest {
            role: "head".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: Some(tensor("per_layer_model_proj.weight")),
            per_layer_proj_norm: Some(tensor("per_layer_proj_norm.weight")),
            per_layer_token_embd: Some(tensor("per_layer_token_embd.weight")),
            layers: vec![layer(0), layer(1)],
            output_norm: None,
            logits: None,
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let stage_plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let begin_plan = stage_plan.begin_token_ids_plan(25, Some(1)).unwrap();
        let summary = begin_plan.summary_label();
        assert!(summary.contains("hidden_dim=4"));
        assert!(summary.contains("tokens=25"));
        assert!(summary.contains("ingress=[token_embd:token_embd.weight rope:rope_freqs.weight]"));
        assert!(summary.contains("core=[qkv:2/2 out:2/2 ffn:2/2]"));
    }

    #[test]
    fn continue_forward_plan_requires_downstream_role_and_hidden_input() {
        let manifest = GgmlStageBindingManifest {
            role: "head".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            per_layer_token_embd: None,
            layers: vec![layer(0)],
            output_norm: None,
            logits: None,
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let stage_plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let err = stage_plan.continue_forward_plan(&hidden_input()).unwrap_err().to_string();
        assert!(err.contains("downstream-capable stage"));
    }

    #[test]
    fn continue_forward_plan_summarizes_tail_execution_surface() {
        let manifest = GgmlStageBindingManifest {
            role: "tail".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: Some(tensor("per_layer_model_proj.weight")),
            per_layer_proj_norm: Some(tensor("per_layer_proj_norm.weight")),
            per_layer_token_embd: Some(tensor("per_layer_token_embd.weight")),
            layers: vec![layer(0), layer(1)],
            output_norm: Some(tensor("output_norm.weight")),
            logits: Some(tensor("output.weight")),
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let stage_plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let mut input = hidden_input();
        input.hidden_dim = 4;
        let continue_plan = stage_plan.continue_forward_plan(&input).unwrap();
        let summary = continue_plan.summary_label();
        assert!(
            summary.contains("input=[kind=HiddenState hidden_dim=4 hidden_bytes=32 aux_bytes=8")
        );
        assert!(summary.contains("tail=[logits:true output_norm:true]"));
    }

    #[test]
    fn continue_forward_plan_with_layer_cap_truncates_layers() {
        let manifest = GgmlStageBindingManifest {
            role: "tail".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: Some(tensor("per_layer_model_proj.weight")),
            per_layer_proj_norm: Some(tensor("per_layer_proj_norm.weight")),
            per_layer_token_embd: Some(tensor("per_layer_token_embd.weight")),
            layers: vec![layer(0), layer(1)],
            output_norm: Some(tensor("output_norm.weight")),
            logits: Some(tensor("output.weight")),
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let stage_plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let mut input = hidden_input();
        input.hidden_dim = 4;
        let continue_plan = stage_plan.continue_forward_plan_with_layer_cap(&input, 1).unwrap();
        assert_eq!(continue_plan.layers.len(), 1);
        assert_eq!(continue_plan.layers[0].layer_index, 0);
    }

    #[test]
    fn sample_tail_plan_requires_tail_role() {
        let manifest = GgmlStageBindingManifest {
            role: "middle".into(),
            token_embd: None,
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            per_layer_token_embd: None,
            layers: vec![layer(0)],
            output_norm: None,
            logits: None,
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let stage_plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let err = stage_plan.sample_tail_plan(&hidden_input()).unwrap_err().to_string();
        assert!(err.contains("tail-capable stage"));
    }

    #[test]
    fn continue_forward_plan_rejects_hidden_dim_mismatch() {
        let manifest = GgmlStageBindingManifest {
            role: "tail".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            per_layer_token_embd: None,
            layers: vec![layer(0)],
            output_norm: Some(tensor("output_norm.weight")),
            logits: Some(tensor("output.weight")),
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let stage_plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let err = stage_plan.continue_forward_plan(&hidden_input()).unwrap_err().to_string();
        assert!(err.contains("hidden_dim mismatch"));
    }

    #[test]
    fn sample_tail_plan_derives_vocab_size() {
        let manifest = GgmlStageBindingManifest {
            role: "tail".into(),
            token_embd: Some(tensor("token_embd.weight")),
            rope_freqs: Some(tensor("rope_freqs.weight")),
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            per_layer_token_embd: None,
            layers: vec![layer(0)],
            output_norm: Some(tensor("output_norm.weight")),
            logits: Some(PackedTensorEntry {
                name: "output.weight".into(),
                pack_offset: 0,
                byte_len: 16,
                source_file_offset: 0,
                dimensions: vec![4, 32],
                ggml_type: 0,
            }),
            shared_extras: vec![],
            tail_extras: vec![],
        };

        let stage_plan = GgmlStageOperatorPlan::from_manifest(&manifest).unwrap();
        let mut input = hidden_input();
        input.hidden_dim = 4;
        let sample_plan = stage_plan.sample_tail_plan(&input).unwrap();
        assert_eq!(sample_plan.vocab_size, 32);
        assert!(sample_plan.summary_label().contains("vocab_size=32"));
    }

    #[test]
    fn execution_recipe_binds_program_ops_in_store_order() {
        let temp = tempdir().unwrap();
        let index = PackedStageIndex {
            model_name: "toy".into(),
            architecture: "gemma4".into(),
            stage_index: 0,
            role: "head".into(),
            total_bytes: 160,
            tensor_count: 10,
            tensors: vec![
                tensor("token_embd.weight"),
                tensor("rope_freqs.weight"),
                tensor("blk.0.attn_norm.weight"),
                tensor("blk.0.attn_q.weight"),
                tensor("blk.0.attn_k.weight"),
                tensor("blk.0.attn_v.weight"),
                tensor("blk.0.attn_output.weight"),
                tensor("blk.0.ffn_gate.weight"),
                tensor("blk.0.ffn_up.weight"),
                tensor("blk.0.ffn_down.weight"),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, mut entry)| {
                entry.pack_offset = (i as u64) * 16;
                entry.byte_len = 16;
                entry
            })
            .collect(),
        };
        let index_path = write_stage(&index, temp.path());
        let load_spec = RealForwardStageLoadSpec {
            config: ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: index_path.clone(),
                start_layer: 0,
                end_layer: 0,
                total_layers: 1,
                is_first_stage: true,
                is_last_stage: false,
                max_batch_size: 16,
                context_length: 8192,
            },
            stage_dir: temp.path().to_path_buf(),
            index_path,
            vocab_path: None,
            vocab_scores_path: None,
            layout: StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-0-0".into(),
                start_layer: 0,
                end_layer: 0,
                is_head: true,
                is_tail: false,
            },
        };

        let recipe = GgmlStageOperatorPlan::execution_recipe_from_load_spec(&load_spec).unwrap();
        let summary = recipe.summary_label();
        assert!(summary.contains("role=head"));
        assert!(summary.contains("AttentionQ"));
        assert!(summary.contains("blk.0.attn_q.weight"));
        assert!(summary.contains("FfnDown"));
    }

    #[test]
    fn materialized_execution_recipe_reads_and_hashes_bound_tensors() {
        let temp = tempdir().unwrap();
        let index = PackedStageIndex {
            model_name: "toy".into(),
            architecture: "gemma4".into(),
            stage_index: 0,
            role: "head".into(),
            total_bytes: 160,
            tensor_count: 10,
            tensors: vec![
                tensor("token_embd.weight"),
                tensor("rope_freqs.weight"),
                tensor("blk.0.attn_norm.weight"),
                tensor("blk.0.attn_q.weight"),
                tensor("blk.0.attn_k.weight"),
                tensor("blk.0.attn_v.weight"),
                tensor("blk.0.attn_output.weight"),
                tensor("blk.0.ffn_gate.weight"),
                tensor("blk.0.ffn_up.weight"),
                tensor("blk.0.ffn_down.weight"),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, mut entry)| {
                entry.pack_offset = (i as u64) * 16;
                entry.byte_len = 16;
                entry
            })
            .collect(),
        };
        let index_path = write_stage(&index, temp.path());
        let pack_path = temp.path().join("stage-required.pack");
        let mut pack = Vec::with_capacity(160);
        for idx in 0..10_u8 {
            pack.extend(std::iter::repeat_n(idx, 16));
        }
        fs::write(&pack_path, pack).unwrap();

        let load_spec = RealForwardStageLoadSpec {
            config: ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: index_path.clone(),
                start_layer: 0,
                end_layer: 0,
                total_layers: 1,
                is_first_stage: true,
                is_last_stage: false,
                max_batch_size: 16,
                context_length: 8192,
            },
            stage_dir: temp.path().to_path_buf(),
            index_path,
            vocab_path: None,
            vocab_scores_path: None,
            layout: StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-0-0".into(),
                start_layer: 0,
                end_layer: 0,
                is_head: true,
                is_tail: false,
            },
        };

        let materialized =
            GgmlStageOperatorPlan::materialized_execution_recipe_from_load_spec(&load_spec)
                .unwrap();
        let summary = materialized.summary_label();
        assert!(summary.contains("unique_tensors=8"));
        assert!(summary.contains("total_tensor_bytes=128"));
        assert!(summary.contains("hash="));
        assert!(summary.contains("blk.0.attn_q.weight"));
    }
}
