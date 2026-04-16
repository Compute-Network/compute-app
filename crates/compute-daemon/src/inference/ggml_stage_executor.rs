use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use stage_forward_lab::prompting::{GemmaPromptMode, format_gemma_prompt};
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::real_math;
use stage_forward_lab::tokenizer::GemmaTokenizer;
use stage_forward_lab::{
    PayloadKind, StageForwardBackend, StageLayout, StageSample, StageTensor, StageTensorStore,
    encode_stage_tensor_bytes, quants, stage_tensor_byte_sections,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

use crate::inference::ggml_graph_runtime::{
    GgmlBatchedSingleOutputGraphRuntime, GgmlFullHeadPrefillLayerRuntime, GgmlGetRowsGraphRuntime,
    GgmlHeadLayerGraphSpec, GgmlPleIngressGraphRuntime, GgmlRopeGraphRuntime,
    GgmlSampleGraphRuntime, GgmlSingleOutputGraphRuntime, GgmlTailLayerRuntime,
    GgmlTailStackRuntime,
};
use crate::inference::ggml_runtime::GgmlRuntimePlan;
use crate::inference::ggml_stage_manifest::GgmlStageBindingManifest;
use crate::inference::ggml_stage_plan::{
    GgmlMaterializedStageExecutionRecipe, GgmlStageExecutionRecipe, GgmlStageOperatorPlan,
};
use crate::inference::ggml_stage_worker::GgmlStageWorkerInitSpec;
use crate::inference::stage_acceleration::StageAccelerationTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlStageExecutorKind {
    ReferenceCpu,
    Ggml,
}

impl GgmlStageExecutorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReferenceCpu => "cpu-ref-worker",
            Self::Ggml => "ggml-worker",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlStageExecutorPlan {
    pub requested: GgmlStageExecutorKind,
    pub active: GgmlStageExecutorKind,
    pub detail: String,
}

impl GgmlStageExecutorPlan {
    pub fn summary_label(&self) -> String {
        if self.requested == self.active {
            format!("{} ({})", self.active.as_str(), self.detail)
        } else {
            format!("{} -> {} ({})", self.requested.as_str(), self.active.as_str(), self.detail)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlHeadIngressProfile {
    pub executor: String,
    pub token_count: usize,
    pub iterations: u32,
    pub total_us: u64,
    pub embed_token_gather_us: Option<u64>,
    pub ple_token_gather_us: Option<u64>,
    pub ple_model_proj_us: Option<u64>,
    pub ple_normalize_combine_us: Option<u64>,
    pub prompt_aux_encode_us: Option<u64>,
    pub hidden_encode_us: Option<u64>,
    pub payload_frame_us: Option<u64>,
    pub other_us: Option<u64>,
    pub hidden_state_bytes: usize,
    pub aux_bytes: usize,
    pub payload_bytes: usize,
}

impl GgmlHeadIngressProfile {
    pub fn avg_total_us(&self) -> f64 {
        self.total_us as f64 / self.iterations.max(1) as f64
    }

    pub fn avg_bucket_us(bucket_us: Option<u64>, iterations: u32) -> Option<f64> {
        bucket_us.map(|value| value as f64 / iterations.max(1) as f64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlHeadLayerExecutionProfile {
    pub layer_index: u32,
    pub total_us: u64,
    pub attention_cpu_us: u64,
    pub attention_matmul_us: u64,
    pub ffn_cpu_us: u64,
    pub ffn_matmul_us: u64,
    pub ple_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlHeadExecutionProfile {
    pub executor: String,
    pub token_count: usize,
    pub iterations: u32,
    pub effective_layer_cap: usize,
    pub ingress_us: u64,
    pub payload_encode_us: u64,
    pub total_us: u64,
    pub layers: Vec<GgmlHeadLayerExecutionProfile>,
    pub hidden_state_bytes: usize,
    pub aux_bytes: usize,
    pub payload_bytes: usize,
}

impl GgmlHeadExecutionProfile {
    pub fn avg_total_us(&self) -> f64 {
        self.total_us as f64 / self.iterations.max(1) as f64
    }

    pub fn avg_ingress_us(&self) -> f64 {
        self.ingress_us as f64 / self.iterations.max(1) as f64
    }

    pub fn avg_payload_encode_us(&self) -> f64 {
        self.payload_encode_us as f64 / self.iterations.max(1) as f64
    }
}

#[derive(Debug, Clone)]
pub struct GgmlProportionalSharedKvLayerDebug {
    pub layer_index: u32,
    pub shared_kv_source_layer: u32,
    pub token_count: usize,
    pub position_offset: u32,
    pub rope_runtime: String,
    pub full_runtime: String,
    pub input_max_abs: f32,
    pub q_rope_ref_candidate_max_abs: f32,
    pub q_input_ggml_max_abs: f32,
    pub q_rope_ggml_candidate_max_abs: f32,
    pub attn_out_max_abs: f32,
    pub layer_out_max_abs: f32,
    pub layer_out_ggml_candidate_max_abs: f32,
}

#[derive(Debug, Clone)]
pub struct GgmlFullGraphHeadLayerDebug {
    pub layer_index: u32,
    pub runtime: String,
    pub local_output_max_abs: f32,
    pub cumulative_output_max_abs: f32,
}

#[derive(Debug, Clone)]
struct BootstrapSharedKvLayerTrace {
    q_pre_rope: Vec<Vec<f32>>,
    q_rope: Vec<Vec<f32>>,
    attn_out: Vec<Vec<f32>>,
    layer_out: Vec<Vec<f32>>,
}

pub trait GgmlStageExecutor: Send {
    fn plan(&self) -> &GgmlStageExecutorPlan;
    fn tokenize_text(&mut self, text: &str) -> Result<Vec<u32>>;
    fn tokenize_generation_prompt(&mut self, text: &str) -> Result<Vec<u32>>;
    fn decode_token_ids(&mut self, token_ids: &[u32]) -> Result<String>;
    fn eos_token_id(&mut self) -> Result<Option<u32>>;
    fn begin_prompt(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor>;
    fn begin_token_ids(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor>;
    fn continue_forward(&mut self, input: StageTensor) -> Result<StageTensor>;
    fn sample_tail(&mut self, input: StageTensor) -> Result<StageSample>;
    fn clear_decode_session(&mut self, request_id: &str);
    fn profile_begin_token_ids_ingress(
        &mut self,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        iterations: u32,
    ) -> Result<GgmlHeadIngressProfile>;
    fn profile_begin_token_ids_execution(
        &mut self,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        iterations: u32,
    ) -> Result<GgmlHeadExecutionProfile>;
}

struct ReferenceCpuGgmlStageExecutor {
    plan: GgmlStageExecutorPlan,
    tokenizer: GemmaTokenizer,
    backend: RealGemmaBackend,
}

struct BootstrapGgmlStageExecutor {
    plan: GgmlStageExecutorPlan,
    tokenizer: GemmaTokenizer,
    model_id: String,
    store: StageTensorStore,
    runtime_plan: GgmlRuntimePlan,
    runtime_summary: String,
    binding_summary: String,
    operator_plan: GgmlStageOperatorPlan,
    execution_recipe: GgmlStageExecutionRecipe,
    materialized_recipe: GgmlMaterializedStageExecutionRecipe,
    stage_id: String,
    debug_layer_cap: Option<usize>,
    sample_graph_runtime: Option<GgmlSampleGraphRuntime>,
    #[allow(dead_code)]
    token_embd_graph_runtime: Option<GgmlGetRowsGraphRuntime>,
    #[allow(dead_code)]
    token_embd_batch_graph_runtimes: HashMap<usize, GgmlGetRowsGraphRuntime>,
    #[allow(dead_code)]
    ple_token_embd_graph_runtime: Option<GgmlGetRowsGraphRuntime>,
    #[allow(dead_code)]
    ple_token_embd_batch_graph_runtimes: HashMap<usize, GgmlGetRowsGraphRuntime>,
    #[allow(dead_code)]
    ple_model_proj_graph_runtime: Option<GgmlSingleOutputGraphRuntime>,
    #[allow(dead_code)]
    ple_model_proj_batch_graph_runtimes: HashMap<usize, GgmlBatchedSingleOutputGraphRuntime>,
    #[allow(dead_code)]
    ple_ingress_graph_runtimes: HashMap<usize, GgmlPleIngressGraphRuntime>,
    head_stack_runtime: Option<GgmlTailStackRuntime>,
    head_prefill_stack_runtimes: HashMap<(usize, usize), GgmlTailStackRuntime>,
    head_full_prefill_layer_runtimes:
        RefCell<HashMap<(u32, usize, bool), GgmlFullHeadPrefillLayerRuntime>>,
    tail_stack_runtime: Option<GgmlTailStackRuntime>,
    decode_sessions: HashMap<String, BootstrapDecodeSession>,
    f32_vector_cache: RefCell<HashMap<String, Vec<f32>>>,
    f32_scalar_cache: RefCell<HashMap<String, f32>>,
}

type AttentionCache = (Vec<Vec<f32>>, Vec<Vec<f32>>);

struct BootstrapDecodeSession {
    seq_len: usize,
    attention_cache_by_layer: HashMap<u32, AttentionCache>,
}

struct BootstrapPromptAuxData {
    seq_len: usize,
    layer_count: usize,
    ple_dim: usize,
    ple_all: Vec<f32>,
    prefix_hashes: Vec<u64>,
}

impl BootstrapPromptAuxData {
    fn from_flat(seq_len: usize, layer_count: usize, ple_dim: usize, ple_all: Vec<f32>) -> Self {
        Self { seq_len, layer_count, ple_dim, ple_all, prefix_hashes: Vec::new() }
    }

    fn with_prefix_hashes(mut self, prefix_hashes: Vec<u64>) -> Self {
        self.prefix_hashes = prefix_hashes;
        self
    }

    fn expected_values(&self) -> usize {
        self.seq_len
            .checked_mul(self.layer_count)
            .and_then(|count| count.checked_mul(self.ple_dim))
            .unwrap_or(0)
    }

    fn token_layer(&self, token_index: usize, layer_index: usize) -> Option<&[f32]> {
        if token_index >= self.seq_len || layer_index >= self.layer_count || self.ple_dim == 0 {
            return None;
        }
        let start = (token_index * self.layer_count + layer_index) * self.ple_dim;
        let end = start + self.ple_dim;
        self.ple_all.get(start..end)
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct BootstrapPleProfileSample {
    token_gather_us: u64,
    model_proj_us: u64,
    normalize_combine_us: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct BootstrapHeadIngressProfileSample {
    total_us: u64,
    embed_token_gather_us: u64,
    ple_token_gather_us: u64,
    ple_model_proj_us: u64,
    ple_normalize_combine_us: u64,
    prompt_aux_encode_us: u64,
    hidden_encode_us: u64,
    payload_frame_us: u64,
}

#[derive(Debug, Default)]
struct BootstrapHeadIngressProfileTotals {
    total_us: u128,
    embed_token_gather_us: u128,
    ple_token_gather_us: u128,
    ple_model_proj_us: u128,
    ple_normalize_combine_us: u128,
    prompt_aux_encode_us: u128,
    hidden_encode_us: u128,
    payload_frame_us: u128,
}

#[derive(Debug, Default, Clone, Copy)]
struct BootstrapLayerExecutionSample {
    layer_index: u32,
    total_us: u64,
    attention_cpu_us: u64,
    attention_matmul_us: u64,
    ffn_cpu_us: u64,
    ffn_matmul_us: u64,
    ple_us: u64,
}

#[derive(Debug, Default, Clone)]
struct BootstrapHeadExecutionProfileSample {
    total_us: u64,
    ingress_us: u64,
    payload_encode_us: u64,
    layers: Vec<BootstrapLayerExecutionSample>,
}

#[derive(Debug, Default)]
struct BootstrapHeadExecutionProfileTotals {
    total_us: u128,
    ingress_us: u128,
    payload_encode_us: u128,
    layer_totals: Vec<BootstrapLayerExecutionSample>,
}

impl BootstrapHeadIngressProfileTotals {
    fn record(&mut self, sample: BootstrapHeadIngressProfileSample) {
        self.total_us += u128::from(sample.total_us);
        self.embed_token_gather_us += u128::from(sample.embed_token_gather_us);
        self.ple_token_gather_us += u128::from(sample.ple_token_gather_us);
        self.ple_model_proj_us += u128::from(sample.ple_model_proj_us);
        self.ple_normalize_combine_us += u128::from(sample.ple_normalize_combine_us);
        self.prompt_aux_encode_us += u128::from(sample.prompt_aux_encode_us);
        self.hidden_encode_us += u128::from(sample.hidden_encode_us);
        self.payload_frame_us += u128::from(sample.payload_frame_us);
    }

    fn finalize(
        self,
        executor: &str,
        token_count: usize,
        iterations: u32,
        tensor: &StageTensor,
    ) -> GgmlHeadIngressProfile {
        let known_total = self.embed_token_gather_us
            + self.ple_token_gather_us
            + self.ple_model_proj_us
            + self.ple_normalize_combine_us
            + self.prompt_aux_encode_us
            + self.hidden_encode_us
            + self.payload_frame_us;
        let other_us = self.total_us.saturating_sub(known_total);
        let sections = stage_tensor_byte_sections(&tensor.bytes);
        let aux_bytes =
            sections.and_then(|parts| parts.aux_bytes.map(|bytes| bytes.len())).unwrap_or(0);
        GgmlHeadIngressProfile {
            executor: executor.to_string(),
            token_count,
            iterations,
            total_us: self.total_us.min(u128::from(u64::MAX)) as u64,
            embed_token_gather_us: Some(self.embed_token_gather_us.min(u128::from(u64::MAX)) as u64),
            ple_token_gather_us: Some(self.ple_token_gather_us.min(u128::from(u64::MAX)) as u64),
            ple_model_proj_us: Some(self.ple_model_proj_us.min(u128::from(u64::MAX)) as u64),
            ple_normalize_combine_us: Some(
                self.ple_normalize_combine_us.min(u128::from(u64::MAX)) as u64
            ),
            prompt_aux_encode_us: Some(self.prompt_aux_encode_us.min(u128::from(u64::MAX)) as u64),
            hidden_encode_us: Some(self.hidden_encode_us.min(u128::from(u64::MAX)) as u64),
            payload_frame_us: Some(self.payload_frame_us.min(u128::from(u64::MAX)) as u64),
            other_us: Some(other_us.min(u128::from(u64::MAX)) as u64),
            hidden_state_bytes: tensor.hidden_state_len(),
            aux_bytes,
            payload_bytes: tensor.bytes.len(),
        }
    }
}

impl BootstrapHeadExecutionProfileTotals {
    fn record(&mut self, sample: &BootstrapHeadExecutionProfileSample) {
        self.total_us += u128::from(sample.total_us);
        self.ingress_us += u128::from(sample.ingress_us);
        self.payload_encode_us += u128::from(sample.payload_encode_us);
        if self.layer_totals.len() < sample.layers.len() {
            self.layer_totals.resize(sample.layers.len(), BootstrapLayerExecutionSample::default());
        }
        for (idx, layer) in sample.layers.iter().enumerate() {
            if self.layer_totals[idx].layer_index == 0 {
                self.layer_totals[idx].layer_index = layer.layer_index;
            }
            self.layer_totals[idx].total_us =
                self.layer_totals[idx].total_us.saturating_add(layer.total_us);
            self.layer_totals[idx].attention_cpu_us =
                self.layer_totals[idx].attention_cpu_us.saturating_add(layer.attention_cpu_us);
            self.layer_totals[idx].attention_matmul_us = self.layer_totals[idx]
                .attention_matmul_us
                .saturating_add(layer.attention_matmul_us);
            self.layer_totals[idx].ffn_cpu_us =
                self.layer_totals[idx].ffn_cpu_us.saturating_add(layer.ffn_cpu_us);
            self.layer_totals[idx].ffn_matmul_us =
                self.layer_totals[idx].ffn_matmul_us.saturating_add(layer.ffn_matmul_us);
            self.layer_totals[idx].ple_us =
                self.layer_totals[idx].ple_us.saturating_add(layer.ple_us);
        }
    }

    fn finalize(
        self,
        executor: &str,
        token_count: usize,
        iterations: u32,
        effective_layer_cap: usize,
        tensor: &StageTensor,
    ) -> GgmlHeadExecutionProfile {
        let sections = stage_tensor_byte_sections(&tensor.bytes);
        let aux_bytes =
            sections.and_then(|parts| parts.aux_bytes.map(|bytes| bytes.len())).unwrap_or(0);
        GgmlHeadExecutionProfile {
            executor: executor.to_string(),
            token_count,
            iterations,
            effective_layer_cap,
            ingress_us: self.ingress_us.min(u128::from(u64::MAX)) as u64,
            payload_encode_us: self.payload_encode_us.min(u128::from(u64::MAX)) as u64,
            total_us: self.total_us.min(u128::from(u64::MAX)) as u64,
            layers: self
                .layer_totals
                .into_iter()
                .take(effective_layer_cap)
                .map(|layer| GgmlHeadLayerExecutionProfile {
                    layer_index: layer.layer_index,
                    total_us: layer.total_us,
                    attention_cpu_us: layer.attention_cpu_us,
                    attention_matmul_us: layer.attention_matmul_us,
                    ffn_cpu_us: layer.ffn_cpu_us,
                    ffn_matmul_us: layer.ffn_matmul_us,
                    ple_us: layer.ple_us,
                })
                .collect(),
            hidden_state_bytes: tensor.hidden_state_len(),
            aux_bytes,
            payload_bytes: tensor.bytes.len(),
        }
    }
}

struct LayerAttentionSpec {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_base_theta: f32,
    rope_rotary_dim: usize,
    proportional_rope: bool,
    sliding_window: Option<usize>,
    shared_kv_source_layer: Option<u32>,
}

impl BootstrapGgmlStageExecutor {
    fn new(init: &GgmlStageWorkerInitSpec) -> Result<Self> {
        let vocab_path = init
            .vocab_path
            .as_deref()
            .context("worker init is missing vocab_path for ggml bootstrap executor")?;
        let tokenizer = GemmaTokenizer::load(vocab_path, init.vocab_scores_path.as_deref())
            .context("load tokenizer for ggml bootstrap executor")?;
        let store = StageTensorStore::load(&init.index_path)
            .context("load stage tensor store for ggml bootstrap executor")?;
        store.validate_offsets().context("validate ggml bootstrap stage offsets")?;
        let bindings = GgmlStageBindingManifest::from_worker_init(init)
            .context("build validated ggml stage binding manifest for worker executor")?;
        let operator_plan = GgmlStageOperatorPlan::from_manifest(&bindings)
            .context("build typed ggml operator plan for worker executor")?;
        let execution_recipe = GgmlStageOperatorPlan::execution_recipe_from_worker_init(init)
            .context("build bound ggml execution recipe for worker executor")?;
        let materialized_recipe =
            GgmlStageOperatorPlan::materialized_execution_recipe_from_worker_init(init)
                .context("build materialized ggml execution recipe for worker executor")?;

        let sample_graph_runtime =
            GgmlSampleGraphRuntime::new(&init.runtime, &operator_plan, &store).ok();
        let sample_runtime_detail = sample_graph_runtime
            .as_ref()
            .map(GgmlSampleGraphRuntime::summary_label)
            .unwrap_or_else(|| "bootstrap-cpu-sample".into());
        let token_embd_graph_runtime = operator_plan
            .shared
            .token_embd
            .as_ref()
            .filter(|_| matches!(operator_plan.role.as_str(), "head" | "single"))
            .and_then(|entry| {
                GgmlGetRowsGraphRuntime::new_single(&init.runtime, entry, &store, "head-token-embd")
                    .ok()
            });
        let token_embd_runtime_detail = token_embd_graph_runtime
            .as_ref()
            .map(GgmlGetRowsGraphRuntime::summary_label)
            .unwrap_or_else(|| "bootstrap-cpu-token-embd".into());
        let ple_token_embd_graph_runtime = operator_plan
            .shared
            .per_layer_token_embd
            .as_ref()
            .filter(|_| matches!(operator_plan.role.as_str(), "head" | "single"))
            .and_then(|entry| {
                GgmlGetRowsGraphRuntime::new_single(
                    &init.runtime,
                    entry,
                    &store,
                    "head-ple-token-embd",
                )
                .ok()
            });
        let ple_token_embd_runtime_detail = ple_token_embd_graph_runtime
            .as_ref()
            .map(GgmlGetRowsGraphRuntime::summary_label)
            .unwrap_or_else(|| "bootstrap-cpu-ple-token-embd".into());
        let ple_model_proj_graph_runtime = operator_plan
            .shared
            .per_layer_model_proj
            .as_ref()
            .filter(|_| matches!(operator_plan.role.as_str(), "head" | "single"))
            .and_then(|entry| {
                GgmlSingleOutputGraphRuntime::new(
                    &init.runtime,
                    entry,
                    &store,
                    "head-ple-model-proj",
                )
                .ok()
            });
        let ple_model_proj_runtime_detail = ple_model_proj_graph_runtime
            .as_ref()
            .map(GgmlSingleOutputGraphRuntime::summary_label)
            .unwrap_or_else(|| "bootstrap-cpu-ple-model-proj".into());
        let head_stack_runtime = if matches!(operator_plan.role.as_str(), "head" | "single") {
            let head_layer_cap = init.debug_layer_cap.unwrap_or(operator_plan.layers.len());
            if head_layer_cap > 0 {
                GgmlTailStackRuntime::new(&init.runtime, &operator_plan, &store, head_layer_cap)
                    .ok()
            } else {
                None
            }
        } else {
            None
        };
        let head_runtime_detail = head_stack_runtime
            .as_ref()
            .map(GgmlTailStackRuntime::summary_label)
            .unwrap_or_else(|| "bootstrap-cpu-head-stack".into());
        let tail_stack_runtime = if matches!(operator_plan.role.as_str(), "tail" | "single") {
            let tail_layer_cap = init.debug_layer_cap.unwrap_or(operator_plan.layers.len());
            if tail_layer_cap > 0 {
                GgmlTailStackRuntime::new(&init.runtime, &operator_plan, &store, tail_layer_cap)
                    .ok()
            } else {
                None
            }
        } else {
            None
        };
        let tail_runtime_detail = tail_stack_runtime
            .as_ref()
            .map(GgmlTailStackRuntime::summary_label)
            .unwrap_or_else(|| "bootstrap-cpu-tail-layer".into());

        Ok(Self {
            plan: GgmlStageExecutorPlan {
                requested: GgmlStageExecutorKind::Ggml,
                active: GgmlStageExecutorKind::Ggml,
                detail: format!(
                    "bootstrap executor with validated stage bindings and {sample_runtime_detail}; {token_embd_runtime_detail}; {ple_token_embd_runtime_detail}; {ple_model_proj_runtime_detail}; {head_runtime_detail}; {tail_runtime_detail}"
                ),
            },
            tokenizer,
            model_id: init.model_id.clone(),
            store,
            runtime_plan: init.runtime.clone(),
            runtime_summary: init.runtime.summary_label(),
            binding_summary: bindings.summary_label(),
            operator_plan,
            execution_recipe,
            materialized_recipe,
            stage_id: init.stage_id.clone(),
            debug_layer_cap: init.debug_layer_cap,
            sample_graph_runtime,
            token_embd_graph_runtime,
            token_embd_batch_graph_runtimes: HashMap::new(),
            ple_token_embd_graph_runtime,
            ple_token_embd_batch_graph_runtimes: HashMap::new(),
            ple_model_proj_graph_runtime,
            ple_model_proj_batch_graph_runtimes: HashMap::new(),
            ple_ingress_graph_runtimes: HashMap::new(),
            head_stack_runtime,
            head_prefill_stack_runtimes: HashMap::new(),
            head_full_prefill_layer_runtimes: RefCell::new(HashMap::new()),
            tail_stack_runtime,
            decode_sessions: HashMap::new(),
            f32_vector_cache: RefCell::new(HashMap::new()),
            f32_scalar_cache: RefCell::new(HashMap::new()),
        })
    }

    fn read_tensor_bytes(&self, tensor_name: &str) -> Result<Vec<u8>> {
        self.store
            .read(tensor_name)
            .with_context(|| format!("read ggml bootstrap tensor `{tensor_name}`"))
    }

    fn debug_layer_compare_enabled() -> bool {
        std::env::var_os("COMPUTE_GGML_DEBUG_LAYER_COMPARE").is_some()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(lhs, rhs)| (lhs - rhs).abs()).fold(0.0f32, f32::max)
    }

    fn max_abs_diff_many(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<f32> {
        if a.len() != b.len() {
            bail!(
                "ggml debug compare batch len mismatch: candidate_batch={} reference_batch={}",
                a.len(),
                b.len()
            );
        }
        let mut max_abs = 0.0f32;
        for (lhs, rhs) in a.iter().zip(b.iter()) {
            if lhs.len() != rhs.len() {
                bail!(
                    "ggml debug compare row len mismatch: candidate_row={} reference_row={}",
                    lhs.len(),
                    rhs.len()
                );
            }
            max_abs = f32::max(max_abs, Self::max_abs_diff(lhs, rhs));
        }
        Ok(max_abs)
    }

    fn debug_compare_op(
        &self,
        op: &str,
        layer_index: u32,
        token_index: usize,
        candidate: &[f32],
        reference: &[f32],
    ) -> Result<()> {
        if candidate.len() != reference.len() {
            bail!(
                "ggml debug compare {} layer {} token {} len mismatch: candidate={} reference={}",
                op,
                layer_index,
                token_index,
                candidate.len(),
                reference.len()
            );
        }
        let max_abs = Self::max_abs_diff(candidate, reference);
        if max_abs > 0.0 {
            bail!(
                "ggml debug compare {} layer {} token {} mismatch: max_abs_diff={}",
                op,
                layer_index,
                token_index,
                max_abs
            );
        }
        Ok(())
    }

    fn debug_compare_op_many(
        &self,
        op: &str,
        layer_index: u32,
        candidate: &[Vec<f32>],
        reference: &[Vec<f32>],
    ) -> Result<()> {
        let max_abs = Self::max_abs_diff_many(candidate, reference)?;
        if max_abs > 0.0 {
            bail!(
                "ggml debug compare {} layer {} batch mismatch: max_abs_diff={}",
                op,
                layer_index,
                max_abs
            );
        }
        Ok(())
    }

    fn full_graph_head_prefill_enabled(
        &self,
        layer: &crate::inference::ggml_stage_plan::GgmlLayerBindings,
        _attn: &LayerAttentionSpec,
        states: &[Vec<f32>],
        position_offset: u32,
        existing_layer_cache: Option<&AttentionCache>,
        _shared_attention_cache: Option<&AttentionCache>,
    ) -> bool {
        let full_graph_local_layer_limit =
            std::env::var("COMPUTE_GGML_FULL_GRAPH_HEAD_LAYER_LIMIT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(4);
        let proportional_shared_kv_enabled =
            std::env::var_os("COMPUTE_GGML_ENABLE_PROPORTIONAL_SHARED_KV_FULL_GRAPH").is_some()
                && _attn.proportional_rope
                && _shared_attention_cache.is_some();
        let local_layer_index = self
            .operator_plan
            .layers
            .iter()
            .position(|candidate| candidate.layer_index == layer.layer_index);
        matches!(self.runtime_plan.target, StageAccelerationTarget::Metal)
            && position_offset == 0
            && states.len() > 1
            && existing_layer_cache.is_none()
            && (!_attn.proportional_rope || proportional_shared_kv_enabled)
            && local_layer_index.is_some_and(|index| index < full_graph_local_layer_limit)
    }

    fn try_run_full_graph_head_prefill_layer(
        &self,
        layer: &crate::inference::ggml_stage_plan::GgmlLayerBindings,
        hidden_dim: usize,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        position_offset: u32,
        attn: &LayerAttentionSpec,
        existing_layer_cache: Option<&AttentionCache>,
        shared_attention_cache: Option<&AttentionCache>,
    ) -> Result<Option<AttentionCache>> {
        if !self.full_graph_head_prefill_enabled(
            layer,
            attn,
            states,
            position_offset,
            existing_layer_cache,
            shared_attention_cache,
        ) {
            return Ok(None);
        }

        let runtime_key = (layer.layer_index, states.len(), shared_attention_cache.is_some());
        let mut runtimes = self.head_full_prefill_layer_runtimes.borrow_mut();
        if !runtimes.contains_key(&runtime_key) {
            let runtime = GgmlFullHeadPrefillLayerRuntime::new(
                &self.runtime_plan,
                layer,
                hidden_dim,
                GgmlHeadLayerGraphSpec {
                    n_heads: attn.n_heads,
                    n_kv_heads: attn.n_kv_heads,
                    head_dim: attn.head_dim,
                    rope_base_theta: attn.rope_base_theta,
                    rope_rotary_dim: attn.rope_rotary_dim,
                    proportional_rope: attn.proportional_rope,
                    sliding_window: attn.sliding_window,
                    uses_shared_kv: shared_attention_cache.is_some(),
                },
                states.len(),
                &self.store,
                "head-layer-full-prefill",
            )?;
            runtimes.insert(runtime_key, runtime);
        }
        let runtime =
            runtimes.get_mut(&runtime_key).expect("inserted full head prefill runtime present");

        let prompt_aux_for_layer = prompt_aux
            .map(|aux| {
                let ple_idx = layer.layer_index as usize;
                (0..states.len())
                    .map(|token_index| {
                        aux.token_layer(token_index, ple_idx)
                            .map(|slice| slice.to_vec())
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "ggml full head prefill prompt-aux missing layer {} token {}",
                                    layer.layer_index,
                                    token_index
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        let result = runtime.run(
            states,
            prompt_aux_for_layer.as_deref(),
            shared_attention_cache
                .map(|(k_cache, v_cache)| (k_cache.as_slice(), v_cache.as_slice())),
        )?;
        for (state, next_state) in states.iter_mut().zip(result.hidden_states.into_iter()) {
            *state = next_state;
        }
        Ok(match (result.k_cache, result.v_cache) {
            (Some(k_cache), Some(v_cache)) => Some((k_cache, v_cache)),
            (None, None) => None,
            _ => bail!(
                "ggml full head prefill layer {} returned incomplete attention cache",
                layer.layer_index
            ),
        })
    }

    fn debug_proportional_shared_kv_layer_exact(
        &self,
        layer: &crate::inference::ggml_stage_plan::GgmlLayerBindings,
        hidden_dim: usize,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        shared_attention_cache: &AttentionCache,
        position_offset: u32,
    ) -> Result<BootstrapSharedKvLayerTrace> {
        let eps = 1e-6f32;
        let seq_len = states.len();
        let attn = self.layer_attention_spec(layer, hidden_dim)?;
        if !attn.proportional_rope || attn.shared_kv_source_layer.is_none() {
            bail!(
                "ggml proportional shared-KV debug requires a proportional shared-KV layer, got layer {}",
                layer.layer_index
            );
        }

        let mut attn_inputs = states.to_vec();
        if let Some(attn_norm) = layer.attn_norm.as_ref() {
            let weight = self.read_f32_vector(&attn_norm.name)?;
            for input in &mut attn_inputs {
                real_math::rms_norm_inplace(input, &weight, eps);
            }
        }

        let mut q_all = self.matmul_many(&layer.attn_q, &attn_inputs)?;
        if let Some(q_norm) = layer.attn_q_norm.as_ref() {
            let weight = self.read_f32_vector(&q_norm.name)?;
            for q in &mut q_all {
                real_math::per_head_rms_norm(q, &weight, attn.n_heads, attn.head_dim);
            }
        }

        let q_pre_rope = q_all.clone();
        let rope_freqs = self.rope_freqs()?;
        let mut q_rope = Vec::with_capacity(seq_len);
        let mut attn_outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut q = std::mem::take(&mut q_all[t]);
            let mut shared_k_scratch = vec![0.0f32; attn.n_kv_heads * attn.head_dim];
            real_math::rope_apply_with_base_and_rotary_dim_mode(
                &mut q,
                &mut shared_k_scratch,
                &rope_freqs,
                position_offset + t as u32,
                attn.n_heads,
                attn.n_kv_heads,
                attn.head_dim,
                attn.rope_base_theta,
                attn.rope_rotary_dim,
                attn.proportional_rope,
            );
            let attn_out = real_math::gqa_attention_seq_with_window_and_limit(
                &q,
                &shared_attention_cache.0,
                &shared_attention_cache.1,
                attn.n_heads,
                attn.n_kv_heads,
                attn.head_dim,
                attn.sliding_window,
                Some(position_offset as usize + t + 1),
            );
            q_rope.push(q);
            attn_outputs.push(attn_out);
        }

        let mut attn_projected = self.matmul_many(&layer.attn_output, &attn_outputs)?;
        for t in 0..seq_len {
            let mut next_state = std::mem::take(&mut attn_projected[t]);
            if let Some(post_attn) = layer.post_attention_norm.as_ref() {
                let weight = self.read_f32_vector(&post_attn.name)?;
                real_math::rms_norm_inplace(&mut next_state, &weight, eps);
            }
            real_math::vec_add_inplace(&mut next_state, &states[t]);
            states[t] = next_state;
        }

        let mut ffn_inputs = states.to_vec();
        if let Some(ffn_norm) = layer.ffn_norm.as_ref() {
            let weight = self.read_f32_vector(&ffn_norm.name)?;
            for input in &mut ffn_inputs {
                real_math::rms_norm_inplace(input, &weight, eps);
            }
        }
        let ffn_input_refs: Vec<&[f32]> = ffn_inputs.iter().map(|input| input.as_slice()).collect();
        let gate_up_dim = layer.ffn_gate.dimensions.get(1).copied().unwrap_or_default() as usize;
        let down_dim = layer.ffn_down.dimensions.get(1).copied().unwrap_or_default() as usize;
        let (mut gate_all, up_all) =
            self.matmul_many_pair_token_major(&layer.ffn_gate, &layer.ffn_up, &ffn_input_refs)?;
        for t in 0..seq_len {
            let start = t * gate_up_dim;
            let end = start + gate_up_dim;
            real_math::gelu_pytorch_tanh_mul_inplace(
                &mut gate_all[start..end],
                &up_all[start..end],
            );
        }
        let gate_refs: Vec<&[f32]> = gate_all.chunks_exact(gate_up_dim).collect();
        let mut down_all = self.matmul_many_token_major(&layer.ffn_down, &gate_refs)?;
        for t in 0..seq_len {
            let start = t * down_dim;
            let end = start + down_dim;
            let next_state = &mut down_all[start..end];
            if let Some(post_ffn) = layer.post_ffw_norm.as_ref() {
                let weight = self.read_f32_vector(&post_ffn.name)?;
                real_math::rms_norm_inplace(next_state, &weight, eps);
            }
            real_math::vec_add_inplace(next_state, &states[t]);
            states[t].copy_from_slice(next_state);
        }

        if let (Some(inp_gate), Some(proj), Some(post_norm), Some(prompt_aux)) =
            (layer.inp_gate.as_ref(), layer.proj.as_ref(), layer.post_norm.as_ref(), prompt_aux)
        {
            let ple_idx = layer.layer_index as usize;
            let ple_inputs: Vec<Vec<f32>> = (0..seq_len)
                .map(|token_index| {
                    prompt_aux
                        .token_layer(token_index, ple_idx)
                        .map(|slice| slice.to_vec())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "ggml proportional shared-KV debug prompt-aux missing layer {} token {}",
                                layer.layer_index,
                                token_index
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut gated_all = self.matmul_many(inp_gate, states)?;
            for (gated, ple) in gated_all.iter_mut().zip(ple_inputs.iter()) {
                real_math::gelu_pytorch_tanh_mul_inplace(gated, ple);
            }
            let mut projected_all = self.matmul_many(proj, &gated_all)?;
            let weight = self.read_f32_vector(&post_norm.name)?;
            for t in 0..seq_len {
                let mut next_state = std::mem::take(&mut projected_all[t]);
                real_math::rms_norm_inplace(&mut next_state, &weight, eps);
                real_math::vec_add_inplace(&mut next_state, &states[t]);
                states[t] = next_state;
            }
        }

        if let Some(layer_scale) = layer.layer_output_scale.as_ref() {
            let scale = self.read_f32_scalar(&layer_scale.name)?;
            for state in states.iter_mut() {
                for value in state {
                    *value *= scale;
                }
            }
        }

        Ok(BootstrapSharedKvLayerTrace {
            q_pre_rope,
            q_rope,
            attn_out: attn_outputs,
            layer_out: states.to_vec(),
        })
    }

    fn read_f32_vector(&self, name: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.f32_vector_cache.borrow().get(name).cloned() {
            return Ok(cached);
        }
        let entry = self
            .store
            .entry(name)
            .ok_or_else(|| anyhow::anyhow!("ggml bootstrap tensor `{name}` is missing"))?;
        if entry.ggml_type != quants::GGML_TYPE_F32 {
            bail!(
                "ggml bootstrap tensor `{}` expected F32, got {}",
                name,
                quants::ggml_type_name(entry.ggml_type)
            );
        }
        let values = quants::dequantize_f32_tensor(&self.read_tensor_bytes(name)?)
            .with_context(|| format!("decode ggml bootstrap tensor `{name}` as f32 vector"))?;
        self.f32_vector_cache.borrow_mut().insert(name.to_string(), values.clone());
        Ok(values)
    }

    fn read_f32_scalar(&self, name: &str) -> Result<f32> {
        if let Some(cached) = self.f32_scalar_cache.borrow().get(name).copied() {
            return Ok(cached);
        }
        let raw = self.read_tensor_bytes(name)?;
        if raw.len() < 4 {
            bail!("ggml bootstrap tensor `{name}` is too small to decode as f32 scalar");
        }
        let value = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        self.f32_scalar_cache.borrow_mut().insert(name.to_string(), value);
        Ok(value)
    }

    fn read_f32_tensor(&self, entry: &stage_forward_lab::PackedTensorEntry) -> Result<Vec<f32>> {
        quants::dequantize_tensor(entry.ggml_type, &self.read_tensor_bytes(&entry.name)?)
            .with_context(|| format!("decode ggml bootstrap tensor `{}` as f32 tensor", entry.name))
    }

    fn decode_row(
        &self,
        entry: &stage_forward_lab::PackedTensorEntry,
        row_index: u32,
        row_elements: usize,
    ) -> Result<Vec<f32>> {
        let raw = self.read_tensor_bytes(&entry.name)?;
        let mut row = vec![0.0f32; row_elements];
        quants::dequantize_row_into(
            entry.ggml_type,
            &raw,
            row_index as usize,
            row_elements,
            &mut row,
        )
        .with_context(|| {
            format!(
                "decode ggml bootstrap row {} from `{}` as {} values",
                row_index, entry.name, row_elements
            )
        })?;
        Ok(row)
    }

    fn decode_hidden_state(bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
        Ok(Self::decode_hidden_states_payload(bytes, hidden_dim)?
            .into_iter()
            .last()
            .unwrap_or_default())
    }

    fn decode_hidden_states_payload(bytes: &[u8], hidden_dim: usize) -> Result<Vec<Vec<f32>>> {
        let hidden_bytes = stage_tensor_byte_sections(bytes)
            .map(|sections| sections.hidden_bytes)
            .unwrap_or(bytes);
        if hidden_dim == 0 {
            bail!("ggml bootstrap hidden_dim must be nonzero");
        }
        let state_stride = hidden_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("ggml bootstrap hidden-state stride overflow"))?;
        if hidden_bytes.len() % state_stride != 0 {
            bail!(
                "ggml bootstrap hidden-state byte length {} is not a multiple of hidden_dim * 4 = {}",
                hidden_bytes.len(),
                state_stride
            );
        }
        let mut states = Vec::with_capacity(hidden_bytes.len() / state_stride);
        for state_bytes in hidden_bytes.chunks_exact(state_stride) {
            states.push(
                state_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            );
        }
        Ok(states)
    }

    fn encode_hidden_states(states: &[Vec<f32>]) -> Vec<u8> {
        let total_values = states.iter().map(Vec::len).sum::<usize>();
        let mut bytes = Vec::with_capacity(total_values * 4);
        for state in states {
            Self::extend_le_f32_bytes(&mut bytes, state);
        }
        bytes
    }

    fn extend_le_f32_bytes(bytes: &mut Vec<u8>, values: &[f32]) {
        #[cfg(target_endian = "little")]
        unsafe {
            let raw = std::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                std::mem::size_of_val(values),
            );
            bytes.extend_from_slice(raw);
        }

        #[cfg(not(target_endian = "little"))]
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn encode_prompt_aux(prompt_aux: Option<&BootstrapPromptAuxData>) -> Result<Option<Vec<u8>>> {
        const REAL_STAGE_AUX_MAGIC: [u8; 4] = *b"rsa1";

        let seq_len = prompt_aux.map_or(0, |aux| aux.seq_len);
        let layer_count = prompt_aux.map_or(0, |aux| aux.layer_count);
        let ple_dim = prompt_aux.map_or(0, |aux| aux.ple_dim);
        let prefix_hashes = prompt_aux.map_or(&[][..], |aux| aux.prefix_hashes.as_slice());
        let expected_values = seq_len
            .checked_mul(layer_count)
            .and_then(|count| count.checked_mul(ple_dim))
            .ok_or_else(|| anyhow::anyhow!("ggml bootstrap prompt-aux dimensions overflow"))?;
        if expected_values == 0 && prefix_hashes.is_empty() {
            return Ok(None);
        }

        let mut bytes = Vec::with_capacity(20 + expected_values * 4 + prefix_hashes.len() * 8);
        bytes.extend_from_slice(&REAL_STAGE_AUX_MAGIC);
        bytes.extend_from_slice(&(seq_len as u32).to_le_bytes());
        bytes.extend_from_slice(&(layer_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(ple_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(prefix_hashes.len() as u32).to_le_bytes());
        if let Some(prompt_aux) = prompt_aux {
            if prompt_aux.expected_values() != expected_values {
                bail!("ggml bootstrap prompt-aux value count mismatch while encoding");
            }
            Self::extend_le_f32_bytes(&mut bytes, &prompt_aux.ple_all);
        }
        for hash in prefix_hashes {
            bytes.extend_from_slice(&hash.to_le_bytes());
        }
        Ok(Some(bytes))
    }

    fn decode_prompt_aux(bytes: &[u8]) -> Result<BootstrapPromptAuxData> {
        const REAL_STAGE_AUX_MAGIC: [u8; 4] = *b"rsa1";

        if bytes.len() < 20 || bytes[..4] != REAL_STAGE_AUX_MAGIC {
            bail!("invalid ggml bootstrap prompt-aux payload");
        }

        let seq_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let layer_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let ple_dim = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let prefix_hash_count =
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let ple_bytes = seq_len
            .checked_mul(layer_count)
            .and_then(|count| count.checked_mul(ple_dim))
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("ggml bootstrap prompt-aux dimensions overflow"))?;
        let expected_bytes = 20
            + ple_bytes
            + prefix_hash_count
                .checked_mul(8)
                .ok_or_else(|| anyhow::anyhow!("ggml bootstrap prompt-aux prefix hash overflow"))?;
        if bytes.len() != expected_bytes {
            bail!(
                "invalid ggml bootstrap prompt-aux byte length {} (expected {})",
                bytes.len(),
                expected_bytes
            );
        }

        let mut offset = 20;
        let value_count = seq_len
            .checked_mul(layer_count)
            .and_then(|count| count.checked_mul(ple_dim))
            .ok_or_else(|| anyhow::anyhow!("ggml bootstrap prompt-aux dimensions overflow"))?;
        let mut ple_all = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            let chunk = &bytes[offset..offset + 4];
            ple_all.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            offset += 4;
        }
        let mut prefix_hashes = Vec::with_capacity(prefix_hash_count);
        for _ in 0..prefix_hash_count {
            let chunk = &bytes[offset..offset + 8];
            prefix_hashes.push(u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
            offset += 8;
        }
        Ok(BootstrapPromptAuxData { seq_len, layer_count, ple_dim, ple_all, prefix_hashes })
    }

    fn prefix_hashes(token_ids: &[u32]) -> Vec<u64> {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        let mut out = Vec::with_capacity(token_ids.len());
        for &token_id in token_ids {
            for byte in token_id.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            out.push(hash);
        }
        out
    }

    #[allow(dead_code)]
    fn get_rows_with_runtime_cache(
        runtime_plan: &GgmlRuntimePlan,
        store: &StageTensorStore,
        batch_runtimes: &mut HashMap<usize, GgmlGetRowsGraphRuntime>,
        entry: &stage_forward_lab::PackedTensorEntry,
        row_indices: &[u32],
        label: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let row_count = row_indices.len();
        if row_count == 0 {
            return Ok(Vec::new());
        }
        if row_count == 1 {
            bail!("ggml batched get_rows requested for single-row input on `{label}`");
        }
        let runtime = if let Some(runtime) = batch_runtimes.get_mut(&row_count) {
            runtime
        } else {
            let runtime =
                GgmlGetRowsGraphRuntime::new(runtime_plan, entry, row_count, store, label)?;
            batch_runtimes.insert(row_count, runtime);
            batch_runtimes.get_mut(&row_count).expect("inserted batched get_rows runtime present")
        };
        runtime.get_rows(row_indices)
    }

    fn embed_tokens_for_begin(
        &mut self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        token_ids: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let scale = (begin_plan.hidden_dim as f32).sqrt();
        let rows = token_ids
            .iter()
            .map(|&token_id| {
                self.decode_row(&begin_plan.token_embd, token_id, begin_plan.hidden_dim)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut embeddings = Vec::with_capacity(rows.len());
        for mut row in rows {
            for value in &mut row {
                *value *= scale;
            }
            embeddings.push(row);
        }
        Ok(embeddings)
    }

    #[allow(dead_code)]
    fn run_single_output_with_runtime_cache(
        runtime_plan: &GgmlRuntimePlan,
        store: &StageTensorStore,
        batch_runtimes: &mut HashMap<usize, GgmlBatchedSingleOutputGraphRuntime>,
        entry: &stage_forward_lab::PackedTensorEntry,
        inputs: &[Vec<f32>],
        label: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let batch_count = inputs.len();
        if batch_count == 0 {
            return Ok(Vec::new());
        }
        if batch_count == 1 {
            bail!("ggml batched single-output requested for single-row input on `{label}`");
        }
        let runtime = if let Some(runtime) = batch_runtimes.get_mut(&batch_count) {
            runtime
        } else {
            let runtime = GgmlBatchedSingleOutputGraphRuntime::new(
                runtime_plan,
                entry,
                batch_count,
                store,
                label,
            )?;
            batch_runtimes.insert(batch_count, runtime);
            batch_runtimes
                .get_mut(&batch_count)
                .expect("inserted batched single-output runtime present")
        };
        runtime.run_many(inputs)
    }

    #[allow(dead_code)]
    fn run_ple_ingress_with_runtime_cache(
        runtime_plan: &GgmlRuntimePlan,
        store: &StageTensorStore,
        batch_runtimes: &mut HashMap<usize, GgmlPleIngressGraphRuntime>,
        model_proj: &stage_forward_lab::PackedTensorEntry,
        proj_norm: &stage_forward_lab::PackedTensorEntry,
        token_embd: &stage_forward_lab::PackedTensorEntry,
        ple_dim: usize,
        token_ids: &[u32],
        embedded_states: &[Vec<f32>],
        label: &str,
    ) -> Result<Vec<f32>> {
        let batch_count = token_ids.len();
        if batch_count == 0 {
            return Ok(Vec::new());
        }
        if batch_count != embedded_states.len() {
            bail!(
                "ggml PLE ingress batch mismatch: token_ids={} embedded_states={}",
                batch_count,
                embedded_states.len()
            );
        }
        let runtime = if let Some(runtime) = batch_runtimes.get_mut(&batch_count) {
            runtime
        } else {
            let runtime = GgmlPleIngressGraphRuntime::new(
                runtime_plan,
                model_proj,
                proj_norm,
                token_embd,
                ple_dim,
                batch_count,
                store,
                label,
            )?;
            batch_runtimes.insert(batch_count, runtime);
            batch_runtimes.get_mut(&batch_count).expect("inserted PLE ingress runtime present")
        };
        runtime.run(token_ids, embedded_states)
    }

    fn compute_ple_inputs_for_begin(
        &mut self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        token_ids: &[u32],
        embedded_states: &[Vec<f32>],
    ) -> Result<Option<BootstrapPromptAuxData>> {
        Ok(self.compute_ple_inputs_for_begin_profiled(begin_plan, token_ids, embedded_states)?.0)
    }

    fn compute_ple_inputs_for_begin_profiled(
        &mut self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        token_ids: &[u32],
        embedded_states: &[Vec<f32>],
    ) -> Result<(Option<BootstrapPromptAuxData>, BootstrapPleProfileSample)> {
        let mut profile = BootstrapPleProfileSample::default();
        let Some(ple_token_embd) = begin_plan.per_layer_token_embd.as_ref() else {
            return Ok((None, profile));
        };
        let Some(ple_model_proj) = begin_plan.per_layer_model_proj.as_ref() else {
            return Ok((None, profile));
        };
        let Some(ple_proj_norm) = begin_plan.per_layer_proj_norm.as_ref() else {
            return Ok((None, profile));
        };
        let ple_dim = ple_proj_norm.dimensions.first().copied().unwrap_or_default() as usize;
        if ple_dim == 0 {
            return Ok((None, profile));
        }
        let total_ple_dim = ple_model_proj.dimensions.get(1).copied().unwrap_or_default() as usize;
        if total_ple_dim == 0 || total_ple_dim % ple_dim != 0 {
            bail!(
                "ggml bootstrap invalid PLE projection shape for `{}`: output {} is not divisible by ple_dim {}",
                ple_model_proj.name,
                total_ple_dim,
                ple_dim
            );
        }
        let num_layers = total_ple_dim / ple_dim;
        let proj_norm = self.read_f32_tensor(ple_proj_norm)?;
        let embed_scale = (ple_dim as f32).sqrt();
        let proj_scale = (embedded_states[0].len() as f32).powf(-0.5);
        let combine_scale = (2.0f32).powf(-0.5);
        let model_proj_start = Instant::now();
        let proj_matrix = self.read_f32_tensor(ple_model_proj)?;
        let input_refs: Vec<&[f32]> =
            embedded_states.iter().map(|state| state.as_slice()).collect();
        let projected = real_math::matmul_many_refs_range(
            &proj_matrix,
            &input_refs,
            0,
            total_ple_dim,
            embedded_states[0].len(),
        );
        profile.model_proj_us =
            model_proj_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        let token_gather_start = Instant::now();
        let token_rows = token_ids
            .iter()
            .map(|&token_id| self.decode_row(ple_token_embd, token_id, total_ple_dim))
            .collect::<Result<Vec<_>>>()?;
        profile.token_gather_us =
            token_gather_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;

        let mut ple_all = Vec::with_capacity(token_ids.len() * num_layers * ple_dim);
        let normalize_combine_start = Instant::now();
        for (token_row, mut combined) in token_rows.into_iter().zip(projected.into_iter()) {
            for value in &mut combined {
                *value *= proj_scale;
            }
            real_math::rms_norm_chunked_inplace(&mut combined, ple_dim, &proj_norm, 1e-6);
            for (out_value, token_value) in combined.iter_mut().zip(token_row.iter()) {
                *out_value = (*out_value + *token_value * embed_scale) * combine_scale;
            }
            ple_all.extend_from_slice(&combined);
        }
        profile.normalize_combine_us =
            normalize_combine_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        Ok((
            Some(BootstrapPromptAuxData::from_flat(token_ids.len(), num_layers, ple_dim, ple_all)),
            profile,
        ))
    }

    fn prepare_begin_token_ids(
        &mut self,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<(
        crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        Vec<Vec<f32>>,
        Option<BootstrapPromptAuxData>,
        Option<Vec<u8>>,
    )> {
        let begin_plan = self.operator_plan.begin_token_ids_plan(token_ids.len(), max_tokens)?;
        let embedded_states = self.embed_tokens_for_begin(&begin_plan, token_ids)?;
        let prompt_aux =
            self.compute_ple_inputs_for_begin(&begin_plan, token_ids, &embedded_states)?;
        let prefix_hashes =
            if token_ids.len() > 1 { Self::prefix_hashes(token_ids) } else { Vec::new() };
        let prompt_aux = match prompt_aux {
            Some(prompt_aux) => Some(prompt_aux.with_prefix_hashes(prefix_hashes)),
            None if !prefix_hashes.is_empty() => Some(
                BootstrapPromptAuxData::from_flat(0, 0, 0, Vec::new())
                    .with_prefix_hashes(prefix_hashes),
            ),
            None => None,
        };
        let prompt_aux_bytes = Self::encode_prompt_aux(prompt_aux.as_ref())?;
        Ok((begin_plan, embedded_states, prompt_aux, prompt_aux_bytes))
    }

    fn begin_token_ids_cap0(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        Ok(self.profiled_begin_token_ids_cap0(request_id, token_ids, max_tokens)?.0)
    }

    fn profiled_begin_token_ids_cap0(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<(StageTensor, BootstrapHeadIngressProfileSample)> {
        let total_start = Instant::now();
        let begin_plan = self.operator_plan.begin_token_ids_plan(token_ids.len(), max_tokens)?;
        let embed_start = Instant::now();
        let embedded_states = self.embed_tokens_for_begin(&begin_plan, token_ids)?;
        let embed_token_gather_us =
            embed_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        let (prompt_aux, ple_profile) =
            self.compute_ple_inputs_for_begin_profiled(&begin_plan, token_ids, &embedded_states)?;
        let prompt_aux = match prompt_aux {
            Some(prompt_aux) => Some(prompt_aux.with_prefix_hashes(if token_ids.len() > 1 {
                Self::prefix_hashes(token_ids)
            } else {
                Vec::new()
            })),
            None if token_ids.len() > 1 => Some(
                BootstrapPromptAuxData::from_flat(0, 0, 0, Vec::new())
                    .with_prefix_hashes(Self::prefix_hashes(token_ids)),
            ),
            None => None,
        };
        let prompt_aux_encode_start = Instant::now();
        let prompt_aux_bytes = Self::encode_prompt_aux(prompt_aux.as_ref())?;
        let prompt_aux_encode_us =
            prompt_aux_encode_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        let hidden_encode_start = Instant::now();
        let hidden_bytes = Self::encode_hidden_states(&embedded_states);
        let hidden_encode_us =
            hidden_encode_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        let payload_frame_start = Instant::now();
        let bytes = encode_stage_tensor_bytes(&hidden_bytes, prompt_aux_bytes.as_deref());
        let payload_frame_us =
            payload_frame_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;

        let tensor = StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![self.stage_id.clone()],
            hidden_dim: begin_plan.hidden_dim,
            bytes,
            prompt_text: None,
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        };
        let sample = BootstrapHeadIngressProfileSample {
            total_us: total_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            embed_token_gather_us,
            ple_token_gather_us: ple_profile.token_gather_us,
            ple_model_proj_us: ple_profile.model_proj_us,
            ple_normalize_combine_us: ple_profile.normalize_combine_us,
            prompt_aux_encode_us,
            hidden_encode_us,
            payload_frame_us,
        };
        Ok((tensor, sample))
    }

    fn matmul_many(
        &self,
        entry: &stage_forward_lab::PackedTensorEntry,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let in_dim = entry.dimensions.first().copied().unwrap_or_default() as usize;
        let out_dim = entry.dimensions.get(1).copied().unwrap_or_default() as usize;
        let input_refs: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        match entry.ggml_type {
            quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q5_K | quants::GGML_TYPE_Q6_K => {
                let raw = self.read_tensor_bytes(&entry.name)?;
                real_math::matmul_quantized_many_refs_range(
                    entry.ggml_type,
                    &raw,
                    &input_refs,
                    0,
                    out_dim,
                    in_dim,
                )
            }
            _ => {
                let matrix = self.read_f32_tensor(entry)?;
                Ok(real_math::matmul_many_refs_range(&matrix, &input_refs, 0, out_dim, in_dim))
            }
        }
    }

    fn matmul_many_token_major(
        &self,
        entry: &stage_forward_lab::PackedTensorEntry,
        inputs: &[&[f32]],
    ) -> Result<Vec<f32>> {
        let in_dim = entry.dimensions.first().copied().unwrap_or_default() as usize;
        let out_dim = entry.dimensions.get(1).copied().unwrap_or_default() as usize;
        match entry.ggml_type {
            quants::GGML_TYPE_Q4_K | quants::GGML_TYPE_Q5_K | quants::GGML_TYPE_Q6_K => {
                let raw = self.read_tensor_bytes(&entry.name)?;
                real_math::matmul_quantized_many_refs_range_token_major(
                    entry.ggml_type,
                    &raw,
                    inputs,
                    0,
                    out_dim,
                    in_dim,
                )
            }
            _ => {
                let matrix = self.read_f32_tensor(entry)?;
                Ok(real_math::matmul_many_refs_range_token_major(
                    &matrix, inputs, 0, out_dim, in_dim,
                ))
            }
        }
    }

    fn matmul_many_pair_token_major(
        &self,
        entry_a: &stage_forward_lab::PackedTensorEntry,
        entry_b: &stage_forward_lab::PackedTensorEntry,
        inputs: &[&[f32]],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let in_dim = entry_a.dimensions.first().copied().unwrap_or_default() as usize;
        let out_dim = entry_a.dimensions.get(1).copied().unwrap_or_default() as usize;
        if entry_a.ggml_type == quants::GGML_TYPE_Q4_K
            && entry_b.ggml_type == quants::GGML_TYPE_Q4_K
            && entry_a.dimensions == entry_b.dimensions
        {
            let raw_a = self.read_tensor_bytes(&entry_a.name)?;
            let raw_b = self.read_tensor_bytes(&entry_b.name)?;
            real_math::matmul_quantized_many_pair_q4_k_refs_range_token_major(
                &raw_a, &raw_b, inputs, 0, out_dim, in_dim,
            )
        } else {
            Ok((
                self.matmul_many_token_major(entry_a, inputs)?,
                self.matmul_many_token_major(entry_b, inputs)?,
            ))
        }
    }

    fn layer_head_dim(layer: &crate::inference::ggml_stage_plan::GgmlLayerBindings) -> usize {
        layer
            .attn_q_norm
            .as_ref()
            .and_then(|entry| entry.dimensions.first().copied())
            .or_else(|| {
                layer.attn_k_norm.as_ref().and_then(|entry| entry.dimensions.first().copied())
            })
            .unwrap_or_default() as usize
    }

    fn layer_attention_spec(
        &self,
        layer: &crate::inference::ggml_stage_plan::GgmlLayerBindings,
        hidden_dim: usize,
    ) -> Result<LayerAttentionSpec> {
        let q_dim = layer.attn_q.dimensions.get(1).copied().unwrap_or_default() as usize;
        let k_dim = layer.attn_k.dimensions.get(1).copied().unwrap_or_default() as usize;
        let head_dim = Self::layer_head_dim(layer);
        if head_dim == 0 {
            bail!("ggml bootstrap could not derive head_dim for layer {}", layer.layer_index);
        }
        if q_dim == 0 || k_dim == 0 || q_dim % head_dim != 0 || k_dim % head_dim != 0 {
            bail!(
                "ggml bootstrap invalid attention dims on layer {}: q_dim={} k_dim={} head_dim={}",
                layer.layer_index,
                q_dim,
                k_dim,
                head_dim
            );
        }
        let n_heads = q_dim / head_dim;
        let n_kv_heads = k_dim / head_dim;
        let gemma4_e4b = self.model_id.contains("gemma-4-e4b")
            && hidden_dim == 2560
            && matches!(head_dim, 256 | 512);
        if gemma4_e4b && layer.layer_index % 6 == 5 {
            Ok(LayerAttentionSpec {
                n_heads,
                n_kv_heads,
                head_dim,
                rope_base_theta: 1_000_000.0,
                rope_rotary_dim: head_dim / 4,
                proportional_rope: true,
                sliding_window: None,
                shared_kv_source_layer: if layer.layer_index >= 24 { Some(23) } else { None },
            })
        } else if gemma4_e4b {
            Ok(LayerAttentionSpec {
                n_heads,
                n_kv_heads,
                head_dim,
                rope_base_theta: 10_000.0,
                rope_rotary_dim: head_dim,
                proportional_rope: false,
                sliding_window: Some(512),
                shared_kv_source_layer: if layer.layer_index >= 24 { Some(22) } else { None },
            })
        } else {
            Ok(LayerAttentionSpec {
                n_heads,
                n_kv_heads,
                head_dim,
                rope_base_theta: 1_000_000.0,
                rope_rotary_dim: head_dim,
                proportional_rope: false,
                sliding_window: None,
                shared_kv_source_layer: None,
            })
        }
    }

    fn rope_freqs(&self) -> Result<Vec<f32>> {
        if self.model_id.contains("gemma-4-e4b") {
            Ok(Vec::new())
        } else {
            self.read_f32_vector("rope_freqs.weight")
        }
    }

    fn run_hidden_layer(
        &self,
        layer: &crate::inference::ggml_stage_plan::GgmlLayerBindings,
        hidden_dim: usize,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        existing_layer_cache: Option<AttentionCache>,
        shared_attention_cache: Option<&AttentionCache>,
        position_offset: u32,
    ) -> Result<Option<AttentionCache>> {
        let eps = 1e-6f32;
        let seq_len = states.len();
        let attn = self.layer_attention_spec(layer, hidden_dim)?;

        if let Some(cache) = self.try_run_full_graph_head_prefill_layer(
            layer,
            hidden_dim,
            states,
            prompt_aux,
            position_offset,
            &attn,
            existing_layer_cache.as_ref(),
            shared_attention_cache,
        )? {
            return Ok(Some(cache));
        }

        let mut attn_inputs = states.to_vec();
        if let Some(attn_norm) = layer.attn_norm.as_ref() {
            let weight = self.read_f32_vector(&attn_norm.name)?;
            for input in &mut attn_inputs {
                real_math::rms_norm_inplace(input, &weight, eps);
            }
        }
        let mut q_all = self.matmul_many(&layer.attn_q, &attn_inputs)?;
        if let Some(q_norm) = layer.attn_q_norm.as_ref() {
            let weight = self.read_f32_vector(&q_norm.name)?;
            for q in &mut q_all {
                real_math::per_head_rms_norm(q, &weight, attn.n_heads, attn.head_dim);
            }
        }
        let mut k_all = if shared_attention_cache.is_none() {
            Some(self.matmul_many(&layer.attn_k, &attn_inputs)?)
        } else {
            None
        };
        if let Some(k_norm) = layer.attn_k_norm.as_ref() {
            let weight = self.read_f32_vector(&k_norm.name)?;
            if let Some(k_all) = k_all.as_mut() {
                for k in k_all {
                    real_math::per_head_rms_norm(k, &weight, attn.n_kv_heads, attn.head_dim);
                }
            }
        }
        let mut v_all = if shared_attention_cache.is_none() {
            Some(self.matmul_many(&layer.attn_v, &attn_inputs)?)
        } else {
            None
        };
        if let Some(v_all) = v_all.as_mut() {
            for v in v_all {
                real_math::per_head_rms_norm_no_scale(v, attn.n_kv_heads, attn.head_dim);
            }
        }

        let rope_freqs = self.rope_freqs()?;
        let (mut k_cache, mut v_cache) = existing_layer_cache.unwrap_or_default();
        let mut attn_outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut q = std::mem::take(&mut q_all[t]);
            let attn_out = if let Some((shared_k_cache, shared_v_cache)) = shared_attention_cache {
                let mut shared_k_scratch = vec![0.0f32; attn.n_kv_heads * attn.head_dim];
                real_math::rope_apply_with_base_and_rotary_dim_mode(
                    &mut q,
                    &mut shared_k_scratch,
                    &rope_freqs,
                    position_offset + t as u32,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.rope_base_theta,
                    attn.rope_rotary_dim,
                    attn.proportional_rope,
                );
                real_math::gqa_attention_seq_with_window_and_limit(
                    &q,
                    shared_k_cache,
                    shared_v_cache,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.sliding_window,
                    Some(position_offset as usize + t + 1),
                )
            } else {
                let mut k = std::mem::take(&mut k_all.as_mut().expect("k cache present")[t]);
                let v = std::mem::take(&mut v_all.as_mut().expect("v cache present")[t]);
                real_math::rope_apply_with_base_and_rotary_dim_mode(
                    &mut q,
                    &mut k,
                    &rope_freqs,
                    position_offset + t as u32,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.rope_base_theta,
                    attn.rope_rotary_dim,
                    attn.proportional_rope,
                );
                k_cache.push(k);
                v_cache.push(v);
                real_math::gqa_attention_seq_with_window_and_limit(
                    &q,
                    &k_cache,
                    &v_cache,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.sliding_window,
                    Some(position_offset as usize + t + 1),
                )
            };
            attn_outputs.push(attn_out);
        }

        let mut attn_projected = self.matmul_many(&layer.attn_output, &attn_outputs)?;
        for t in 0..seq_len {
            let mut next_state = std::mem::take(&mut attn_projected[t]);
            if let Some(post_attn) = layer.post_attention_norm.as_ref() {
                let weight = self.read_f32_vector(&post_attn.name)?;
                real_math::rms_norm_inplace(&mut next_state, &weight, eps);
            }
            real_math::vec_add_inplace(&mut next_state, &states[t]);
            states[t] = next_state;
        }

        let mut ffn_inputs = states.to_vec();
        if let Some(ffn_norm) = layer.ffn_norm.as_ref() {
            let weight = self.read_f32_vector(&ffn_norm.name)?;
            for input in &mut ffn_inputs {
                real_math::rms_norm_inplace(input, &weight, eps);
            }
        }
        let ffn_input_refs: Vec<&[f32]> = ffn_inputs.iter().map(|input| input.as_slice()).collect();
        let gate_up_dim = layer.ffn_gate.dimensions.get(1).copied().unwrap_or_default() as usize;
        let down_dim = layer.ffn_down.dimensions.get(1).copied().unwrap_or_default() as usize;
        let (mut gate_all, up_all) =
            self.matmul_many_pair_token_major(&layer.ffn_gate, &layer.ffn_up, &ffn_input_refs)?;
        for t in 0..seq_len {
            let start = t * gate_up_dim;
            let end = start + gate_up_dim;
            real_math::gelu_pytorch_tanh_mul_inplace(
                &mut gate_all[start..end],
                &up_all[start..end],
            );
        }
        let gate_refs: Vec<&[f32]> = gate_all.chunks_exact(gate_up_dim).collect();
        let mut down_all = self.matmul_many_token_major(&layer.ffn_down, &gate_refs)?;
        for t in 0..seq_len {
            let start = t * down_dim;
            let end = start + down_dim;
            let next_state = &mut down_all[start..end];
            if let Some(post_ffn) = layer.post_ffw_norm.as_ref() {
                let weight = self.read_f32_vector(&post_ffn.name)?;
                real_math::rms_norm_inplace(next_state, &weight, eps);
            }
            real_math::vec_add_inplace(next_state, &states[t]);
            states[t].copy_from_slice(next_state);
        }

        if let (Some(inp_gate), Some(proj), Some(post_norm), Some(prompt_aux)) =
            (layer.inp_gate.as_ref(), layer.proj.as_ref(), layer.post_norm.as_ref(), prompt_aux)
        {
            let ple_idx = layer.layer_index as usize;
            let ple_inputs: Vec<Vec<f32>> = (0..seq_len)
                .map(|token_index| {
                    prompt_aux
                        .token_layer(token_index, ple_idx)
                        .map(|slice| slice.to_vec())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "ggml bootstrap prompt-aux missing layer {} token {} while running cap1",
                                layer.layer_index,
                                token_index
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut gated_all = self.matmul_many(inp_gate, &states)?;
            for (gated, ple) in gated_all.iter_mut().zip(ple_inputs.iter()) {
                real_math::gelu_pytorch_tanh_mul_inplace(gated, ple);
            }
            let mut projected_all = self.matmul_many(proj, &gated_all)?;
            let weight = self.read_f32_vector(&post_norm.name)?;
            for t in 0..seq_len {
                let mut next_state = std::mem::take(&mut projected_all[t]);
                real_math::rms_norm_inplace(&mut next_state, &weight, eps);
                real_math::vec_add_inplace(&mut next_state, &states[t]);
                states[t] = next_state;
            }
        }

        if let Some(layer_scale) = layer.layer_output_scale.as_ref() {
            let scale = self.read_f32_scalar(&layer_scale.name)?;
            for state in states.iter_mut() {
                for value in state {
                    *value *= scale;
                }
            }
        }

        Ok(if shared_attention_cache.is_none() { Some((k_cache, v_cache)) } else { None })
    }

    fn run_begin_layers_with_cache(
        &self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        layer_cap: usize,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
    ) -> Result<()> {
        for layer in begin_plan.layers.iter().take(layer_cap) {
            let attn = self.layer_attention_spec(layer, begin_plan.hidden_dim)?;
            let shared_attention_cache = attn
                .shared_kv_source_layer
                .and_then(|layer_index| attention_cache_by_layer.get(&layer_index).cloned());
            let existing_layer_cache = if shared_attention_cache.is_none() {
                attention_cache_by_layer.remove(&layer.layer_index)
            } else {
                None
            };
            if let Some(cache) = self.run_hidden_layer(
                layer,
                begin_plan.hidden_dim,
                states,
                prompt_aux,
                existing_layer_cache,
                shared_attention_cache.as_ref(),
                position_offset,
            )? {
                attention_cache_by_layer.insert(layer.layer_index, cache);
            }
        }
        Ok(())
    }

    fn run_begin_layers_with_cache_profiled(
        &self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        layer_cap: usize,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
    ) -> Result<Vec<BootstrapLayerExecutionSample>> {
        let mut samples = Vec::with_capacity(layer_cap.min(begin_plan.layers.len()));
        for layer in begin_plan.layers.iter().take(layer_cap) {
            let layer_start = Instant::now();
            let attn = self.layer_attention_spec(layer, begin_plan.hidden_dim)?;
            let shared_attention_cache = attn
                .shared_kv_source_layer
                .and_then(|layer_index| attention_cache_by_layer.get(&layer_index).cloned());
            let existing_layer_cache = if shared_attention_cache.is_none() {
                attention_cache_by_layer.remove(&layer.layer_index)
            } else {
                None
            };
            if let Some(cache) = self.run_hidden_layer(
                layer,
                begin_plan.hidden_dim,
                states,
                prompt_aux,
                existing_layer_cache,
                shared_attention_cache.as_ref(),
                position_offset,
            )? {
                attention_cache_by_layer.insert(layer.layer_index, cache);
            }
            samples.push(BootstrapLayerExecutionSample {
                layer_index: layer.layer_index,
                total_us: layer_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                ..Default::default()
            });
        }
        Ok(samples)
    }

    fn run_begin_layers_with_runtimes(
        &self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        layer_cap: usize,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
        runtimes: &mut [GgmlTailLayerRuntime],
    ) -> Result<()> {
        let layers = &begin_plan.layers[..layer_cap.min(begin_plan.layers.len())];
        self.run_layers_with_runtimes(
            layers,
            begin_plan.hidden_dim,
            states,
            prompt_aux,
            position_offset,
            attention_cache_by_layer,
            runtimes,
        )
    }

    fn run_begin_layers_with_runtimes_profiled(
        &self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        layer_cap: usize,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
        runtimes: &mut [GgmlTailLayerRuntime],
    ) -> Result<Vec<BootstrapLayerExecutionSample>> {
        let layers = &begin_plan.layers[..layer_cap.min(begin_plan.layers.len())];
        self.run_layers_with_runtimes_profiled(
            layers,
            begin_plan.hidden_dim,
            states,
            prompt_aux,
            position_offset,
            attention_cache_by_layer,
            runtimes,
        )
    }

    fn run_begin_layers_with_batched_runtimes(
        &self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        layer_cap: usize,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
        runtimes: &mut [GgmlTailLayerRuntime],
    ) -> Result<()> {
        let layers = &begin_plan.layers[..layer_cap.min(begin_plan.layers.len())];
        let _ = self.run_layers_with_batched_runtimes_profiled(
            layers,
            begin_plan.hidden_dim,
            states,
            prompt_aux,
            position_offset,
            attention_cache_by_layer,
            runtimes,
        )?;
        Ok(())
    }

    fn run_begin_layers_with_batched_runtimes_profiled(
        &self,
        begin_plan: &crate::inference::ggml_stage_plan::GgmlBeginTokenIdsPlan,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        layer_cap: usize,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
        runtimes: &mut [GgmlTailLayerRuntime],
    ) -> Result<Vec<BootstrapLayerExecutionSample>> {
        let layers = &begin_plan.layers[..layer_cap.min(begin_plan.layers.len())];
        self.run_layers_with_batched_runtimes_profiled(
            layers,
            begin_plan.hidden_dim,
            states,
            prompt_aux,
            position_offset,
            attention_cache_by_layer,
            runtimes,
        )
    }

    fn run_continue_layers_with_cache(
        &self,
        continue_plan: &crate::inference::ggml_stage_plan::GgmlContinueForwardPlan,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
    ) -> Result<()> {
        for layer in &continue_plan.layers {
            let attn = self.layer_attention_spec(layer, continue_plan.input.hidden_dim)?;
            let shared_attention_cache = attn
                .shared_kv_source_layer
                .and_then(|layer_index| attention_cache_by_layer.get(&layer_index).cloned());
            let existing_layer_cache = if shared_attention_cache.is_none() {
                attention_cache_by_layer.remove(&layer.layer_index)
            } else {
                None
            };
            if let Some(cache) = self.run_hidden_layer(
                layer,
                continue_plan.input.hidden_dim,
                states,
                prompt_aux,
                existing_layer_cache,
                shared_attention_cache.as_ref(),
                position_offset,
            )? {
                attention_cache_by_layer.insert(layer.layer_index, cache);
            }
        }
        Ok(())
    }

    fn run_layers_with_runtimes(
        &self,
        layers: &[crate::inference::ggml_stage_plan::GgmlLayerBindings],
        hidden_dim: usize,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
        runtimes: &mut [GgmlTailLayerRuntime],
    ) -> Result<()> {
        if runtimes.len() != layers.len() {
            bail!(
                "ggml layer runtime count mismatch: runtimes={} plan_layers={}",
                runtimes.len(),
                layers.len()
            );
        }

        let eps = 1e-6f32;
        let rope_freqs = self.rope_freqs()?;

        for (layer, runtime) in layers.iter().zip(runtimes.iter_mut()) {
            if runtime.layer_index() != layer.layer_index {
                bail!(
                    "ggml layer runtime mismatch: runtime={} plan={}",
                    runtime.layer_index(),
                    layer.layer_index
                );
            }

            let attn = self.layer_attention_spec(layer, hidden_dim)?;
            if attn.shared_kv_source_layer.is_some() {
                bail!(
                    "ggml layer runtime does not support shared-KV layer {} yet",
                    layer.layer_index
                );
            }
            if let Some(cache) = self.try_run_full_graph_head_prefill_layer(
                layer,
                hidden_dim,
                states,
                prompt_aux,
                position_offset,
                &attn,
                attention_cache_by_layer.get(&layer.layer_index),
                None,
            )? {
                attention_cache_by_layer.insert(layer.layer_index, cache);
                continue;
            }
            let mut layer_cache =
                attention_cache_by_layer.remove(&layer.layer_index).unwrap_or_default();

            for (token_index, state) in states.iter_mut().enumerate() {
                let original_state = state.clone();

                let mut attn_input = original_state.clone();
                if let Some(attn_norm) = layer.attn_norm.as_ref() {
                    let weight = self.read_f32_vector(&attn_norm.name)?;
                    real_math::rms_norm_inplace(&mut attn_input, &weight, eps);
                }

                let (mut q, mut k, mut v) = runtime.qkv(&attn_input)?;
                if Self::debug_layer_compare_enabled() {
                    let q_ref = self.matmul_many(&layer.attn_q, &[attn_input.clone()])?;
                    let k_ref = self.matmul_many(&layer.attn_k, &[attn_input.clone()])?;
                    let v_ref = self.matmul_many(&layer.attn_v, &[attn_input.clone()])?;
                    self.debug_compare_op(
                        "attn_q",
                        layer.layer_index,
                        token_index,
                        &q,
                        q_ref.first().expect("single q ref present"),
                    )?;
                    self.debug_compare_op(
                        "attn_k",
                        layer.layer_index,
                        token_index,
                        &k,
                        k_ref.first().expect("single k ref present"),
                    )?;
                    self.debug_compare_op(
                        "attn_v",
                        layer.layer_index,
                        token_index,
                        &v,
                        v_ref.first().expect("single v ref present"),
                    )?;
                }
                if let Some(q_norm) = layer.attn_q_norm.as_ref() {
                    let weight = self.read_f32_vector(&q_norm.name)?;
                    real_math::per_head_rms_norm(&mut q, &weight, attn.n_heads, attn.head_dim);
                }
                if let Some(k_norm) = layer.attn_k_norm.as_ref() {
                    let weight = self.read_f32_vector(&k_norm.name)?;
                    real_math::per_head_rms_norm(&mut k, &weight, attn.n_kv_heads, attn.head_dim);
                }
                real_math::per_head_rms_norm_no_scale(&mut v, attn.n_kv_heads, attn.head_dim);

                real_math::rope_apply_with_base_and_rotary_dim_mode(
                    &mut q,
                    &mut k,
                    &rope_freqs,
                    position_offset + token_index as u32,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.rope_base_theta,
                    attn.rope_rotary_dim,
                    attn.proportional_rope,
                );
                layer_cache.0.push(k);
                layer_cache.1.push(v);

                let attn_out = real_math::gqa_attention_seq_with_window_and_limit(
                    &q,
                    &layer_cache.0,
                    &layer_cache.1,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.sliding_window,
                    Some(position_offset as usize + token_index + 1),
                );
                let mut next_state = runtime.attn_output(&attn_out)?;
                if Self::debug_layer_compare_enabled() {
                    let attn_output_ref =
                        self.matmul_many(&layer.attn_output, &[attn_out.clone()])?;
                    self.debug_compare_op(
                        "attn_output",
                        layer.layer_index,
                        token_index,
                        &next_state,
                        attn_output_ref.first().expect("single attn_output ref present"),
                    )?;
                }
                if let Some(post_attn) = layer.post_attention_norm.as_ref() {
                    let weight = self.read_f32_vector(&post_attn.name)?;
                    real_math::rms_norm_inplace(&mut next_state, &weight, eps);
                }
                real_math::vec_add_inplace(&mut next_state, &original_state);

                let mut ffn_input = next_state.clone();
                if let Some(ffn_norm) = layer.ffn_norm.as_ref() {
                    let weight = self.read_f32_vector(&ffn_norm.name)?;
                    real_math::rms_norm_inplace(&mut ffn_input, &weight, eps);
                }
                let (mut gate, up) = runtime.gate_up(&ffn_input)?;
                if Self::debug_layer_compare_enabled() {
                    let gate_ref = self.matmul_many(&layer.ffn_gate, &[ffn_input.clone()])?;
                    let up_ref = self.matmul_many(&layer.ffn_up, &[ffn_input.clone()])?;
                    self.debug_compare_op(
                        "ffn_gate",
                        layer.layer_index,
                        token_index,
                        &gate,
                        gate_ref.first().expect("single gate ref present"),
                    )?;
                    self.debug_compare_op(
                        "ffn_up",
                        layer.layer_index,
                        token_index,
                        &up,
                        up_ref.first().expect("single up ref present"),
                    )?;
                }
                real_math::gelu_pytorch_tanh_mul_inplace(&mut gate, &up);
                let mut ffn_state = runtime.down(&gate)?;
                if Self::debug_layer_compare_enabled() {
                    let down_ref = self.matmul_many(&layer.ffn_down, &[gate.clone()])?;
                    self.debug_compare_op(
                        "ffn_down",
                        layer.layer_index,
                        token_index,
                        &ffn_state,
                        down_ref.first().expect("single down ref present"),
                    )?;
                }
                if let Some(post_ffn) = layer.post_ffw_norm.as_ref() {
                    let weight = self.read_f32_vector(&post_ffn.name)?;
                    real_math::rms_norm_inplace(&mut ffn_state, &weight, eps);
                }
                real_math::vec_add_inplace(&mut ffn_state, &next_state);

                if let (Some(_inp_gate), Some(_proj), Some(post_norm), Some(prompt_aux)) = (
                    layer.inp_gate.as_ref(),
                    layer.proj.as_ref(),
                    layer.post_norm.as_ref(),
                    prompt_aux,
                ) {
                    let ple_idx = layer.layer_index as usize;
                    let ple_input = prompt_aux
                        .token_layer(token_index, ple_idx)
                        .map(|slice| slice.to_vec())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "ggml layer runtime prompt-aux missing layer {} token {}",
                                layer.layer_index,
                                token_index
                            )
                        })?;
                    let mut gated = runtime.inp_gate(&ffn_state)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "ggml layer runtime missing inp_gate graph for layer {}",
                            layer.layer_index
                        )
                    })?;
                    real_math::gelu_pytorch_tanh_mul_inplace(&mut gated, &ple_input);
                    let mut projected = runtime.proj(&gated)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "ggml layer runtime missing proj graph for layer {}",
                            layer.layer_index
                        )
                    })?;
                    let weight = self.read_f32_vector(&post_norm.name)?;
                    real_math::rms_norm_inplace(&mut projected, &weight, eps);
                    real_math::vec_add_inplace(&mut projected, &ffn_state);
                    ffn_state = projected;
                }

                if let Some(layer_scale) = layer.layer_output_scale.as_ref() {
                    let scale = self.read_f32_scalar(&layer_scale.name)?;
                    for value in &mut ffn_state {
                        *value *= scale;
                    }
                }

                *state = ffn_state;
            }

            attention_cache_by_layer.insert(layer.layer_index, layer_cache);
        }
        Ok(())
    }

    fn run_layers_with_runtimes_profiled(
        &self,
        layers: &[crate::inference::ggml_stage_plan::GgmlLayerBindings],
        hidden_dim: usize,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
        runtimes: &mut [GgmlTailLayerRuntime],
    ) -> Result<Vec<BootstrapLayerExecutionSample>> {
        if runtimes.len() != layers.len() {
            bail!(
                "ggml layer runtime count mismatch: runtimes={} plan_layers={}",
                runtimes.len(),
                layers.len()
            );
        }

        let eps = 1e-6f32;
        let rope_freqs = self.rope_freqs()?;
        let mut samples = Vec::with_capacity(layers.len());

        for (layer, runtime) in layers.iter().zip(runtimes.iter_mut()) {
            let layer_start = Instant::now();
            let mut sample = BootstrapLayerExecutionSample {
                layer_index: layer.layer_index,
                ..Default::default()
            };
            if runtime.layer_index() != layer.layer_index {
                bail!(
                    "ggml layer runtime mismatch: runtime={} plan={}",
                    runtime.layer_index(),
                    layer.layer_index
                );
            }

            let attn = self.layer_attention_spec(layer, hidden_dim)?;
            if attn.shared_kv_source_layer.is_some() {
                bail!(
                    "ggml layer runtime does not support shared-KV layer {} yet",
                    layer.layer_index
                );
            }
            if let Some(cache) = self.try_run_full_graph_head_prefill_layer(
                layer,
                hidden_dim,
                states,
                prompt_aux,
                position_offset,
                &attn,
                attention_cache_by_layer.get(&layer.layer_index),
                None,
            )? {
                attention_cache_by_layer.insert(layer.layer_index, cache);
                sample.total_us =
                    layer_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                sample.attention_matmul_us = sample.total_us;
                samples.push(sample);
                continue;
            }
            let mut layer_cache =
                attention_cache_by_layer.remove(&layer.layer_index).unwrap_or_default();

            for (token_index, state) in states.iter_mut().enumerate() {
                let attn_cpu_start = Instant::now();
                let original_state = state.clone();

                let mut attn_input = original_state.clone();
                if let Some(attn_norm) = layer.attn_norm.as_ref() {
                    let weight = self.read_f32_vector(&attn_norm.name)?;
                    real_math::rms_norm_inplace(&mut attn_input, &weight, eps);
                }
                sample.attention_cpu_us = sample.attention_cpu_us.saturating_add(
                    attn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let attn_matmul_start = Instant::now();
                let (mut q, mut k, mut v) = runtime.qkv(&attn_input)?;
                sample.attention_matmul_us = sample.attention_matmul_us.saturating_add(
                    attn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let attn_cpu_start = Instant::now();
                if let Some(q_norm) = layer.attn_q_norm.as_ref() {
                    let weight = self.read_f32_vector(&q_norm.name)?;
                    real_math::per_head_rms_norm(&mut q, &weight, attn.n_heads, attn.head_dim);
                }
                if let Some(k_norm) = layer.attn_k_norm.as_ref() {
                    let weight = self.read_f32_vector(&k_norm.name)?;
                    real_math::per_head_rms_norm(&mut k, &weight, attn.n_kv_heads, attn.head_dim);
                }
                real_math::per_head_rms_norm_no_scale(&mut v, attn.n_kv_heads, attn.head_dim);

                real_math::rope_apply_with_base_and_rotary_dim_mode(
                    &mut q,
                    &mut k,
                    &rope_freqs,
                    position_offset + token_index as u32,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.rope_base_theta,
                    attn.rope_rotary_dim,
                    attn.proportional_rope,
                );
                layer_cache.0.push(k);
                layer_cache.1.push(v);

                let attn_out = real_math::gqa_attention_seq_with_window_and_limit(
                    &q,
                    &layer_cache.0,
                    &layer_cache.1,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.sliding_window,
                    Some(position_offset as usize + token_index + 1),
                );
                sample.attention_cpu_us = sample.attention_cpu_us.saturating_add(
                    attn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let attn_matmul_start = Instant::now();
                let mut next_state = runtime.attn_output(&attn_out)?;
                sample.attention_matmul_us = sample.attention_matmul_us.saturating_add(
                    attn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let attn_cpu_start = Instant::now();
                if let Some(post_attn) = layer.post_attention_norm.as_ref() {
                    let weight = self.read_f32_vector(&post_attn.name)?;
                    real_math::rms_norm_inplace(&mut next_state, &weight, eps);
                }
                real_math::vec_add_inplace(&mut next_state, &original_state);
                sample.attention_cpu_us = sample.attention_cpu_us.saturating_add(
                    attn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let ffn_cpu_start = Instant::now();
                let mut ffn_input = next_state.clone();
                if let Some(ffn_norm) = layer.ffn_norm.as_ref() {
                    let weight = self.read_f32_vector(&ffn_norm.name)?;
                    real_math::rms_norm_inplace(&mut ffn_input, &weight, eps);
                }
                sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                    ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let ffn_matmul_start = Instant::now();
                let (mut gate, up) = runtime.gate_up(&ffn_input)?;
                sample.ffn_matmul_us = sample.ffn_matmul_us.saturating_add(
                    ffn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let ffn_cpu_start = Instant::now();
                real_math::gelu_pytorch_tanh_mul_inplace(&mut gate, &up);
                sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                    ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let ffn_matmul_start = Instant::now();
                let mut ffn_state = runtime.down(&gate)?;
                sample.ffn_matmul_us = sample.ffn_matmul_us.saturating_add(
                    ffn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                let ffn_cpu_start = Instant::now();
                if let Some(post_ffn) = layer.post_ffw_norm.as_ref() {
                    let weight = self.read_f32_vector(&post_ffn.name)?;
                    real_math::rms_norm_inplace(&mut ffn_state, &weight, eps);
                }
                real_math::vec_add_inplace(&mut ffn_state, &next_state);
                sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                    ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );

                if let (Some(_inp_gate), Some(_proj), Some(post_norm), Some(prompt_aux)) = (
                    layer.inp_gate.as_ref(),
                    layer.proj.as_ref(),
                    layer.post_norm.as_ref(),
                    prompt_aux,
                ) {
                    let ple_start = Instant::now();
                    let ple_idx = layer.layer_index as usize;
                    let ple_input = prompt_aux
                        .token_layer(token_index, ple_idx)
                        .map(|slice| slice.to_vec())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "ggml layer runtime prompt-aux missing layer {} token {}",
                                layer.layer_index,
                                token_index
                            )
                        })?;
                    let mut gated = runtime.inp_gate(&ffn_state)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "ggml layer runtime missing inp_gate graph for layer {}",
                            layer.layer_index
                        )
                    })?;
                    real_math::gelu_pytorch_tanh_mul_inplace(&mut gated, &ple_input);
                    let mut projected = runtime.proj(&gated)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "ggml layer runtime missing proj graph for layer {}",
                            layer.layer_index
                        )
                    })?;
                    let weight = self.read_f32_vector(&post_norm.name)?;
                    real_math::rms_norm_inplace(&mut projected, &weight, eps);
                    real_math::vec_add_inplace(&mut projected, &ffn_state);
                    ffn_state = projected;
                    sample.ple_us = sample.ple_us.saturating_add(
                        ple_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                    );
                }

                if let Some(layer_scale) = layer.layer_output_scale.as_ref() {
                    let ffn_cpu_start = Instant::now();
                    let scale = self.read_f32_scalar(&layer_scale.name)?;
                    for value in &mut ffn_state {
                        *value *= scale;
                    }
                    sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                        ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                    );
                }

                *state = ffn_state;
            }

            attention_cache_by_layer.insert(layer.layer_index, layer_cache);
            sample.total_us = layer_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
            samples.push(sample);
        }
        Ok(samples)
    }

    fn run_layers_with_batched_runtimes_profiled(
        &self,
        layers: &[crate::inference::ggml_stage_plan::GgmlLayerBindings],
        hidden_dim: usize,
        states: &mut [Vec<f32>],
        prompt_aux: Option<&BootstrapPromptAuxData>,
        position_offset: u32,
        attention_cache_by_layer: &mut HashMap<u32, AttentionCache>,
        runtimes: &mut [GgmlTailLayerRuntime],
    ) -> Result<Vec<BootstrapLayerExecutionSample>> {
        if runtimes.len() != layers.len() {
            bail!(
                "ggml batched layer runtime count mismatch: runtimes={} plan_layers={}",
                runtimes.len(),
                layers.len()
            );
        }
        let seq_len = states.len();
        let eps = 1e-6f32;
        let rope_freqs = self.rope_freqs()?;
        let mut samples = Vec::with_capacity(layers.len());

        for (layer, runtime) in layers.iter().zip(runtimes.iter_mut()) {
            let layer_start = Instant::now();
            let mut sample = BootstrapLayerExecutionSample {
                layer_index: layer.layer_index,
                ..Default::default()
            };
            if runtime.layer_index() != layer.layer_index {
                bail!(
                    "ggml batched layer runtime mismatch: runtime={} plan={}",
                    runtime.layer_index(),
                    layer.layer_index
                );
            }
            if runtime.batch_count() != seq_len {
                bail!(
                    "ggml batched layer runtime batch mismatch: runtime={} seq_len={}",
                    runtime.batch_count(),
                    seq_len
                );
            }

            let attn = self.layer_attention_spec(layer, hidden_dim)?;
            if attn.shared_kv_source_layer.is_some() {
                bail!(
                    "ggml batched layer runtime does not support shared-KV layer {} yet",
                    layer.layer_index
                );
            }
            let mut layer_cache =
                attention_cache_by_layer.remove(&layer.layer_index).unwrap_or_default();

            let attn_cpu_start = Instant::now();
            let mut attn_inputs = states.to_vec();
            if let Some(attn_norm) = layer.attn_norm.as_ref() {
                let weight = self.read_f32_vector(&attn_norm.name)?;
                for input in &mut attn_inputs {
                    real_math::rms_norm_inplace(input, &weight, eps);
                }
            }
            sample.attention_cpu_us = sample.attention_cpu_us.saturating_add(
                attn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );

            let attn_matmul_start = Instant::now();
            let (mut q_all, mut k_all, mut v_all) = runtime.qkv_many(&attn_inputs)?;
            sample.attention_matmul_us = sample.attention_matmul_us.saturating_add(
                attn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );
            if Self::debug_layer_compare_enabled() {
                let q_ref = self.matmul_many(&layer.attn_q, &attn_inputs)?;
                let k_ref = self.matmul_many(&layer.attn_k, &attn_inputs)?;
                let v_ref = self.matmul_many(&layer.attn_v, &attn_inputs)?;
                self.debug_compare_op_many("attn_q_many", layer.layer_index, &q_all, &q_ref)?;
                self.debug_compare_op_many("attn_k_many", layer.layer_index, &k_all, &k_ref)?;
                self.debug_compare_op_many("attn_v_many", layer.layer_index, &v_all, &v_ref)?;
            }

            let attn_cpu_start = Instant::now();
            if let Some(q_norm) = layer.attn_q_norm.as_ref() {
                let weight = self.read_f32_vector(&q_norm.name)?;
                for q in &mut q_all {
                    real_math::per_head_rms_norm(q, &weight, attn.n_heads, attn.head_dim);
                }
            }
            if let Some(k_norm) = layer.attn_k_norm.as_ref() {
                let weight = self.read_f32_vector(&k_norm.name)?;
                for k in &mut k_all {
                    real_math::per_head_rms_norm(k, &weight, attn.n_kv_heads, attn.head_dim);
                }
            }
            for v in &mut v_all {
                real_math::per_head_rms_norm_no_scale(v, attn.n_kv_heads, attn.head_dim);
            }

            let mut attn_outputs = Vec::with_capacity(seq_len);
            for token_index in 0..seq_len {
                let mut q = std::mem::take(&mut q_all[token_index]);
                let mut k = std::mem::take(&mut k_all[token_index]);
                let v = std::mem::take(&mut v_all[token_index]);
                real_math::rope_apply_with_base_and_rotary_dim_mode(
                    &mut q,
                    &mut k,
                    &rope_freqs,
                    position_offset + token_index as u32,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.rope_base_theta,
                    attn.rope_rotary_dim,
                    attn.proportional_rope,
                );
                layer_cache.0.push(k);
                layer_cache.1.push(v);
                attn_outputs.push(real_math::gqa_attention_seq_with_window_and_limit(
                    &q,
                    &layer_cache.0,
                    &layer_cache.1,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.sliding_window,
                    Some(position_offset as usize + token_index + 1),
                ));
            }
            sample.attention_cpu_us = sample.attention_cpu_us.saturating_add(
                attn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );

            let attn_matmul_start = Instant::now();
            let mut attn_projected = runtime.attn_output_many(&attn_outputs)?;
            sample.attention_matmul_us = sample.attention_matmul_us.saturating_add(
                attn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );
            if Self::debug_layer_compare_enabled() {
                let attn_output_ref = self.matmul_many(&layer.attn_output, &attn_outputs)?;
                self.debug_compare_op_many(
                    "attn_output_many",
                    layer.layer_index,
                    &attn_projected,
                    &attn_output_ref,
                )?;
            }

            let attn_cpu_start = Instant::now();
            for token_index in 0..seq_len {
                let mut next_state = std::mem::take(&mut attn_projected[token_index]);
                if let Some(post_attn) = layer.post_attention_norm.as_ref() {
                    let weight = self.read_f32_vector(&post_attn.name)?;
                    real_math::rms_norm_inplace(&mut next_state, &weight, eps);
                }
                real_math::vec_add_inplace(&mut next_state, &states[token_index]);
                states[token_index] = next_state;
            }
            sample.attention_cpu_us = sample.attention_cpu_us.saturating_add(
                attn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );

            let ffn_cpu_start = Instant::now();
            let mut ffn_inputs = states.to_vec();
            if let Some(ffn_norm) = layer.ffn_norm.as_ref() {
                let weight = self.read_f32_vector(&ffn_norm.name)?;
                for input in &mut ffn_inputs {
                    real_math::rms_norm_inplace(input, &weight, eps);
                }
            }
            sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );

            let ffn_matmul_start = Instant::now();
            let (mut gate_all, up_all) = runtime.gate_up_many(&ffn_inputs)?;
            sample.ffn_matmul_us = sample.ffn_matmul_us.saturating_add(
                ffn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );
            if Self::debug_layer_compare_enabled() {
                let gate_ref = self.matmul_many(&layer.ffn_gate, &ffn_inputs)?;
                let up_ref = self.matmul_many(&layer.ffn_up, &ffn_inputs)?;
                self.debug_compare_op_many(
                    "ffn_gate_many",
                    layer.layer_index,
                    &gate_all,
                    &gate_ref,
                )?;
                self.debug_compare_op_many("ffn_up_many", layer.layer_index, &up_all, &up_ref)?;
            }

            let ffn_cpu_start = Instant::now();
            for (gate, up) in gate_all.iter_mut().zip(up_all.iter()) {
                real_math::gelu_pytorch_tanh_mul_inplace(gate, up);
            }
            sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );

            let ffn_matmul_start = Instant::now();
            let mut down_all = runtime.down_many(&gate_all)?;
            sample.ffn_matmul_us = sample.ffn_matmul_us.saturating_add(
                ffn_matmul_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );
            if Self::debug_layer_compare_enabled() {
                let down_ref = self.matmul_many(&layer.ffn_down, &gate_all)?;
                self.debug_compare_op_many(
                    "ffn_down_many",
                    layer.layer_index,
                    &down_all,
                    &down_ref,
                )?;
            }

            let ffn_cpu_start = Instant::now();
            for token_index in 0..seq_len {
                let next_state = &mut down_all[token_index];
                if let Some(post_ffn) = layer.post_ffw_norm.as_ref() {
                    let weight = self.read_f32_vector(&post_ffn.name)?;
                    real_math::rms_norm_inplace(next_state, &weight, eps);
                }
                real_math::vec_add_inplace(next_state, &states[token_index]);
            }
            sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
            );
            for (state, next_state) in states.iter_mut().zip(down_all.into_iter()) {
                *state = next_state;
            }

            if let (Some(inp_gate), Some(proj), Some(post_norm), Some(prompt_aux)) =
                (layer.inp_gate.as_ref(), layer.proj.as_ref(), layer.post_norm.as_ref(), prompt_aux)
            {
                let ple_start = Instant::now();
                let ple_idx = layer.layer_index as usize;
                let ple_inputs: Vec<Vec<f32>> = (0..seq_len)
                    .map(|token_index| {
                        prompt_aux
                            .token_layer(token_index, ple_idx)
                            .map(|slice| slice.to_vec())
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "ggml batched layer runtime prompt-aux missing layer {} token {}",
                                    layer.layer_index,
                                    token_index
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut gated_all = self.matmul_many(inp_gate, states)?;
                for (gated, ple) in gated_all.iter_mut().zip(ple_inputs.iter()) {
                    real_math::gelu_pytorch_tanh_mul_inplace(gated, ple);
                }
                let mut projected_all = self.matmul_many(proj, &gated_all)?;
                let weight = self.read_f32_vector(&post_norm.name)?;
                for token_index in 0..seq_len {
                    let mut next_state = std::mem::take(&mut projected_all[token_index]);
                    real_math::rms_norm_inplace(&mut next_state, &weight, eps);
                    real_math::vec_add_inplace(&mut next_state, &states[token_index]);
                    states[token_index] = next_state;
                }
                sample.ple_us = sample.ple_us.saturating_add(
                    ple_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );
            }

            if let Some(layer_scale) = layer.layer_output_scale.as_ref() {
                let ffn_cpu_start = Instant::now();
                let scale = self.read_f32_scalar(&layer_scale.name)?;
                for state in states.iter_mut() {
                    for value in state {
                        *value *= scale;
                    }
                }
                sample.ffn_cpu_us = sample.ffn_cpu_us.saturating_add(
                    ffn_cpu_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                );
            }

            attention_cache_by_layer.insert(layer.layer_index, layer_cache);
            sample.total_us = layer_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
            samples.push(sample);
        }

        Ok(samples)
    }

    fn begin_token_ids_with_layer_cap(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        layer_cap: usize,
    ) -> Result<StageTensor> {
        let current_token_count = token_ids.len();
        let mut session = if current_token_count == 1 {
            self.decode_sessions.remove(request_id)
        } else {
            self.decode_sessions.remove(request_id);
            None
        };
        let position_offset = session.as_ref().map(|session| session.seq_len as u32).unwrap_or(0);
        let (begin_plan, mut states, prompt_aux, _) =
            self.prepare_begin_token_ids(token_ids, max_tokens)?;
        let mut attention_cache_by_layer =
            session.take().map(|session| session.attention_cache_by_layer).unwrap_or_default();
        let effective_layer_cap = layer_cap.min(begin_plan.layers.len());
        let use_head_prefill_runtime =
            std::env::var_os("COMPUTE_GGML_ENABLE_BATCHED_HEAD_PREFILL_RUNTIME").is_some()
                && effective_layer_cap > 0
                && current_token_count > 1
                && matches!(begin_plan.role.as_str(), "head" | "single");
        let use_head_layer_runtime = effective_layer_cap > 0
            && matches!(begin_plan.role.as_str(), "head" | "single")
            && self
                .head_stack_runtime
                .as_ref()
                .is_some_and(|runtime| runtime.len() == effective_layer_cap);

        if use_head_prefill_runtime {
            let key = (effective_layer_cap, current_token_count);
            if !self.head_prefill_stack_runtimes.contains_key(&key) {
                let runtime = GgmlTailStackRuntime::new_with_batch_count(
                    &self.runtime_plan,
                    &self.operator_plan,
                    &self.store,
                    effective_layer_cap,
                    current_token_count,
                )?;
                self.head_prefill_stack_runtimes.insert(key, runtime);
            }
            let mut runtime = self
                .head_prefill_stack_runtimes
                .remove(&key)
                .expect("head prefill stack runtime present");
            let result = self.run_begin_layers_with_batched_runtimes(
                &begin_plan,
                &mut states,
                prompt_aux.as_ref(),
                effective_layer_cap,
                position_offset,
                &mut attention_cache_by_layer,
                runtime.layers_mut(),
            );
            self.head_prefill_stack_runtimes.insert(key, runtime);
            result?;
        } else if use_head_layer_runtime {
            let mut runtime = self.head_stack_runtime.take().expect("head stack runtime present");
            let result = self.run_begin_layers_with_runtimes(
                &begin_plan,
                &mut states,
                prompt_aux.as_ref(),
                effective_layer_cap,
                position_offset,
                &mut attention_cache_by_layer,
                runtime.layers_mut(),
            );
            self.head_stack_runtime = Some(runtime);
            result?;
        } else {
            self.run_begin_layers_with_cache(
                &begin_plan,
                &mut states,
                prompt_aux.as_ref(),
                effective_layer_cap,
                position_offset,
                &mut attention_cache_by_layer,
            )?;
        }

        let prefix_hashes =
            if current_token_count > 1 { Self::prefix_hashes(token_ids) } else { Vec::new() };
        let prompt_aux = match prompt_aux {
            Some(prompt_aux) => Some(prompt_aux.with_prefix_hashes(prefix_hashes)),
            None if !prefix_hashes.is_empty() => Some(
                BootstrapPromptAuxData::from_flat(0, 0, 0, Vec::new())
                    .with_prefix_hashes(prefix_hashes),
            ),
            None => None,
        };
        let prompt_aux_bytes = Self::encode_prompt_aux(prompt_aux.as_ref())?;
        let hidden_bytes = Self::encode_hidden_states(&states);
        let bytes = encode_stage_tensor_bytes(&hidden_bytes, prompt_aux_bytes.as_deref());
        self.decode_sessions.insert(
            request_id.to_string(),
            BootstrapDecodeSession {
                seq_len: position_offset as usize + current_token_count,
                attention_cache_by_layer,
            },
        );
        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![self.stage_id.clone()],
            hidden_dim: begin_plan.hidden_dim,
            bytes,
            prompt_text: None,
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn profiled_begin_token_ids_with_layer_cap(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        layer_cap: usize,
    ) -> Result<(StageTensor, BootstrapHeadExecutionProfileSample)> {
        let total_start = Instant::now();
        let current_token_count = token_ids.len();
        let mut session = if current_token_count == 1 {
            self.decode_sessions.remove(request_id)
        } else {
            self.decode_sessions.remove(request_id);
            None
        };
        let position_offset = session.as_ref().map(|session| session.seq_len as u32).unwrap_or(0);
        let ingress_start = Instant::now();
        let (begin_plan, mut states, prompt_aux, _) =
            self.prepare_begin_token_ids(token_ids, max_tokens)?;
        let ingress_us = ingress_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        let mut attention_cache_by_layer =
            session.take().map(|session| session.attention_cache_by_layer).unwrap_or_default();
        let effective_layer_cap = layer_cap.min(begin_plan.layers.len());
        let use_head_prefill_runtime =
            std::env::var_os("COMPUTE_GGML_ENABLE_BATCHED_HEAD_PREFILL_RUNTIME").is_some()
                && effective_layer_cap > 0
                && current_token_count > 1
                && matches!(begin_plan.role.as_str(), "head" | "single");
        let use_head_layer_runtime = effective_layer_cap > 0
            && matches!(begin_plan.role.as_str(), "head" | "single")
            && self
                .head_stack_runtime
                .as_ref()
                .is_some_and(|runtime| runtime.len() == effective_layer_cap);

        let layer_samples = if use_head_prefill_runtime {
            let key = (effective_layer_cap, current_token_count);
            if !self.head_prefill_stack_runtimes.contains_key(&key) {
                let runtime = GgmlTailStackRuntime::new_with_batch_count(
                    &self.runtime_plan,
                    &self.operator_plan,
                    &self.store,
                    effective_layer_cap,
                    current_token_count,
                )?;
                self.head_prefill_stack_runtimes.insert(key, runtime);
            }
            let mut runtime = self
                .head_prefill_stack_runtimes
                .remove(&key)
                .expect("head prefill stack runtime present");
            let result = self.run_begin_layers_with_batched_runtimes_profiled(
                &begin_plan,
                &mut states,
                prompt_aux.as_ref(),
                effective_layer_cap,
                position_offset,
                &mut attention_cache_by_layer,
                runtime.layers_mut(),
            );
            self.head_prefill_stack_runtimes.insert(key, runtime);
            result?
        } else if use_head_layer_runtime {
            let mut runtime = self.head_stack_runtime.take().expect("head stack runtime present");
            let result = self.run_begin_layers_with_runtimes_profiled(
                &begin_plan,
                &mut states,
                prompt_aux.as_ref(),
                effective_layer_cap,
                position_offset,
                &mut attention_cache_by_layer,
                runtime.layers_mut(),
            );
            self.head_stack_runtime = Some(runtime);
            result?
        } else {
            self.run_begin_layers_with_cache_profiled(
                &begin_plan,
                &mut states,
                prompt_aux.as_ref(),
                effective_layer_cap,
                position_offset,
                &mut attention_cache_by_layer,
            )?
        };

        let prefix_hashes =
            if current_token_count > 1 { Self::prefix_hashes(token_ids) } else { Vec::new() };
        let prompt_aux = match prompt_aux {
            Some(prompt_aux) => Some(prompt_aux.with_prefix_hashes(prefix_hashes)),
            None if !prefix_hashes.is_empty() => Some(
                BootstrapPromptAuxData::from_flat(0, 0, 0, Vec::new())
                    .with_prefix_hashes(prefix_hashes),
            ),
            None => None,
        };
        let payload_encode_start = Instant::now();
        let prompt_aux_bytes = Self::encode_prompt_aux(prompt_aux.as_ref())?;
        let hidden_bytes = Self::encode_hidden_states(&states);
        let bytes = encode_stage_tensor_bytes(&hidden_bytes, prompt_aux_bytes.as_deref());
        let payload_encode_us =
            payload_encode_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        self.decode_sessions.insert(
            request_id.to_string(),
            BootstrapDecodeSession {
                seq_len: position_offset as usize + current_token_count,
                attention_cache_by_layer,
            },
        );
        Ok((
            StageTensor {
                request_id: request_id.to_string(),
                kind: PayloadKind::HiddenState,
                stage_trace: vec![self.stage_id.clone()],
                hidden_dim: begin_plan.hidden_dim,
                bytes,
                prompt_text: None,
                max_tokens,
                continuation: None,
                transient: None,
                carry: None,
            },
            BootstrapHeadExecutionProfileSample {
                total_us: total_start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                ingress_us,
                payload_encode_us,
                layers: layer_samples,
            },
        ))
    }

    fn begin_token_ids_cap1(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.begin_token_ids_with_layer_cap(request_id, token_ids, max_tokens, 1)
    }

    fn continue_forward_with_layer_cap(
        &mut self,
        input: StageTensor,
        layer_cap: Option<usize>,
    ) -> Result<StageTensor> {
        let continue_plan = if let Some(layer_cap) = layer_cap {
            self.operator_plan.continue_forward_plan_with_layer_cap(&input, layer_cap)?
        } else {
            self.operator_plan.continue_forward_plan(&input)?
        };
        let prompt_aux_bytes = stage_tensor_byte_sections(&input.bytes)
            .and_then(|sections| sections.aux_bytes)
            .map(|bytes| bytes.to_vec());
        let prompt_aux = prompt_aux_bytes.as_deref().map(Self::decode_prompt_aux).transpose()?;
        let _prefix_hash_count =
            prompt_aux.as_ref().map(|aux| aux.prefix_hashes.len()).unwrap_or(0);
        let state_count = Self::decode_hidden_states_payload(&input.bytes, input.hidden_dim)?.len();
        let use_cached_decode = state_count == 1 && prompt_aux.is_some();
        let mut session = if use_cached_decode {
            self.decode_sessions.remove(&input.request_id)
        } else {
            self.decode_sessions.remove(&input.request_id);
            None
        };

        let mut states = Self::decode_hidden_states_payload(&input.bytes, input.hidden_dim)?;
        let position_offset = session.as_ref().map(|session| session.seq_len as u32).unwrap_or(0);
        let mut attention_cache_by_layer =
            session.take().map(|session| session.attention_cache_by_layer).unwrap_or_default();

        let use_tail_layer_runtime = continue_plan.layers.len()
            == self.tail_stack_runtime.as_ref().map_or(0, |rt| rt.len())
            && matches!(continue_plan.role.as_str(), "tail" | "single")
            && self.tail_stack_runtime.is_some();

        if use_tail_layer_runtime {
            let mut runtime = self.tail_stack_runtime.take().expect("tail stack runtime present");
            let result = self.run_layers_with_runtimes(
                &continue_plan.layers,
                continue_plan.input.hidden_dim,
                &mut states,
                prompt_aux.as_ref(),
                position_offset,
                &mut attention_cache_by_layer,
                runtime.layers_mut(),
            );
            self.tail_stack_runtime = Some(runtime);
            result?;
        } else {
            self.run_continue_layers_with_cache(
                &continue_plan,
                &mut states,
                prompt_aux.as_ref(),
                position_offset,
                &mut attention_cache_by_layer,
            )?;
        }

        self.decode_sessions.insert(
            input.request_id.clone(),
            BootstrapDecodeSession {
                seq_len: position_offset as usize + state_count,
                attention_cache_by_layer,
            },
        );

        let mut stage_trace = input.stage_trace;
        stage_trace.push(self.stage_id.clone());
        let bytes = encode_stage_tensor_bytes(
            &Self::encode_hidden_states(&states),
            prompt_aux_bytes.as_deref(),
        );
        Ok(StageTensor {
            request_id: input.request_id,
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim: continue_plan.input.hidden_dim,
            bytes,
            prompt_text: None,
            max_tokens: input.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn sample_tail_from_materialized_plan(&self, input: StageTensor) -> Result<StageSample> {
        let sample_plan = self.operator_plan.sample_tail_plan(&input)?;
        let mut state = Self::decode_hidden_state(&input.bytes, sample_plan.hidden_dim)?;

        if let Some(output_norm) = sample_plan.output_norm.as_ref() {
            let weight = self.read_f32_vector(&output_norm.name)?;
            real_math::rms_norm_inplace(&mut state, &weight, 1e-6);
        }

        let logits_raw = self.read_tensor_bytes(&sample_plan.logits.name)?;
        let logits = real_math::matmul_raw_top_k(
            sample_plan.logits.ggml_type,
            &logits_raw,
            &state,
            0,
            sample_plan.vocab_size,
            sample_plan.hidden_dim,
            1,
            Some(30.0),
        )
        .with_context(|| {
            format!("ggml bootstrap sample_tail matmul for `{}`", sample_plan.logits.name)
        })?;

        let token_count = input.max_tokens.unwrap_or(1).max(1) as usize;
        let selected = logits.argmax_idx as u32;
        let token_ids = vec![selected; token_count];
        let text = self.tokenizer.decode_ids(&token_ids);

        Ok(StageSample {
            request_id: input.request_id,
            model_id: self.model_id.clone(),
            text,
            token_ids,
            completion_tokens: token_count as u32,
        })
    }

    fn sample_tail_with_active_runtime(&mut self, input: StageTensor) -> Result<StageSample> {
        if let Some(runtime) = self.sample_graph_runtime.as_mut() {
            let sample_plan = self.operator_plan.sample_tail_plan(&input)?;
            let state = Self::decode_hidden_state(&input.bytes, sample_plan.hidden_dim)?;
            let token_count = input.max_tokens.unwrap_or(1).max(1) as usize;
            let selected = runtime.sample_argmax(&state)? as u32;
            let token_ids = vec![selected; token_count];
            let text = self.tokenizer.decode_ids(&token_ids);
            return Ok(StageSample {
                request_id: input.request_id,
                model_id: self.model_id.clone(),
                text,
                token_ids,
                completion_tokens: token_count as u32,
            });
        }
        self.sample_tail_from_materialized_plan(input)
    }
}

impl ReferenceCpuGgmlStageExecutor {
    fn new(init: &GgmlStageWorkerInitSpec) -> Result<Self> {
        let vocab_path = init.vocab_path.as_deref().context("worker init is missing vocab_path")?;
        let tokenizer = GemmaTokenizer::load(vocab_path, init.vocab_scores_path.as_deref())
            .context("load tokenizer for ggml stage executor")?;

        let mut backend = RealGemmaBackend::new(&init.index_path);
        if let Some(vocab_path) = init.vocab_path.as_deref() {
            backend.load_tokenizer(vocab_path, init.vocab_scores_path.as_deref())?;
        }
        backend.load_layout(StageLayout {
            model_id: init.model_id.clone(),
            stage_id: init.stage_id.clone(),
            start_layer: init.start_layer,
            end_layer: init.end_layer,
            is_head: matches!(init.role.as_str(), "head" | "single"),
            is_tail: matches!(init.role.as_str(), "tail" | "single"),
        })?;
        backend.set_debug_layer_cap(init.debug_layer_cap);

        Ok(Self {
            plan: GgmlStageExecutorPlan {
                requested: init.requested_executor,
                active: GgmlStageExecutorKind::ReferenceCpu,
                detail: "real_forward reference executor inside ggml worker".into(),
            },
            tokenizer,
            backend,
        })
    }
}

impl GgmlStageExecutor for ReferenceCpuGgmlStageExecutor {
    fn plan(&self) -> &GgmlStageExecutorPlan {
        &self.plan
    }

    fn tokenize_text(&mut self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode_with_bos(text))
    }

    fn tokenize_generation_prompt(&mut self, text: &str) -> Result<Vec<u32>> {
        let formatted = format_gemma_prompt(GemmaPromptMode::GemmaInstruct, text);
        Ok(self.tokenizer.encode_with_bos(&formatted))
    }

    fn decode_token_ids(&mut self, token_ids: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode_ids(token_ids))
    }

    fn eos_token_id(&mut self) -> Result<Option<u32>> {
        Ok(Some(self.tokenizer.eos_id()))
    }

    fn begin_prompt(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.backend.begin_prompt(request_id, prompt, max_tokens, 0)
    }

    fn begin_token_ids(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.backend.begin_token_ids(request_id, token_ids, max_tokens, 0)
    }

    fn continue_forward(&mut self, input: StageTensor) -> Result<StageTensor> {
        self.backend.continue_forward(input)
    }

    fn sample_tail(&mut self, input: StageTensor) -> Result<StageSample> {
        self.backend.sample_tail(input)
    }

    fn clear_decode_session(&mut self, request_id: &str) {
        self.backend.clear_decode_session(request_id);
    }

    fn profile_begin_token_ids_ingress(
        &mut self,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        iterations: u32,
    ) -> Result<GgmlHeadIngressProfile> {
        let iterations = iterations.max(1);
        let mut total_us = 0u128;
        let mut last_tensor = None;
        for iter_idx in 0..iterations {
            let request_id = format!("cpu-ref-ingress-bench-{iter_idx}");
            let start = Instant::now();
            let tensor = self.backend.begin_token_ids(&request_id, token_ids, max_tokens, 0)?;
            total_us += start.elapsed().as_micros();
            self.backend.clear_decode_session(&request_id);
            last_tensor = Some(tensor);
        }
        let tensor = last_tensor.expect("head ingress profile produced at least one tensor");
        let sections = stage_tensor_byte_sections(&tensor.bytes);
        let aux_bytes =
            sections.and_then(|parts| parts.aux_bytes.map(|bytes| bytes.len())).unwrap_or(0);
        Ok(GgmlHeadIngressProfile {
            executor: self.plan.active.as_str().to_string(),
            token_count: token_ids.len(),
            iterations,
            total_us: total_us.min(u128::from(u64::MAX)) as u64,
            embed_token_gather_us: None,
            ple_token_gather_us: None,
            ple_model_proj_us: None,
            ple_normalize_combine_us: None,
            prompt_aux_encode_us: None,
            hidden_encode_us: None,
            payload_frame_us: None,
            other_us: None,
            hidden_state_bytes: tensor.hidden_state_len(),
            aux_bytes,
            payload_bytes: tensor.bytes.len(),
        })
    }

    fn profile_begin_token_ids_execution(
        &mut self,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        iterations: u32,
    ) -> Result<GgmlHeadExecutionProfile> {
        let iterations = iterations.max(1);
        let mut total_us = 0u128;
        let mut last_tensor = None;
        for iter_idx in 0..iterations {
            let request_id = format!("cpu-ref-exec-bench-{iter_idx}");
            let start = Instant::now();
            let tensor = self.backend.begin_token_ids(&request_id, token_ids, max_tokens, 0)?;
            total_us += start.elapsed().as_micros();
            self.backend.clear_decode_session(&request_id);
            last_tensor = Some(tensor);
        }
        let tensor = last_tensor.expect("head execution profile produced at least one tensor");
        let sections = stage_tensor_byte_sections(&tensor.bytes);
        let aux_bytes =
            sections.and_then(|parts| parts.aux_bytes.map(|bytes| bytes.len())).unwrap_or(0);
        Ok(GgmlHeadExecutionProfile {
            executor: self.plan.active.as_str().to_string(),
            token_count: token_ids.len(),
            iterations,
            effective_layer_cap: 0,
            ingress_us: 0,
            payload_encode_us: 0,
            total_us: total_us.min(u128::from(u64::MAX)) as u64,
            layers: Vec::new(),
            hidden_state_bytes: tensor.hidden_state_len(),
            aux_bytes,
            payload_bytes: tensor.bytes.len(),
        })
    }
}

impl GgmlStageExecutor for BootstrapGgmlStageExecutor {
    fn plan(&self) -> &GgmlStageExecutorPlan {
        &self.plan
    }

    fn tokenize_text(&mut self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode_with_bos(text))
    }

    fn tokenize_generation_prompt(&mut self, text: &str) -> Result<Vec<u32>> {
        let formatted = format_gemma_prompt(GemmaPromptMode::GemmaInstruct, text);
        Ok(self.tokenizer.encode_with_bos(&formatted))
    }

    fn decode_token_ids(&mut self, token_ids: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode_ids(token_ids))
    }

    fn eos_token_id(&mut self) -> Result<Option<u32>> {
        Ok(Some(self.tokenizer.eos_id()))
    }

    fn begin_prompt(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        let token_ids = self.tokenize_generation_prompt(prompt)?;
        if self.debug_layer_cap == Some(0) {
            return self.begin_token_ids_cap0(request_id, &token_ids, max_tokens);
        }
        if self.debug_layer_cap == Some(1) {
            return self.begin_token_ids_cap1(request_id, &token_ids, max_tokens);
        }
        if let Some(layer_cap) = self.debug_layer_cap {
            return self
                .begin_token_ids_with_layer_cap(request_id, &token_ids, max_tokens, layer_cap);
        }
        if self.head_stack_runtime.is_some()
            && matches!(self.operator_plan.role.as_str(), "head" | "single")
        {
            return self.begin_token_ids_with_layer_cap(
                request_id,
                &token_ids,
                max_tokens,
                self.operator_plan.layers.len(),
            );
        }
        let token_count = token_ids.len();
        let begin_plan = self
            .operator_plan
            .begin_token_ids_plan(token_count, max_tokens)
            .map(|plan| plan.summary_label())
            .unwrap_or_else(|err| format!("unavailable ({err})"));
        bail!(
            "ggml-worker bootstrap executor for stage {} does not implement `begin_token_ids` yet (runtime={}; bindings={}; ops={}; begin_plan={})",
            self.stage_id,
            self.runtime_summary,
            self.binding_summary,
            format!(
                "{}; recipe={}; materialized={}",
                self.operator_plan.summary_label(),
                self.execution_recipe.summary_label(),
                self.materialized_recipe.summary_label(),
            ),
            begin_plan
        )
    }

    fn begin_token_ids(
        &mut self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        if self.debug_layer_cap == Some(0) {
            return self.begin_token_ids_cap0(request_id, token_ids, max_tokens);
        }
        if self.debug_layer_cap == Some(1) {
            return self.begin_token_ids_cap1(request_id, token_ids, max_tokens);
        }
        if let Some(layer_cap) = self.debug_layer_cap {
            return self
                .begin_token_ids_with_layer_cap(request_id, token_ids, max_tokens, layer_cap);
        }
        if self.head_stack_runtime.is_some()
            && matches!(self.operator_plan.role.as_str(), "head" | "single")
        {
            return self.begin_token_ids_with_layer_cap(
                request_id,
                token_ids,
                max_tokens,
                self.operator_plan.layers.len(),
            );
        }
        let begin_plan = self
            .operator_plan
            .begin_token_ids_plan(token_ids.len(), max_tokens)
            .map(|plan| plan.summary_label())
            .unwrap_or_else(|err| format!("unavailable ({err})"));
        bail!(
            "ggml-worker bootstrap executor for stage {} does not implement `begin_token_ids` yet (runtime={}; bindings={}; ops={}; begin_plan={})",
            self.stage_id,
            self.runtime_summary,
            self.binding_summary,
            format!(
                "{}; recipe={}; materialized={}",
                self.operator_plan.summary_label(),
                self.execution_recipe.summary_label(),
                self.materialized_recipe.summary_label(),
            ),
            begin_plan
        )
    }

    fn continue_forward(&mut self, input: StageTensor) -> Result<StageTensor> {
        self.continue_forward_with_layer_cap(input, self.debug_layer_cap)
    }

    fn sample_tail(&mut self, input: StageTensor) -> Result<StageSample> {
        self.sample_tail_with_active_runtime(input)
    }

    fn clear_decode_session(&mut self, request_id: &str) {
        self.decode_sessions.remove(request_id);
    }

    fn profile_begin_token_ids_ingress(
        &mut self,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        iterations: u32,
    ) -> Result<GgmlHeadIngressProfile> {
        let iterations = iterations.max(1);
        let mut totals = BootstrapHeadIngressProfileTotals::default();
        let mut last_tensor = None;
        for iter_idx in 0..iterations {
            let request_id = format!("ggml-ingress-bench-{iter_idx}");
            let (tensor, sample) =
                self.profiled_begin_token_ids_cap0(&request_id, token_ids, max_tokens)?;
            totals.record(sample);
            last_tensor = Some(tensor);
        }
        Ok(totals.finalize(
            self.plan.active.as_str(),
            token_ids.len(),
            iterations,
            &last_tensor.expect("head ingress profile produced at least one tensor"),
        ))
    }

    fn profile_begin_token_ids_execution(
        &mut self,
        token_ids: &[u32],
        max_tokens: Option<u32>,
        iterations: u32,
    ) -> Result<GgmlHeadExecutionProfile> {
        let iterations = iterations.max(1);
        let effective_layer_cap = self
            .debug_layer_cap
            .unwrap_or(self.operator_plan.layers.len())
            .min(self.operator_plan.layers.len());
        let mut totals = BootstrapHeadExecutionProfileTotals::default();
        let mut last_tensor = None;
        for iter_idx in 0..iterations {
            let request_id = format!("ggml-exec-bench-{iter_idx}");
            let (tensor, sample) = self.profiled_begin_token_ids_with_layer_cap(
                &request_id,
                token_ids,
                max_tokens,
                effective_layer_cap,
            )?;
            self.clear_decode_session(&request_id);
            totals.record(&sample);
            last_tensor = Some(tensor);
        }
        Ok(totals.finalize(
            self.plan.active.as_str(),
            token_ids.len(),
            iterations,
            effective_layer_cap,
            &last_tensor.expect("head execution profile produced at least one tensor"),
        ))
    }
}

pub fn build_ggml_stage_executor(
    init: &GgmlStageWorkerInitSpec,
) -> Result<Box<dyn GgmlStageExecutor>> {
    match init.requested_executor {
        GgmlStageExecutorKind::ReferenceCpu => {
            Ok(Box::new(ReferenceCpuGgmlStageExecutor::new(init)?))
        }
        GgmlStageExecutorKind::Ggml => Ok(Box::new(BootstrapGgmlStageExecutor::new(init)?)),
    }
}

pub fn debug_proportional_shared_kv_layer(
    init: &GgmlStageWorkerInitSpec,
    reference_input: &StageTensor,
    candidate_input: &StageTensor,
    layer_index: u32,
) -> Result<GgmlProportionalSharedKvLayerDebug> {
    if reference_input.hidden_dim != candidate_input.hidden_dim {
        bail!(
            "ggml proportional shared-KV debug hidden_dim mismatch: reference={} candidate={}",
            reference_input.hidden_dim,
            candidate_input.hidden_dim
        );
    }

    let mut exact_init = init.clone();
    exact_init.runtime.target = StageAccelerationTarget::Cpu;
    let exact = BootstrapGgmlStageExecutor::new(&exact_init)?;
    if !matches!(exact.operator_plan.role.as_str(), "tail" | "single") {
        bail!(
            "ggml proportional shared-KV debug requires a tail/single stage, got role `{}`",
            exact.operator_plan.role
        );
    }

    let mut reference_states = BootstrapGgmlStageExecutor::decode_hidden_states_payload(
        &reference_input.bytes,
        reference_input.hidden_dim,
    )?;
    let mut candidate_states = BootstrapGgmlStageExecutor::decode_hidden_states_payload(
        &candidate_input.bytes,
        candidate_input.hidden_dim,
    )?;
    if reference_states.len() != candidate_states.len() {
        bail!(
            "ggml proportional shared-KV debug token count mismatch: reference={} candidate={}",
            reference_states.len(),
            candidate_states.len()
        );
    }

    let reference_prompt_aux = stage_tensor_byte_sections(&reference_input.bytes)
        .and_then(|sections| sections.aux_bytes)
        .map(BootstrapGgmlStageExecutor::decode_prompt_aux)
        .transpose()?;
    let candidate_prompt_aux = stage_tensor_byte_sections(&candidate_input.bytes)
        .and_then(|sections| sections.aux_bytes)
        .map(BootstrapGgmlStageExecutor::decode_prompt_aux)
        .transpose()?;

    let target_index = exact
        .operator_plan
        .layers
        .iter()
        .position(|layer| layer.layer_index == layer_index)
        .ok_or_else(|| {
            anyhow::anyhow!("ggml proportional shared-KV debug missing layer {}", layer_index)
        })?;
    let target_layer = &exact.operator_plan.layers[target_index];
    let target_attn = exact.layer_attention_spec(target_layer, reference_input.hidden_dim)?;
    let shared_kv_source_layer = target_attn.shared_kv_source_layer.ok_or_else(|| {
        anyhow::anyhow!(
            "ggml proportional shared-KV debug target layer {} is not a shared-KV layer",
            layer_index
        )
    })?;
    if !target_attn.proportional_rope {
        bail!(
            "ggml proportional shared-KV debug target layer {} is not proportional-rope",
            layer_index
        );
    }

    let mut reference_caches = HashMap::new();
    let mut candidate_caches = HashMap::new();
    for layer in exact.operator_plan.layers.iter().take(target_index) {
        let attn = exact.layer_attention_spec(layer, reference_input.hidden_dim)?;
        let reference_shared =
            attn.shared_kv_source_layer.and_then(|index| reference_caches.get(&index).cloned());
        let reference_existing = if reference_shared.is_none() {
            reference_caches.remove(&layer.layer_index)
        } else {
            None
        };
        if let Some(cache) = exact.run_hidden_layer(
            layer,
            reference_input.hidden_dim,
            &mut reference_states,
            reference_prompt_aux.as_ref(),
            reference_existing,
            reference_shared.as_ref(),
            0,
        )? {
            reference_caches.insert(layer.layer_index, cache);
        }

        let candidate_shared =
            attn.shared_kv_source_layer.and_then(|index| candidate_caches.get(&index).cloned());
        let candidate_existing = if candidate_shared.is_none() {
            candidate_caches.remove(&layer.layer_index)
        } else {
            None
        };
        if let Some(cache) = exact.run_hidden_layer(
            layer,
            candidate_input.hidden_dim,
            &mut candidate_states,
            candidate_prompt_aux.as_ref(),
            candidate_existing,
            candidate_shared.as_ref(),
            0,
        )? {
            candidate_caches.insert(layer.layer_index, cache);
        }
    }

    let input_max_abs =
        BootstrapGgmlStageExecutor::max_abs_diff_many(&reference_states, &candidate_states)?;
    let candidate_states_before_layer = candidate_states.clone();
    let reference_shared_cache = reference_caches.get(&shared_kv_source_layer).ok_or_else(|| {
        anyhow::anyhow!(
            "ggml proportional shared-KV debug missing reference shared cache for source layer {}",
            shared_kv_source_layer
        )
    })?;
    let candidate_shared_cache = candidate_caches.get(&shared_kv_source_layer).ok_or_else(|| {
        anyhow::anyhow!(
            "ggml proportional shared-KV debug missing candidate shared cache for source layer {}",
            shared_kv_source_layer
        )
    })?;

    let reference_trace = exact.debug_proportional_shared_kv_layer_exact(
        target_layer,
        reference_input.hidden_dim,
        &mut reference_states,
        reference_prompt_aux.as_ref(),
        reference_shared_cache,
        0,
    )?;
    let candidate_trace = exact.debug_proportional_shared_kv_layer_exact(
        target_layer,
        candidate_input.hidden_dim,
        &mut candidate_states,
        candidate_prompt_aux.as_ref(),
        candidate_shared_cache,
        0,
    )?;

    let mut rope_runtime = GgmlRopeGraphRuntime::new(
        &init.runtime,
        GgmlHeadLayerGraphSpec {
            n_heads: target_attn.n_heads,
            n_kv_heads: target_attn.n_kv_heads,
            head_dim: target_attn.head_dim,
            rope_base_theta: target_attn.rope_base_theta,
            rope_rotary_dim: target_attn.rope_rotary_dim,
            proportional_rope: target_attn.proportional_rope,
            sliding_window: target_attn.sliding_window,
            uses_shared_kv: true,
        },
        candidate_trace.q_pre_rope.len(),
        "proportional-shared-kv-debug",
    )?;
    let rope_runtime_label = rope_runtime.summary_label();
    let candidate_ggml_q_rope =
        rope_runtime.apply_with_position_offset(&candidate_trace.q_pre_rope, 0)?;

    let prompt_aux_for_layer = candidate_prompt_aux
        .as_ref()
        .map(|aux| {
            let ple_idx = target_layer.layer_index as usize;
            (0..candidate_states_before_layer.len())
                .map(|token_index| {
                    aux.token_layer(token_index, ple_idx)
                        .map(|slice| slice.to_vec())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "ggml proportional shared-KV debug prompt-aux missing layer {} token {}",
                                target_layer.layer_index,
                                token_index
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut full_runtime = GgmlFullHeadPrefillLayerRuntime::new(
        &init.runtime,
        target_layer,
        candidate_input.hidden_dim,
        GgmlHeadLayerGraphSpec {
            n_heads: target_attn.n_heads,
            n_kv_heads: target_attn.n_kv_heads,
            head_dim: target_attn.head_dim,
            rope_base_theta: target_attn.rope_base_theta,
            rope_rotary_dim: target_attn.rope_rotary_dim,
            proportional_rope: target_attn.proportional_rope,
            sliding_window: target_attn.sliding_window,
            uses_shared_kv: true,
        },
        candidate_states_before_layer.len(),
        &exact.store,
        "proportional-shared-kv-full-debug",
    )?;
    let full_runtime_label = full_runtime.summary_label();
    let full_runtime_result = full_runtime.run(
        &candidate_states_before_layer,
        prompt_aux_for_layer.as_deref(),
        Some((candidate_shared_cache.0.as_slice(), candidate_shared_cache.1.as_slice())),
    )?;

    Ok(GgmlProportionalSharedKvLayerDebug {
        layer_index,
        shared_kv_source_layer,
        token_count: candidate_trace.q_pre_rope.len(),
        position_offset: 0,
        rope_runtime: rope_runtime_label,
        full_runtime: full_runtime_label,
        input_max_abs,
        q_rope_ref_candidate_max_abs: BootstrapGgmlStageExecutor::max_abs_diff_many(
            &reference_trace.q_rope,
            &candidate_trace.q_rope,
        )?,
        q_input_ggml_max_abs: BootstrapGgmlStageExecutor::max_abs_diff_many(
            &candidate_trace.q_pre_rope,
            &candidate_ggml_q_rope,
        )?,
        q_rope_ggml_candidate_max_abs: BootstrapGgmlStageExecutor::max_abs_diff_many(
            &candidate_trace.q_rope,
            &candidate_ggml_q_rope,
        )?,
        attn_out_max_abs: BootstrapGgmlStageExecutor::max_abs_diff_many(
            &reference_trace.attn_out,
            &candidate_trace.attn_out,
        )?,
        layer_out_max_abs: BootstrapGgmlStageExecutor::max_abs_diff_many(
            &reference_trace.layer_out,
            &candidate_trace.layer_out,
        )?,
        layer_out_ggml_candidate_max_abs: BootstrapGgmlStageExecutor::max_abs_diff_many(
            &candidate_trace.layer_out,
            &full_runtime_result.hidden_states,
        )?,
    })
}

pub fn debug_full_graph_head_layers(
    init: &GgmlStageWorkerInitSpec,
    token_ids: &[u32],
    max_tokens: Option<u32>,
) -> Result<Vec<GgmlFullGraphHeadLayerDebug>> {
    let mut ingress_init = init.clone();
    ingress_init.runtime.target = StageAccelerationTarget::Cpu;
    ingress_init.requested_executor = GgmlStageExecutorKind::Ggml;
    let mut ingress = BootstrapGgmlStageExecutor::new(&ingress_init)?;
    if !matches!(ingress.operator_plan.role.as_str(), "head" | "single") {
        bail!(
            "ggml full-graph head debug requires a head/single stage, got role `{}`",
            ingress.operator_plan.role
        );
    }

    let mut reference_init = init.clone();
    reference_init.requested_executor = GgmlStageExecutorKind::ReferenceCpu;
    let mut reference = ReferenceCpuGgmlStageExecutor::new(&reference_init)?;

    let (_begin_plan, ingress_states, prompt_aux, _) =
        ingress.prepare_begin_token_ids(token_ids, max_tokens)?;
    let hidden_dim = ingress_states.first().map(|state| state.len()).unwrap_or(0);
    let batch_count = ingress_states.len();
    let mut exact_input_states = ingress_states.clone();
    let mut cumulative_states = ingress_states;

    let mut runtimes = HashMap::<(u32, usize, bool), GgmlFullHeadPrefillLayerRuntime>::new();
    let mut debug_rows = Vec::with_capacity(ingress.operator_plan.layers.len());

    for (local_layer_index, layer) in ingress.operator_plan.layers.iter().enumerate() {
        let layer_cap = local_layer_index + 1;
        reference.backend.set_debug_layer_cap(Some(layer_cap));
        let request_id = format!("ggml-full-graph-head-debug-{layer_cap}");
        let exact_tensor =
            reference.backend.begin_token_ids(&request_id, token_ids, max_tokens, 0)?;
        reference.backend.clear_decode_session(&request_id);
        let exact_output_states = BootstrapGgmlStageExecutor::decode_hidden_states_payload(
            &exact_tensor.bytes,
            exact_tensor.hidden_dim,
        )?;

        let attn = ingress.layer_attention_spec(layer, hidden_dim)?;
        if attn.shared_kv_source_layer.is_some() {
            bail!(
                "ggml full-graph head debug does not support shared-KV head layer {}",
                layer.layer_index
            );
        }
        let runtime_key = (layer.layer_index, batch_count, false);
        if !runtimes.contains_key(&runtime_key) {
            let runtime = GgmlFullHeadPrefillLayerRuntime::new(
                &init.runtime,
                layer,
                hidden_dim,
                GgmlHeadLayerGraphSpec {
                    n_heads: attn.n_heads,
                    n_kv_heads: attn.n_kv_heads,
                    head_dim: attn.head_dim,
                    rope_base_theta: attn.rope_base_theta,
                    rope_rotary_dim: attn.rope_rotary_dim,
                    proportional_rope: attn.proportional_rope,
                    sliding_window: attn.sliding_window,
                    uses_shared_kv: false,
                },
                batch_count,
                &ingress.store,
                "full-graph-head-layer-debug",
            )?;
            runtimes.insert(runtime_key, runtime);
        }
        let runtime =
            runtimes.get_mut(&runtime_key).expect("inserted head full-graph runtime present");

        let prompt_aux_for_layer = prompt_aux
            .as_ref()
            .map(|aux| {
                let ple_idx = layer.layer_index as usize;
                (0..batch_count)
                    .map(|token_index| {
                        aux.token_layer(token_index, ple_idx)
                            .map(|slice| slice.to_vec())
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "ggml full-graph head debug prompt-aux missing layer {} token {}",
                                    layer.layer_index,
                                    token_index
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        let local_result =
            runtime.run(&exact_input_states, prompt_aux_for_layer.as_deref(), None)?;
        let local_output_max_abs = BootstrapGgmlStageExecutor::max_abs_diff_many(
            &exact_output_states,
            &local_result.hidden_states,
        )?;

        let cumulative_result =
            runtime.run(&cumulative_states, prompt_aux_for_layer.as_deref(), None)?;
        cumulative_states = cumulative_result.hidden_states;
        let cumulative_output_max_abs = BootstrapGgmlStageExecutor::max_abs_diff_many(
            &exact_output_states,
            &cumulative_states,
        )?;

        debug_rows.push(GgmlFullGraphHeadLayerDebug {
            layer_index: layer.layer_index,
            runtime: runtime.summary_label(),
            local_output_max_abs,
            cumulative_output_max_abs,
        });
        exact_input_states = exact_output_states;
    }

    Ok(debug_rows)
}

#[cfg(test)]
fn executor_kind_from_value(value: Option<&str>) -> GgmlStageExecutorKind {
    match value {
        Some("ggml") => GgmlStageExecutorKind::Ggml,
        _ => GgmlStageExecutorKind::ReferenceCpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_executor_selection_uses_reference_cpu() {
        assert_eq!(executor_kind_from_value(None), GgmlStageExecutorKind::ReferenceCpu);
    }

    #[test]
    fn ggml_executor_selection_can_be_requested_explicitly() {
        assert_eq!(executor_kind_from_value(Some("ggml")), GgmlStageExecutorKind::Ggml);
    }
}
