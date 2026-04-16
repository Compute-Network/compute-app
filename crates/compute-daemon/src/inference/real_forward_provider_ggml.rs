use anyhow::Result;
use stage_forward_lab::{PackedStageIndex, StageSample, StageTensor, quants};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

use crate::inference::engine::ShardConfig;
use crate::inference::ggml_runtime::{GgmlRuntimePlan, detect_ggml_runtime_plan};
use crate::inference::ggml_stage_executor::GgmlStageExecutorKind;
use crate::inference::ggml_stage_manifest::GgmlStageBindingManifest;
use crate::inference::ggml_stage_plan::GgmlStageOperatorPlan;
use crate::inference::ggml_stage_worker::{
    GgmlStageWorkerContract, GgmlStageWorkerHostLaunchSpec, GgmlStageWorkerInitSpec,
    GgmlStageWorkerRequest, GgmlStageWorkerResponse, GgmlStageWorkerSession,
    run_stage_worker_session_request, spawn_in_process_metadata_session,
    spawn_in_process_stage_worker_session,
};
use crate::inference::real_forward_artifact::RealForwardStageLoadSpec;
use crate::inference::real_forward_provider::{
    RealForwardProviderCapabilities, RealForwardStageProvider,
};
use crate::inference::stage_acceleration::StageAccelerationTarget;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlStageExecutionPlan {
    pub target: StageAccelerationTarget,
    pub model_name: String,
    pub architecture: String,
    pub stage_index: u32,
    pub role: String,
    pub stage_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub ggml_type_counts: BTreeMap<u32, usize>,
}

impl GgmlStageExecutionPlan {
    pub fn from_load_spec(
        target: StageAccelerationTarget,
        load_spec: &RealForwardStageLoadSpec,
    ) -> Result<Self> {
        let index: PackedStageIndex = serde_json::from_slice(&fs::read(&load_spec.index_path)?)?;
        let mut ggml_type_counts = BTreeMap::<u32, usize>::new();
        for tensor in &index.tensors {
            *ggml_type_counts.entry(tensor.ggml_type).or_default() += 1;
        }
        Ok(Self {
            target,
            model_name: index.model_name,
            architecture: index.architecture,
            stage_index: index.stage_index,
            role: index.role,
            stage_id: load_spec.layout.stage_id.clone(),
            start_layer: load_spec.layout.start_layer,
            end_layer: load_spec.layout.end_layer,
            tensor_count: index.tensor_count,
            total_bytes: index.total_bytes,
            ggml_type_counts,
        })
    }

    pub fn summary_label(&self) -> String {
        let types = self
            .ggml_type_counts
            .iter()
            .map(|(ggml_type, count)| format!("{}x{}", count, quants::ggml_type_name(*ggml_type)))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "target={} arch={} stage={} role={} layers={}-{} tensors={} bytes={} types=[{}]",
            self.target.as_str(),
            self.architecture,
            self.stage_index,
            self.role,
            self.start_layer,
            self.end_layer,
            self.tensor_count,
            self.total_bytes,
            types
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlProviderCompatibility {
    pub artifact_kind: String,
    pub stock_runtime_compatible: bool,
    pub reasons: Vec<String>,
}

impl GgmlProviderCompatibility {
    pub fn from_load_spec(load_spec: &RealForwardStageLoadSpec) -> Self {
        let mut reasons = vec![
            "current real_forward stage artifact is a packed stage directory with a .index.json manifest, not a GGUF model file consumable by stock llama-server".into(),
            "stage-local execution needs a split-boundary forward endpoint that can stop at the stage boundary and emit hidden states, which stock llama-server does not expose".into(),
            "stage-local decode requires per-stage KV/session ownership across continuation steps, which needs a custom stage worker contract".into(),
        ];
        if !load_spec.layout.is_head {
            reasons.push(
                "downstream stages require hidden-state ingress at the split boundary, which stock llama-server does not accept".into(),
            );
        }
        if !load_spec.layout.is_tail {
            reasons.push(
                "non-tail stages must forward split-boundary activations instead of sampling tokens, which stock llama-server cannot do directly".into(),
            );
        }
        Self { artifact_kind: "packed-stage-dir".into(), stock_runtime_compatible: false, reasons }
    }

    pub fn summary_label(&self) -> String {
        format!(
            "artifact={} stock-llama-server-compatible={} reasons=[{}]",
            self.artifact_kind,
            self.stock_runtime_compatible,
            self.reasons.join(" | ")
        )
    }
}

pub struct GgmlRealForwardProvider {
    requested_target: StageAccelerationTarget,
    metadata_executor: GgmlStageExecutorKind,
    execution_executor: GgmlStageExecutorKind,
    sample_executor: GgmlStageExecutorKind,
    last_load_spec: Option<RealForwardStageLoadSpec>,
    last_plan: Option<GgmlStageExecutionPlan>,
    last_runtime_plan: Option<GgmlRuntimePlan>,
    last_compatibility: Option<GgmlProviderCompatibility>,
    last_binding_manifest: Option<GgmlStageBindingManifest>,
    last_operator_plan: Option<GgmlStageOperatorPlan>,
    last_worker_contract: Option<GgmlStageWorkerContract>,
    last_metadata_worker_init: Option<GgmlStageWorkerInitSpec>,
    last_metadata_worker_launch: Option<GgmlStageWorkerHostLaunchSpec>,
    last_execution_worker_init: Option<GgmlStageWorkerInitSpec>,
    last_execution_worker_launch: Option<GgmlStageWorkerHostLaunchSpec>,
    last_sample_worker_init: Option<GgmlStageWorkerInitSpec>,
    last_sample_worker_launch: Option<GgmlStageWorkerHostLaunchSpec>,
    metadata_worker_session: Mutex<Option<GgmlStageWorkerSession>>,
    execution_worker_session: Mutex<Option<GgmlStageWorkerSession>>,
    sample_worker_session: Mutex<Option<GgmlStageWorkerSession>>,
}

impl GgmlRealForwardProvider {
    pub fn new(
        requested_target: StageAccelerationTarget,
        metadata_executor: GgmlStageExecutorKind,
        execution_executor: GgmlStageExecutorKind,
        sample_executor: GgmlStageExecutorKind,
    ) -> Self {
        Self {
            requested_target,
            metadata_executor,
            execution_executor,
            sample_executor,
            last_load_spec: None,
            last_plan: None,
            last_runtime_plan: None,
            last_compatibility: None,
            last_binding_manifest: None,
            last_operator_plan: None,
            last_worker_contract: None,
            last_metadata_worker_init: None,
            last_metadata_worker_launch: None,
            last_execution_worker_init: None,
            last_execution_worker_launch: None,
            last_sample_worker_init: None,
            last_sample_worker_launch: None,
            metadata_worker_session: Mutex::new(None),
            execution_worker_session: Mutex::new(None),
            sample_worker_session: Mutex::new(None),
        }
    }

    fn error(&self) -> anyhow::Error {
        let plan = self
            .last_plan
            .as_ref()
            .map(GgmlStageExecutionPlan::summary_label)
            .unwrap_or_else(|| format!("target={}", self.requested_target.as_str()));
        let runtime = self
            .last_runtime_plan
            .as_ref()
            .map(GgmlRuntimePlan::summary_label)
            .unwrap_or_else(|| "runtime=unresolved".to_string());
        let compatibility = self
            .last_compatibility
            .as_ref()
            .map(GgmlProviderCompatibility::summary_label)
            .unwrap_or_else(|| "compatibility=unresolved".to_string());
        let bindings = self
            .last_binding_manifest
            .as_ref()
            .map(GgmlStageBindingManifest::summary_label)
            .unwrap_or_else(|| "bindings=unresolved".to_string());
        let operator_plan = self
            .last_operator_plan
            .as_ref()
            .map(GgmlStageOperatorPlan::summary_label)
            .unwrap_or_else(|| "ops=unresolved".to_string());
        let worker = self
            .last_worker_contract
            .as_ref()
            .map(GgmlStageWorkerContract::summary_label)
            .unwrap_or_else(|| "worker=unresolved".to_string());
        let metadata_init = self
            .last_metadata_worker_init
            .as_ref()
            .map(GgmlStageWorkerInitSpec::summary_label)
            .unwrap_or_else(|| "metadata_init=unresolved".to_string());
        let metadata_launch = self
            .last_metadata_worker_launch
            .as_ref()
            .map(GgmlStageWorkerHostLaunchSpec::summary_label)
            .unwrap_or_else(|| "metadata_launch=unresolved".to_string());
        let execution_init = self
            .last_execution_worker_init
            .as_ref()
            .map(GgmlStageWorkerInitSpec::summary_label)
            .unwrap_or_else(|| "execution_init=unresolved".to_string());
        let execution_launch = self
            .last_execution_worker_launch
            .as_ref()
            .map(GgmlStageWorkerHostLaunchSpec::summary_label)
            .unwrap_or_else(|| "execution_launch=unresolved".to_string());
        let sample_init = self
            .last_sample_worker_init
            .as_ref()
            .map(GgmlStageWorkerInitSpec::summary_label)
            .unwrap_or_else(|| "sample_init=unresolved".to_string());
        let sample_launch = self
            .last_sample_worker_launch
            .as_ref()
            .map(GgmlStageWorkerHostLaunchSpec::summary_label)
            .unwrap_or_else(|| "sample_launch=unresolved".to_string());
        let execution_route = self
            .last_execution_worker_init
            .as_ref()
            .map(|init| match (init.role.as_str(), init.requested_executor) {
                ("head" | "single", GgmlStageExecutorKind::Ggml) => {
                    "head-stage token ingress / hidden-state egress routes through `ggml-worker`"
                        .to_string()
                }
                ("tail", GgmlStageExecutorKind::Ggml) => {
                    "downstream hidden-state forward routes through `ggml-worker`".to_string()
                }
                (_, GgmlStageExecutorKind::ReferenceCpu) => {
                    "this stage's hidden-state execution currently routes through `cpu-ref-worker`"
                        .to_string()
                }
                _ => "this stage's hidden-state execution route is unresolved".to_string(),
            })
            .unwrap_or_else(|| "execution route unresolved".to_string());
        anyhow::anyhow!(
            "Requested `ggml` real_forward provider; bootstrap worker path is available, `ggml-worker` currently covers metadata, head `begin_token_ids`, downstream `continue_forward`, and tail `sample_tail`, and the hidden-state execution route is stage-aware under the worker contract ({execution_route}; {plan}; runtime={runtime}; compatibility={compatibility}; bindings={bindings}; operator_plan={operator_plan}; worker={worker}; metadata={metadata_init}; metadata_launch={metadata_launch}; execution={execution_init}; execution_launch={execution_launch}; sample={sample_init}; sample_launch={sample_launch})"
        )
    }

    fn effective_execution_executor(
        &self,
        load_spec: &RealForwardStageLoadSpec,
    ) -> GgmlStageExecutorKind {
        match self.execution_executor {
            GgmlStageExecutorKind::Ggml
                if load_spec.layout.is_head && !load_spec.layout.is_tail =>
            {
                GgmlStageExecutorKind::Ggml
            }
            GgmlStageExecutorKind::Ggml => GgmlStageExecutorKind::ReferenceCpu,
            GgmlStageExecutorKind::ReferenceCpu => GgmlStageExecutorKind::ReferenceCpu,
        }
    }

    fn execution_debug_layer_cap(
        load_spec: &RealForwardStageLoadSpec,
        executor: GgmlStageExecutorKind,
    ) -> Option<usize> {
        if executor != GgmlStageExecutorKind::Ggml || !load_spec.layout.is_head {
            return None;
        }
        Some((load_spec.layout.end_layer - load_spec.layout.start_layer + 1) as usize)
    }

    fn metadata_worker_launch(&self) -> Result<&GgmlStageWorkerHostLaunchSpec> {
        self.last_metadata_worker_launch.as_ref().ok_or_else(|| self.error())
    }

    fn metadata_worker_init(&self) -> Result<&GgmlStageWorkerInitSpec> {
        self.last_metadata_worker_init.as_ref().ok_or_else(|| self.error())
    }

    fn execution_worker_launch(&self) -> Result<&GgmlStageWorkerHostLaunchSpec> {
        self.last_execution_worker_launch.as_ref().ok_or_else(|| self.error())
    }

    fn execution_worker_init(&self) -> Result<&GgmlStageWorkerInitSpec> {
        self.last_execution_worker_init.as_ref().ok_or_else(|| self.error())
    }

    fn sample_worker_launch(&self) -> Result<&GgmlStageWorkerHostLaunchSpec> {
        self.last_sample_worker_launch.as_ref().ok_or_else(|| self.error())
    }

    fn sample_worker_init(&self) -> Result<&GgmlStageWorkerInitSpec> {
        self.last_sample_worker_init.as_ref().ok_or_else(|| self.error())
    }

    fn shutdown_worker_session(session: &Mutex<Option<GgmlStageWorkerSession>>) {
        let mut session = session.lock().expect("ggml worker session mutex poisoned");
        if let Some(mut active) = session.take() {
            active.shutdown();
        }
    }

    fn maybe_prewarm_worker_session(
        session: &Mutex<Option<GgmlStageWorkerSession>>,
        launch: Option<&GgmlStageWorkerHostLaunchSpec>,
    ) -> Result<()> {
        let _ = session;
        let _ = launch;
        Ok(())
    }

    fn maybe_prewarm_head_execution(&self) -> Result<()> {
        Ok(())
    }

    fn metadata_session_request(
        &self,
        request: &GgmlStageWorkerRequest,
    ) -> Result<GgmlStageWorkerResponse> {
        let init = self.metadata_worker_init()?.clone();
        let mut session = self
            .metadata_worker_session
            .lock()
            .expect("ggml metadata worker session mutex poisoned");
        if session.is_none() {
            *session = Some(spawn_in_process_metadata_session(&init)?);
        }
        let response = run_stage_worker_session_request(
            session.as_mut().expect("metadata worker session initialized"),
            request,
        );
        if response.is_ok() {
            return response;
        }

        if let Some(mut active) = session.take() {
            active.shutdown();
        }
        *session = Some(spawn_in_process_metadata_session(&init)?);
        run_stage_worker_session_request(
            session.as_mut().expect("metadata worker session reinitialized"),
            request,
        )
    }

    fn execution_session_request(
        &self,
        request: &GgmlStageWorkerRequest,
    ) -> Result<GgmlStageWorkerResponse> {
        let init = self.execution_worker_init()?.clone();
        let mut session = self
            .execution_worker_session
            .lock()
            .expect("ggml execution worker session mutex poisoned");
        if session.is_none() {
            *session = Some(spawn_in_process_stage_worker_session(&init)?);
        }
        let response = run_stage_worker_session_request(
            session.as_mut().expect("execution worker session initialized"),
            request,
        );
        if response.is_ok() {
            return response;
        }

        if let Some(mut active) = session.take() {
            active.shutdown();
        }
        *session = Some(spawn_in_process_stage_worker_session(&init)?);
        run_stage_worker_session_request(
            session.as_mut().expect("execution worker session reinitialized"),
            request,
        )
    }

    fn sample_session_request(
        &self,
        request: &GgmlStageWorkerRequest,
    ) -> Result<GgmlStageWorkerResponse> {
        let init = self.sample_worker_init()?.clone();
        let mut session =
            self.sample_worker_session.lock().expect("ggml sample worker session mutex poisoned");
        if session.is_none() {
            *session = Some(spawn_in_process_stage_worker_session(&init)?);
        }
        let response = run_stage_worker_session_request(
            session.as_mut().expect("sample worker session initialized"),
            request,
        );
        if response.is_ok() {
            return response;
        }

        if let Some(mut active) = session.take() {
            active.shutdown();
        }
        *session = Some(spawn_in_process_stage_worker_session(&init)?);
        run_stage_worker_session_request(
            session.as_mut().expect("sample worker session reinitialized"),
            request,
        )
    }

    fn partial_capabilities(&self) -> RealForwardProviderCapabilities {
        if let Some(contract) = &self.last_worker_contract {
            return contract.capabilities;
        }
        RealForwardProviderCapabilities::default()
    }
}

impl RealForwardStageProvider for GgmlRealForwardProvider {
    fn provider_name(&self) -> &'static str {
        "ggml-bootstrap"
    }

    fn capabilities(&self) -> RealForwardProviderCapabilities {
        self.partial_capabilities()
    }

    fn load_shard(&mut self, config: &ShardConfig) -> Result<()> {
        Self::shutdown_worker_session(&self.metadata_worker_session);
        Self::shutdown_worker_session(&self.execution_worker_session);
        Self::shutdown_worker_session(&self.sample_worker_session);
        let load_spec = RealForwardStageLoadSpec::from_shard_config(config)?;
        let plan = GgmlStageExecutionPlan::from_load_spec(self.requested_target, &load_spec)?;
        let runtime_plan = detect_ggml_runtime_plan(self.requested_target);
        let compatibility = GgmlProviderCompatibility::from_load_spec(&load_spec);
        let binding_manifest = GgmlStageBindingManifest::from_load_spec(&load_spec)?;
        let operator_plan = GgmlStageOperatorPlan::from_manifest(&binding_manifest)?;
        let worker_contract = GgmlStageWorkerContract::from_layout(&load_spec.layout);
        let execution_executor = self.effective_execution_executor(&load_spec);
        let metadata_worker_init = GgmlStageWorkerInitSpec::from_load_spec(
            &load_spec,
            &runtime_plan,
            self.metadata_executor,
        );
        let mut execution_worker_init =
            GgmlStageWorkerInitSpec::from_load_spec(&load_spec, &runtime_plan, execution_executor);
        execution_worker_init.debug_layer_cap =
            Self::execution_debug_layer_cap(&load_spec, execution_executor);
        let sample_worker_init = GgmlStageWorkerInitSpec::from_load_spec(
            &load_spec,
            &runtime_plan,
            self.sample_executor,
        );
        let metadata_worker_launch =
            Some(GgmlStageWorkerHostLaunchSpec::from_init_spec(&metadata_worker_init)?);
        let execution_worker_launch =
            Some(GgmlStageWorkerHostLaunchSpec::from_init_spec(&execution_worker_init)?);
        let sample_worker_launch =
            Some(GgmlStageWorkerHostLaunchSpec::from_init_spec(&sample_worker_init)?);
        self.last_load_spec = Some(load_spec);
        self.last_plan = Some(plan);
        self.last_runtime_plan = Some(runtime_plan);
        self.last_compatibility = Some(compatibility);
        self.last_binding_manifest = Some(binding_manifest);
        self.last_operator_plan = Some(operator_plan);
        self.last_worker_contract = Some(worker_contract);
        self.last_metadata_worker_init = Some(metadata_worker_init);
        self.last_metadata_worker_launch = metadata_worker_launch;
        self.last_execution_worker_init = Some(execution_worker_init);
        self.last_execution_worker_launch = execution_worker_launch;
        self.last_sample_worker_init = Some(sample_worker_init);
        self.last_sample_worker_launch = sample_worker_launch;
        Self::maybe_prewarm_worker_session(
            &self.metadata_worker_session,
            self.last_metadata_worker_launch.as_ref(),
        )?;
        Self::maybe_prewarm_worker_session(
            &self.execution_worker_session,
            self.last_execution_worker_launch.as_ref(),
        )?;
        Self::maybe_prewarm_worker_session(
            &self.sample_worker_session,
            self.last_sample_worker_launch.as_ref(),
        )?;
        self.maybe_prewarm_head_execution()?;
        Ok(())
    }

    fn unload(&mut self) -> Result<()> {
        self.last_load_spec = None;
        self.last_plan = None;
        self.last_runtime_plan = None;
        self.last_compatibility = None;
        self.last_binding_manifest = None;
        self.last_operator_plan = None;
        self.last_worker_contract = None;
        self.last_metadata_worker_init = None;
        self.last_metadata_worker_launch = None;
        self.last_execution_worker_init = None;
        self.last_execution_worker_launch = None;
        self.last_sample_worker_init = None;
        self.last_sample_worker_launch = None;
        Self::shutdown_worker_session(&self.metadata_worker_session);
        Self::shutdown_worker_session(&self.execution_worker_session);
        Self::shutdown_worker_session(&self.sample_worker_session);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        let response = self.execution_session_request(&GgmlStageWorkerRequest::BeginPrompt {
            request_id: request_id.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
        })?;
        match response {
            GgmlStageWorkerResponse::Tensor { tensor } => Ok(tensor),
            other => anyhow::bail!("unexpected ggml worker response for begin_prompt: {:?}", other),
        }
    }

    fn begin_token_ids(
        &self,
        request_id: &str,
        token_ids: &[u32],
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        let response = self.execution_session_request(&GgmlStageWorkerRequest::BeginTokenIds {
            request_id: request_id.to_string(),
            token_ids: token_ids.to_vec(),
            max_tokens,
        })?;
        match response {
            GgmlStageWorkerResponse::Tensor { tensor } => Ok(tensor),
            other => {
                anyhow::bail!("unexpected ggml worker response for begin_token_ids: {:?}", other)
            }
        }
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        let response =
            self.execution_session_request(&GgmlStageWorkerRequest::ContinueForward { input })?;
        match response {
            GgmlStageWorkerResponse::Tensor { tensor } => Ok(tensor),
            other => {
                anyhow::bail!("unexpected ggml worker response for continue_forward: {:?}", other)
            }
        }
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let response =
            self.sample_session_request(&GgmlStageWorkerRequest::SampleTail { input })?;
        match response {
            GgmlStageWorkerResponse::Sample { sample } => Ok(sample),
            other => anyhow::bail!("unexpected ggml worker response for sample_tail: {:?}", other),
        }
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
        let response = self.metadata_session_request(&GgmlStageWorkerRequest::TokenizeText {
            text: text.to_string(),
        })?;
        match response {
            GgmlStageWorkerResponse::TokenIds { token_ids } => Ok(token_ids),
            other => {
                anyhow::bail!("unexpected ggml worker response for tokenize_text: {:?}", other)
            }
        }
    }

    fn tokenize_generation_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let response =
            self.metadata_session_request(&GgmlStageWorkerRequest::TokenizeGenerationPrompt {
                text: text.to_string(),
            })?;
        match response {
            GgmlStageWorkerResponse::TokenIds { token_ids } => Ok(token_ids),
            other => anyhow::bail!(
                "unexpected ggml worker response for tokenize_generation_prompt: {:?}",
                other
            ),
        }
    }

    fn decode_token_ids(&self, tokens: &[u32]) -> Result<String> {
        let response = self.metadata_session_request(&GgmlStageWorkerRequest::DecodeTokenIds {
            token_ids: tokens.to_vec(),
        })?;
        match response {
            GgmlStageWorkerResponse::Text { text } => Ok(text),
            other => {
                anyhow::bail!("unexpected ggml worker response for decode_token_ids: {:?}", other)
            }
        }
    }

    fn eos_token_id(&self) -> Option<u32> {
        match self.metadata_session_request(&GgmlStageWorkerRequest::EosTokenId).ok()? {
            GgmlStageWorkerResponse::EosTokenId { token_id } => token_id,
            _ => None,
        }
    }

    fn clear_decode_session(&self, request_id: &str) {
        let _ = self.execution_session_request(&GgmlStageWorkerRequest::ClearDecodeSession {
            request_id: request_id.to_string(),
        });
        if self
            .last_execution_worker_init
            .as_ref()
            .map(|init| init.requested_executor == GgmlStageExecutorKind::ReferenceCpu)
            .unwrap_or(false)
        {
            Self::shutdown_worker_session(&self.execution_worker_session);
        }
    }
}

impl Drop for GgmlRealForwardProvider {
    fn drop(&mut self) {
        Self::shutdown_worker_session(&self.metadata_worker_session);
        Self::shutdown_worker_session(&self.execution_worker_session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stage_forward_lab::PackedTensorEntry;
    use tempfile::tempdir;

    fn pin_test_worker_host() {
        unsafe {
            std::env::set_var("COMPUTE_GGML_STAGE_WORKER_HOST", "/bin/sh");
        }
    }

    fn write_minimal_head_stage(stage_dir: &std::path::Path) {
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
    }

    #[test]
    fn ggml_provider_loads_artifact_spec_and_plan_before_becoming_available() {
        pin_test_worker_host();
        let temp = tempdir().unwrap();
        let stage_dir = temp.path().join("packed-stage-0-20");
        std::fs::create_dir_all(&stage_dir).unwrap();
        write_minimal_head_stage(&stage_dir);

        let mut provider = GgmlRealForwardProvider::new(
            StageAccelerationTarget::Metal,
            GgmlStageExecutorKind::Ggml,
            GgmlStageExecutorKind::Ggml,
            GgmlStageExecutorKind::Ggml,
        );
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
        assert!(provider.last_load_spec.is_some());
        assert!(provider.last_plan.is_some());
        assert!(provider.last_runtime_plan.is_some());
        assert!(provider.last_compatibility.is_some());
        assert!(provider.last_operator_plan.is_some());
        assert!(provider.last_worker_contract.is_some());
        assert!(provider.last_metadata_worker_init.is_some());
        assert!(provider.last_execution_worker_init.is_some());
        assert!(provider.last_sample_worker_init.is_some());
        assert!(provider.last_metadata_worker_launch.is_some());
        assert!(provider.last_execution_worker_launch.is_some());
        assert!(provider.last_sample_worker_launch.is_some());
        assert_eq!(provider.provider_name(), "ggml-bootstrap");
        assert_eq!(
            provider.capabilities(),
            RealForwardProviderCapabilities {
                hidden_state_ingress: false,
                hidden_state_egress: true,
                token_id_prompt_ingress: true,
                tail_sampling: false,
                per_stage_decode_sessions: true,
            }
        );
        let plan = provider.last_plan.as_ref().unwrap().summary_label();
        assert!(plan.contains("target=metal arch=gemma4 stage=0 role=head layers=0-20"));
        assert!(plan.contains("types=[1xF32,7xQ4_K]") || plan.contains("types=[7xQ4_K,1xF32]"));
        let compatibility = provider.last_compatibility.as_ref().unwrap().summary_label();
        assert!(compatibility.contains("stock-llama-server-compatible=false"));
        let execution_init = provider.last_execution_worker_init.as_ref().unwrap();
        assert_eq!(execution_init.requested_executor, GgmlStageExecutorKind::Ggml);
        assert_eq!(execution_init.debug_layer_cap, Some(21));
    }

    #[test]
    fn ggml_provider_exposes_bootstrap_capabilities_after_successful_load() {
        pin_test_worker_host();
        let temp = tempdir().unwrap();
        let stage_dir = temp.path().join("packed-stage-0-20");
        std::fs::create_dir_all(&stage_dir).unwrap();
        write_minimal_head_stage(&stage_dir);

        let mut provider = GgmlRealForwardProvider::new(
            StageAccelerationTarget::Metal,
            GgmlStageExecutorKind::Ggml,
            GgmlStageExecutorKind::Ggml,
            GgmlStageExecutorKind::Ggml,
        );
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

        assert_eq!(provider.provider_name(), "ggml-bootstrap");
        assert_eq!(
            provider.capabilities(),
            RealForwardProviderCapabilities {
                hidden_state_ingress: false,
                hidden_state_egress: true,
                token_id_prompt_ingress: true,
                tail_sampling: false,
                per_stage_decode_sessions: true,
            }
        );
    }

    #[test]
    fn ggml_stage_execution_plan_summarizes_tensor_types() {
        let temp = tempdir().unwrap();
        let stage_dir = temp.path().join("packed-stage-21-41");
        std::fs::create_dir_all(&stage_dir).unwrap();
        let index_path = stage_dir.join("stage-2-required.index.json");
        std::fs::write(
            &index_path,
            serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "gemma-4-e4b-q4".into(),
                architecture: "gemma4".into(),
                stage_index: 1,
                role: "tail".into(),
                total_bytes: 8192,
                tensor_count: 3,
                tensors: vec![
                    PackedTensorEntry {
                        name: "blk.21.ffn_gate.weight".into(),
                        pack_offset: 0,
                        byte_len: 2048,
                        source_file_offset: 0,
                        dimensions: vec![1],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                    PackedTensorEntry {
                        name: "blk.21.ffn_down.weight".into(),
                        pack_offset: 2048,
                        byte_len: 2048,
                        source_file_offset: 2048,
                        dimensions: vec![1],
                        ggml_type: quants::GGML_TYPE_Q6_K,
                    },
                    PackedTensorEntry {
                        name: "output.weight".into(),
                        pack_offset: 4096,
                        byte_len: 4096,
                        source_file_offset: 4096,
                        dimensions: vec![1],
                        ggml_type: quants::GGML_TYPE_Q4_K,
                    },
                ],
            })
            .unwrap(),
        )
        .unwrap();

        let load_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
            model_id: "gemma-4-e4b-q4".into(),
            shard_path: index_path,
            start_layer: 21,
            end_layer: 41,
            total_layers: 42,
            is_first_stage: false,
            is_last_stage: true,
            max_batch_size: 16,
            context_length: 8192,
        })
        .unwrap();

        let plan =
            GgmlStageExecutionPlan::from_load_spec(StageAccelerationTarget::Cuda, &load_spec)
                .unwrap();
        assert_eq!(plan.ggml_type_counts.get(&quants::GGML_TYPE_Q4_K), Some(&2));
        assert_eq!(plan.ggml_type_counts.get(&quants::GGML_TYPE_Q6_K), Some(&1));
        assert!(plan.summary_label().contains("target=cuda"));
        assert!(plan.summary_label().contains("layers=21-41"));
    }

    #[test]
    fn ggml_provider_compatibility_calls_out_split_boundary_limits() {
        let load_spec = RealForwardStageLoadSpec {
            config: ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: std::env::temp_dir(),
                start_layer: 21,
                end_layer: 41,
                total_layers: 42,
                is_first_stage: false,
                is_last_stage: true,
                max_batch_size: 16,
                context_length: 8192,
            },
            stage_dir: std::env::temp_dir(),
            index_path: std::env::temp_dir().join("stage-2-required.index.json"),
            vocab_path: None,
            vocab_scores_path: None,
            layout: stage_forward_lab::StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-21-41".into(),
                start_layer: 21,
                end_layer: 41,
                is_head: false,
                is_tail: true,
            },
        };

        let compatibility = GgmlProviderCompatibility::from_load_spec(&load_spec);
        assert_eq!(compatibility.artifact_kind, "packed-stage-dir");
        assert!(!compatibility.stock_runtime_compatible);
        assert!(compatibility.summary_label().contains("stock-llama-server-compatible=false"));
        assert!(compatibility.reasons.iter().any(|reason| reason.contains("hidden-state ingress")));
    }
}
