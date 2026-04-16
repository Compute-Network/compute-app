use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use stage_forward_lab::StageLayout;
use stage_forward_lab::prompting::{GemmaPromptMode, format_gemma_prompt};
use stage_forward_lab::tokenizer::GemmaTokenizer;

use crate::inference::ggml_runtime::GgmlRuntimePlan;
use crate::inference::ggml_stage_executor::{
    GgmlStageExecutor, GgmlStageExecutorKind, build_ggml_stage_executor,
};
use crate::inference::real_forward_artifact::RealForwardStageLoadSpec;
use crate::inference::real_forward_provider::RealForwardProviderCapabilities;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlStageWorkerOperation {
    BeginPrompt,
    BeginTokenIds,
    ContinueForward,
    SampleTail,
    TokenizeText,
    TokenizeGenerationPrompt,
    DecodeTokenIds,
    EosTokenId,
    ClearDecodeSession,
}

impl GgmlStageWorkerOperation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BeginPrompt => "begin_prompt",
            Self::BeginTokenIds => "begin_token_ids",
            Self::ContinueForward => "continue_forward",
            Self::SampleTail => "sample_tail",
            Self::TokenizeText => "tokenize_text",
            Self::TokenizeGenerationPrompt => "tokenize_generation_prompt",
            Self::DecodeTokenIds => "decode_token_ids",
            Self::EosTokenId => "eos_token_id",
            Self::ClearDecodeSession => "clear_decode_session",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlStageWorkerContract {
    pub stage_id: String,
    pub role: String,
    pub capabilities: RealForwardProviderCapabilities,
    pub required_operations: Vec<GgmlStageWorkerOperation>,
}

impl GgmlStageWorkerContract {
    pub fn from_layout(layout: &StageLayout) -> Self {
        let mut required_operations = vec![
            GgmlStageWorkerOperation::EosTokenId,
            GgmlStageWorkerOperation::ClearDecodeSession,
        ];
        if layout.is_head {
            required_operations.extend([
                GgmlStageWorkerOperation::BeginPrompt,
                GgmlStageWorkerOperation::BeginTokenIds,
                GgmlStageWorkerOperation::TokenizeText,
                GgmlStageWorkerOperation::TokenizeGenerationPrompt,
            ]);
        }
        if !layout.is_head {
            required_operations.push(GgmlStageWorkerOperation::ContinueForward);
        }
        if layout.is_tail {
            required_operations.extend([
                GgmlStageWorkerOperation::SampleTail,
                GgmlStageWorkerOperation::DecodeTokenIds,
            ]);
        }
        if !layout.is_head
            && !layout.is_tail
            && !required_operations.contains(&GgmlStageWorkerOperation::ContinueForward)
        {
            required_operations.push(GgmlStageWorkerOperation::ContinueForward);
        }

        let role = match (layout.is_head, layout.is_tail) {
            (true, true) => "single",
            (true, false) => "head",
            (false, true) => "tail",
            (false, false) => "middle",
        }
        .to_string();

        let capabilities = RealForwardProviderCapabilities {
            hidden_state_ingress: !layout.is_head,
            hidden_state_egress: !layout.is_tail,
            token_id_prompt_ingress: layout.is_head,
            tail_sampling: layout.is_tail,
            per_stage_decode_sessions: true,
        };

        Self { stage_id: layout.stage_id.clone(), role, capabilities, required_operations }
    }

    pub fn summary_label(&self) -> String {
        let ops = self
            .required_operations
            .iter()
            .map(GgmlStageWorkerOperation::as_str)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "stage={} role={} caps=[hidden_in={} hidden_out={} token_prompt={} tail_sample={} decode_sessions={}] ops=[{}]",
            self.stage_id,
            self.role,
            self.capabilities.hidden_state_ingress,
            self.capabilities.hidden_state_egress,
            self.capabilities.token_id_prompt_ingress,
            self.capabilities.tail_sampling,
            self.capabilities.per_stage_decode_sessions,
            ops
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlStageWorkerInitSpec {
    pub model_id: String,
    pub stage_id: String,
    pub role: String,
    pub start_layer: u32,
    pub end_layer: u32,
    #[serde(default)]
    pub debug_layer_cap: Option<usize>,
    pub requested_executor: GgmlStageExecutorKind,
    pub stage_dir: PathBuf,
    pub index_path: PathBuf,
    pub vocab_path: Option<PathBuf>,
    pub vocab_scores_path: Option<PathBuf>,
    pub runtime: GgmlRuntimePlan,
    pub contract: GgmlStageWorkerContract,
}

impl GgmlStageWorkerInitSpec {
    pub fn from_load_spec(
        load_spec: &RealForwardStageLoadSpec,
        runtime: &GgmlRuntimePlan,
        requested_executor: GgmlStageExecutorKind,
    ) -> Self {
        Self {
            model_id: load_spec.config.model_id.clone(),
            stage_id: load_spec.layout.stage_id.clone(),
            role: match (load_spec.layout.is_head, load_spec.layout.is_tail) {
                (true, true) => "single",
                (true, false) => "head",
                (false, true) => "tail",
                (false, false) => "middle",
            }
            .into(),
            start_layer: load_spec.layout.start_layer,
            end_layer: load_spec.layout.end_layer,
            debug_layer_cap: None,
            requested_executor,
            stage_dir: load_spec.stage_dir.clone(),
            index_path: load_spec.index_path.clone(),
            vocab_path: load_spec.vocab_path.clone(),
            vocab_scores_path: load_spec.vocab_scores_path.clone(),
            runtime: runtime.clone(),
            contract: GgmlStageWorkerContract::from_layout(&load_spec.layout),
        }
    }

    pub fn summary_label(&self) -> String {
        let vocab = self
            .vocab_path
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "-".into());
        let vocab_scores = self
            .vocab_scores_path
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "-".into());
        let debug =
            self.debug_layer_cap.map(|cap| format!(" debug_layer_cap={}", cap)).unwrap_or_default();
        format!(
            "model={} stage={} role={} layers={}-{}{} executor={} stage_dir={} index={} vocab={} vocab_scores={} runtime={} contract={}",
            self.model_id,
            self.stage_id,
            self.role,
            self.start_layer,
            self.end_layer,
            debug,
            self.requested_executor.as_str(),
            self.stage_dir.display(),
            self.index_path.display(),
            vocab,
            vocab_scores,
            self.runtime.summary_label(),
            self.contract.summary_label()
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlStageWorkerHostLaunchSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
}

impl GgmlStageWorkerHostLaunchSpec {
    pub fn from_init_spec(init: &GgmlStageWorkerInitSpec) -> Result<Self> {
        let program = find_ggml_stage_worker_host()?;
        let init_json = serde_json::to_string(init)?;
        Ok(Self { program, args: vec!["--init-json".into(), init_json] })
    }

    pub fn summary_label(&self) -> String {
        let init_bytes = self.args.get(1).map(String::len).unwrap_or(0);
        format!("{} --init-json <{} bytes>", self.program.display(), init_bytes)
    }

    pub fn into_command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        command
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum GgmlStageWorkerRequest {
    TokenizeText { text: String },
    TokenizeGenerationPrompt { text: String },
    DecodeTokenIds { token_ids: Vec<u32> },
    EosTokenId,
    BeginPrompt { request_id: String, prompt: String, max_tokens: Option<u32> },
    BeginTokenIds { request_id: String, token_ids: Vec<u32>, max_tokens: Option<u32> },
    BeginTokenIdsSummary { request_id: String, token_ids: Vec<u32>, max_tokens: Option<u32> },
    ContinueForward { input: stage_forward_lab::StageTensor },
    ContinueForwardSummary { input: stage_forward_lab::StageTensor },
    SampleTail { input: stage_forward_lab::StageTensor },
    ClearDecodeSession { request_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlStageWorkerTensorSummary {
    pub kind: String,
    pub hidden_dim: usize,
    pub hidden_state_bytes: usize,
    pub aux_bytes: usize,
    pub stage_trace_depth: usize,
    pub hidden_bytes_hash: u64,
    pub aux_bytes_hash: Option<u64>,
    pub prompt_text: Option<String>,
    pub max_tokens: Option<u32>,
}

impl GgmlStageWorkerTensorSummary {
    pub fn from_tensor(tensor: &stage_forward_lab::StageTensor) -> Self {
        let sections = stage_forward_lab::stage_tensor_byte_sections(&tensor.bytes);
        let (hidden_bytes, aux_bytes) = if let Some(sections) = sections {
            (sections.hidden_bytes, sections.aux_bytes.unwrap_or(&[]))
        } else {
            (&tensor.bytes[..], &[][..])
        };
        Self {
            kind: format!("{:?}", tensor.kind),
            hidden_dim: tensor.hidden_dim,
            hidden_state_bytes: tensor.hidden_state_len(),
            aux_bytes: aux_bytes.len(),
            stage_trace_depth: tensor.stage_trace.len(),
            hidden_bytes_hash: stable_stage_byte_hash(hidden_bytes),
            aux_bytes_hash: (!aux_bytes.is_empty()).then(|| stable_stage_byte_hash(aux_bytes)),
            prompt_text: tensor.prompt_text.clone(),
            max_tokens: tensor.max_tokens,
        }
    }

    pub fn hidden_contract_matches(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.hidden_dim == other.hidden_dim
            && self.hidden_state_bytes == other.hidden_state_bytes
            && self.aux_bytes == other.aux_bytes
            && self.stage_trace_depth == other.stage_trace_depth
            && self.hidden_bytes_hash == other.hidden_bytes_hash
            && self.aux_bytes_hash == other.aux_bytes_hash
            && self.prompt_text == other.prompt_text
            && self.max_tokens == other.max_tokens
    }

    pub fn summary_label(&self) -> String {
        format!(
            "kind={} hidden_dim={} hidden_state_bytes={} aux_bytes={} trace_depth={} hidden_hash={} aux_hash={:?} prompt={} max_tokens={:?}",
            self.kind,
            self.hidden_dim,
            self.hidden_state_bytes,
            self.aux_bytes,
            self.stage_trace_depth,
            self.hidden_bytes_hash,
            self.aux_bytes_hash,
            self.prompt_text.as_deref().unwrap_or("-"),
            self.max_tokens
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GgmlStageWorkerResponse {
    Ack,
    TokenIds { token_ids: Vec<u32> },
    Text { text: String },
    EosTokenId { token_id: Option<u32> },
    Tensor { tensor: stage_forward_lab::StageTensor },
    TensorSummary { summary: GgmlStageWorkerTensorSummary },
    Sample { sample: stage_forward_lab::StageSample },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgmlStageWorkerTensorWire {
    request_id: String,
    kind: stage_forward_lab::PayloadKind,
    stage_trace: Vec<String>,
    hidden_dim: usize,
    bytes_b64: String,
    prompt_text: Option<String>,
    max_tokens: Option<u32>,
    continuation: Option<stage_forward_lab::StageContinuation>,
    transient: Option<stage_forward_lab::StageTransientState>,
    carry: Option<stage_forward_lab::StageCarryState>,
}

impl From<stage_forward_lab::StageTensor> for GgmlStageWorkerTensorWire {
    fn from(tensor: stage_forward_lab::StageTensor) -> Self {
        Self {
            request_id: tensor.request_id,
            kind: tensor.kind,
            stage_trace: tensor.stage_trace,
            hidden_dim: tensor.hidden_dim,
            bytes_b64: base64::engine::general_purpose::STANDARD.encode(tensor.bytes),
            prompt_text: tensor.prompt_text,
            max_tokens: tensor.max_tokens,
            continuation: tensor.continuation,
            transient: tensor.transient,
            carry: tensor.carry,
        }
    }
}

impl TryFrom<GgmlStageWorkerTensorWire> for stage_forward_lab::StageTensor {
    type Error = anyhow::Error;

    fn try_from(tensor: GgmlStageWorkerTensorWire) -> Result<Self> {
        Ok(Self {
            request_id: tensor.request_id,
            kind: tensor.kind,
            stage_trace: tensor.stage_trace,
            hidden_dim: tensor.hidden_dim,
            bytes: base64::engine::general_purpose::STANDARD
                .decode(tensor.bytes_b64)
                .context("decode ggml worker tensor bytes from base64")?,
            prompt_text: tensor.prompt_text,
            max_tokens: tensor.max_tokens,
            continuation: tensor.continuation,
            transient: tensor.transient,
            carry: tensor.carry,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GgmlStageWorkerWireResponse {
    Ack,
    TokenIds { token_ids: Vec<u32> },
    Text { text: String },
    EosTokenId { token_id: Option<u32> },
    Tensor { tensor: GgmlStageWorkerTensorWire },
    TensorSummary { summary: GgmlStageWorkerTensorSummary },
    Sample { sample: stage_forward_lab::StageSample },
}

impl From<GgmlStageWorkerResponse> for GgmlStageWorkerWireResponse {
    fn from(response: GgmlStageWorkerResponse) -> Self {
        match response {
            GgmlStageWorkerResponse::Ack => Self::Ack,
            GgmlStageWorkerResponse::TokenIds { token_ids } => Self::TokenIds { token_ids },
            GgmlStageWorkerResponse::Text { text } => Self::Text { text },
            GgmlStageWorkerResponse::EosTokenId { token_id } => Self::EosTokenId { token_id },
            GgmlStageWorkerResponse::Tensor { tensor } => Self::Tensor { tensor: tensor.into() },
            GgmlStageWorkerResponse::TensorSummary { summary } => Self::TensorSummary { summary },
            GgmlStageWorkerResponse::Sample { sample } => Self::Sample { sample },
        }
    }
}

impl TryFrom<GgmlStageWorkerWireResponse> for GgmlStageWorkerResponse {
    type Error = anyhow::Error;

    fn try_from(response: GgmlStageWorkerWireResponse) -> Result<Self> {
        Ok(match response {
            GgmlStageWorkerWireResponse::Ack => Self::Ack,
            GgmlStageWorkerWireResponse::TokenIds { token_ids } => Self::TokenIds { token_ids },
            GgmlStageWorkerWireResponse::Text { text } => Self::Text { text },
            GgmlStageWorkerWireResponse::EosTokenId { token_id } => Self::EosTokenId { token_id },
            GgmlStageWorkerWireResponse::Tensor { tensor } => {
                Self::Tensor { tensor: tensor.try_into()? }
            }
            GgmlStageWorkerWireResponse::TensorSummary { summary } => {
                Self::TensorSummary { summary }
            }
            GgmlStageWorkerWireResponse::Sample { sample } => Self::Sample { sample },
        })
    }
}

pub fn run_stage_worker_request(
    launch: &GgmlStageWorkerHostLaunchSpec,
    request: &GgmlStageWorkerRequest,
) -> Result<GgmlStageWorkerResponse> {
    let request_json = serde_json::to_string(request)?;
    let request_file = request_file_path();
    let response_file = exchange_file_path("response");
    fs::write(&request_file, request_json).with_context(|| {
        format!("write ggml stage worker request file to {}", request_file.display())
    })?;

    let mut command = Command::new(&launch.program);
    command.args(&launch.args);
    command.arg("--request-file");
    command.arg(&request_file);
    command.arg("--response-file");
    command.arg(&response_file);
    let output = command
        .output()
        .with_context(|| format!("launch ggml stage worker host via {}", launch.program.display()));
    let _ = fs::remove_file(&request_file);
    let output = output?;
    if !output.status.success() {
        anyhow::bail!(
            "ggml stage worker host failed (status={}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let response_json = fs::read_to_string(&response_file).with_context(|| {
        format!("read ggml stage worker response file from {}", response_file.display())
    });
    let _ = fs::remove_file(&response_file);
    let response_json = response_json?;
    let response: GgmlStageWorkerWireResponse =
        serde_json::from_str(response_json.trim()).context("parse ggml stage worker response")?;
    response.try_into()
}

fn request_file_path() -> PathBuf {
    exchange_file_path("request")
}

pub fn handle_stage_worker_request(
    executor: &mut dyn GgmlStageExecutor,
    request: GgmlStageWorkerRequest,
) -> Result<GgmlStageWorkerResponse> {
    Ok(match request {
        GgmlStageWorkerRequest::TokenizeText { text } => {
            GgmlStageWorkerResponse::TokenIds { token_ids: executor.tokenize_text(&text)? }
        }
        GgmlStageWorkerRequest::TokenizeGenerationPrompt { text } => {
            GgmlStageWorkerResponse::TokenIds {
                token_ids: executor.tokenize_generation_prompt(&text)?,
            }
        }
        GgmlStageWorkerRequest::DecodeTokenIds { token_ids } => {
            GgmlStageWorkerResponse::Text { text: executor.decode_token_ids(&token_ids)? }
        }
        GgmlStageWorkerRequest::EosTokenId => {
            GgmlStageWorkerResponse::EosTokenId { token_id: executor.eos_token_id()? }
        }
        GgmlStageWorkerRequest::BeginPrompt { request_id, prompt, max_tokens } => {
            let tensor = executor.begin_prompt(&request_id, &prompt, max_tokens)?;
            GgmlStageWorkerResponse::Tensor { tensor }
        }
        GgmlStageWorkerRequest::BeginTokenIds { request_id, token_ids, max_tokens } => {
            let tensor = executor.begin_token_ids(&request_id, &token_ids, max_tokens)?;
            GgmlStageWorkerResponse::Tensor { tensor }
        }
        GgmlStageWorkerRequest::BeginTokenIdsSummary { request_id, token_ids, max_tokens } => {
            let tensor = executor.begin_token_ids(&request_id, &token_ids, max_tokens)?;
            GgmlStageWorkerResponse::TensorSummary {
                summary: GgmlStageWorkerTensorSummary::from_tensor(&tensor),
            }
        }
        GgmlStageWorkerRequest::ContinueForward { input } => {
            let tensor = executor.continue_forward(input)?;
            GgmlStageWorkerResponse::Tensor { tensor }
        }
        GgmlStageWorkerRequest::ContinueForwardSummary { input } => {
            let tensor = executor.continue_forward(input)?;
            GgmlStageWorkerResponse::TensorSummary {
                summary: GgmlStageWorkerTensorSummary::from_tensor(&tensor),
            }
        }
        GgmlStageWorkerRequest::SampleTail { input } => {
            let sample = executor.sample_tail(input)?;
            GgmlStageWorkerResponse::Sample { sample }
        }
        GgmlStageWorkerRequest::ClearDecodeSession { request_id } => {
            executor.clear_decode_session(&request_id);
            GgmlStageWorkerResponse::Ack
        }
    })
}

enum GgmlStageWorkerSessionKind {
    Host {
        addr: String,
        child: Child,
    },
    InProcessMetadata {
        stage_id: String,
        role: String,
        tokenizer: GemmaTokenizer,
    },
    InProcessThread {
        stage_id: String,
        role: String,
        executor_summary: String,
        tx: mpsc::Sender<GgmlStageWorkerThreadCommand>,
        join: Option<thread::JoinHandle<()>>,
    },
}

enum GgmlStageWorkerThreadCommand {
    Request {
        request: GgmlStageWorkerRequest,
        reply: mpsc::Sender<Result<GgmlStageWorkerResponse, String>>,
    },
    Shutdown,
}

pub struct GgmlStageWorkerSession {
    kind: GgmlStageWorkerSessionKind,
}

impl std::fmt::Debug for GgmlStageWorkerSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.summary_label())
    }
}

impl GgmlStageWorkerSession {
    pub fn summary_label(&self) -> String {
        match &self.kind {
            GgmlStageWorkerSessionKind::Host { addr, child } => {
                format!("persistent@{} pid={}", addr, child.id())
            }
            GgmlStageWorkerSessionKind::InProcessMetadata { stage_id, role, .. } => {
                format!("in-process-metadata stage={} role={}", stage_id, role)
            }
            GgmlStageWorkerSessionKind::InProcessThread {
                stage_id,
                role,
                executor_summary,
                ..
            } => {
                format!(
                    "in-process-thread stage={} role={} executor={}",
                    stage_id, role, executor_summary
                )
            }
        }
    }

    pub fn shutdown(&mut self) {
        match &mut self.kind {
            GgmlStageWorkerSessionKind::Host { child, .. } => {
                let _ = child.kill();
                let _ = child.wait();
            }
            GgmlStageWorkerSessionKind::InProcessThread { tx, join, .. } => {
                let _ = tx.send(GgmlStageWorkerThreadCommand::Shutdown);
                if let Some(join) = join.take() {
                    let _ = join.join();
                }
            }
            GgmlStageWorkerSessionKind::InProcessMetadata { .. } => {}
        }
    }
}

impl Drop for GgmlStageWorkerSession {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl GgmlStageWorkerHostLaunchSpec {
    pub fn spawn_persistent_session(&self) -> Result<GgmlStageWorkerSession> {
        let ready_file = exchange_file_path("ready");
        let mut command = self.into_command();
        command.arg("--listen-addr");
        command.arg("127.0.0.1:0");
        command.arg("--ready-file");
        command.arg(&ready_file);
        command.stdout(Stdio::null());
        command.stderr(Stdio::null());
        let mut child = command.spawn().with_context(|| {
            format!("launch persistent ggml stage worker host via {}", self.program.display())
        })?;

        let mut last_error = None;
        for _ in 0..300 {
            if let Ok(addr) = fs::read_to_string(&ready_file) {
                let _ = fs::remove_file(&ready_file);
                let addr = addr.trim().to_string();
                if !addr.is_empty() {
                    return Ok(GgmlStageWorkerSession {
                        kind: GgmlStageWorkerSessionKind::Host { addr, child },
                    });
                }
            }
            if let Some(status) = child.try_wait()? {
                last_error = Some(format!("worker exited before ready with status {status}"));
                break;
            }
            thread::sleep(std::time::Duration::from_millis(100));
        }
        let _ = fs::remove_file(&ready_file);
        let message =
            last_error.unwrap_or_else(|| "worker did not publish ready address in time".into());
        let _ = child.kill();
        let _ = child.wait();
        anyhow::bail!("failed to start persistent ggml stage worker session: {message}");
    }
}

pub fn spawn_in_process_stage_worker_session(
    init: &GgmlStageWorkerInitSpec,
) -> Result<GgmlStageWorkerSession> {
    let init = init.clone();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<String, String>>();
    let (tx, rx) = mpsc::channel::<GgmlStageWorkerThreadCommand>();
    let stage_id = init.stage_id.clone();
    let role = init.role.clone();
    let join = thread::Builder::new()
        .name(format!("ggml-inproc-{}", stage_id))
        .spawn(move || {
            let mut executor = match build_ggml_stage_executor(&init) {
                Ok(executor) => {
                    let summary = executor.plan().summary_label();
                    let _ = ready_tx.send(Ok(summary));
                    executor
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("{err:#}")));
                    return;
                }
            };
            while let Ok(command) = rx.recv() {
                match command {
                    GgmlStageWorkerThreadCommand::Request { request, reply } => {
                        let result = handle_stage_worker_request(executor.as_mut(), request)
                            .map_err(|err| format!("{err:#}"));
                        let _ = reply.send(result);
                    }
                    GgmlStageWorkerThreadCommand::Shutdown => break,
                }
            }
        })
        .context("spawn in-process ggml stage worker thread")?;
    let executor_summary = ready_rx
        .recv()
        .context("wait for in-process ggml stage worker thread ready state")?
        .map_err(|err| anyhow::anyhow!("in-process ggml stage worker init failed: {err}"))?;
    Ok(GgmlStageWorkerSession {
        kind: GgmlStageWorkerSessionKind::InProcessThread {
            stage_id,
            role,
            executor_summary,
            tx,
            join: Some(join),
        },
    })
}

pub fn spawn_in_process_metadata_session(
    init: &GgmlStageWorkerInitSpec,
) -> Result<GgmlStageWorkerSession> {
    let vocab_path = init
        .vocab_path
        .as_deref()
        .context("worker init is missing vocab_path for in-process metadata session")?;
    let tokenizer = GemmaTokenizer::load(vocab_path, init.vocab_scores_path.as_deref())
        .context("load tokenizer for in-process metadata session")?;
    Ok(GgmlStageWorkerSession {
        kind: GgmlStageWorkerSessionKind::InProcessMetadata {
            stage_id: init.stage_id.clone(),
            role: init.role.clone(),
            tokenizer,
        },
    })
}

pub fn run_stage_worker_session_request(
    session: &mut GgmlStageWorkerSession,
    request: &GgmlStageWorkerRequest,
) -> Result<GgmlStageWorkerResponse> {
    match &mut session.kind {
        GgmlStageWorkerSessionKind::Host { addr, .. } => {
            let mut stream = TcpStream::connect(addr.as_str())
                .with_context(|| format!("connect to ggml stage worker at {}", addr))?;
            write_framed_json(&mut stream, request)
                .context("write ggml stage worker request frame")?;
            let response: GgmlStageWorkerWireResponse =
                read_framed_json(&mut stream).context("read ggml stage worker response frame")?;
            response.try_into()
        }
        GgmlStageWorkerSessionKind::InProcessMetadata { tokenizer, .. } => match request.clone() {
            GgmlStageWorkerRequest::TokenizeText { text } => {
                Ok(GgmlStageWorkerResponse::TokenIds {
                    token_ids: tokenizer.encode_with_bos(&text),
                })
            }
            GgmlStageWorkerRequest::TokenizeGenerationPrompt { text } => {
                let formatted = format_gemma_prompt(GemmaPromptMode::GemmaInstruct, &text);
                Ok(GgmlStageWorkerResponse::TokenIds {
                    token_ids: tokenizer.encode_with_bos(&formatted),
                })
            }
            GgmlStageWorkerRequest::DecodeTokenIds { token_ids } => {
                Ok(GgmlStageWorkerResponse::Text { text: tokenizer.decode_ids(&token_ids) })
            }
            GgmlStageWorkerRequest::EosTokenId => {
                Ok(GgmlStageWorkerResponse::EosTokenId { token_id: Some(tokenizer.eos_id()) })
            }
            GgmlStageWorkerRequest::ClearDecodeSession { .. } => Ok(GgmlStageWorkerResponse::Ack),
            other => bail!("unsupported request for in-process metadata session: {:?}", other),
        },
        GgmlStageWorkerSessionKind::InProcessThread { tx, .. } => {
            let (reply_tx, reply_rx) = mpsc::channel();
            tx.send(GgmlStageWorkerThreadCommand::Request {
                request: request.clone(),
                reply: reply_tx,
            })
            .context("send request to in-process ggml stage worker thread")?;
            reply_rx
                .recv()
                .context("receive response from in-process ggml stage worker thread")?
                .map_err(|err| anyhow::anyhow!(err))
        }
    }
}

fn write_framed_json<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<()> {
    let payload = serde_json::to_vec(value)?;
    let len = u64::try_from(payload.len()).context("frame payload exceeds u64")?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

pub fn read_framed_json<T: DeserializeOwned>(reader: &mut impl Read) -> Result<T> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let len = usize::try_from(u64::from_le_bytes(len_buf)).context("frame length exceeds usize")?;
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;
    serde_json::from_slice(&payload).context("parse framed ggml stage worker json")
}

fn exchange_file_path(kind: &str) -> PathBuf {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    std::env::temp_dir().join(format!(
        "compute-ggml-stage-worker-{kind}-{}-{}.json",
        std::process::id(),
        ts
    ))
}

pub fn find_ggml_stage_worker_host() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("COMPUTE_GGML_STAGE_WORKER_HOST") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        anyhow::bail!("COMPUTE_GGML_STAGE_WORKER_HOST points to missing path {}", path.display());
    }

    let current = std::env::current_exe()
        .context("resolve current executable for ggml stage worker host lookup")?;
    let Some(bin_dir) = current.parent() else {
        anyhow::bail!("could not resolve binary directory for ggml stage worker host lookup");
    };
    let mut candidates =
        vec![bin_dir.join("ggml_stage_worker_host"), bin_dir.join("ggml_stage_worker_host.exe")];
    if let Some(parent) = bin_dir.parent() {
        for profile in ["debug", "release"] {
            candidates.push(parent.join(profile).join("ggml_stage_worker_host"));
            candidates.push(parent.join(profile).join("ggml_stage_worker_host.exe"));
        }
    }
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    anyhow::bail!(
        "ggml stage worker host not found next to {}; build or point COMPUTE_GGML_STAGE_WORKER_HOST at the host binary",
        current.display()
    )
}

fn stable_stage_byte_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ggml_runtime::{GgmlRuntimeMode, GgmlRuntimePlan};
    use crate::inference::stage_acceleration::StageAccelerationTarget;

    #[test]
    fn head_stage_contract_requires_prompt_entry_and_hidden_state_egress() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-0-20".into(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
        };
        let contract = GgmlStageWorkerContract::from_layout(&layout);
        assert!(contract.capabilities.token_id_prompt_ingress);
        assert!(contract.capabilities.hidden_state_egress);
        assert!(!contract.capabilities.hidden_state_ingress);
        assert!(contract.required_operations.contains(&GgmlStageWorkerOperation::BeginPrompt));
        assert!(contract.summary_label().contains("role=head"));
    }

    #[test]
    fn tail_stage_contract_requires_hidden_state_ingress_and_sampling() {
        let layout = StageLayout {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-21-41".into(),
            start_layer: 21,
            end_layer: 41,
            is_head: false,
            is_tail: true,
        };
        let contract = GgmlStageWorkerContract::from_layout(&layout);
        assert!(contract.capabilities.hidden_state_ingress);
        assert!(!contract.capabilities.hidden_state_egress);
        assert!(contract.capabilities.tail_sampling);
        assert!(contract.required_operations.contains(&GgmlStageWorkerOperation::ContinueForward));
        assert!(contract.required_operations.contains(&GgmlStageWorkerOperation::SampleTail));
    }

    #[test]
    fn init_spec_resolves_runtime_and_tokenizer_paths_from_load_spec() {
        let load_spec = RealForwardStageLoadSpec {
            config: crate::inference::engine::ShardConfig {
                model_id: "gemma-4-e4b-q4".into(),
                shard_path: PathBuf::from("/tmp/packed-stage-0-20/stage-1-required.index.json"),
                start_layer: 0,
                end_layer: 20,
                total_layers: 42,
                is_first_stage: true,
                is_last_stage: false,
                max_batch_size: 16,
                context_length: 8192,
            },
            stage_dir: PathBuf::from("/tmp/packed-stage-0-20"),
            index_path: PathBuf::from("/tmp/packed-stage-0-20/stage-1-required.index.json"),
            vocab_path: Some(PathBuf::from("/tmp/packed-stage-0-20/vocab.json")),
            vocab_scores_path: Some(PathBuf::from("/tmp/packed-stage-0-20/vocab_scores.json")),
            layout: StageLayout {
                model_id: "gemma-4-e4b-q4".into(),
                stage_id: "stage-0-20".into(),
                start_layer: 0,
                end_layer: 20,
                is_head: true,
                is_tail: false,
            },
        };
        let runtime = GgmlRuntimePlan {
            target: StageAccelerationTarget::Metal,
            mode: GgmlRuntimeMode::NativeLlamaServer {
                path: PathBuf::from("/opt/homebrew/bin/llama-server"),
            },
            detail: "macos target via native llama-server".into(),
        };
        let init = GgmlStageWorkerInitSpec::from_load_spec(
            &load_spec,
            &runtime,
            GgmlStageExecutorKind::ReferenceCpu,
        );
        assert_eq!(init.role, "head");
        assert_eq!(init.requested_executor, GgmlStageExecutorKind::ReferenceCpu);
        assert!(init.contract.capabilities.token_id_prompt_ingress);
        assert!(init.summary_label().contains("vocab=vocab.json"));
        assert!(init.summary_label().contains("runtime=native-llama-server"));
        assert!(init.summary_label().contains("executor=cpu-ref-worker"));
        assert!(!init.summary_label().contains("debug_layer_cap="));
    }

    #[test]
    fn host_launch_spec_uses_explicit_host_override() {
        let temp = tempfile::tempdir().unwrap();
        let host_path = temp.path().join("ggml_stage_worker_host");
        std::fs::write(&host_path, "#!/bin/sh\n").unwrap();
        let previous = std::env::var_os("COMPUTE_GGML_STAGE_WORKER_HOST");
        unsafe {
            std::env::set_var("COMPUTE_GGML_STAGE_WORKER_HOST", &host_path);
        }

        let init = GgmlStageWorkerInitSpec {
            model_id: "gemma-4-e4b-q4".into(),
            stage_id: "stage-0-20".into(),
            role: "head".into(),
            start_layer: 0,
            end_layer: 20,
            debug_layer_cap: None,
            requested_executor: GgmlStageExecutorKind::ReferenceCpu,
            stage_dir: PathBuf::from("/tmp/packed-stage-0-20"),
            index_path: PathBuf::from("/tmp/packed-stage-0-20/stage-1-required.index.json"),
            vocab_path: Some(PathBuf::from("/tmp/packed-stage-0-20/vocab.json")),
            vocab_scores_path: Some(PathBuf::from("/tmp/packed-stage-0-20/vocab_scores.json")),
            runtime: GgmlRuntimePlan {
                target: StageAccelerationTarget::Metal,
                mode: GgmlRuntimeMode::NativeLlamaServer {
                    path: PathBuf::from("/opt/homebrew/bin/llama-server"),
                },
                detail: "macos target via native llama-server".into(),
            },
            contract: GgmlStageWorkerContract {
                stage_id: "stage-0-20".into(),
                role: "head".into(),
                capabilities: RealForwardProviderCapabilities {
                    hidden_state_ingress: false,
                    hidden_state_egress: true,
                    token_id_prompt_ingress: true,
                    tail_sampling: false,
                    per_stage_decode_sessions: true,
                },
                required_operations: vec![GgmlStageWorkerOperation::BeginPrompt],
            },
        };
        let launch = GgmlStageWorkerHostLaunchSpec::from_init_spec(&init).unwrap();
        assert_eq!(launch.program, host_path);
        assert_eq!(launch.args.first().map(String::as_str), Some("--init-json"));
        assert!(launch.summary_label().contains("ggml_stage_worker_host"));

        match previous {
            Some(value) => unsafe {
                std::env::set_var("COMPUTE_GGML_STAGE_WORKER_HOST", value);
            },
            None => unsafe {
                std::env::remove_var("COMPUTE_GGML_STAGE_WORKER_HOST");
            },
        }
    }

    #[test]
    fn worker_request_roundtrips_with_stable_snake_case_tags() {
        let request = GgmlStageWorkerRequest::TokenizeGenerationPrompt { text: "hello".into() };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"op\":\"tokenize_generation_prompt\""));

        let response = GgmlStageWorkerResponse::EosTokenId { token_id: Some(4) };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"kind\":\"eos_token_id\""));
    }

    #[test]
    fn tensor_wire_response_roundtrips_stage_tensor_bytes_exactly() {
        let tensor = stage_forward_lab::StageTensor {
            request_id: "req".into(),
            kind: stage_forward_lab::PayloadKind::HiddenState,
            stage_trace: vec!["stage-0-20".into()],
            hidden_dim: 4,
            bytes: stage_forward_lab::encode_stage_tensor_bytes(
                &[0, 1, 2, 3, 127, 128, 254, 255],
                Some(&[9, 8, 7, 6, 5]),
            ),
            prompt_text: Some("hi".into()),
            max_tokens: Some(1),
            continuation: None,
            transient: None,
            carry: None,
        };
        let wire = GgmlStageWorkerWireResponse::from(GgmlStageWorkerResponse::Tensor {
            tensor: tensor.clone(),
        });
        let json = serde_json::to_vec(&wire).unwrap();
        let decoded: GgmlStageWorkerWireResponse = serde_json::from_slice(&json).unwrap();
        let roundtrip = GgmlStageWorkerResponse::try_from(decoded).unwrap();
        assert_eq!(roundtrip, GgmlStageWorkerResponse::Tensor { tensor });
    }

    #[test]
    fn tensor_summary_hash_is_stable_for_identical_bytes() {
        let tensor = stage_forward_lab::StageTensor {
            request_id: "req".into(),
            kind: stage_forward_lab::PayloadKind::HiddenState,
            stage_trace: vec!["stage-0-20".into()],
            hidden_dim: 4,
            bytes: vec![1, 2, 3, 4],
            prompt_text: Some("hi".into()),
            max_tokens: Some(1),
            continuation: None,
            transient: None,
            carry: None,
        };
        let summary_a = GgmlStageWorkerTensorSummary::from_tensor(&tensor);
        let summary_b = GgmlStageWorkerTensorSummary::from_tensor(&tensor);
        assert_eq!(summary_a.hidden_bytes_hash, summary_b.hidden_bytes_hash);
        assert!(summary_a.hidden_contract_matches(&summary_b));
        assert!(summary_a.summary_label().contains("hidden_dim=4"));
    }
}
