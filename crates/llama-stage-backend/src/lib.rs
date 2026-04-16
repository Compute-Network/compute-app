use anyhow::{Context, Result, bail};
use libloading::Library;
use serde::{Deserialize, Serialize};
use stage_forward_lab::{PayloadKind, StageForwardBackend, StageLayout, StageSample, StageTensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::env::consts::EXE_SUFFIX;
use std::ffi::{CString, c_char, c_void};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::slice;
use std::thread;
use std::time::{Duration, Instant};

#[allow(non_camel_case_types, dead_code)]
mod ffi {
    use super::{c_char, c_void};

    pub enum LlamaModel {}
    pub enum LlamaContext {}
    pub enum LlamaVocab {}
    pub enum LlamaMemory {}

    pub type GgmlBackendDev = *mut c_void;
    pub type GgmlBackendBufferType = *mut c_void;
    pub type GgmlBackendSchedEvalCallback = Option<unsafe extern "C" fn()>;
    pub type GgmlAbortCallback = Option<unsafe extern "C" fn(*mut c_void) -> bool>;
    pub type LlamaProgressCallback = Option<unsafe extern "C" fn(f32, *mut c_void) -> bool>;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_model_tensor_buft_override {
        pub pattern: *const c_char,
        pub buft: GgmlBackendBufferType,
    }

    #[repr(C)]
    pub struct llama_model_kv_override {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct llama_sampler_seq_config {
        _private: [u8; 0],
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_split_mode {
        None = 0,
        Layer = 1,
        Row = 2,
        Tensor = 3,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_rope_scaling_type {
        Unspecified = -1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_pooling_type {
        Unspecified = -1,
        None = 0,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_attention_type {
        Unspecified = -1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum llama_flash_attn_type {
        Auto = -1,
        Disabled = 0,
        Enabled = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum ggml_type {
        F32 = 0,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_model_params {
        pub devices: *mut GgmlBackendDev,
        pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
        pub n_gpu_layers: i32,
        pub split_mode: llama_split_mode,
        pub main_gpu: i32,
        pub tensor_split: *const f32,
        pub progress_callback: LlamaProgressCallback,
        pub progress_callback_user_data: *mut c_void,
        pub kv_overrides: *const llama_model_kv_override,
        pub vocab_only: bool,
        pub use_mmap: bool,
        pub use_direct_io: bool,
        pub use_mlock: bool,
        pub check_tensors: bool,
        pub use_extra_bufts: bool,
        pub no_host: bool,
        pub no_alloc: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_context_params {
        pub n_ctx: u32,
        pub n_batch: u32,
        pub n_ubatch: u32,
        pub n_seq_max: u32,
        pub n_threads: i32,
        pub n_threads_batch: i32,
        pub rope_scaling_type: llama_rope_scaling_type,
        pub pooling_type: llama_pooling_type,
        pub attention_type: llama_attention_type,
        pub flash_attn_type: llama_flash_attn_type,
        pub rope_freq_base: f32,
        pub rope_freq_scale: f32,
        pub yarn_ext_factor: f32,
        pub yarn_attn_factor: f32,
        pub yarn_beta_fast: f32,
        pub yarn_beta_slow: f32,
        pub yarn_orig_ctx: u32,
        pub defrag_thold: f32,
        pub cb_eval: GgmlBackendSchedEvalCallback,
        pub cb_eval_user_data: *mut c_void,
        pub type_k: ggml_type,
        pub type_v: ggml_type,
        pub abort_callback: GgmlAbortCallback,
        pub abort_callback_data: *mut c_void,
        pub embeddings: bool,
        pub offload_kqv: bool,
        pub no_perf: bool,
        pub op_offload: bool,
        pub swa_full: bool,
        pub kv_unified: bool,
        pub samplers: *mut llama_sampler_seq_config,
        pub n_samplers: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct llama_batch {
        pub n_tokens: i32,
        pub token: *mut i32,
        pub embd: *mut f32,
        pub pos: *mut i32,
        pub n_seq_id: *mut i32,
        pub seq_id: *mut *mut i32,
        pub logits: *mut i8,
    }
}

type FnBackendInit = unsafe extern "C" fn();
type FnModelDefaultParams = unsafe extern "C" fn() -> ffi::llama_model_params;
type FnContextDefaultParams = unsafe extern "C" fn() -> ffi::llama_context_params;
type FnModelLoadFromFile =
    unsafe extern "C" fn(*const c_char, ffi::llama_model_params) -> *mut ffi::LlamaModel;
type FnModelFree = unsafe extern "C" fn(*mut ffi::LlamaModel);
type FnInitFromModel =
    unsafe extern "C" fn(*mut ffi::LlamaModel, ffi::llama_context_params) -> *mut ffi::LlamaContext;
type FnContextFree = unsafe extern "C" fn(*mut ffi::LlamaContext);
type FnGetMemory = unsafe extern "C" fn(*const ffi::LlamaContext) -> *mut ffi::LlamaMemory;
type FnMemoryClear = unsafe extern "C" fn(*mut ffi::LlamaMemory, bool);
type FnModelGetVocab = unsafe extern "C" fn(*const ffi::LlamaModel) -> *const ffi::LlamaVocab;
type FnModelNEmbdOut = unsafe extern "C" fn(*const ffi::LlamaModel) -> i32;
type FnVocabNTokens = unsafe extern "C" fn(*const ffi::LlamaVocab) -> i32;
type FnTokenize = unsafe extern "C" fn(
    *const ffi::LlamaVocab,
    *const c_char,
    i32,
    *mut i32,
    i32,
    bool,
    bool,
) -> i32;
type FnTokenToPiece =
    unsafe extern "C" fn(*const ffi::LlamaVocab, i32, *mut c_char, i32, i32, bool) -> i32;
type FnDecode = unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch) -> i32;
type FnDecodeHead = unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch, i32) -> i32;
type FnDecodeMiddle =
    unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch, i32, i32) -> i32;
type FnDecodeTail = unsafe extern "C" fn(*mut ffi::LlamaContext, ffi::llama_batch, i32) -> i32;
type FnGetEmbeddings = unsafe extern "C" fn(*mut ffi::LlamaContext) -> *mut f32;
type FnGetLogitsIth = unsafe extern "C" fn(*mut ffi::LlamaContext, i32) -> *mut f32;
type FnVocabIsEog = unsafe extern "C" fn(*const ffi::LlamaVocab, i32) -> bool;

struct LlamaApi {
    _deps: Vec<Library>,
    _llama: Library,
    backend_init: FnBackendInit,
    model_default_params: FnModelDefaultParams,
    context_default_params: FnContextDefaultParams,
    model_load_from_file: FnModelLoadFromFile,
    model_free: FnModelFree,
    init_from_model: FnInitFromModel,
    context_free: FnContextFree,
    get_memory: FnGetMemory,
    memory_clear: FnMemoryClear,
    model_get_vocab: FnModelGetVocab,
    model_n_embd_out: FnModelNEmbdOut,
    vocab_n_tokens: FnVocabNTokens,
    tokenize: FnTokenize,
    token_to_piece: FnTokenToPiece,
    decode: FnDecode,
    decode_head: FnDecodeHead,
    decode_middle: FnDecodeMiddle,
    decode_tail: FnDecodeTail,
    get_embeddings: FnGetEmbeddings,
    get_logits_ith: FnGetLogitsIth,
    vocab_is_eog: FnVocabIsEog,
}

impl LlamaApi {
    fn load() -> Result<Self> {
        let lib_dir = resolve_vendor_lib_dir()?;
        let mut dylibs: Vec<PathBuf> = fs::read_dir(&lib_dir)
            .with_context(|| format!("reading {}", lib_dir.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| {
                        let ext = ext.to_ascii_lowercase();
                        ext == "dylib" || ext == "so" || ext == "dll"
                    })
                    .unwrap_or(false)
            })
            .collect();
        dylibs.sort();

        let llama_path = dylibs
            .iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.contains("llama"))
                    .unwrap_or(false)
            })
            .cloned()
            .with_context(|| format!("no libllama found under {}", lib_dir.display()))?;

        let mut deps = Vec::new();
        for path in dylibs.iter().filter(|path| *path != &llama_path) {
            let lib = unsafe { Library::new(path) }
                .with_context(|| format!("loading dependency {}", path.display()))?;
            deps.push(lib);
        }

        let llama = unsafe { Library::new(&llama_path) }
            .with_context(|| format!("loading {}", llama_path.display()))?;

        unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
            Ok(*unsafe { lib.get::<T>(name)? })
        }

        let api = unsafe {
            Self {
                backend_init: load_symbol(&llama, b"llama_backend_init\0")?,
                model_default_params: load_symbol(&llama, b"llama_model_default_params\0")?,
                context_default_params: load_symbol(&llama, b"llama_context_default_params\0")?,
                model_load_from_file: load_symbol(&llama, b"llama_model_load_from_file\0")?,
                model_free: load_symbol(&llama, b"llama_model_free\0")?,
                init_from_model: load_symbol(&llama, b"llama_init_from_model\0")?,
                context_free: load_symbol(&llama, b"llama_free\0")?,
                get_memory: load_symbol(&llama, b"llama_get_memory\0")?,
                memory_clear: load_symbol(&llama, b"llama_memory_clear\0")?,
                model_get_vocab: load_symbol(&llama, b"llama_model_get_vocab\0")?,
                model_n_embd_out: load_symbol(&llama, b"llama_model_n_embd_out\0")?,
                vocab_n_tokens: load_symbol(&llama, b"llama_vocab_n_tokens\0")?,
                tokenize: load_symbol(&llama, b"llama_tokenize\0")?,
                token_to_piece: load_symbol(&llama, b"llama_token_to_piece\0")?,
                decode: load_symbol(&llama, b"llama_decode\0")?,
                decode_head: load_symbol(&llama, b"llama_decode_head\0")?,
                decode_middle: load_symbol(&llama, b"llama_decode_middle\0")?,
                decode_tail: load_symbol(&llama, b"llama_decode_tail\0")?,
                get_embeddings: load_symbol(&llama, b"llama_get_embeddings\0")?,
                get_logits_ith: load_symbol(&llama, b"llama_get_logits_ith\0")?,
                vocab_is_eog: load_symbol(&llama, b"llama_vocab_is_eog\0")?,
                _deps: deps,
                _llama: llama,
            }
        };

        unsafe { (api.backend_init)() };

        Ok(api)
    }
}

struct LlamaModelHandle {
    model: *mut ffi::LlamaModel,
}

impl LlamaModelHandle {
    fn new(api: &LlamaApi, model_path: &Path) -> Result<Self> {
        let path = CString::new(model_path.to_string_lossy().as_bytes())?;
        let force_cpu = std::env::var_os("LLAMA_STAGE_FORCE_CPU").is_some();

        let mut mparams = unsafe { (api.model_default_params)() };
        mparams.n_gpu_layers = if force_cpu { 0 } else { -1 };
        mparams.split_mode = ffi::llama_split_mode::None;
        mparams.use_mmap = true;
        mparams.use_mlock = false;

        let model = unsafe { (api.model_load_from_file)(path.as_ptr(), mparams) };
        if model.is_null() {
            bail!("failed to load model {}", model_path.display());
        }

        Ok(Self { model })
    }

    fn vocab(&self, api: &LlamaApi) -> *const ffi::LlamaVocab {
        unsafe { (api.model_get_vocab)(self.model) }
    }

    fn hidden_dim(&self, api: &LlamaApi) -> usize {
        unsafe { (api.model_n_embd_out)(self.model) as usize }
    }

    fn create_session(&self, api: &LlamaApi) -> Result<LlamaSession> {
        let force_cpu = std::env::var_os("LLAMA_STAGE_FORCE_CPU").is_some();

        let mut cparams = unsafe { (api.context_default_params)() };
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        cparams.n_ctx = 8192;
        cparams.n_batch = 2048;
        cparams.n_ubatch = 2048;
        cparams.n_seq_max = 1;
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
        cparams.pooling_type = ffi::llama_pooling_type::None;
        cparams.rope_scaling_type = ffi::llama_rope_scaling_type::Unspecified;
        cparams.attention_type = ffi::llama_attention_type::Unspecified;
        cparams.flash_attn_type = ffi::llama_flash_attn_type::Enabled;
        cparams.type_k = ffi::ggml_type::F32;
        cparams.type_v = ffi::ggml_type::F32;
        cparams.offload_kqv = !force_cpu;
        cparams.op_offload = !force_cpu;
        cparams.kv_unified = true;
        cparams.embeddings = false;

        let ctx = unsafe { (api.init_from_model)(self.model, cparams) };
        if ctx.is_null() {
            bail!("failed to create context from loaded model");
        }

        Ok(LlamaSession { ctx })
    }
}

struct LlamaSession {
    ctx: *mut ffi::LlamaContext,
}

impl LlamaSession {
    fn clear_memory(&self, api: &LlamaApi) {
        let memory = unsafe { (api.get_memory)(self.ctx) };
        if !memory.is_null() {
            unsafe { (api.memory_clear)(memory, true) };
        }
    }

    fn destroy(self, api: &LlamaApi) {
        unsafe { (api.context_free)(self.ctx) };
    }
}

struct OwnedBatch {
    _tokens: Option<Vec<i32>>,
    _embd: Option<Vec<f32>>,
    raw: ffi::llama_batch,
}

impl OwnedBatch {
    fn token_only(tokens: Vec<i32>) -> Self {
        let n_tokens = tokens.len() as i32;
        let mut tokens = tokens;
        let raw = ffi::llama_batch {
            n_tokens,
            token: tokens.as_mut_ptr(),
            embd: std::ptr::null_mut(),
            pos: std::ptr::null_mut(),
            n_seq_id: std::ptr::null_mut(),
            seq_id: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
        };
        Self {
            _tokens: Some(tokens),
            _embd: None,
            raw,
        }
    }

    fn token_and_hidden(tokens: Option<Vec<i32>>, hidden: Vec<f32>, token_count: usize) -> Self {
        let mut token_buf = tokens;
        let mut embd = hidden;
        let n_tokens = token_count as i32;

        let raw = ffi::llama_batch {
            n_tokens,
            token: token_buf
                .as_mut()
                .map(|tokens| tokens.as_mut_ptr())
                .unwrap_or(std::ptr::null_mut()),
            embd: embd.as_mut_ptr(),
            pos: std::ptr::null_mut(),
            n_seq_id: std::ptr::null_mut(),
            seq_id: std::ptr::null_mut(),
            logits: std::ptr::null_mut(),
        };

        Self {
            _tokens: token_buf,
            _embd: Some(embd),
            raw,
        }
    }
}

#[derive(Clone)]
struct CachedSample {
    sample: StageSample,
    token_id: i32,
    is_eog: bool,
}

struct SessionState {
    session: LlamaSession,
    cached_sample: Option<CachedSample>,
}

struct BackendState {
    layout: Option<StageLayout>,
    model: Option<LlamaModelHandle>,
    sessions: HashMap<String, SessionState>,
}

pub struct LlamaStageBackend {
    api: LlamaApi,
    model_path: PathBuf,
    state: RefCell<BackendState>,
}

// SAFETY:
// Access to a backend instance is serialized by the caller. The sidecar binaries
// handle one request at a time, and the daemon stage runtime wraps the backend
// behind a mutex before sharing it across async tasks. The underlying llama
// handles are not thread-safe for concurrent mutation; callers must not use the
// same backend instance concurrently without external synchronization.
unsafe impl Send for LlamaStageBackend {}
unsafe impl Sync for LlamaStageBackend {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GreedyTokenSample {
    pub token_id: i32,
    pub piece: String,
    pub is_eog: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GreedyCompletion {
    pub text: String,
    pub completion_tokens: u32,
    pub token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StageNodeRequest {
    Info,
    Tokenize {
        text: String,
    },
    BeginPrompt {
        request_id: String,
        prompt: String,
        max_tokens: Option<u32>,
    },
    ContinueHeadTokens {
        request_id: String,
        token_ids: Vec<i32>,
        max_tokens: Option<u32>,
    },
    ContinueForward {
        tensor: StageTensor,
    },
    ContinueForwardTokens {
        tensor: StageTensor,
        token_ids: Vec<i32>,
        clear_memory: bool,
    },
    SampleTail {
        tensor: StageTensor,
    },
    SampleTailToken {
        tensor: StageTensor,
    },
    ClearDecodeSession {
        request_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageNodeResponse {
    Info { info: StageNodeInfo },
    TokenIds { token_ids: Vec<i32> },
    Tensor { tensor: StageTensor },
    Sample { sample: StageSample },
    TokenSample { sample: GreedyTokenSample },
    Ack,
    Error { message: String },
}

#[derive(Debug, Clone)]
pub struct StageNodeConfig {
    pub model_path: PathBuf,
    pub stage_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub is_head: bool,
    pub is_tail: bool,
}

pub const LLAMA_STAGE_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageNodeInfo {
    pub protocol_version: u32,
    pub model_id: String,
    pub stage_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub is_head: bool,
    pub is_tail: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteStageTimings {
    pub head_prefill_ms: u64,
    pub head_decode_ms: u64,
    pub tail_decode_ms: u64,
    pub sample_ms: u64,
    pub transfer_bytes: usize,
    pub ttft_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteStageCompletion {
    pub text: String,
    pub completion_tokens: u32,
    pub token_ids: Vec<i32>,
    pub timings: RemoteStageTimings,
}

pub struct TcpStageClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

pub struct TcpGatewayClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayServiceInfo {
    pub protocol_version: u32,
    pub head_info: StageNodeInfo,
    pub tail_info: StageNodeInfo,
    pub reconnect_after_prompt: bool,
}

pub struct GatewayServiceClient {
    client: TcpGatewayClient,
    info: GatewayServiceInfo,
}

pub struct RemoteStageNodeClient {
    addr: String,
    client: Option<TcpStageClient>,
}

pub struct RemoteStagePair {
    head: RemoteStageNodeClient,
    tail: RemoteStageNodeClient,
    pub head_info: StageNodeInfo,
    pub tail_info: StageNodeInfo,
}

struct GatewaySessionState {
    max_tokens: u32,
    head_tensor: StageTensor,
    tail_tensor: StageTensor,
    text: String,
    token_ids: Vec<i32>,
    timings: RemoteStageTimings,
}

pub struct RemoteStageGateway {
    pair: RemoteStagePair,
    reconnect_after_prompt: bool,
    sessions: HashMap<String, GatewaySessionState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GatewayStep {
    Token {
        request_id: String,
        sample: GreedyTokenSample,
        text: String,
        token_ids: Vec<i32>,
    },
    Complete {
        request_id: String,
        completion: RemoteStageCompletion,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StageGatewayRequest {
    Info,
    Tokenize {
        text: String,
    },
    Complete {
        request_id: String,
        prompt: String,
        max_tokens: u32,
    },
    BeginCompletion {
        request_id: String,
        prompt: String,
        max_tokens: u32,
    },
    StepCompletion {
        request_id: String,
    },
    ClearCompletion {
        request_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageGatewayResponse {
    Info {
        protocol_version: u32,
        head_info: StageNodeInfo,
        tail_info: StageNodeInfo,
        reconnect_after_prompt: bool,
    },
    TokenIds {
        token_ids: Vec<i32>,
    },
    Completion {
        completion: RemoteStageCompletion,
    },
    Started {
        request_id: String,
    },
    Step {
        step: GatewayStep,
    },
    Ack,
    Error {
        message: String,
    },
}

impl GatewayServiceClient {
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_timeout(addr, None)
    }

    pub fn connect_with_timeout(addr: &str, timeout: Option<Duration>) -> Result<Self> {
        let mut client = TcpGatewayClient::connect_with_timeout(addr, timeout)?;
        let info = match client.request(&StageGatewayRequest::Info)? {
            StageGatewayResponse::Info {
                protocol_version,
                head_info,
                tail_info,
                reconnect_after_prompt,
            } => GatewayServiceInfo {
                protocol_version,
                head_info,
                tail_info,
                reconnect_after_prompt,
            },
            other => bail!("expected info response, got {other:?}"),
        };
        if info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
            bail!(
                "gateway protocol mismatch: expected {}, got {}",
                LLAMA_STAGE_PROTOCOL_VERSION,
                info.protocol_version
            );
        }
        if info.head_info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
            bail!(
                "head protocol mismatch: expected {}, got {}",
                LLAMA_STAGE_PROTOCOL_VERSION,
                info.head_info.protocol_version
            );
        }
        if info.tail_info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
            bail!(
                "tail protocol mismatch: expected {}, got {}",
                LLAMA_STAGE_PROTOCOL_VERSION,
                info.tail_info.protocol_version
            );
        }
        Ok(Self { client, info })
    }

    pub fn info(&self) -> &GatewayServiceInfo {
        &self.info
    }

    pub fn complete(
        &mut self,
        request_id: impl Into<String>,
        prompt: impl Into<String>,
        max_tokens: u32,
    ) -> Result<RemoteStageCompletion> {
        match self.client.request(&StageGatewayRequest::Complete {
            request_id: request_id.into(),
            prompt: prompt.into(),
            max_tokens,
        })? {
            StageGatewayResponse::Completion { completion } => Ok(completion),
            other => bail!("expected completion response, got {other:?}"),
        }
    }

    pub fn begin_completion(
        &mut self,
        request_id: impl Into<String>,
        prompt: impl Into<String>,
        max_tokens: u32,
    ) -> Result<String> {
        let request_id = request_id.into();
        match self.client.request(&StageGatewayRequest::BeginCompletion {
            request_id: request_id.clone(),
            prompt: prompt.into(),
            max_tokens,
        })? {
            StageGatewayResponse::Started { request_id } => Ok(request_id),
            other => bail!("expected started response, got {other:?}"),
        }
    }

    pub fn step_completion(&mut self, request_id: impl Into<String>) -> Result<GatewayStep> {
        match self.client.request(&StageGatewayRequest::StepCompletion {
            request_id: request_id.into(),
        })? {
            StageGatewayResponse::Step { step } => Ok(step),
            other => bail!("expected step response, got {other:?}"),
        }
    }

    pub fn clear_completion(&mut self, request_id: impl Into<String>) -> Result<()> {
        match self.client.request(&StageGatewayRequest::ClearCompletion {
            request_id: request_id.into(),
        })? {
            StageGatewayResponse::Ack => Ok(()),
            other => bail!("expected ack response, got {other:?}"),
        }
    }

    pub fn tokenize(&mut self, text: impl Into<String>) -> Result<Vec<i32>> {
        match self
            .client
            .request(&StageGatewayRequest::Tokenize { text: text.into() })?
        {
            StageGatewayResponse::TokenIds { token_ids } => Ok(token_ids),
            other => bail!("expected token_ids response, got {other:?}"),
        }
    }

    pub fn request(&mut self, request: &StageGatewayRequest) -> Result<StageGatewayResponse> {
        self.client.request(request)
    }
}

pub fn handle_gateway_service_client_request(
    client: &mut GatewayServiceClient,
    request: StageGatewayRequest,
) -> StageGatewayResponse {
    let result: Result<StageGatewayResponse> = (|| match request {
        StageGatewayRequest::Info => Ok(StageGatewayResponse::Info {
            protocol_version: client.info.protocol_version,
            head_info: client.info.head_info.clone(),
            tail_info: client.info.tail_info.clone(),
            reconnect_after_prompt: client.info.reconnect_after_prompt,
        }),
        StageGatewayRequest::Tokenize { text } => Ok(StageGatewayResponse::TokenIds {
            token_ids: client.tokenize(text)?,
        }),
        StageGatewayRequest::Complete {
            request_id,
            prompt,
            max_tokens,
        } => Ok(StageGatewayResponse::Completion {
            completion: client.complete(request_id, prompt, max_tokens)?,
        }),
        StageGatewayRequest::BeginCompletion {
            request_id,
            prompt,
            max_tokens,
        } => Ok(StageGatewayResponse::Started {
            request_id: client.begin_completion(request_id, prompt, max_tokens)?,
        }),
        StageGatewayRequest::StepCompletion { request_id } => Ok(StageGatewayResponse::Step {
            step: client.step_completion(request_id)?,
        }),
        StageGatewayRequest::ClearCompletion { request_id } => {
            client.clear_completion(request_id)?;
            Ok(StageGatewayResponse::Ack)
        }
    })();

    match result {
        Ok(response) => response,
        Err(err) => StageGatewayResponse::Error {
            message: err.to_string(),
        },
    }
}

impl LlamaStageBackend {
    pub fn new(model_path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            api: LlamaApi::load()?,
            model_path: model_path.into(),
            state: RefCell::new(BackendState {
                layout: None,
                model: None,
                sessions: HashMap::new(),
            }),
        })
    }

    fn debug_flow_enabled() -> bool {
        std::env::var_os("LLAMA_STAGE_DEBUG_FLOW").is_some()
    }

    fn layout<'a>(&self, state: &'a BackendState) -> Result<&'a StageLayout> {
        state
            .layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no stage layout loaded"))
    }

    fn ensure_model<'a>(&'a self, state: &'a mut BackendState) -> Result<&'a LlamaModelHandle> {
        if state.model.is_none() {
            state.model = Some(LlamaModelHandle::new(&self.api, &self.model_path)?);
        }
        Ok(state.model.as_ref().expect("model initialized"))
    }

    fn tokenize_prompt(&self, model: &LlamaModelHandle, prompt: &str) -> Result<Vec<i32>> {
        let vocab = model.vocab(&self.api);
        if vocab.is_null() {
            bail!("model vocabulary is not available");
        }

        let prompt = CString::new(prompt)?;
        let mut tokens = vec![0i32; prompt.as_bytes().len() + 256];
        let mut n = unsafe {
            (self.api.tokenize)(
                vocab,
                prompt.as_ptr(),
                prompt.as_bytes().len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
            )
        };

        if n < 0 {
            let need = (-n) as usize;
            tokens.resize(need, 0);
            n = unsafe {
                (self.api.tokenize)(
                    vocab,
                    prompt.as_ptr(),
                    prompt.as_bytes().len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    true,
                    true,
                )
            };
        }

        if n < 0 {
            bail!("tokenization failed for prompt");
        }

        tokens.truncate(n as usize);
        Ok(tokens)
    }

    fn tokenize_text(&self, prompt: &str) -> Result<Vec<i32>> {
        let mut state = self.state.borrow_mut();
        let model = self.ensure_model(&mut state)?;
        self.tokenize_prompt(model, prompt)
    }

    fn ensure_session<'a>(
        &'a self,
        state: &'a mut BackendState,
        request_id: &str,
    ) -> Result<&'a mut SessionState> {
        if state.model.is_none() {
            state.model = Some(LlamaModelHandle::new(&self.api, &self.model_path)?);
        }
        if !state.sessions.contains_key(request_id) {
            let session = state
                .model
                .as_ref()
                .expect("model initialized")
                .create_session(&self.api)?;
            state.sessions.insert(
                request_id.to_string(),
                SessionState {
                    session,
                    cached_sample: None,
                },
            );
        }
        Ok(state
            .sessions
            .get_mut(request_id)
            .expect("session initialized"))
    }

    fn clear_decode_session_inner(state: &mut BackendState, api: &LlamaApi, request_id: &str) {
        if let Some(session) = state.sessions.remove(request_id) {
            session.session.destroy(api);
        }
    }

    fn embeddings_to_tensor(
        &self,
        session: &LlamaSession,
        hidden_dim: usize,
        request_id: &str,
        prompt_text: Option<String>,
        stage_trace: Vec<String>,
        max_tokens: Option<u32>,
        token_count: usize,
    ) -> Result<StageTensor> {
        let ptr = unsafe { (self.api.get_embeddings)(session.ctx) };
        if ptr.is_null() {
            bail!("llama_get_embeddings returned null");
        }

        let floats = unsafe { slice::from_raw_parts(ptr, token_count * hidden_dim) };
        let mut bytes = Vec::with_capacity(floats.len() * std::mem::size_of::<f32>());
        for value in floats {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim,
            bytes,
            prompt_text,
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn greedy_sample(
        &self,
        model: *mut ffi::LlamaModel,
        session: &LlamaSession,
        request_id: &str,
        model_id: &str,
    ) -> Result<CachedSample> {
        let vocab = unsafe { (self.api.model_get_vocab)(model) };
        let logits = unsafe { (self.api.get_logits_ith)(session.ctx, -1) };
        if logits.is_null() {
            bail!("no logits available for sampling");
        }

        let n_vocab = unsafe { (self.api.vocab_n_tokens)(vocab) as usize };
        let logits = unsafe { slice::from_raw_parts(logits, n_vocab) };
        let (token_id, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|lhs, rhs| {
                lhs.1
                    .partial_cmp(&rhs.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .context("empty logits buffer")?;

        let token_id = token_id as i32;
        let text = self.token_to_piece(vocab, token_id)?;
        let is_eog = unsafe { (self.api.vocab_is_eog)(vocab, token_id) };
        Ok(CachedSample {
            sample: StageSample {
                request_id: request_id.to_string(),
                model_id: model_id.to_string(),
                text,
                token_ids: vec![token_id as u32],
                completion_tokens: 1,
            },
            token_id,
            is_eog,
        })
    }

    fn token_to_piece(&self, vocab: *const ffi::LlamaVocab, token: i32) -> Result<String> {
        let mut buf = vec![0i8; 256];
        let mut n = unsafe {
            (self.api.token_to_piece)(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
        };

        if n < 0 {
            let need = (-n) as usize;
            buf.resize(need, 0);
            n = unsafe {
                (self.api.token_to_piece)(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
            };
        }

        if n < 0 {
            bail!("token_to_piece failed for token {}", token);
        }

        let bytes = buf[..n as usize]
            .iter()
            .map(|b| *b as u8)
            .collect::<Vec<u8>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn hidden_bytes_to_f32(input: &StageTensor) -> Result<Vec<f32>> {
        if input.hidden_dim == 0 {
            bail!("hidden_dim must be non-zero");
        }
        if input.bytes.len() % 4 != 0 {
            bail!("hidden-state bytes must be a multiple of 4");
        }
        let float_count = input.bytes.len() / 4;
        if float_count % input.hidden_dim != 0 {
            bail!(
                "hidden-state float count {} is not divisible by hidden_dim {}",
                float_count,
                input.hidden_dim
            );
        }
        Ok(input
            .bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn forward_head_tokens_impl(
        &self,
        request_id: &str,
        token_ids: Vec<i32>,
        prompt_text: Option<String>,
        max_tokens: Option<u32>,
        clear_memory: bool,
    ) -> Result<StageTensor> {
        let mut state = self.state.borrow_mut();
        let layout = self.layout(&state)?.clone();
        if !layout.is_head {
            bail!(
                "head token ingress called on non-head stage {}",
                layout.stage_id
            );
        }

        let hidden_dim = self.ensure_model(&mut state)?.hidden_dim(&self.api);
        let session_state = self.ensure_session(&mut state, request_id)?;
        if clear_memory {
            session_state.cached_sample = None;
            session_state.session.clear_memory(&self.api);
        }

        let batch = OwnedBatch::token_only(token_ids.clone());
        let rc = unsafe {
            (self.api.decode_head)(
                session_state.session.ctx,
                batch.raw,
                layout.end_layer as i32,
            )
        };
        if rc != 0 && rc != 1 {
            bail!("llama_decode_head failed with {}", rc);
        }

        self.embeddings_to_tensor(
            &session_state.session,
            hidden_dim,
            request_id,
            prompt_text,
            vec![layout.stage_id],
            max_tokens,
            token_ids.len(),
        )
    }

    fn continue_forward_impl(
        &self,
        input: StageTensor,
        token_ids: Option<Vec<i32>>,
        clear_memory: bool,
    ) -> Result<StageTensor> {
        let mut state = self.state.borrow_mut();
        let layout = self.layout(&state)?.clone();
        let hidden_dim = self.ensure_model(&mut state)?.hidden_dim(&self.api);
        let model_ptr = self.ensure_model(&mut state)?.model;

        let hidden = Self::hidden_bytes_to_f32(&input)?;
        let token_count = hidden.len() / input.hidden_dim;
        if token_count == 0 {
            bail!("empty hidden-state payload");
        }

        let tokens = if let Some(token_ids) = token_ids {
            Some(token_ids)
        } else if let Some(prompt) = input.prompt_text.as_ref() {
            let model = self.ensure_model(&mut state)?;
            Some(self.tokenize_prompt(model, prompt)?)
        } else {
            None
        };

        if layout.start_layer > 0 && tokens.is_none() {
            bail!("downstream llama stage requires token ids or prompt_text");
        }

        if let Some(tokens) = tokens.as_ref() {
            if tokens.len() != token_count {
                bail!(
                    "token count {} does not match hidden token count {}",
                    tokens.len(),
                    token_count
                );
            }
        }

        let batch = OwnedBatch::token_and_hidden(tokens.clone(), hidden, token_count);
        let request_id = input.request_id.clone();
        let session_state = self.ensure_session(&mut state, &request_id)?;
        if clear_memory {
            session_state.cached_sample = None;
            session_state.session.clear_memory(&self.api);
        }

        if layout.is_tail {
            let rc = unsafe {
                (self.api.decode_tail)(
                    session_state.session.ctx,
                    batch.raw,
                    layout.start_layer as i32,
                )
            };
            if rc != 0 && rc != 1 {
                bail!("llama_decode_tail failed with {}", rc);
            }
            session_state.cached_sample = Some(self.greedy_sample(
                model_ptr,
                &session_state.session,
                &request_id,
                &layout.model_id,
            )?);

            let mut stage_trace = input.stage_trace.clone();
            stage_trace.push(layout.stage_id);
            return Ok(StageTensor {
                request_id,
                kind: PayloadKind::HiddenState,
                stage_trace,
                hidden_dim: input.hidden_dim,
                bytes: input.bytes,
                prompt_text: input.prompt_text,
                max_tokens: input.max_tokens,
                continuation: input.continuation,
                transient: input.transient,
                carry: input.carry,
            });
        }

        let rc = unsafe {
            (self.api.decode_middle)(
                session_state.session.ctx,
                batch.raw,
                layout.start_layer as i32,
                layout.end_layer as i32,
            )
        };
        if rc != 0 && rc != 1 {
            bail!("llama_decode_middle failed with {}", rc);
        }

        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());
        self.embeddings_to_tensor(
            &session_state.session,
            hidden_dim,
            &request_id,
            input.prompt_text,
            stage_trace,
            input.max_tokens,
            token_count,
        )
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        self.tokenize_text(prompt)
    }

    pub fn decode_token_ids(&self, tokens: &[u32]) -> Result<String> {
        let mut state = self.state.borrow_mut();
        let model = self.ensure_model(&mut state)?;
        let vocab = unsafe { (self.api.model_get_vocab)(model.model) };
        let mut text = String::new();
        for &token in tokens {
            let token = i32::try_from(token).context("token id exceeded i32 range")?;
            text.push_str(&self.token_to_piece(vocab, token)?);
        }
        Ok(text)
    }

    pub fn eos_token_id(&self) -> Result<Option<u32>> {
        let mut state = self.state.borrow_mut();
        let model = self.ensure_model(&mut state)?;
        let vocab = unsafe { (self.api.model_get_vocab)(model.model) };
        let n_vocab = unsafe { (self.api.vocab_n_tokens)(vocab) as usize };
        for token in 0..n_vocab {
            let token = token as i32;
            if unsafe { (self.api.vocab_is_eog)(vocab, token) } {
                return Ok(Some(token as u32));
            }
        }
        Ok(None)
    }

    pub fn clear_decode_session(&self, request_id: &str) -> Result<()> {
        let mut state = self.state.borrow_mut();
        Self::clear_decode_session_inner(&mut state, &self.api, request_id);
        Ok(())
    }

    pub fn node_info(&self) -> Result<StageNodeInfo> {
        let state = self.state.borrow();
        let layout = self.layout(&state)?;
        Ok(StageNodeInfo {
            protocol_version: LLAMA_STAGE_PROTOCOL_VERSION,
            model_id: layout.model_id.clone(),
            stage_id: layout.stage_id.clone(),
            start_layer: layout.start_layer,
            end_layer: layout.end_layer,
            is_head: layout.is_head,
            is_tail: layout.is_tail,
        })
    }

    pub fn begin_prompt_session(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.clear_decode_session(request_id)?;
        let token_ids = self.tokenize(prompt)?;
        self.forward_head_tokens_impl(
            request_id,
            token_ids,
            Some(prompt.to_string()),
            max_tokens,
            true,
        )
    }

    pub fn continue_head_tokens(
        &self,
        request_id: &str,
        token_ids: Vec<i32>,
        max_tokens: Option<u32>,
    ) -> Result<StageTensor> {
        self.forward_head_tokens_impl(request_id, token_ids, None, max_tokens, false)
    }

    pub fn continue_forward_with_tokens(
        &self,
        input: StageTensor,
        token_ids: Vec<i32>,
        clear_memory: bool,
    ) -> Result<StageTensor> {
        self.continue_forward_impl(input, Some(token_ids), clear_memory)
    }

    pub fn sample_tail_token(&self, input: StageTensor) -> Result<GreedyTokenSample> {
        let sample = self.sample_tail(input)?;
        let state = self.state.borrow();
        let cached = state
            .sessions
            .get(&sample.request_id)
            .and_then(|session| session.cached_sample.as_ref())
            .context("no cached tail token; call continue_forward first")?;
        Ok(GreedyTokenSample {
            token_id: cached.token_id,
            piece: sample.text,
            is_eog: cached.is_eog,
        })
    }
}

pub fn build_stage_backend(config: &StageNodeConfig) -> Result<LlamaStageBackend> {
    let mut backend = LlamaStageBackend::new(&config.model_path)?;
    backend.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: config.stage_id.clone(),
        start_layer: config.start_layer,
        end_layer: config.end_layer,
        is_head: config.is_head,
        is_tail: config.is_tail,
    })?;
    Ok(backend)
}

impl TcpStageClient {
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_timeout(addr, None)
    }

    pub fn connect_with_timeout(addr: &str, timeout: Option<Duration>) -> Result<Self> {
        let stream = connect_tcp_stream(addr, timeout)?;
        stream.set_nodelay(true)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }

    pub fn request(&mut self, request: &StageNodeRequest) -> Result<StageNodeResponse> {
        serde_json::to_writer(&mut self.stream, request)?;
        self.stream.write_all(b"\n")?;
        self.stream.flush()?;

        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            bail!("tcp stage returned empty response");
        }

        let response: StageNodeResponse = serde_json::from_str(line.trim())?;
        if let StageNodeResponse::Error { message } = &response {
            bail!("tcp stage error: {message}");
        }
        Ok(response)
    }
}

impl TcpGatewayClient {
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_timeout(addr, None)
    }

    pub fn connect_with_timeout(addr: &str, timeout: Option<Duration>) -> Result<Self> {
        let stream = connect_tcp_stream(addr, timeout)?;
        stream.set_nodelay(true)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }

    pub fn request(&mut self, request: &StageGatewayRequest) -> Result<StageGatewayResponse> {
        serde_json::to_writer(&mut self.stream, request)?;
        self.stream.write_all(b"\n")?;
        self.stream.flush()?;

        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            bail!("tcp gateway returned empty response");
        }

        let response: StageGatewayResponse = serde_json::from_str(line.trim())?;
        if let StageGatewayResponse::Error { message } = &response {
            bail!("tcp gateway error: {message}");
        }
        Ok(response)
    }
}

fn connect_tcp_stream(addr: &str, timeout: Option<Duration>) -> Result<TcpStream> {
    match timeout {
        Some(timeout) => {
            let socket_addr = addr
                .to_socket_addrs()
                .with_context(|| format!("resolving {addr}"))?
                .next()
                .with_context(|| format!("no socket addresses resolved for {addr}"))?;
            TcpStream::connect_timeout(&socket_addr, timeout).with_context(|| {
                format!(
                    "connecting to {addr} with timeout {}ms",
                    timeout.as_millis()
                )
            })
        }
        None => TcpStream::connect(addr).with_context(|| format!("connecting to {addr}")),
    }
}

impl RemoteStageNodeClient {
    pub fn connect(addr: impl Into<String>) -> Result<Self> {
        let addr = addr.into();
        let client = TcpStageClient::connect(&addr)?;
        Ok(Self {
            addr,
            client: Some(client),
        })
    }

    fn ensure_client(&mut self) -> Result<&mut TcpStageClient> {
        if self.client.is_none() {
            self.client = Some(TcpStageClient::connect(&self.addr)?);
        }
        Ok(self.client.as_mut().expect("client initialized"))
    }

    pub fn reconnect(&mut self) -> Result<()> {
        self.client = Some(TcpStageClient::connect(&self.addr)?);
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.client = None;
    }

    pub fn request(&mut self, request: &StageNodeRequest) -> Result<StageNodeResponse> {
        let first_try = self.ensure_client()?.request(request);
        match first_try {
            Ok(response) => Ok(response),
            Err(_) => {
                self.reconnect()?;
                self.ensure_client()?.request(request)
            }
        }
    }

    pub fn info(&mut self) -> Result<StageNodeInfo> {
        match self.request(&StageNodeRequest::Info)? {
            StageNodeResponse::Info { info } => {
                if info.protocol_version != LLAMA_STAGE_PROTOCOL_VERSION {
                    bail!(
                        "stage protocol mismatch at {}: expected {}, got {}",
                        self.addr,
                        LLAMA_STAGE_PROTOCOL_VERSION,
                        info.protocol_version
                    );
                }
                Ok(info)
            }
            other => bail!("expected info response, got {other:?}"),
        }
    }
}

impl RemoteStagePair {
    pub fn connect(head_addr: impl Into<String>, tail_addr: impl Into<String>) -> Result<Self> {
        let mut head = RemoteStageNodeClient::connect(head_addr)?;
        let mut tail = RemoteStageNodeClient::connect(tail_addr)?;
        let head_info = head.info()?;
        let tail_info = tail.info()?;

        if !head_info.is_head {
            bail!("head endpoint {} is not marked as a head stage", head.addr);
        }
        if !tail_info.is_tail {
            bail!("tail endpoint {} is not marked as a tail stage", tail.addr);
        }
        if head_info.model_id != tail_info.model_id {
            bail!(
                "head model {} does not match tail model {}",
                head_info.model_id,
                tail_info.model_id
            );
        }
        if head_info.end_layer + 1 != tail_info.start_layer {
            bail!(
                "head/tail layer ranges are not contiguous: {}-{} then {}-{}",
                head_info.start_layer,
                head_info.end_layer,
                tail_info.start_layer,
                tail_info.end_layer
            );
        }

        Ok(Self {
            head,
            tail,
            head_info,
            tail_info,
        })
    }

    pub fn reconnect(&mut self) -> Result<()> {
        self.head.reconnect()?;
        self.tail.reconnect()?;
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.head.disconnect();
        self.tail.disconnect();
    }

    pub fn clear_decode_session(&mut self, request_id: &str) -> Result<()> {
        self.head.request(&StageNodeRequest::ClearDecodeSession {
            request_id: request_id.to_string(),
        })?;
        self.tail.request(&StageNodeRequest::ClearDecodeSession {
            request_id: request_id.to_string(),
        })?;
        Ok(())
    }

    pub fn run_greedy_completion(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        reconnect_after_prompt: bool,
    ) -> Result<RemoteStageCompletion> {
        let request_id = format!(
            "remote-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        self.clear_decode_session(&request_id)?;

        let prompt_tokens = match self.head.request(&StageNodeRequest::Tokenize {
            text: prompt.to_string(),
        })? {
            StageNodeResponse::TokenIds { token_ids } => token_ids,
            other => bail!("expected token_ids response, got {other:?}"),
        };

        let t_head = std::time::Instant::now();
        let mut head_tensor = match self.head.request(&StageNodeRequest::BeginPrompt {
            request_id: request_id.clone(),
            prompt: prompt.to_string(),
            max_tokens: Some(max_tokens),
        })? {
            StageNodeResponse::Tensor { tensor } => tensor,
            other => bail!("expected tensor response, got {other:?}"),
        };
        let head_prefill_ms = t_head.elapsed().as_millis() as u64;
        let transfer_bytes = head_tensor.bytes.len();

        let t_tail = std::time::Instant::now();
        let mut tail_tensor = match self
            .tail
            .request(&StageNodeRequest::ContinueForwardTokens {
                tensor: head_tensor.clone(),
                token_ids: prompt_tokens,
                clear_memory: true,
            })? {
            StageNodeResponse::Tensor { tensor } => tensor,
            other => bail!("expected tensor response, got {other:?}"),
        };
        let mut tail_decode_ms_total = t_tail.elapsed().as_millis() as u64;

        if reconnect_after_prompt {
            self.disconnect();
            self.reconnect()?;
        }

        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut head_decode_ms_total = 0u64;
        let mut sample_ms_total = 0u64;

        for step in 0..max_tokens {
            let t_sample = std::time::Instant::now();
            let sampled = match self.tail.request(&StageNodeRequest::SampleTailToken {
                tensor: tail_tensor.clone(),
            })? {
                StageNodeResponse::TokenSample { sample } => sample,
                other => bail!("expected token_sample response, got {other:?}"),
            };
            sample_ms_total += t_sample.elapsed().as_millis() as u64;

            text.push_str(&sampled.piece);
            token_ids.push(sampled.token_id);

            if sampled.is_eog || step + 1 >= max_tokens {
                break;
            }

            let t_head_step = std::time::Instant::now();
            head_tensor = match self.head.request(&StageNodeRequest::ContinueHeadTokens {
                request_id: request_id.clone(),
                token_ids: vec![sampled.token_id],
                max_tokens: Some(max_tokens),
            })? {
                StageNodeResponse::Tensor { tensor } => tensor,
                other => bail!("expected tensor response, got {other:?}"),
            };
            head_decode_ms_total += t_head_step.elapsed().as_millis() as u64;

            let t_tail_step = std::time::Instant::now();
            tail_tensor = match self
                .tail
                .request(&StageNodeRequest::ContinueForwardTokens {
                    tensor: head_tensor.clone(),
                    token_ids: vec![sampled.token_id],
                    clear_memory: false,
                })? {
                StageNodeResponse::Tensor { tensor } => tensor,
                other => bail!("expected tensor response, got {other:?}"),
            };
            tail_decode_ms_total += t_tail_step.elapsed().as_millis() as u64;
        }

        let ttft_ms = head_prefill_ms + tail_decode_ms_total + sample_ms_total;
        let total_ms =
            head_prefill_ms + head_decode_ms_total + tail_decode_ms_total + sample_ms_total;

        let result = RemoteStageCompletion {
            text,
            completion_tokens: token_ids.len() as u32,
            token_ids,
            timings: RemoteStageTimings {
                head_prefill_ms,
                head_decode_ms: head_decode_ms_total,
                tail_decode_ms: tail_decode_ms_total,
                sample_ms: sample_ms_total,
                transfer_bytes,
                ttft_ms,
                total_ms,
            },
        };

        self.clear_decode_session(&request_id)?;
        Ok(result)
    }
}

impl RemoteStageGateway {
    pub fn connect(
        head_addr: impl Into<String>,
        tail_addr: impl Into<String>,
        reconnect_after_prompt: bool,
    ) -> Result<Self> {
        Ok(Self {
            pair: RemoteStagePair::connect(head_addr, tail_addr)?,
            reconnect_after_prompt,
            sessions: HashMap::new(),
        })
    }

    pub fn head_info(&self) -> &StageNodeInfo {
        &self.pair.head_info
    }

    pub fn tail_info(&self) -> &StageNodeInfo {
        &self.pair.tail_info
    }

    pub fn reconnect_after_prompt(&self) -> bool {
        self.reconnect_after_prompt
    }

    pub fn clear_completion(&mut self, request_id: &str) -> Result<()> {
        self.sessions.remove(request_id);
        self.pair.clear_decode_session(request_id)
    }

    pub fn begin_completion(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<()> {
        if self.sessions.contains_key(request_id) {
            bail!("completion session already exists for request_id {request_id}");
        }

        self.pair.clear_decode_session(request_id)?;

        let prompt_tokens = match self.pair.head.request(&StageNodeRequest::Tokenize {
            text: prompt.to_string(),
        })? {
            StageNodeResponse::TokenIds { token_ids } => token_ids,
            other => bail!("expected token_ids response, got {other:?}"),
        };

        let t_head = std::time::Instant::now();
        let head_tensor = match self.pair.head.request(&StageNodeRequest::BeginPrompt {
            request_id: request_id.to_string(),
            prompt: prompt.to_string(),
            max_tokens: Some(max_tokens),
        })? {
            StageNodeResponse::Tensor { tensor } => tensor,
            other => bail!("expected tensor response, got {other:?}"),
        };
        let head_prefill_ms = t_head.elapsed().as_millis() as u64;
        let transfer_bytes = head_tensor.bytes.len();

        let t_tail = std::time::Instant::now();
        let tail_tensor =
            match self
                .pair
                .tail
                .request(&StageNodeRequest::ContinueForwardTokens {
                    tensor: head_tensor.clone(),
                    token_ids: prompt_tokens,
                    clear_memory: true,
                })? {
                StageNodeResponse::Tensor { tensor } => tensor,
                other => bail!("expected tensor response, got {other:?}"),
            };
        let tail_decode_ms = t_tail.elapsed().as_millis() as u64;

        if self.reconnect_after_prompt {
            self.pair.disconnect();
            self.pair.reconnect()?;
        }

        self.sessions.insert(
            request_id.to_string(),
            GatewaySessionState {
                max_tokens,
                head_tensor,
                tail_tensor,
                text: String::new(),
                token_ids: Vec::new(),
                timings: RemoteStageTimings {
                    head_prefill_ms,
                    head_decode_ms: 0,
                    tail_decode_ms,
                    sample_ms: 0,
                    transfer_bytes,
                    ttft_ms: 0,
                    total_ms: 0,
                },
            },
        );

        Ok(())
    }

    pub fn step_completion(&mut self, request_id: &str) -> Result<GatewayStep> {
        let mut session = self
            .sessions
            .remove(request_id)
            .with_context(|| format!("no completion session for request_id {request_id}"))?;

        let t_sample = std::time::Instant::now();
        let sampled = match self.pair.tail.request(&StageNodeRequest::SampleTailToken {
            tensor: session.tail_tensor.clone(),
        })? {
            StageNodeResponse::TokenSample { sample } => sample,
            other => bail!("expected token_sample response, got {other:?}"),
        };
        session.timings.sample_ms += t_sample.elapsed().as_millis() as u64;

        session.text.push_str(&sampled.piece);
        session.token_ids.push(sampled.token_id);

        let done = sampled.is_eog || session.token_ids.len() as u32 >= session.max_tokens;
        if done {
            session.timings.ttft_ms = session.timings.head_prefill_ms
                + session.timings.tail_decode_ms
                + session.timings.sample_ms;
            session.timings.total_ms = session.timings.head_prefill_ms
                + session.timings.head_decode_ms
                + session.timings.tail_decode_ms
                + session.timings.sample_ms;
            let completion = RemoteStageCompletion {
                text: session.text,
                completion_tokens: session.token_ids.len() as u32,
                token_ids: session.token_ids,
                timings: session.timings,
            };
            self.pair.clear_decode_session(request_id)?;
            return Ok(GatewayStep::Complete {
                request_id: request_id.to_string(),
                completion,
            });
        }

        let t_head = std::time::Instant::now();
        session.head_tensor =
            match self
                .pair
                .head
                .request(&StageNodeRequest::ContinueHeadTokens {
                    request_id: request_id.to_string(),
                    token_ids: vec![sampled.token_id],
                    max_tokens: Some(session.max_tokens),
                })? {
                StageNodeResponse::Tensor { tensor } => tensor,
                other => bail!("expected tensor response, got {other:?}"),
            };
        session.timings.head_decode_ms += t_head.elapsed().as_millis() as u64;

        let t_tail = std::time::Instant::now();
        session.tail_tensor =
            match self
                .pair
                .tail
                .request(&StageNodeRequest::ContinueForwardTokens {
                    tensor: session.head_tensor.clone(),
                    token_ids: vec![sampled.token_id],
                    clear_memory: false,
                })? {
                StageNodeResponse::Tensor { tensor } => tensor,
                other => bail!("expected tensor response, got {other:?}"),
            };
        session.timings.tail_decode_ms += t_tail.elapsed().as_millis() as u64;

        let text = session.text.clone();
        let token_ids = session.token_ids.clone();
        self.sessions.insert(request_id.to_string(), session);

        Ok(GatewayStep::Token {
            request_id: request_id.to_string(),
            sample: sampled,
            text,
            token_ids,
        })
    }

    pub fn complete(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<RemoteStageCompletion> {
        self.begin_completion(request_id, prompt, max_tokens)?;
        loop {
            match self.step_completion(request_id)? {
                GatewayStep::Token { .. } => continue,
                GatewayStep::Complete { completion, .. } => return Ok(completion),
            }
        }
    }
}

pub fn handle_stage_gateway_request(
    gateway: &mut RemoteStageGateway,
    request: StageGatewayRequest,
) -> StageGatewayResponse {
    let result: Result<StageGatewayResponse> = (|| match request {
        StageGatewayRequest::Info => Ok(StageGatewayResponse::Info {
            protocol_version: LLAMA_STAGE_PROTOCOL_VERSION,
            head_info: gateway.head_info().clone(),
            tail_info: gateway.tail_info().clone(),
            reconnect_after_prompt: gateway.reconnect_after_prompt(),
        }),
        StageGatewayRequest::Tokenize { text } => Ok(StageGatewayResponse::TokenIds {
            token_ids: gateway
                .pair
                .head
                .request(&StageNodeRequest::Tokenize { text })
                .and_then(|response| match response {
                    StageNodeResponse::TokenIds { token_ids } => Ok(token_ids),
                    other => bail!("expected token_ids response, got {other:?}"),
                })?,
        }),
        StageGatewayRequest::Complete {
            request_id,
            prompt,
            max_tokens,
        } => Ok(StageGatewayResponse::Completion {
            completion: gateway.complete(&request_id, &prompt, max_tokens)?,
        }),
        StageGatewayRequest::BeginCompletion {
            request_id,
            prompt,
            max_tokens,
        } => {
            gateway.begin_completion(&request_id, &prompt, max_tokens)?;
            Ok(StageGatewayResponse::Started { request_id })
        }
        StageGatewayRequest::StepCompletion { request_id } => Ok(StageGatewayResponse::Step {
            step: gateway.step_completion(&request_id)?,
        }),
        StageGatewayRequest::ClearCompletion { request_id } => {
            gateway.clear_completion(&request_id)?;
            Ok(StageGatewayResponse::Ack)
        }
    })();

    match result {
        Ok(response) => response,
        Err(err) => StageGatewayResponse::Error {
            message: err.to_string(),
        },
    }
}

pub fn handle_stage_node_request(
    backend: &LlamaStageBackend,
    request: StageNodeRequest,
) -> StageNodeResponse {
    let result: Result<StageNodeResponse> = (|| match request {
        StageNodeRequest::Info => Ok(StageNodeResponse::Info {
            info: backend.node_info()?,
        }),
        StageNodeRequest::Tokenize { text } => Ok(StageNodeResponse::TokenIds {
            token_ids: backend.tokenize(&text)?,
        }),
        StageNodeRequest::BeginPrompt {
            request_id,
            prompt,
            max_tokens,
        } => Ok(StageNodeResponse::Tensor {
            tensor: backend.begin_prompt_session(&request_id, &prompt, max_tokens)?,
        }),
        StageNodeRequest::ContinueHeadTokens {
            request_id,
            token_ids,
            max_tokens,
        } => Ok(StageNodeResponse::Tensor {
            tensor: backend.continue_head_tokens(&request_id, token_ids, max_tokens)?,
        }),
        StageNodeRequest::ContinueForward { tensor } => Ok(StageNodeResponse::Tensor {
            tensor: backend.continue_forward(tensor)?,
        }),
        StageNodeRequest::ContinueForwardTokens {
            tensor,
            token_ids,
            clear_memory,
        } => Ok(StageNodeResponse::Tensor {
            tensor: backend.continue_forward_with_tokens(tensor, token_ids, clear_memory)?,
        }),
        StageNodeRequest::SampleTail { tensor } => Ok(StageNodeResponse::Sample {
            sample: backend.sample_tail(tensor)?,
        }),
        StageNodeRequest::SampleTailToken { tensor } => Ok(StageNodeResponse::TokenSample {
            sample: backend.sample_tail_token(tensor)?,
        }),
        StageNodeRequest::ClearDecodeSession { request_id } => {
            backend.clear_decode_session(&request_id)?;
            Ok(StageNodeResponse::Ack)
        }
    })();

    match result {
        Ok(response) => response,
        Err(err) => StageNodeResponse::Error {
            message: err.to_string(),
        },
    }
}

pub fn default_gemma_model_path() -> PathBuf {
    if let Some(home) = env::var_os("HOME") {
        let cached = PathBuf::from(home).join(".compute/models/gemma-4-E4B-it-Q4_K_M.gguf");
        if cached.exists() {
            return cached;
        }
    }

    PathBuf::from("models/gemma-4-E4B-it-Q4_K_M.gguf")
}

pub fn default_compute_bin_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| PathBuf::from(home).join(".compute/bin"))
}

fn dir_has_libllama(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if !lower.contains("llama") {
            continue;
        }
        let is_dylib = lower.ends_with(".dylib")
            || lower.ends_with(".so")
            || lower.ends_with(".dll")
            || lower.contains(".so.");
        if is_dylib {
            return true;
        }
    }
    false
}

fn resolve_vendor_lib_dir() -> Result<PathBuf> {
    if let Some(override_dir) = env::var_os("LLAMA_STAGE_VENDOR_LIB_DIR") {
        let path = PathBuf::from(override_dir);
        if dir_has_libllama(&path) {
            return Ok(path);
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            for candidate in [
                parent.to_path_buf(),
                parent.join("lib"),
                parent.join("../lib"),
            ] {
                if dir_has_libllama(&candidate) {
                    return Ok(candidate);
                }
            }
        }
    }

    if let Some(bin_dir) = default_compute_bin_dir() {
        for candidate in [bin_dir.clone(), bin_dir.join("lib")] {
            if dir_has_libllama(&candidate) {
                return Ok(candidate);
            }
        }
    }

    let baked = PathBuf::from(env!("LLAMA_STAGE_VENDOR_LIB_DIR"));
    if dir_has_libllama(&baked) {
        return Ok(baked);
    }

    bail!(
        "could not locate libllama dylib; checked LLAMA_STAGE_VENDOR_LIB_DIR, executable dir, ~/.compute/bin, and the build-time path {}",
        baked.display()
    );
}

pub fn resolve_model_arg(args: &[String]) -> (PathBuf, usize) {
    if let Some(candidate) = args.get(1) {
        let path = PathBuf::from(candidate);
        let looks_like_gguf = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if looks_like_gguf {
            return (path, 2);
        }
    }

    (default_gemma_model_path(), 1)
}

fn compute_workspace_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .context("failed to resolve compute-app workspace root")
}

fn resolve_managed_binary_path(explicit: Option<&Path>, name: &str) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        bail!("configured binary path does not exist: {}", path.display());
    }

    let file_name = format!("{name}{EXE_SUFFIX}");
    if let Some(bin_dir) = default_compute_bin_dir() {
        let installed = bin_dir.join(&file_name);
        if installed.exists() {
            return Ok(installed);
        }
    }

    let root = compute_workspace_root()?;
    for profile in ["debug", "release"] {
        let candidate = root.join("target").join(profile).join(&file_name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    bail!(
        "could not find {name} binary in ~/.compute/bin or under {}/target/{{debug,release}}; install sidecars into ~/.compute/bin, build compute-app with `cargo build -p llama-stage-backend --bins`, or configure an explicit binary path",
        root.display()
    )
}

fn read_listening_addr(stderr: ChildStderr) -> Result<String> {
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            bail!("child exited before announcing listening address");
        }
        let trimmed = line.trim();
        if let Some(addr) = trimmed.strip_prefix("listening=") {
            return Ok(addr.to_string());
        }
    }
}

struct ManagedServiceChild {
    child: Child,
    addr: String,
}

impl ManagedServiceChild {
    fn spawn_stage_from_bin(
        bin_path: &Path,
        model_path: &Path,
        bind_addr: &str,
        stage_id: &str,
        start_layer: u32,
        end_layer: u32,
        is_head: bool,
        is_tail: bool,
    ) -> Result<Self> {
        let mut command = Command::new(bin_path);
        command
            .arg("--model")
            .arg(model_path)
            .arg("--bind")
            .arg(bind_addr)
            .arg("--stage-id")
            .arg(stage_id)
            .arg("--start-layer")
            .arg(start_layer.to_string())
            .arg("--end-layer")
            .arg(end_layer.to_string());
        if is_head {
            command.arg("--head");
        }
        if is_tail {
            command.arg("--tail");
        }

        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning tcp stage node from {}", bin_path.display()))?;
        let stderr = child.stderr.take().context("missing child stderr")?;
        let addr = read_listening_addr(stderr)?;
        Ok(Self { child, addr })
    }

    fn spawn_gateway_from_bin(
        bin_path: &Path,
        bind_addr: &str,
        head_addr: &str,
        tail_addr: &str,
        reconnect_after_prompt: bool,
    ) -> Result<Self> {
        let mut command = Command::new(bin_path);
        command
            .arg("--bind")
            .arg(bind_addr)
            .arg("--head")
            .arg(head_addr)
            .arg("--tail")
            .arg(tail_addr);
        if reconnect_after_prompt {
            command.arg("--reconnect-after-prompt");
        }

        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning tcp gateway node from {}", bin_path.display()))?;
        let stderr = child.stderr.take().context("missing gateway stderr")?;
        let addr = read_listening_addr(stderr)?;
        Ok(Self { child, addr })
    }
}

impl Drop for ManagedServiceChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct ManagedGatewayStack {
    _head: ManagedServiceChild,
    _tail: ManagedServiceChild,
    gateway: ManagedServiceChild,
}

#[derive(Debug, Clone, Default)]
pub struct ManagedGatewayLaunchSpec {
    pub stage_node_bin: Option<PathBuf>,
    pub gateway_bin: Option<PathBuf>,
    pub head_bind: Option<String>,
    pub tail_bind: Option<String>,
    pub gateway_bind: Option<String>,
}

impl ManagedGatewayStack {
    pub fn spawn_local(
        model_path: impl Into<PathBuf>,
        reconnect_after_prompt: bool,
    ) -> Result<Self> {
        Self::spawn_local_with_spec(
            model_path,
            reconnect_after_prompt,
            &ManagedGatewayLaunchSpec::default(),
        )
    }

    pub fn spawn_local_with_spec(
        model_path: impl Into<PathBuf>,
        reconnect_after_prompt: bool,
        launch_spec: &ManagedGatewayLaunchSpec,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let stage_node_bin = resolve_managed_binary_path(
            launch_spec.stage_node_bin.as_deref(),
            "llama_stage_tcp_node",
        )?;
        let gateway_bin = resolve_managed_binary_path(
            launch_spec.gateway_bin.as_deref(),
            "llama_stage_gateway_tcp_node",
        )?;
        let head_bind = launch_spec.head_bind.as_deref().unwrap_or("127.0.0.1:0");
        let tail_bind = launch_spec.tail_bind.as_deref().unwrap_or("127.0.0.1:0");
        let gateway_bind = launch_spec.gateway_bind.as_deref().unwrap_or("127.0.0.1:0");

        let head = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            head_bind,
            "stage-1",
            0,
            20,
            true,
            false,
        )?;
        let tail = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            tail_bind,
            "stage-2",
            21,
            41,
            false,
            true,
        )?;
        let gateway = ManagedServiceChild::spawn_gateway_from_bin(
            &gateway_bin,
            gateway_bind,
            &head.addr,
            &tail.addr,
            reconnect_after_prompt,
        )?;

        let deadline = Instant::now() + Duration::from_secs(30);
        let mut last_err = None;
        while Instant::now() < deadline {
            match GatewayServiceClient::connect(&gateway.addr) {
                Ok(_) => {
                    return Ok(Self {
                        _head: head,
                        _tail: tail,
                        gateway,
                    });
                }
                Err(err) => {
                    last_err = Some(err);
                    thread::sleep(Duration::from_millis(250));
                }
            }
        }

        let err = last_err
            .map(|err| err.to_string())
            .unwrap_or_else(|| "gateway did not become ready".to_string());
        bail!("timed out waiting for gateway readiness: {err}")
    }

    pub fn gateway_addr(&self) -> &str {
        &self.gateway.addr
    }
}

/// A standalone tail-only stage worker (the rear half of a 2-machine split).
/// Spawns just `llama_stage_tcp_node --tail` so a remote head can connect.
pub struct ManagedTailNode {
    _child: ManagedServiceChild,
    addr: String,
}

impl ManagedTailNode {
    pub fn spawn(
        model_path: impl Into<PathBuf>,
        bind_addr: impl Into<String>,
        start_layer: u32,
        end_layer: u32,
        launch_spec: &ManagedGatewayLaunchSpec,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let bind_addr = bind_addr.into();
        let stage_node_bin = resolve_managed_binary_path(
            launch_spec.stage_node_bin.as_deref(),
            "llama_stage_tcp_node",
        )?;
        let child = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            &bind_addr,
            "stage-tail",
            start_layer,
            end_layer,
            false,
            true,
        )?;
        let addr = child.addr.clone();
        Ok(Self { _child: child, addr })
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }
}

/// A head + gateway stack that talks to a remote tail worker over TCP.
/// Spawns the local head stage node and a gateway pointing at `tail_remote_addr`.
pub struct ManagedHeadGatewayStack {
    _head: ManagedServiceChild,
    gateway: ManagedServiceChild,
}

impl ManagedHeadGatewayStack {
    pub fn spawn_with_remote_tail(
        model_path: impl Into<PathBuf>,
        head_start_layer: u32,
        head_end_layer: u32,
        tail_remote_addr: impl Into<String>,
        reconnect_after_prompt: bool,
        launch_spec: &ManagedGatewayLaunchSpec,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let tail_remote_addr = tail_remote_addr.into();
        let stage_node_bin = resolve_managed_binary_path(
            launch_spec.stage_node_bin.as_deref(),
            "llama_stage_tcp_node",
        )?;
        let gateway_bin = resolve_managed_binary_path(
            launch_spec.gateway_bin.as_deref(),
            "llama_stage_gateway_tcp_node",
        )?;
        let head_bind = launch_spec.head_bind.as_deref().unwrap_or("127.0.0.1:0");
        let gateway_bind = launch_spec.gateway_bind.as_deref().unwrap_or("127.0.0.1:0");

        let head = ManagedServiceChild::spawn_stage_from_bin(
            &stage_node_bin,
            &model_path,
            head_bind,
            "stage-head",
            head_start_layer,
            head_end_layer,
            true,
            false,
        )?;
        let gateway = ManagedServiceChild::spawn_gateway_from_bin(
            &gateway_bin,
            gateway_bind,
            &head.addr,
            &tail_remote_addr,
            reconnect_after_prompt,
        )?;

        let deadline = Instant::now() + Duration::from_secs(60);
        let mut last_err = None;
        while Instant::now() < deadline {
            match GatewayServiceClient::connect(&gateway.addr) {
                Ok(_) => {
                    return Ok(Self {
                        _head: head,
                        gateway,
                    });
                }
                Err(err) => {
                    last_err = Some(err);
                    thread::sleep(Duration::from_millis(250));
                }
            }
        }

        let err = last_err
            .map(|err| err.to_string())
            .unwrap_or_else(|| "gateway did not become ready".to_string());
        bail!("timed out waiting for head-only gateway readiness: {err}")
    }

    pub fn gateway_addr(&self) -> &str {
        &self.gateway.addr
    }
}

impl Drop for LlamaStageBackend {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        for (_, session) in state.sessions.drain() {
            session.session.destroy(&self.api);
        }
        if let Some(model) = state.model.take() {
            unsafe {
                (self.api.model_free)(model.model);
            }
        }
    }
}

impl StageForwardBackend for LlamaStageBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        let mut state = self.state.borrow_mut();
        for (_, session) in state.sessions.drain() {
            session.session.destroy(&self.api);
        }
        state.layout = Some(layout);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let mut state = self.state.borrow_mut();
        let layout = self.layout(&state)?.clone();
        if !layout.is_head {
            bail!("begin_prompt called on non-head stage {}", layout.stage_id);
        }

        let model = self.ensure_model(&mut state)?;
        let tokens = self.tokenize_prompt(model, prompt)?;
        drop(state);
        if Self::debug_flow_enabled() {
            eprintln!(
                "[llama-stage] begin_prompt stage={} tokens={}",
                layout.stage_id,
                tokens.len()
            );
        }
        self.forward_head_tokens_impl(
            request_id,
            tokens,
            Some(prompt.to_string()),
            max_tokens,
            true,
        )
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        self.continue_forward_impl(input, None, true)
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let state = self.state.borrow();
        let layout = self.layout(&state)?.clone();
        if !layout.is_tail {
            bail!("sample_tail called on non-tail stage {}", layout.stage_id);
        }

        let cached = state
            .sessions
            .get(&input.request_id)
            .and_then(|session| session.cached_sample.clone())
            .context("no cached tail sample; call continue_forward first")?;

        if Self::debug_flow_enabled() {
            eprintln!(
                "[llama-stage] sample_tail returning stage={} text={:?}",
                layout.stage_id, cached.sample.text
            );
        }

        Ok(cached.sample)
    }
}

pub fn greedy_single_node_baseline(
    model_path: impl Into<PathBuf>,
    prompt: &str,
) -> Result<StageSample> {
    let api = LlamaApi::load()?;
    let model_path = model_path.into();
    let model = LlamaModelHandle::new(&api, &model_path)?;
    let session = model.create_session(&api)?;
    session.clear_memory(&api);

    let backend = LlamaStageBackend {
        api,
        model_path,
        state: RefCell::new(BackendState {
            layout: None,
            model: None,
            sessions: HashMap::new(),
        }),
    };

    let tokens = backend.tokenize_prompt(&model, prompt)?;
    let batch = OwnedBatch::token_only(tokens);
    let rc = unsafe { (backend.api.decode)(session.ctx, batch.raw) };
    if rc != 0 && rc != 1 {
        bail!("llama_decode baseline failed with {}", rc);
    }

    let sample = backend.greedy_sample(model.model, &session, "baseline", "baseline")?;
    session.destroy(&backend.api);
    unsafe { (backend.api.model_free)(model.model) };
    Ok(sample.sample)
}

pub fn greedy_single_node_completion(
    model_path: impl Into<PathBuf>,
    prompt: &str,
    max_tokens: u32,
) -> Result<GreedyCompletion> {
    let api = LlamaApi::load()?;
    let model_path = model_path.into();
    let model = LlamaModelHandle::new(&api, &model_path)?;
    let session = model.create_session(&api)?;
    session.clear_memory(&api);

    let backend = LlamaStageBackend {
        api,
        model_path,
        state: RefCell::new(BackendState {
            layout: None,
            model: None,
            sessions: HashMap::new(),
        }),
    };

    let prompt_tokens = backend.tokenize_prompt(&model, prompt)?;
    let prompt_batch = OwnedBatch::token_only(prompt_tokens);
    let rc = unsafe { (backend.api.decode)(session.ctx, prompt_batch.raw) };
    if rc != 0 && rc != 1 {
        bail!("llama_decode baseline prompt failed with {}", rc);
    }

    let mut text = String::new();
    let mut token_ids = Vec::new();
    for _ in 0..max_tokens.max(1) {
        let token = backend.greedy_sample(model.model, &session, "baseline", "baseline")?;
        text.push_str(&token.sample.text);
        token_ids.push(token.token_id);
        if token.is_eog {
            break;
        }

        let step_batch = OwnedBatch::token_only(vec![token.token_id]);
        let rc = unsafe { (backend.api.decode)(session.ctx, step_batch.raw) };
        if rc != 0 && rc != 1 {
            bail!("llama_decode baseline continuation failed with {}", rc);
        }
    }

    session.destroy(&backend.api);
    unsafe { (backend.api.model_free)(model.model) };

    Ok(GreedyCompletion {
        completion_tokens: token_ids.len() as u32,
        text,
        token_ids,
    })
}
