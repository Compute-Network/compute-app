use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::PathBuf;
use std::ptr::null;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use stage_forward_lab::{PackedTensorEntry, StageTensorStore, quants, real_math::rope_angles};

use crate::inference::ggml_runtime::GgmlRuntimePlan;
use crate::inference::ggml_stage_plan::GgmlStageOperatorPlan;
use crate::inference::stage_acceleration::StageAccelerationTarget;

const GGML_TYPE_F32: c_int = 0;
const GGML_TYPE_I32: c_int = 26;

const GGML_STATUS_SUCCESS: c_int = 0;
const GGML_PREC_F32: c_int = 10;
const GGML_ROPE_TYPE_NEOX: c_int = 2;

const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;
const GGML_BACKEND_DEVICE_TYPE_GPU: c_int = 1;
const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int = 2;

const GGML_MEM_SIZE: usize = 64 * 1024 * 1024;

#[repr(C)]
struct GgmlInitParams {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}

#[repr(C)]
struct GgmlContext {
    _private: [u8; 0],
}

#[repr(C)]
struct GgmlTensor {
    type_: c_int,
    buffer: *mut GgmlBackendBuffer,
    ne: [i64; 4],
    nb: [usize; 4],
}

#[repr(C)]
struct GgmlGraph {
    _private: [u8; 0],
}

#[repr(C)]
struct GgmlBackend {
    _private: [u8; 0],
}

#[repr(C)]
struct GgmlBackendBuffer {
    _private: [u8; 0],
}

type GgmlInitFn = unsafe extern "C" fn(GgmlInitParams) -> *mut GgmlContext;
type GgmlFreeFn = unsafe extern "C" fn(*mut GgmlContext);
type GgmlFp32ToFp16RowFn = unsafe extern "C" fn(*const f32, *mut u16, i64);
type GgmlNewTensor1dFn = unsafe extern "C" fn(*mut GgmlContext, c_int, i64) -> *mut GgmlTensor;
type GgmlNewTensor2dFn = unsafe extern "C" fn(*mut GgmlContext, c_int, i64, i64) -> *mut GgmlTensor;
type GgmlNewGraphFn = unsafe extern "C" fn(*mut GgmlContext) -> *mut GgmlGraph;
type GgmlBuildForwardExpandFn = unsafe extern "C" fn(*mut GgmlGraph, *mut GgmlTensor);
type GgmlRmsNormFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, f32) -> *mut GgmlTensor;
type GgmlContFn = unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlCont2dFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, i64, i64) -> *mut GgmlTensor;
type GgmlRepeatFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlAddFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlAdd1Fn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlMulFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlTanhFn = unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlSoftMaxFn = unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlScaleFn = unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, f32) -> *mut GgmlTensor;
type GgmlConcatFn = unsafe extern "C" fn(
    *mut GgmlContext,
    *mut GgmlTensor,
    *mut GgmlTensor,
    c_int,
) -> *mut GgmlTensor;
type GgmlMulMatFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlMulMatSetPrecFn = unsafe extern "C" fn(*mut GgmlTensor, c_int);
type GgmlGetRowsFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, *mut GgmlTensor) -> *mut GgmlTensor;
type GgmlReshape3dFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, i64, i64, i64) -> *mut GgmlTensor;
type GgmlReshape4dFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlTensor, i64, i64, i64, i64) -> *mut GgmlTensor;
type GgmlPermuteFn = unsafe extern "C" fn(
    *mut GgmlContext,
    *mut GgmlTensor,
    c_int,
    c_int,
    c_int,
    c_int,
) -> *mut GgmlTensor;
type GgmlView4dFn = unsafe extern "C" fn(
    *mut GgmlContext,
    *mut GgmlTensor,
    i64,
    i64,
    i64,
    i64,
    usize,
    usize,
    usize,
    usize,
) -> *mut GgmlTensor;
type GgmlRopeExtFn = unsafe extern "C" fn(
    *mut GgmlContext,
    *mut GgmlTensor,
    *mut GgmlTensor,
    *mut GgmlTensor,
    c_int,
    c_int,
    c_int,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
) -> *mut GgmlTensor;
type GgmlFlashAttnExtFn = unsafe extern "C" fn(
    *mut GgmlContext,
    *mut GgmlTensor,
    *mut GgmlTensor,
    *mut GgmlTensor,
    *mut GgmlTensor,
    f32,
    f32,
    f32,
) -> *mut GgmlTensor;
type GgmlFlashAttnExtSetPrecFn = unsafe extern "C" fn(*mut GgmlTensor, c_int);

type GgmlBackendLoadAllFn = unsafe extern "C" fn();
type GgmlBackendInitByTypeFn = unsafe extern "C" fn(c_int, *const c_char) -> *mut GgmlBackend;
type GgmlBackendInitBestFn = unsafe extern "C" fn() -> *mut GgmlBackend;
type GgmlBackendNameFn = unsafe extern "C" fn(*mut GgmlBackend) -> *const c_char;
type GgmlBackendFreeFn = unsafe extern "C" fn(*mut GgmlBackend);
type GgmlBackendAllocCtxTensorsFn =
    unsafe extern "C" fn(*mut GgmlContext, *mut GgmlBackend) -> *mut GgmlBackendBuffer;
type GgmlBackendBufferFreeFn = unsafe extern "C" fn(*mut GgmlBackendBuffer);
type GgmlBackendTensorSetFn = unsafe extern "C" fn(*mut GgmlTensor, *const c_void, usize, usize);
type GgmlBackendTensorGetFn = unsafe extern "C" fn(*const GgmlTensor, *mut c_void, usize, usize);
type GgmlBackendGraphComputeFn = unsafe extern "C" fn(*mut GgmlBackend, *mut GgmlGraph) -> c_int;
type GgmlBackendSynchronizeFn = unsafe extern "C" fn(*mut GgmlBackend);

#[derive(Clone, Copy)]
struct GgmlDynamicApi {
    ggml_init: GgmlInitFn,
    ggml_free: GgmlFreeFn,
    ggml_fp32_to_fp16_row: GgmlFp32ToFp16RowFn,
    ggml_new_tensor_1d: GgmlNewTensor1dFn,
    ggml_new_tensor_2d: GgmlNewTensor2dFn,
    ggml_new_graph: GgmlNewGraphFn,
    ggml_build_forward_expand: GgmlBuildForwardExpandFn,
    ggml_rms_norm: GgmlRmsNormFn,
    ggml_cont: GgmlContFn,
    ggml_cont_2d: GgmlCont2dFn,
    ggml_repeat: GgmlRepeatFn,
    ggml_add: GgmlAddFn,
    ggml_add1: GgmlAdd1Fn,
    ggml_mul: GgmlMulFn,
    ggml_tanh: GgmlTanhFn,
    ggml_soft_max: GgmlSoftMaxFn,
    ggml_scale: GgmlScaleFn,
    ggml_concat: GgmlConcatFn,
    ggml_mul_mat: GgmlMulMatFn,
    ggml_mul_mat_set_prec: GgmlMulMatSetPrecFn,
    ggml_get_rows: GgmlGetRowsFn,
    ggml_reshape_3d: GgmlReshape3dFn,
    ggml_reshape_4d: GgmlReshape4dFn,
    ggml_permute: GgmlPermuteFn,
    ggml_view_4d: GgmlView4dFn,
    ggml_rope_ext: GgmlRopeExtFn,
    ggml_flash_attn_ext: GgmlFlashAttnExtFn,
    ggml_flash_attn_ext_set_prec: GgmlFlashAttnExtSetPrecFn,
    ggml_backend_load_all: GgmlBackendLoadAllFn,
    ggml_backend_init_by_type: GgmlBackendInitByTypeFn,
    ggml_backend_init_best: GgmlBackendInitBestFn,
    ggml_backend_name: GgmlBackendNameFn,
    ggml_backend_free: GgmlBackendFreeFn,
    ggml_backend_alloc_ctx_tensors: GgmlBackendAllocCtxTensorsFn,
    ggml_backend_buffer_free: GgmlBackendBufferFreeFn,
    ggml_backend_tensor_set: GgmlBackendTensorSetFn,
    ggml_backend_tensor_get: GgmlBackendTensorGetFn,
    ggml_backend_graph_compute: GgmlBackendGraphComputeFn,
    ggml_backend_synchronize: GgmlBackendSynchronizeFn,
}

static GGML_BASE_LIBRARY: OnceLock<Result<&'static Library, String>> = OnceLock::new();
static GGML_BACKEND_LIBRARY: OnceLock<Result<&'static Library, String>> = OnceLock::new();
static GGML_DYNAMIC_API: OnceLock<Result<GgmlDynamicApi, String>> = OnceLock::new();

fn library_candidates(file_name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if cfg!(target_os = "macos") {
        candidates.push(PathBuf::from(format!("/opt/homebrew/lib/{file_name}")));
        candidates.push(PathBuf::from(format!("/usr/local/lib/{file_name}")));
    } else if cfg!(target_os = "linux") {
        candidates.push(PathBuf::from(format!("/usr/local/lib/{file_name}")));
        candidates.push(PathBuf::from(format!("/usr/lib/{file_name}")));
        candidates.push(PathBuf::from(format!("/usr/lib/x86_64-linux-gnu/{file_name}")));
    } else if cfg!(target_os = "windows") {
        candidates.push(PathBuf::from(file_name));
    }
    candidates.push(PathBuf::from(file_name));
    candidates
}

fn base_library_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "libggml-base.dylib"
    } else if cfg!(target_os = "windows") {
        "ggml-base.dll"
    } else {
        "libggml-base.so"
    }
}

fn backend_library_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "libggml.dylib"
    } else if cfg!(target_os = "windows") {
        "ggml.dll"
    } else {
        "libggml.so"
    }
}

fn leak_loaded_library(candidates: &[PathBuf]) -> Result<&'static Library> {
    let mut errors = Vec::new();
    for candidate in candidates {
        match unsafe { Library::new(candidate) } {
            Ok(lib) => return Ok(Box::leak(Box::new(lib))),
            Err(err) => errors.push(format!("{} ({err})", candidate.display())),
        }
    }
    bail!("failed to load ggml shared library: {}", errors.join(" | "))
}

unsafe fn load_symbol<T: Copy>(library: &'static Library, symbol: &[u8]) -> Result<T> {
    Ok(*unsafe { library.get::<T>(symbol)? })
}

impl GgmlDynamicApi {
    fn shared() -> Result<&'static Self> {
        let result = GGML_DYNAMIC_API.get_or_init(|| Self::load().map_err(|err| err.to_string()));
        match result {
            Ok(api) => Ok(api),
            Err(err) => Err(anyhow!(err.clone())),
        }
    }

    fn load() -> Result<Self> {
        let base = match GGML_BASE_LIBRARY.get_or_init(|| {
            leak_loaded_library(&library_candidates(base_library_filename()))
                .map_err(|err| err.to_string())
        }) {
            Ok(lib) => *lib,
            Err(err) => return Err(anyhow!(err.clone())),
        };
        let backend = match GGML_BACKEND_LIBRARY.get_or_init(|| {
            leak_loaded_library(&library_candidates(backend_library_filename()))
                .map_err(|err| err.to_string())
        }) {
            Ok(lib) => *lib,
            Err(err) => return Err(anyhow!(err.clone())),
        };

        unsafe {
            Ok(Self {
                ggml_init: load_symbol(base, b"ggml_init\0")?,
                ggml_free: load_symbol(base, b"ggml_free\0")?,
                ggml_fp32_to_fp16_row: load_symbol(base, b"ggml_fp32_to_fp16_row\0")?,
                ggml_new_tensor_1d: load_symbol(base, b"ggml_new_tensor_1d\0")?,
                ggml_new_tensor_2d: load_symbol(base, b"ggml_new_tensor_2d\0")?,
                ggml_new_graph: load_symbol(base, b"ggml_new_graph\0")?,
                ggml_build_forward_expand: load_symbol(base, b"ggml_build_forward_expand\0")?,
                ggml_rms_norm: load_symbol(base, b"ggml_rms_norm\0")?,
                ggml_cont: load_symbol(base, b"ggml_cont\0")?,
                ggml_cont_2d: load_symbol(base, b"ggml_cont_2d\0")?,
                ggml_repeat: load_symbol(base, b"ggml_repeat\0")?,
                ggml_add: load_symbol(base, b"ggml_add\0")?,
                ggml_add1: load_symbol(base, b"ggml_add1\0")?,
                ggml_mul: load_symbol(base, b"ggml_mul\0")?,
                ggml_tanh: load_symbol(base, b"ggml_tanh\0")?,
                ggml_soft_max: load_symbol(base, b"ggml_soft_max\0")?,
                ggml_scale: load_symbol(base, b"ggml_scale\0")?,
                ggml_concat: load_symbol(base, b"ggml_concat\0")?,
                ggml_mul_mat: load_symbol(base, b"ggml_mul_mat\0")?,
                ggml_mul_mat_set_prec: load_symbol(base, b"ggml_mul_mat_set_prec\0")?,
                ggml_get_rows: load_symbol(base, b"ggml_get_rows\0")?,
                ggml_reshape_3d: load_symbol(base, b"ggml_reshape_3d\0")?,
                ggml_reshape_4d: load_symbol(base, b"ggml_reshape_4d\0")?,
                ggml_permute: load_symbol(base, b"ggml_permute\0")?,
                ggml_view_4d: load_symbol(base, b"ggml_view_4d\0")?,
                ggml_rope_ext: load_symbol(base, b"ggml_rope_ext\0")?,
                ggml_flash_attn_ext: load_symbol(base, b"ggml_flash_attn_ext\0")?,
                ggml_flash_attn_ext_set_prec: load_symbol(base, b"ggml_flash_attn_ext_set_prec\0")?,
                ggml_backend_load_all: load_symbol(backend, b"ggml_backend_load_all\0")?,
                ggml_backend_init_by_type: load_symbol(backend, b"ggml_backend_init_by_type\0")?,
                ggml_backend_init_best: load_symbol(backend, b"ggml_backend_init_best\0")?,
                ggml_backend_name: load_symbol(base, b"ggml_backend_name\0")?,
                ggml_backend_free: load_symbol(base, b"ggml_backend_free\0")?,
                ggml_backend_alloc_ctx_tensors: load_symbol(
                    base,
                    b"ggml_backend_alloc_ctx_tensors\0",
                )?,
                ggml_backend_buffer_free: load_symbol(base, b"ggml_backend_buffer_free\0")?,
                ggml_backend_tensor_set: load_symbol(base, b"ggml_backend_tensor_set\0")?,
                ggml_backend_tensor_get: load_symbol(base, b"ggml_backend_tensor_get\0")?,
                ggml_backend_graph_compute: load_symbol(base, b"ggml_backend_graph_compute\0")?,
                ggml_backend_synchronize: load_symbol(base, b"ggml_backend_synchronize\0")?,
            })
        }
    }

    fn init_backend(&self, target: StageAccelerationTarget) -> Result<(*mut GgmlBackend, String)> {
        unsafe { (self.ggml_backend_load_all)() };
        let mut backend = match target {
            StageAccelerationTarget::Cpu => unsafe {
                (self.ggml_backend_init_by_type)(GGML_BACKEND_DEVICE_TYPE_CPU, null())
            },
            StageAccelerationTarget::Metal => unsafe {
                let igpu = (self.ggml_backend_init_by_type)(GGML_BACKEND_DEVICE_TYPE_IGPU, null());
                if !igpu.is_null() {
                    igpu
                } else {
                    let gpu =
                        (self.ggml_backend_init_by_type)(GGML_BACKEND_DEVICE_TYPE_GPU, null());
                    if !gpu.is_null() {
                        gpu
                    } else {
                        (self.ggml_backend_init_by_type)(GGML_BACKEND_DEVICE_TYPE_CPU, null())
                    }
                }
            },
            StageAccelerationTarget::Cuda
            | StageAccelerationTarget::Vulkan
            | StageAccelerationTarget::DirectMl => unsafe {
                let gpu = (self.ggml_backend_init_by_type)(GGML_BACKEND_DEVICE_TYPE_GPU, null());
                if !gpu.is_null() {
                    gpu
                } else {
                    (self.ggml_backend_init_by_type)(GGML_BACKEND_DEVICE_TYPE_CPU, null())
                }
            },
        };
        if backend.is_null() {
            backend = unsafe { (self.ggml_backend_init_best)() };
        }
        if backend.is_null() {
            bail!("ggml backend init failed for target {}", target.as_str());
        }
        let name = unsafe {
            let raw = (self.ggml_backend_name)(backend);
            if raw.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(raw).to_string_lossy().to_string()
            }
        };
        Ok((backend, name))
    }
}

fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_f16_bytes(api: &'static GgmlDynamicApi, values: &[f32]) -> Vec<u8> {
    let mut halves = vec![0u16; values.len()];
    unsafe {
        (api.ggml_fp32_to_fp16_row)(
            values.as_ptr(),
            halves.as_mut_ptr(),
            values.len().try_into().expect("f16 encode len fits in i64"),
        );
    }
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in halves {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn tensor_f32_bytes(store: &StageTensorStore, entry: &PackedTensorEntry) -> Result<Vec<u8>> {
    if entry.ggml_type == quants::GGML_TYPE_F32 {
        store.read(&entry.name)
    } else {
        Ok(encode_f32_bytes(&quants::dequantize_tensor(
            entry.ggml_type,
            &store.read(&entry.name)?,
        )?))
    }
}

fn build_causal_mask(batch_count: usize, sliding_window: Option<usize>) -> Vec<f32> {
    let mut values = vec![-1.0e30f32; batch_count * batch_count];
    for query_index in 0..batch_count {
        let window_start =
            sliding_window.map(|size| (query_index + 1).saturating_sub(size)).unwrap_or(0);
        for key_index in window_start..=query_index {
            values[key_index + query_index * batch_count] = 0.0;
        }
    }
    values
}

fn build_proportional_rope_freq_factors(
    base_theta: f32,
    head_dim: usize,
    rotary_dim: usize,
) -> Vec<f32> {
    let head_dim = head_dim & !1usize;
    let rotary_dim = rotary_dim.min(head_dim) & !1usize;
    if rotary_dim == 0 || head_dim == 0 {
        return Vec::new();
    }
    (0..(rotary_dim / 2))
        .map(|i| {
            base_theta.powf(2.0 * i as f32 * (1.0 / rotary_dim as f32 - 1.0 / head_dim as f32))
        })
        .collect()
}

unsafe fn alloc_linear_weight_tensor(
    api: &'static GgmlDynamicApi,
    ctx: *mut GgmlContext,
    uploads: &mut Vec<(*mut GgmlTensor, Vec<u8>)>,
    store: &StageTensorStore,
    entry: &PackedTensorEntry,
    input_dim: usize,
    output_dim: usize,
    label: &str,
) -> Result<*mut GgmlTensor> {
    let use_f32_weight = force_f32_linear_weights() && entry.ggml_type != quants::GGML_TYPE_F32;
    let tensor = unsafe {
        (api.ggml_new_tensor_2d)(
            ctx,
            if use_f32_weight { GGML_TYPE_F32 } else { entry.ggml_type as c_int },
            input_dim as i64,
            output_dim as i64,
        )
    };
    if tensor.is_null() {
        bail!(
            "ggml full head prefill graph `{label}` failed to allocate weight tensor `{}`",
            entry.name
        );
    }
    let raw =
        if use_f32_weight { tensor_f32_bytes(store, entry)? } else { store.read(&entry.name)? };
    uploads.push((tensor, raw));
    Ok(tensor)
}

unsafe fn alloc_f32_vector_tensor(
    api: &'static GgmlDynamicApi,
    ctx: *mut GgmlContext,
    uploads: &mut Vec<(*mut GgmlTensor, Vec<u8>)>,
    store: &StageTensorStore,
    entry: &PackedTensorEntry,
    len: usize,
    label: &str,
) -> Result<*mut GgmlTensor> {
    let tensor = unsafe { (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_F32, len as i64) };
    if tensor.is_null() {
        bail!(
            "ggml full head prefill graph `{label}` failed to allocate vector tensor `{}`",
            entry.name
        );
    }
    uploads.push((tensor, tensor_f32_bytes(store, entry)?));
    Ok(tensor)
}

fn force_f32_linear_weights() -> bool {
    std::env::var_os("COMPUTE_GGML_FORCE_F32_LINEAR_WEIGHTS").is_some()
}

pub struct GgmlSampleGraphRuntime {
    api: &'static GgmlDynamicApi,
    backend: *mut GgmlBackend,
    ctx: *mut GgmlContext,
    buffer: *mut GgmlBackendBuffer,
    graph: *mut GgmlGraph,
    input: *mut GgmlTensor,
    output: *mut GgmlTensor,
    hidden_dim: usize,
    vocab_size: usize,
    backend_name: String,
}

unsafe impl Send for GgmlSampleGraphRuntime {}

pub struct GgmlGetRowsGraphRuntime {
    api: &'static GgmlDynamicApi,
    backend: *mut GgmlBackend,
    ctx: *mut GgmlContext,
    buffer: *mut GgmlBackendBuffer,
    graph: *mut GgmlGraph,
    indices: *mut GgmlTensor,
    output: *mut GgmlTensor,
    row_width: usize,
    row_count: usize,
    backend_name: String,
}

unsafe impl Send for GgmlGetRowsGraphRuntime {}

struct GgmlLinearGraphRuntime {
    api: &'static GgmlDynamicApi,
    backend: *mut GgmlBackend,
    ctx: *mut GgmlContext,
    buffer: *mut GgmlBackendBuffer,
    graph: *mut GgmlGraph,
    input: *mut GgmlTensor,
    outputs: Vec<*mut GgmlTensor>,
    input_dim: usize,
    batch_count: usize,
    output_dims: Vec<usize>,
}

unsafe impl Send for GgmlLinearGraphRuntime {}

pub struct GgmlSingleOutputGraphRuntime {
    _backend: GgmlOwnedBackend,
    runtime: GgmlLinearGraphRuntime,
    output_dim: usize,
    backend_name: String,
}

unsafe impl Send for GgmlSingleOutputGraphRuntime {}

pub struct GgmlBatchedSingleOutputGraphRuntime {
    _backend: GgmlOwnedBackend,
    runtime: GgmlLinearGraphRuntime,
    output_dim: usize,
    batch_count: usize,
    backend_name: String,
}

unsafe impl Send for GgmlBatchedSingleOutputGraphRuntime {}

pub struct GgmlPleIngressGraphRuntime {
    api: &'static GgmlDynamicApi,
    backend: *mut GgmlBackend,
    ctx: *mut GgmlContext,
    buffer: *mut GgmlBackendBuffer,
    graph: *mut GgmlGraph,
    embedded_states: *mut GgmlTensor,
    token_indices: *mut GgmlTensor,
    output: *mut GgmlTensor,
    hidden_dim: usize,
    total_ple_dim: usize,
    ple_dim: usize,
    num_layers: usize,
    batch_count: usize,
    backend_name: String,
}

unsafe impl Send for GgmlPleIngressGraphRuntime {}

pub struct GgmlRopeGraphRuntime {
    api: &'static GgmlDynamicApi,
    backend: *mut GgmlBackend,
    ctx: *mut GgmlContext,
    buffer: *mut GgmlBackendBuffer,
    graph: *mut GgmlGraph,
    input: *mut GgmlTensor,
    positions: *mut GgmlTensor,
    freq_factors: Option<*mut GgmlTensor>,
    custom_cos: Option<*mut GgmlTensor>,
    custom_sin: Option<*mut GgmlTensor>,
    output: *mut GgmlTensor,
    width: usize,
    batch_count: usize,
    n_heads: usize,
    head_dim: usize,
    rope_base_theta: f32,
    rope_rotary_dim: usize,
    backend_name: String,
    factor_mode: String,
}

unsafe impl Send for GgmlRopeGraphRuntime {}

fn debug_proportional_factor_mode() -> String {
    std::env::var("COMPUTE_GGML_DEBUG_PROPORTIONAL_FACTOR_MODE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "default".to_string())
}

fn build_debug_proportional_rope_freq_factors(
    base_theta: f32,
    head_dim: usize,
    rotary_dim: usize,
    mode: &str,
) -> Vec<f32> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "default" | "none" | "ones" | "identity" => {
            vec![1.0; rotary_dim.min(head_dim & !1usize) / 2]
        }
        "derived" | "legacy" => {
            build_proportional_rope_freq_factors(base_theta, head_dim, rotary_dim)
        }
        "inverse" | "inv" => {
            let mut factors =
                build_proportional_rope_freq_factors(base_theta, head_dim, rotary_dim);
            for value in &mut factors {
                if *value != 0.0 {
                    *value = value.recip();
                }
            }
            factors
        }
        _ => vec![1.0; rotary_dim.min(head_dim & !1usize) / 2],
    }
}

fn build_proportional_rope_tables(
    base_theta: f32,
    head_dim: usize,
    rotary_dim: usize,
    batch_count: usize,
    position_offset: u32,
    factor_mode: Option<&str>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if let Some(mode) = factor_mode {
        match mode.trim().to_ascii_lowercase().as_str() {
            "identity" | "identity-rot" | "unit" => {
                let pair_count = rotary_dim.min(head_dim & !1usize) / 2;
                return Ok((
                    vec![1.0; pair_count * batch_count],
                    vec![0.0; pair_count * batch_count],
                ));
            }
            _ => {}
        }
    }
    let factors = match factor_mode {
        Some(mode) => {
            build_debug_proportional_rope_freq_factors(base_theta, head_dim, rotary_dim, mode)
        }
        None => vec![1.0; rotary_dim.min(head_dim & !1usize) / 2],
    };
    let pair_count = rotary_dim.min(head_dim & !1usize) / 2;
    let mut cos = Vec::with_capacity(pair_count * batch_count);
    let mut sin = Vec::with_capacity(pair_count * batch_count);
    for batch_index in 0..batch_count {
        let position = position_offset
            .checked_add(batch_index as u32)
            .ok_or_else(|| anyhow!("ggml proportional rope position overflow"))?;
        for (cos_t, sin_t) in
            rope_angles(&factors, position, head_dim, base_theta, rotary_dim, true)
        {
            cos.push(cos_t);
            sin.push(sin_t);
        }
    }
    Ok((cos, sin))
}

fn build_proportional_rope_head_tables(
    base_theta: f32,
    head_dim: usize,
    rotary_dim: usize,
    n_heads: usize,
    batch_count: usize,
    position_offset: u32,
    factor_mode: Option<&str>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let pair_count = rotary_dim.min(head_dim & !1usize) / 2;
    let (cos, sin) = build_proportional_rope_tables(
        base_theta,
        head_dim,
        rotary_dim,
        batch_count,
        position_offset,
        factor_mode,
    )?;
    let mut cos_full = Vec::with_capacity(cos.len() * n_heads);
    let mut sin_full = Vec::with_capacity(sin.len() * n_heads);
    for (cos_row, sin_row) in cos.chunks_exact(pair_count).zip(sin.chunks_exact(pair_count)) {
        for _ in 0..n_heads {
            cos_full.extend_from_slice(cos_row);
            sin_full.extend_from_slice(sin_row);
        }
    }
    Ok((cos_full, sin_full))
}

unsafe fn build_proportional_neox_rope_tensor(
    api: &'static GgmlDynamicApi,
    ctx: *mut GgmlContext,
    input_4d: *mut GgmlTensor,
    cos_input: *mut GgmlTensor,
    sin_input: *mut GgmlTensor,
    head_dim: usize,
    rotary_dim: usize,
    label: &str,
) -> Result<*mut GgmlTensor> {
    let head_dim = head_dim & !1usize;
    let rotary_dim = rotary_dim.min(head_dim) & !1usize;
    let pair_count = rotary_dim / 2;
    let half = head_dim / 2;
    if pair_count == 0 {
        return Ok(input_4d);
    }
    if pair_count > half {
        bail!(
            "ggml proportional rope `{label}` invalid dims: pair_count={} half={} head_dim={} rotary_dim={}",
            pair_count,
            half,
            head_dim,
            rotary_dim
        );
    }

    let ne = unsafe { (*input_4d).ne };
    let nb = unsafe { (*input_4d).nb };
    let view = |offset_elems: usize, width: usize, tag: &str| -> Result<*mut GgmlTensor> {
        let offset = offset_elems
            .checked_mul(nb[0])
            .ok_or_else(|| anyhow!("ggml proportional rope `{label}` offset overflow for {tag}"))?;
        let tensor = unsafe {
            (api.ggml_view_4d)(
                ctx,
                input_4d,
                width as i64,
                ne[1],
                ne[2],
                ne[3],
                nb[1],
                nb[2],
                nb[3],
                offset,
            )
        };
        if tensor.is_null() {
            bail!("ggml proportional rope `{label}` failed to build view for {tag}");
        }
        Ok(tensor)
    };

    let x0 = view(0, pair_count, "x0")?;
    let x1 = view(half, pair_count, "x1")?;
    let middle = half - pair_count;
    let low_tail = if middle > 0 { Some(view(pair_count, middle, "low_tail")?) } else { None };
    let high_tail = if head_dim > half + pair_count {
        Some(view(half + pair_count, head_dim - (half + pair_count), "high_tail")?)
    } else {
        None
    };

    let x0_cos = unsafe { (api.ggml_mul)(ctx, x0, cos_input) };
    let x1_sin = unsafe { (api.ggml_mul)(ctx, x1, sin_input) };
    let x0_sin = unsafe { (api.ggml_mul)(ctx, x0, sin_input) };
    let x1_cos = unsafe { (api.ggml_mul)(ctx, x1, cos_input) };
    if x0_cos.is_null() || x1_sin.is_null() || x0_sin.is_null() || x1_cos.is_null() {
        bail!("ggml proportional rope `{label}` failed to build trig mul ops");
    }

    let rotated_low = unsafe {
        let neg_x1_sin = (api.ggml_scale)(ctx, x1_sin, -1.0);
        if neg_x1_sin.is_null() {
            bail!("ggml proportional rope `{label}` failed to scale sin branch");
        }
        (api.ggml_add)(ctx, x0_cos, neg_x1_sin)
    };
    let rotated_high = unsafe { (api.ggml_add)(ctx, x0_sin, x1_cos) };
    if rotated_low.is_null() || rotated_high.is_null() {
        bail!("ggml proportional rope `{label}` failed to combine rotated halves");
    }

    let mut combined = rotated_low;
    if let Some(low_tail) = low_tail {
        combined = unsafe { (api.ggml_concat)(ctx, combined, low_tail, 0) };
        if combined.is_null() {
            bail!("ggml proportional rope `{label}` failed to concat low tail");
        }
    }
    combined = unsafe { (api.ggml_concat)(ctx, combined, rotated_high, 0) };
    if combined.is_null() {
        bail!("ggml proportional rope `{label}` failed to concat rotated high");
    }
    if let Some(high_tail) = high_tail {
        combined = unsafe { (api.ggml_concat)(ctx, combined, high_tail, 0) };
        if combined.is_null() {
            bail!("ggml proportional rope `{label}` failed to concat high tail");
        }
    }

    let combined = unsafe { (api.ggml_cont)(ctx, combined) };
    if combined.is_null() {
        bail!("ggml proportional rope `{label}` failed to materialize output");
    }
    Ok(combined)
}

struct GgmlOwnedBackend {
    api: &'static GgmlDynamicApi,
    backend: *mut GgmlBackend,
    backend_name: String,
}

unsafe impl Send for GgmlOwnedBackend {}

pub struct GgmlTailLayerRuntime {
    qkv: GgmlLinearGraphRuntime,
    attn_output: GgmlLinearGraphRuntime,
    gate_up: GgmlLinearGraphRuntime,
    down: GgmlLinearGraphRuntime,
    inp_gate: Option<GgmlLinearGraphRuntime>,
    proj: Option<GgmlLinearGraphRuntime>,
    layer_index: u32,
    hidden_dim: usize,
    batch_count: usize,
    backend: GgmlOwnedBackend,
}

unsafe impl Send for GgmlTailLayerRuntime {}

pub struct GgmlTailStackRuntime {
    layers: Vec<GgmlTailLayerRuntime>,
    backend_name: String,
    batch_count: usize,
}

unsafe impl Send for GgmlTailStackRuntime {}

#[derive(Debug, Clone, Copy)]
pub struct GgmlHeadLayerGraphSpec {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_base_theta: f32,
    pub rope_rotary_dim: usize,
    pub proportional_rope: bool,
    pub sliding_window: Option<usize>,
    pub uses_shared_kv: bool,
}

pub struct GgmlFullHeadPrefillLayerRunResult {
    pub hidden_states: Vec<Vec<f32>>,
    pub k_cache: Option<Vec<Vec<f32>>>,
    pub v_cache: Option<Vec<Vec<f32>>>,
}

pub struct GgmlFullHeadPrefillLayerRuntime {
    api: &'static GgmlDynamicApi,
    backend: *mut GgmlBackend,
    ctx: *mut GgmlContext,
    buffer: *mut GgmlBackendBuffer,
    graph: *mut GgmlGraph,
    hidden_input: *mut GgmlTensor,
    prompt_aux_input: Option<*mut GgmlTensor>,
    shared_k_input: Option<*mut GgmlTensor>,
    shared_v_input: Option<*mut GgmlTensor>,
    hidden_output: *mut GgmlTensor,
    k_cache_output: Option<*mut GgmlTensor>,
    v_cache_output: Option<*mut GgmlTensor>,
    hidden_dim: usize,
    k_dim: usize,
    ple_dim: Option<usize>,
    batch_count: usize,
    layer_index: u32,
    uses_shared_kv: bool,
    backend_name: String,
}

unsafe impl Send for GgmlFullHeadPrefillLayerRuntime {}

impl GgmlSampleGraphRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        operator_plan: &GgmlStageOperatorPlan,
        store: &StageTensorStore,
    ) -> Result<Self> {
        let sample_plan = operator_plan.sample_tail_static_plan()?;
        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;

        let ctx = unsafe {
            (api.ggml_init)(GgmlInitParams {
                mem_size: GGML_MEM_SIZE,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { (api.ggml_backend_free)(backend) };
            bail!("ggml_init failed while building sample graph runtime");
        }

        let result = (|| -> Result<Self> {
            let input = unsafe {
                (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, sample_plan.hidden_dim as i64, 1)
            };
            if input.is_null() {
                bail!("ggml sample graph failed to allocate input tensor");
            }

            let mut state = input;
            if let Some(output_norm) = sample_plan.output_norm.as_ref() {
                let norm_weight = unsafe {
                    (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_F32, sample_plan.hidden_dim as i64)
                };
                if norm_weight.is_null() {
                    bail!("ggml sample graph failed to allocate output_norm tensor");
                }
                let normed = unsafe { (api.ggml_rms_norm)(ctx, input, 1e-6) };
                if normed.is_null() {
                    bail!("ggml sample graph failed to build rms_norm op");
                }
                let repeated = unsafe { (api.ggml_repeat)(ctx, norm_weight, normed) };
                if repeated.is_null() {
                    bail!("ggml sample graph failed to build repeat op for output_norm");
                }
                let scaled = unsafe { (api.ggml_mul)(ctx, normed, repeated) };
                if scaled.is_null() {
                    bail!("ggml sample graph failed to build output_norm scale op");
                }
                let raw = if output_norm.ggml_type == quants::GGML_TYPE_F32 {
                    store.read(&output_norm.name)?
                } else {
                    encode_f32_bytes(&quants::dequantize_tensor(
                        output_norm.ggml_type,
                        &store.read(&output_norm.name)?,
                    )?)
                };
                state = scaled;
                // upload after buffer allocation
                let logits_weight = unsafe {
                    (api.ggml_new_tensor_2d)(
                        ctx,
                        sample_plan.logits.ggml_type as c_int,
                        sample_plan.hidden_dim as i64,
                        sample_plan.vocab_size as i64,
                    )
                };
                if logits_weight.is_null() {
                    bail!("ggml sample graph failed to allocate logits tensor");
                }
                let output = unsafe { (api.ggml_mul_mat)(ctx, logits_weight, state) };
                if output.is_null() {
                    bail!("ggml sample graph failed to build logits matmul");
                }
                unsafe { (api.ggml_mul_mat_set_prec)(output, 10) };
                let graph = unsafe { (api.ggml_new_graph)(ctx) };
                if graph.is_null() {
                    bail!("ggml sample graph failed to allocate compute graph");
                }
                unsafe { (api.ggml_build_forward_expand)(graph, output) };

                let buffer = unsafe { (api.ggml_backend_alloc_ctx_tensors)(ctx, backend) };
                if buffer.is_null() {
                    bail!("ggml sample graph failed to allocate backend tensors");
                }

                unsafe {
                    (api.ggml_backend_tensor_set)(
                        norm_weight,
                        raw.as_ptr().cast::<c_void>(),
                        0,
                        raw.len(),
                    );
                }

                let logits_raw = store.read(&sample_plan.logits.name).with_context(|| {
                    format!("read ggml sample logits tensor `{}`", sample_plan.logits.name)
                })?;
                unsafe {
                    (api.ggml_backend_tensor_set)(
                        logits_weight,
                        logits_raw.as_ptr().cast::<c_void>(),
                        0,
                        logits_raw.len(),
                    );
                }

                Ok(Self {
                    api,
                    backend,
                    ctx,
                    buffer,
                    graph,
                    input,
                    output,
                    hidden_dim: sample_plan.hidden_dim,
                    vocab_size: sample_plan.vocab_size,
                    backend_name,
                })
            } else {
                let logits_weight = unsafe {
                    (api.ggml_new_tensor_2d)(
                        ctx,
                        sample_plan.logits.ggml_type as c_int,
                        sample_plan.hidden_dim as i64,
                        sample_plan.vocab_size as i64,
                    )
                };
                if logits_weight.is_null() {
                    bail!("ggml sample graph failed to allocate logits tensor");
                }
                let output = unsafe { (api.ggml_mul_mat)(ctx, logits_weight, state) };
                if output.is_null() {
                    bail!("ggml sample graph failed to build logits matmul");
                }
                unsafe { (api.ggml_mul_mat_set_prec)(output, 10) };
                let graph = unsafe { (api.ggml_new_graph)(ctx) };
                if graph.is_null() {
                    bail!("ggml sample graph failed to allocate compute graph");
                }
                unsafe { (api.ggml_build_forward_expand)(graph, output) };

                let buffer = unsafe { (api.ggml_backend_alloc_ctx_tensors)(ctx, backend) };
                if buffer.is_null() {
                    bail!("ggml sample graph failed to allocate backend tensors");
                }

                let logits_raw = store.read(&sample_plan.logits.name).with_context(|| {
                    format!("read ggml sample logits tensor `{}`", sample_plan.logits.name)
                })?;
                unsafe {
                    (api.ggml_backend_tensor_set)(
                        logits_weight,
                        logits_raw.as_ptr().cast::<c_void>(),
                        0,
                        logits_raw.len(),
                    );
                }

                Ok(Self {
                    api,
                    backend,
                    ctx,
                    buffer,
                    graph,
                    input,
                    output,
                    hidden_dim: sample_plan.hidden_dim,
                    vocab_size: sample_plan.vocab_size,
                    backend_name,
                })
            }
        })();

        if result.is_err() {
            unsafe {
                (api.ggml_free)(ctx);
                (api.ggml_backend_free)(backend);
            }
        }
        result
    }

    pub fn summary_label(&self) -> String {
        format!(
            "ggml-graph-sample backend={} hidden_dim={} vocab_size={}",
            self.backend_name, self.hidden_dim, self.vocab_size
        )
    }

    pub fn sample_argmax(&mut self, hidden_state: &[f32]) -> Result<usize> {
        if hidden_state.len() != self.hidden_dim {
            bail!(
                "ggml sample graph hidden_state len mismatch: got {} expected {}",
                hidden_state.len(),
                self.hidden_dim
            );
        }
        unsafe {
            (self.api.ggml_backend_tensor_set)(
                self.input,
                f32_bytes(hidden_state).as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(hidden_state),
            );
        }
        let status = unsafe { (self.api.ggml_backend_graph_compute)(self.backend, self.graph) };
        if status != GGML_STATUS_SUCCESS {
            bail!("ggml sample graph compute failed with status {}", status);
        }
        unsafe { (self.api.ggml_backend_synchronize)(self.backend) };

        let mut logits = vec![0.0f32; self.vocab_size];
        unsafe {
            (self.api.ggml_backend_tensor_get)(
                self.output,
                logits.as_mut_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(logits.as_slice()),
            );
        }
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("ggml sample graph produced no logits"))
    }
}

impl GgmlGetRowsGraphRuntime {
    pub fn new_single(
        runtime: &GgmlRuntimePlan,
        entry: &PackedTensorEntry,
        store: &StageTensorStore,
        label: &str,
    ) -> Result<Self> {
        Self::new(runtime, entry, 1, store, label)
    }

    pub fn new(
        runtime: &GgmlRuntimePlan,
        entry: &PackedTensorEntry,
        row_count: usize,
        store: &StageTensorStore,
        label: &str,
    ) -> Result<Self> {
        if row_count == 0 {
            bail!("ggml get-rows graph `{label}` requires row_count > 0");
        }
        let row_width = entry.dimensions.first().copied().unwrap_or_default() as usize;
        let total_rows = entry.dimensions.get(1).copied().unwrap_or_default() as usize;
        if row_width == 0 || total_rows == 0 {
            bail!(
                "ggml get-rows graph `{label}` tensor `{}` has invalid dims {:?}",
                entry.name,
                entry.dimensions
            );
        }

        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;

        let ctx = unsafe {
            (api.ggml_init)(GgmlInitParams {
                mem_size: GGML_MEM_SIZE,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { (api.ggml_backend_free)(backend) };
            bail!("ggml_init failed while building get-rows graph `{label}`");
        }

        let result = (|| -> Result<Self> {
            let weight = unsafe {
                (api.ggml_new_tensor_2d)(
                    ctx,
                    entry.ggml_type as c_int,
                    row_width as i64,
                    total_rows as i64,
                )
            };
            if weight.is_null() {
                bail!(
                    "ggml get-rows graph `{label}` failed to allocate weight tensor `{}`",
                    entry.name
                );
            }

            let indices = unsafe { (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_I32, row_count as i64) };
            if indices.is_null() {
                bail!("ggml get-rows graph `{label}` failed to allocate indices tensor");
            }

            let output = unsafe { (api.ggml_get_rows)(ctx, weight, indices) };
            if output.is_null() {
                bail!(
                    "ggml get-rows graph `{label}` failed to build gather op for `{}`",
                    entry.name
                );
            }

            let graph = unsafe { (api.ggml_new_graph)(ctx) };
            if graph.is_null() {
                bail!("ggml get-rows graph `{label}` failed to allocate compute graph");
            }
            unsafe { (api.ggml_build_forward_expand)(graph, output) };

            let buffer = unsafe { (api.ggml_backend_alloc_ctx_tensors)(ctx, backend) };
            if buffer.is_null() {
                bail!("ggml get-rows graph `{label}` failed to allocate backend tensors");
            }

            let raw = store.read(&entry.name).with_context(|| {
                format!("read ggml get-rows tensor `{}` for `{label}`", entry.name)
            })?;
            unsafe {
                (api.ggml_backend_tensor_set)(weight, raw.as_ptr().cast::<c_void>(), 0, raw.len());
            }

            Ok(Self {
                api,
                backend,
                ctx,
                buffer,
                graph,
                indices,
                output,
                row_width,
                row_count,
                backend_name,
            })
        })();

        if result.is_err() {
            unsafe {
                (api.ggml_free)(ctx);
                (api.ggml_backend_free)(backend);
            }
        }
        result
    }

    pub fn summary_label(&self) -> String {
        format!(
            "ggml-get-rows backend={} row_width={} row_count={}",
            self.backend_name, self.row_width, self.row_count
        )
    }

    pub fn get_row(&mut self, row_index: u32) -> Result<Vec<f32>> {
        let mut rows = self.get_rows(&[row_index])?;
        Ok(rows.pop().expect("single gathered row present"))
    }

    pub fn get_rows(&mut self, row_indices: &[u32]) -> Result<Vec<Vec<f32>>> {
        if row_indices.len() != self.row_count {
            bail!(
                "ggml get-rows input len mismatch: got {} expected {}",
                row_indices.len(),
                self.row_count
            );
        }
        let indices_i32: Vec<i32> = row_indices
            .iter()
            .map(|&value| {
                i32::try_from(value)
                    .map_err(|_| anyhow!("ggml get-rows token id {} does not fit in i32", value))
            })
            .collect::<Result<Vec<_>>>()?;

        unsafe {
            (self.api.ggml_backend_tensor_set)(
                self.indices,
                indices_i32.as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(indices_i32.as_slice()),
            );
        }
        let status = unsafe { (self.api.ggml_backend_graph_compute)(self.backend, self.graph) };
        if status != GGML_STATUS_SUCCESS {
            bail!("ggml get-rows graph compute failed with status {}", status);
        }
        unsafe { (self.api.ggml_backend_synchronize)(self.backend) };

        let mut flat = vec![0.0f32; self.row_width * self.row_count];
        unsafe {
            (self.api.ggml_backend_tensor_get)(
                self.output,
                flat.as_mut_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(flat.as_slice()),
            );
        }
        Ok(flat.chunks_exact(self.row_width).map(|chunk| chunk.to_vec()).collect())
    }
}

impl GgmlRopeGraphRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        spec: GgmlHeadLayerGraphSpec,
        batch_count: usize,
        label: &str,
    ) -> Result<Self> {
        if batch_count == 0 {
            bail!("ggml rope graph `{label}` requires batch_count > 0");
        }
        let width = spec
            .n_heads
            .checked_mul(spec.head_dim)
            .ok_or_else(|| anyhow!("ggml rope graph `{label}` width overflow"))?;
        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;

        let ctx = unsafe {
            (api.ggml_init)(GgmlInitParams {
                mem_size: GGML_MEM_SIZE,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { (api.ggml_backend_free)(backend) };
            bail!("ggml_init failed while building rope graph `{label}`");
        }

        let result = (|| -> Result<Self> {
            let factor_mode = debug_proportional_factor_mode();
            let input = unsafe {
                (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, width as i64, batch_count as i64)
            };
            if input.is_null() {
                bail!("ggml rope graph `{label}` failed to allocate input");
            }
            let input_3d = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    input,
                    spec.head_dim as i64,
                    spec.n_heads as i64,
                    batch_count as i64,
                )
            };
            if input_3d.is_null() {
                bail!("ggml rope graph `{label}` failed to reshape input to 3d");
            }
            let input_4d = unsafe {
                (api.ggml_reshape_4d)(
                    ctx,
                    input_3d,
                    spec.head_dim as i64,
                    spec.n_heads as i64,
                    batch_count as i64,
                    1,
                )
            };
            if input_4d.is_null() {
                bail!("ggml rope graph `{label}` failed to reshape input");
            }
            let positions =
                unsafe { (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_I32, batch_count as i64) };
            if positions.is_null() {
                bail!("ggml rope graph `{label}` failed to allocate positions");
            }
            let freq_factors = None;
            let (custom_cos, custom_sin) = if spec.proportional_rope {
                let pair_count = spec.rope_rotary_dim / 2;
                let cos_base = unsafe {
                    (api.ggml_new_tensor_2d)(
                        ctx,
                        GGML_TYPE_F32,
                        (pair_count * spec.n_heads) as i64,
                        batch_count as i64,
                    )
                };
                let sin_base = unsafe {
                    (api.ggml_new_tensor_2d)(
                        ctx,
                        GGML_TYPE_F32,
                        (pair_count * spec.n_heads) as i64,
                        batch_count as i64,
                    )
                };
                if cos_base.is_null() || sin_base.is_null() {
                    bail!("ggml rope graph `{label}` failed to allocate proportional trig tables");
                }
                (Some(cos_base), Some(sin_base))
            } else {
                (None, None)
            };
            let output = if spec.proportional_rope {
                let pair_count = spec.rope_rotary_dim / 2;
                let cos_4d = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        custom_cos.expect("proportional cos allocated"),
                        pair_count as i64,
                        spec.n_heads as i64,
                        batch_count as i64,
                        1,
                    )
                };
                let sin_4d = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        custom_sin.expect("proportional sin allocated"),
                        pair_count as i64,
                        spec.n_heads as i64,
                        batch_count as i64,
                        1,
                    )
                };
                let output_4d = unsafe {
                    build_proportional_neox_rope_tensor(
                        api,
                        ctx,
                        input_4d,
                        cos_4d,
                        sin_4d,
                        spec.head_dim,
                        spec.rope_rotary_dim,
                        label,
                    )?
                };
                unsafe { (api.ggml_cont_2d)(ctx, output_4d, width as i64, batch_count as i64) }
            } else {
                unsafe {
                    (api.ggml_rope_ext)(
                        ctx,
                        input_4d,
                        positions,
                        std::ptr::null_mut(),
                        spec.rope_rotary_dim as c_int,
                        GGML_ROPE_TYPE_NEOX,
                        0,
                        spec.rope_base_theta,
                        1.0,
                        0.0,
                        1.0,
                        32.0,
                        1.0,
                    )
                }
            };
            if output.is_null() {
                bail!("ggml rope graph `{label}` failed to build rope op");
            }

            let graph = unsafe { (api.ggml_new_graph)(ctx) };
            if graph.is_null() {
                bail!("ggml rope graph `{label}` failed to allocate graph");
            }
            unsafe { (api.ggml_build_forward_expand)(graph, output) };

            let buffer = unsafe { (api.ggml_backend_alloc_ctx_tensors)(ctx, backend) };
            if buffer.is_null() {
                bail!("ggml rope graph `{label}` failed to allocate backend tensors");
            }

            Ok(Self {
                api,
                backend,
                ctx,
                buffer,
                graph,
                input,
                positions,
                freq_factors,
                custom_cos,
                custom_sin,
                output,
                width,
                batch_count,
                n_heads: spec.n_heads,
                head_dim: spec.head_dim,
                rope_base_theta: spec.rope_base_theta,
                rope_rotary_dim: spec.rope_rotary_dim,
                backend_name,
                factor_mode,
            })
        })();

        if result.is_err() {
            unsafe {
                (api.ggml_free)(ctx);
                (api.ggml_backend_free)(backend);
            }
        }
        result
    }

    pub fn summary_label(&self) -> String {
        format!(
            "ggml-rope backend={} width={} batch_count={} n_heads={} head_dim={} proportional={} factor_mode={}",
            self.backend_name,
            self.width,
            self.batch_count,
            self.n_heads,
            self.head_dim,
            self.custom_cos.is_some() || self.freq_factors.is_some(),
            self.factor_mode,
        )
    }

    pub fn apply_with_position_offset(
        &mut self,
        values: &[Vec<f32>],
        position_offset: u32,
    ) -> Result<Vec<Vec<f32>>> {
        if values.len() != self.batch_count {
            bail!(
                "ggml rope graph batch mismatch: got {} expected {}",
                values.len(),
                self.batch_count
            );
        }
        let mut flat = Vec::with_capacity(self.width * self.batch_count);
        for value in values {
            if value.len() != self.width {
                bail!(
                    "ggml rope graph width mismatch: got {} expected {}",
                    value.len(),
                    self.width
                );
            }
            flat.extend_from_slice(value);
        }
        unsafe {
            (self.api.ggml_backend_tensor_set)(
                self.input,
                f32_bytes(&flat).as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(flat.as_slice()),
            );
        }
        if let (Some(custom_cos), Some(custom_sin)) = (self.custom_cos, self.custom_sin) {
            let (cos, sin) = build_proportional_rope_head_tables(
                self.rope_base_theta,
                self.head_dim,
                self.rope_rotary_dim,
                self.n_heads,
                self.batch_count,
                position_offset,
                Some(&self.factor_mode),
            )?;
            unsafe {
                (self.api.ggml_backend_tensor_set)(
                    custom_cos,
                    f32_bytes(&cos).as_ptr().cast::<c_void>(),
                    0,
                    std::mem::size_of_val(cos.as_slice()),
                );
                (self.api.ggml_backend_tensor_set)(
                    custom_sin,
                    f32_bytes(&sin).as_ptr().cast::<c_void>(),
                    0,
                    std::mem::size_of_val(sin.as_slice()),
                );
            }
        } else {
            let positions: Vec<i32> = (0..self.batch_count)
                .map(|idx| {
                    let pos = position_offset
                        .checked_add(idx as u32)
                        .ok_or_else(|| anyhow!("ggml rope graph position overflow"))?;
                    i32::try_from(pos).map_err(|_| anyhow!("ggml rope graph position overflow"))
                })
                .collect::<Result<Vec<_>>>()?;
            unsafe {
                (self.api.ggml_backend_tensor_set)(
                    self.positions,
                    positions.as_ptr().cast::<c_void>(),
                    0,
                    std::mem::size_of_val(positions.as_slice()),
                );
            }
        }
        let status = unsafe { (self.api.ggml_backend_graph_compute)(self.backend, self.graph) };
        if status != GGML_STATUS_SUCCESS {
            bail!("ggml rope graph compute failed with status {}", status);
        }
        unsafe { (self.api.ggml_backend_synchronize)(self.backend) };

        let mut out = vec![0.0f32; self.width * self.batch_count];
        unsafe {
            (self.api.ggml_backend_tensor_get)(
                self.output,
                out.as_mut_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(out.as_slice()),
            );
        }
        Ok(out.chunks_exact(self.width).map(|chunk| chunk.to_vec()).collect())
    }
}

impl GgmlLinearGraphRuntime {
    fn new(
        api: &'static GgmlDynamicApi,
        backend: *mut GgmlBackend,
        input_dim: usize,
        batch_count: usize,
        outputs: &[PackedTensorEntry],
        store: &StageTensorStore,
        label: &str,
    ) -> Result<Self> {
        if outputs.is_empty() {
            bail!("ggml linear graph `{label}` requires at least one output tensor");
        }
        if batch_count == 0 {
            bail!("ggml linear graph `{label}` requires batch_count > 0");
        }

        let ctx = unsafe {
            (api.ggml_init)(GgmlInitParams {
                mem_size: GGML_MEM_SIZE,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            bail!("ggml_init failed while building linear graph `{label}`");
        }

        let result = (|| -> Result<Self> {
            let input = unsafe {
                (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, input_dim as i64, batch_count as i64)
            };
            if input.is_null() {
                bail!("ggml linear graph `{label}` failed to allocate input tensor");
            }

            let graph = unsafe { (api.ggml_new_graph)(ctx) };
            if graph.is_null() {
                bail!("ggml linear graph `{label}` failed to allocate compute graph");
            }

            let mut output_tensors = Vec::with_capacity(outputs.len());
            let mut output_dims = Vec::with_capacity(outputs.len());
            let mut weight_tensors = Vec::with_capacity(outputs.len());
            let mut weight_bytes = Vec::with_capacity(outputs.len());

            for entry in outputs {
                let entry_input_dim =
                    entry.dimensions.first().copied().unwrap_or_default() as usize;
                let entry_output_dim =
                    entry.dimensions.get(1).copied().unwrap_or_default() as usize;
                if entry_input_dim != input_dim || entry_output_dim == 0 {
                    bail!(
                        "ggml linear graph `{label}` tensor `{}` has incompatible dims {:?} for input_dim {}",
                        entry.name,
                        entry.dimensions,
                        input_dim
                    );
                }
                let use_f32_weight =
                    force_f32_linear_weights() && entry.ggml_type != quants::GGML_TYPE_F32;
                let weight = unsafe {
                    (api.ggml_new_tensor_2d)(
                        ctx,
                        if use_f32_weight { GGML_TYPE_F32 } else { entry.ggml_type as c_int },
                        input_dim as i64,
                        entry_output_dim as i64,
                    )
                };
                if weight.is_null() {
                    bail!(
                        "ggml linear graph `{label}` failed to allocate weight tensor `{}`",
                        entry.name
                    );
                }
                let output = unsafe { (api.ggml_mul_mat)(ctx, weight, input) };
                if output.is_null() {
                    bail!(
                        "ggml linear graph `{label}` failed to build matmul for `{}`",
                        entry.name
                    );
                }
                unsafe { (api.ggml_mul_mat_set_prec)(output, 10) };
                unsafe { (api.ggml_build_forward_expand)(graph, output) };
                output_tensors.push(output);
                output_dims.push(entry_output_dim);
                weight_tensors.push(weight);
                let raw = store.read(&entry.name).with_context(|| {
                    format!("read ggml linear graph tensor `{}` for `{label}`", entry.name)
                })?;
                let bytes = if use_f32_weight {
                    encode_f32_bytes(
                        &quants::dequantize_tensor(entry.ggml_type, &raw).with_context(|| {
                            format!(
                                "dequantize ggml linear graph tensor `{}` to f32 for `{label}`",
                                entry.name
                            )
                        })?,
                    )
                } else {
                    raw
                };
                weight_bytes.push(bytes);
            }

            let buffer = unsafe { (api.ggml_backend_alloc_ctx_tensors)(ctx, backend) };
            if buffer.is_null() {
                bail!("ggml linear graph `{label}` failed to allocate backend tensors");
            }

            for (weight, raw) in weight_tensors.into_iter().zip(weight_bytes.into_iter()) {
                unsafe {
                    (api.ggml_backend_tensor_set)(
                        weight,
                        raw.as_ptr().cast::<c_void>(),
                        0,
                        raw.len(),
                    );
                }
            }

            Ok(Self {
                api,
                backend,
                ctx,
                buffer,
                graph,
                input,
                outputs: output_tensors,
                input_dim,
                batch_count,
                output_dims,
            })
        })();

        if result.is_err() {
            unsafe { (api.ggml_free)(ctx) };
        }
        result
    }

    fn run(&mut self, input_values: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.run_flat(input_values)
    }

    fn run_flat(&mut self, input_values: &[f32]) -> Result<Vec<Vec<f32>>> {
        let expected_len = self.input_dim * self.batch_count;
        if input_values.len() != expected_len {
            if self.batch_count == 1 {
                bail!(
                    "ggml linear graph input len mismatch: got {} expected {}",
                    input_values.len(),
                    self.input_dim
                );
            }
            bail!(
                "ggml linear graph input len mismatch: got {} expected {} ({} x {})",
                input_values.len(),
                expected_len,
                self.input_dim,
                self.batch_count
            );
        }
        let input_storage;
        let input_slice = if self.batch_count > 1
            && std::env::var_os("COMPUTE_GGML_DEBUG_LINEAR_TRANSPOSED_INPUT").is_some()
        {
            let mut transposed = vec![0.0f32; expected_len];
            for batch_index in 0..self.batch_count {
                for input_index in 0..self.input_dim {
                    transposed[input_index * self.batch_count + batch_index] =
                        input_values[batch_index * self.input_dim + input_index];
                }
            }
            input_storage = transposed;
            input_storage.as_slice()
        } else {
            input_values
        };
        unsafe {
            (self.api.ggml_backend_tensor_set)(
                self.input,
                f32_bytes(input_slice).as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(input_slice),
            );
        }
        let status = unsafe { (self.api.ggml_backend_graph_compute)(self.backend, self.graph) };
        if status != GGML_STATUS_SUCCESS {
            bail!("ggml linear graph compute failed with status {}", status);
        }
        unsafe { (self.api.ggml_backend_synchronize)(self.backend) };

        let mut out = Vec::with_capacity(self.outputs.len());
        for (&tensor, &output_dim) in self.outputs.iter().zip(self.output_dims.iter()) {
            let mut values = vec![0.0f32; output_dim * self.batch_count];
            unsafe {
                (self.api.ggml_backend_tensor_get)(
                    tensor,
                    values.as_mut_ptr().cast::<c_void>(),
                    0,
                    std::mem::size_of_val(values.as_slice()),
                );
            }
            if self.batch_count == 1 {
                out.push(values);
            } else {
                if std::env::var_os("COMPUTE_GGML_DEBUG_LINEAR_TRANSPOSED_OUTPUT").is_some() {
                    let mut rows = vec![vec![0.0f32; output_dim]; self.batch_count];
                    for out_index in 0..output_dim {
                        for (batch_index, row) in rows.iter_mut().enumerate() {
                            row[out_index] = values[batch_index + out_index * self.batch_count];
                        }
                    }
                    out.extend(rows);
                } else {
                    for chunk in values.chunks_exact(output_dim) {
                        out.push(chunk.to_vec());
                    }
                }
            }
        }
        Ok(out)
    }
}

impl GgmlSingleOutputGraphRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        entry: &PackedTensorEntry,
        store: &StageTensorStore,
        label: &str,
    ) -> Result<Self> {
        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;
        let owned_backend = GgmlOwnedBackend { api, backend, backend_name: backend_name.clone() };
        let runtime = GgmlLinearGraphRuntime::new(
            api,
            owned_backend.backend,
            entry.dimensions.first().copied().unwrap_or_default() as usize,
            1,
            std::slice::from_ref(entry),
            store,
            label,
        )?;
        let output_dim = entry.dimensions.get(1).copied().unwrap_or_default() as usize;
        Ok(Self { _backend: owned_backend, runtime, output_dim, backend_name })
    }

    pub fn summary_label(&self) -> String {
        format!("ggml-single-output backend={} output_dim={}", self.backend_name, self.output_dim)
    }

    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut outputs = self.runtime.run(input)?;
        Ok(outputs.pop().expect("single output present"))
    }
}

impl GgmlBatchedSingleOutputGraphRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        entry: &PackedTensorEntry,
        batch_count: usize,
        store: &StageTensorStore,
        label: &str,
    ) -> Result<Self> {
        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;
        let owned_backend = GgmlOwnedBackend { api, backend, backend_name: backend_name.clone() };
        let runtime = GgmlLinearGraphRuntime::new(
            api,
            owned_backend.backend,
            entry.dimensions.first().copied().unwrap_or_default() as usize,
            batch_count,
            std::slice::from_ref(entry),
            store,
            label,
        )?;
        let output_dim = entry.dimensions.get(1).copied().unwrap_or_default() as usize;
        Ok(Self { _backend: owned_backend, runtime, output_dim, batch_count, backend_name })
    }

    pub fn summary_label(&self) -> String {
        format!(
            "ggml-batched-single-output backend={} output_dim={} batch_count={}",
            self.backend_name, self.output_dim, self.batch_count
        )
    }

    pub fn run_many(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if inputs.len() != self.batch_count {
            bail!(
                "ggml batched single-output input count mismatch: got {} expected {}",
                inputs.len(),
                self.batch_count
            );
        }
        let mut flat = Vec::with_capacity(self.batch_count * self.runtime.input_dim);
        for input in inputs {
            if input.len() != self.runtime.input_dim {
                bail!(
                    "ggml batched single-output input width mismatch: got {} expected {}",
                    input.len(),
                    self.runtime.input_dim
                );
            }
            flat.extend_from_slice(input);
        }
        let mut outputs = self.runtime.run_flat(&flat)?;
        if outputs.len() != self.batch_count {
            bail!(
                "ggml batched single-output returned {} outputs for batch_count {}",
                outputs.len(),
                self.batch_count
            );
        }
        Ok(std::mem::take(&mut outputs))
    }
}

impl GgmlPleIngressGraphRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        model_proj: &PackedTensorEntry,
        proj_norm: &PackedTensorEntry,
        token_embd: &PackedTensorEntry,
        ple_dim: usize,
        batch_count: usize,
        store: &StageTensorStore,
        label: &str,
    ) -> Result<Self> {
        if batch_count == 0 {
            bail!("ggml PLE ingress graph `{label}` requires batch_count > 0");
        }
        let hidden_dim = model_proj.dimensions.first().copied().unwrap_or_default() as usize;
        let total_ple_dim = model_proj.dimensions.get(1).copied().unwrap_or_default() as usize;
        if hidden_dim == 0 || total_ple_dim == 0 || ple_dim == 0 || total_ple_dim % ple_dim != 0 {
            bail!(
                "ggml PLE ingress graph `{label}` has invalid dims: model_proj={:?} ple_dim={}",
                model_proj.dimensions,
                ple_dim
            );
        }
        let num_layers = total_ple_dim / ple_dim;
        let token_row_width = token_embd.dimensions.first().copied().unwrap_or_default() as usize;
        let token_row_count = token_embd.dimensions.get(1).copied().unwrap_or_default() as usize;
        if token_row_width != total_ple_dim || token_row_count == 0 {
            bail!(
                "ggml PLE ingress graph `{label}` token_embd `{}` has incompatible dims {:?} for total_ple_dim {}",
                token_embd.name,
                token_embd.dimensions,
                total_ple_dim
            );
        }

        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;

        let ctx = unsafe {
            (api.ggml_init)(GgmlInitParams {
                mem_size: GGML_MEM_SIZE,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { (api.ggml_backend_free)(backend) };
            bail!("ggml_init failed while building PLE ingress graph `{label}`");
        }

        let result = (|| -> Result<Self> {
            let embedded_states = unsafe {
                (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, hidden_dim as i64, batch_count as i64)
            };
            if embedded_states.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to allocate embedded state input");
            }
            let token_indices =
                unsafe { (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_I32, batch_count as i64) };
            if token_indices.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to allocate token indices");
            }
            let model_proj_weight = unsafe {
                (api.ggml_new_tensor_2d)(
                    ctx,
                    model_proj.ggml_type as c_int,
                    hidden_dim as i64,
                    total_ple_dim as i64,
                )
            };
            if model_proj_weight.is_null() {
                bail!(
                    "ggml PLE ingress graph `{label}` failed to allocate model_proj `{}`",
                    model_proj.name
                );
            }
            let token_embd_weight = unsafe {
                (api.ggml_new_tensor_2d)(
                    ctx,
                    token_embd.ggml_type as c_int,
                    total_ple_dim as i64,
                    token_row_count as i64,
                )
            };
            if token_embd_weight.is_null() {
                bail!(
                    "ggml PLE ingress graph `{label}` failed to allocate token_embd `{}`",
                    token_embd.name
                );
            }
            let proj_norm_weight =
                unsafe { (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_F32, ple_dim as i64) };
            if proj_norm_weight.is_null() {
                bail!(
                    "ggml PLE ingress graph `{label}` failed to allocate proj_norm `{}`",
                    proj_norm.name
                );
            }

            let projected = unsafe { (api.ggml_mul_mat)(ctx, model_proj_weight, embedded_states) };
            if projected.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to build model_proj matmul");
            }
            unsafe { (api.ggml_mul_mat_set_prec)(projected, 10) };
            let projected_scaled =
                unsafe { (api.ggml_scale)(ctx, projected, (hidden_dim as f32).powf(-0.5)) };
            if projected_scaled.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to build projected scale op");
            }
            let projected_3d = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    projected_scaled,
                    ple_dim as i64,
                    num_layers as i64,
                    batch_count as i64,
                )
            };
            if projected_3d.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to reshape projected output");
            }
            let normed = unsafe { (api.ggml_rms_norm)(ctx, projected_3d, 1e-6) };
            if normed.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to build proj rms_norm");
            }
            let norm_weight_repeated = unsafe { (api.ggml_repeat)(ctx, proj_norm_weight, normed) };
            if norm_weight_repeated.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to repeat proj_norm weights");
            }
            let projected_normed = unsafe { (api.ggml_mul)(ctx, normed, norm_weight_repeated) };
            if projected_normed.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to apply proj_norm weights");
            }

            let token_rows = unsafe { (api.ggml_get_rows)(ctx, token_embd_weight, token_indices) };
            if token_rows.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to build token get_rows");
            }
            let token_scaled =
                unsafe { (api.ggml_scale)(ctx, token_rows, (ple_dim as f32).sqrt()) };
            if token_scaled.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to build token scale op");
            }
            let token_3d = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    token_scaled,
                    ple_dim as i64,
                    num_layers as i64,
                    batch_count as i64,
                )
            };
            if token_3d.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to reshape token rows");
            }

            let combined = unsafe { (api.ggml_add)(ctx, projected_normed, token_3d) };
            if combined.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to add projected and token rows");
            }
            let output = unsafe { (api.ggml_scale)(ctx, combined, (2.0f32).powf(-0.5)) };
            if output.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to apply combine scale");
            }

            let graph = unsafe { (api.ggml_new_graph)(ctx) };
            if graph.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to allocate compute graph");
            }
            unsafe { (api.ggml_build_forward_expand)(graph, output) };

            let buffer = unsafe { (api.ggml_backend_alloc_ctx_tensors)(ctx, backend) };
            if buffer.is_null() {
                bail!("ggml PLE ingress graph `{label}` failed to allocate backend tensors");
            }

            let model_proj_raw = store.read(&model_proj.name).with_context(|| {
                format!("read ggml PLE ingress model_proj `{}` for `{label}`", model_proj.name)
            })?;
            let token_embd_raw = store.read(&token_embd.name).with_context(|| {
                format!("read ggml PLE ingress token_embd `{}` for `{label}`", token_embd.name)
            })?;
            let proj_norm_raw = if proj_norm.ggml_type == quants::GGML_TYPE_F32 {
                store.read(&proj_norm.name).with_context(|| {
                    format!("read ggml PLE ingress proj_norm `{}` for `{label}`", proj_norm.name)
                })?
            } else {
                encode_f32_bytes(&quants::dequantize_tensor(
                    proj_norm.ggml_type,
                    &store.read(&proj_norm.name).with_context(|| {
                        format!(
                            "read ggml PLE ingress proj_norm `{}` for `{label}`",
                            proj_norm.name
                        )
                    })?,
                )?)
            };

            unsafe {
                (api.ggml_backend_tensor_set)(
                    model_proj_weight,
                    model_proj_raw.as_ptr().cast::<c_void>(),
                    0,
                    model_proj_raw.len(),
                );
                (api.ggml_backend_tensor_set)(
                    token_embd_weight,
                    token_embd_raw.as_ptr().cast::<c_void>(),
                    0,
                    token_embd_raw.len(),
                );
                (api.ggml_backend_tensor_set)(
                    proj_norm_weight,
                    proj_norm_raw.as_ptr().cast::<c_void>(),
                    0,
                    proj_norm_raw.len(),
                );
            }

            Ok(Self {
                api,
                backend,
                ctx,
                buffer,
                graph,
                embedded_states,
                token_indices,
                output,
                hidden_dim,
                total_ple_dim,
                ple_dim,
                num_layers,
                batch_count,
                backend_name,
            })
        })();

        if result.is_err() {
            unsafe {
                (api.ggml_free)(ctx);
                (api.ggml_backend_free)(backend);
            }
        }
        result
    }

    pub fn summary_label(&self) -> String {
        format!(
            "ggml-ple-ingress backend={} hidden_dim={} ple_dim={} layers={} batch_count={}",
            self.backend_name, self.hidden_dim, self.ple_dim, self.num_layers, self.batch_count
        )
    }

    pub fn run(&mut self, token_ids: &[u32], embedded_states: &[Vec<f32>]) -> Result<Vec<f32>> {
        if token_ids.len() != self.batch_count || embedded_states.len() != self.batch_count {
            bail!(
                "ggml PLE ingress batch mismatch: token_ids={} embedded_states={} expected={}",
                token_ids.len(),
                embedded_states.len(),
                self.batch_count
            );
        }
        let mut flat_states = Vec::with_capacity(self.hidden_dim * self.batch_count);
        for state in embedded_states {
            if state.len() != self.hidden_dim {
                bail!(
                    "ggml PLE ingress hidden_state width mismatch: got {} expected {}",
                    state.len(),
                    self.hidden_dim
                );
            }
            flat_states.extend_from_slice(state);
        }
        let indices_i32: Vec<i32> = token_ids
            .iter()
            .map(|&value| {
                i32::try_from(value)
                    .map_err(|_| anyhow!("ggml PLE ingress token id {} does not fit in i32", value))
            })
            .collect::<Result<Vec<_>>>()?;

        unsafe {
            (self.api.ggml_backend_tensor_set)(
                self.embedded_states,
                f32_bytes(&flat_states).as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(flat_states.as_slice()),
            );
            (self.api.ggml_backend_tensor_set)(
                self.token_indices,
                indices_i32.as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(indices_i32.as_slice()),
            );
        }
        let status = unsafe { (self.api.ggml_backend_graph_compute)(self.backend, self.graph) };
        if status != GGML_STATUS_SUCCESS {
            bail!("ggml PLE ingress graph compute failed with status {}", status);
        }
        unsafe { (self.api.ggml_backend_synchronize)(self.backend) };

        let mut values = vec![0.0f32; self.total_ple_dim * self.batch_count];
        unsafe {
            (self.api.ggml_backend_tensor_get)(
                self.output,
                values.as_mut_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(values.as_slice()),
            );
        }
        Ok(values)
    }
}

impl GgmlFullHeadPrefillLayerRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        layer: &crate::inference::ggml_stage_plan::GgmlLayerBindings,
        hidden_dim: usize,
        spec: GgmlHeadLayerGraphSpec,
        batch_count: usize,
        store: &StageTensorStore,
        label: &str,
    ) -> Result<Self> {
        if batch_count == 0 {
            bail!("ggml full head prefill graph `{label}` requires batch_count > 0");
        }
        let q_dim = layer.attn_q.dimensions.get(1).copied().unwrap_or_default() as usize;
        let k_dim = layer.attn_k.dimensions.get(1).copied().unwrap_or_default() as usize;
        let v_dim = layer.attn_v.dimensions.get(1).copied().unwrap_or_default() as usize;
        if q_dim != spec.n_heads * spec.head_dim
            || k_dim != spec.n_kv_heads * spec.head_dim
            || v_dim != spec.n_kv_heads * spec.head_dim
        {
            bail!(
                "ggml full head prefill graph `{label}` incompatible attention dims on layer {}: q={} k={} v={} heads={} kv_heads={} head_dim={}",
                layer.layer_index,
                q_dim,
                k_dim,
                v_dim,
                spec.n_heads,
                spec.n_kv_heads,
                spec.head_dim
            );
        }

        let ple_dim = layer
            .inp_gate
            .as_ref()
            .map(|entry| entry.dimensions.get(1).copied().unwrap_or_default() as usize)
            .filter(|dim| *dim > 0);

        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;

        let ctx = unsafe {
            (api.ggml_init)(GgmlInitParams {
                mem_size: GGML_MEM_SIZE,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { (api.ggml_backend_free)(backend) };
            bail!("ggml_init failed while building full head prefill graph `{label}`");
        }

        let result = (|| -> Result<Self> {
            let hidden_input = unsafe {
                (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, hidden_dim as i64, batch_count as i64)
            };
            if hidden_input.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to allocate hidden input");
            }
            let positions_input =
                unsafe { (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_I32, batch_count as i64) };
            if positions_input.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to allocate positions input");
            }
            let mask_input = unsafe {
                (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, batch_count as i64, batch_count as i64)
            };
            if mask_input.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to allocate mask input");
            }
            let scalar_one = unsafe { (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_F32, 1) };
            if scalar_one.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to allocate scalar tensor");
            }

            let prompt_aux_input = if let Some(ple_dim) = ple_dim {
                let tensor = unsafe {
                    (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, ple_dim as i64, batch_count as i64)
                };
                if tensor.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to allocate prompt aux input"
                    );
                }
                Some(tensor)
            } else {
                None
            };
            let rope_freq_factors = if spec.proportional_rope && !spec.uses_shared_kv {
                let tensor = unsafe {
                    (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_F32, (spec.rope_rotary_dim / 2) as i64)
                };
                if tensor.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to allocate rope freq factors"
                    );
                }
                Some(tensor)
            } else {
                None
            };
            let (proportional_q_cos, proportional_q_sin) = if spec.proportional_rope
                && spec.uses_shared_kv
            {
                let pair_count = spec.rope_rotary_dim / 2;
                let cos_base = unsafe {
                    (api.ggml_new_tensor_2d)(
                        ctx,
                        GGML_TYPE_F32,
                        (pair_count * spec.n_heads) as i64,
                        batch_count as i64,
                    )
                };
                let sin_base = unsafe {
                    (api.ggml_new_tensor_2d)(
                        ctx,
                        GGML_TYPE_F32,
                        (pair_count * spec.n_heads) as i64,
                        batch_count as i64,
                    )
                };
                if cos_base.is_null() || sin_base.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to allocate proportional Q trig tables"
                    );
                }
                (Some(cos_base), Some(sin_base))
            } else {
                (None, None)
            };
            let shared_k_input = if spec.uses_shared_kv {
                let tensor = unsafe {
                    (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, k_dim as i64, batch_count as i64)
                };
                if tensor.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to allocate shared K input"
                    );
                }
                Some(tensor)
            } else {
                None
            };
            let shared_v_input = if spec.uses_shared_kv {
                let tensor = unsafe {
                    (api.ggml_new_tensor_2d)(ctx, GGML_TYPE_F32, k_dim as i64, batch_count as i64)
                };
                if tensor.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to allocate shared V input"
                    );
                }
                Some(tensor)
            } else {
                None
            };

            let mut uploads: Vec<(*mut GgmlTensor, Vec<u8>)> = Vec::new();

            let attn_q_w = unsafe {
                alloc_linear_weight_tensor(
                    api,
                    ctx,
                    &mut uploads,
                    store,
                    &layer.attn_q,
                    hidden_dim,
                    q_dim,
                    label,
                )?
            };
            let attn_k_w = if spec.uses_shared_kv {
                None
            } else {
                Some(unsafe {
                    alloc_linear_weight_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        &layer.attn_k,
                        hidden_dim,
                        k_dim,
                        label,
                    )?
                })
            };
            let attn_v_w = if spec.uses_shared_kv {
                None
            } else {
                Some(unsafe {
                    alloc_linear_weight_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        &layer.attn_v,
                        hidden_dim,
                        v_dim,
                        label,
                    )?
                })
            };
            let attn_output_w = unsafe {
                alloc_linear_weight_tensor(
                    api,
                    ctx,
                    &mut uploads,
                    store,
                    &layer.attn_output,
                    q_dim,
                    hidden_dim,
                    label,
                )?
            };
            let ffn_gate_dim =
                layer.ffn_gate.dimensions.get(1).copied().unwrap_or_default() as usize;
            let ffn_up_dim = layer.ffn_up.dimensions.get(1).copied().unwrap_or_default() as usize;
            let ffn_down_in =
                layer.ffn_down.dimensions.first().copied().unwrap_or_default() as usize;
            let ffn_down_out =
                layer.ffn_down.dimensions.get(1).copied().unwrap_or_default() as usize;
            if ffn_gate_dim == 0
                || ffn_gate_dim != ffn_up_dim
                || ffn_down_in != ffn_gate_dim
                || ffn_down_out != hidden_dim
            {
                bail!(
                    "ggml full head prefill graph `{label}` invalid FFN dims on layer {}: gate={:?} up={:?} down={:?}",
                    layer.layer_index,
                    layer.ffn_gate.dimensions,
                    layer.ffn_up.dimensions,
                    layer.ffn_down.dimensions
                );
            }
            let ffn_gate_w = unsafe {
                alloc_linear_weight_tensor(
                    api,
                    ctx,
                    &mut uploads,
                    store,
                    &layer.ffn_gate,
                    hidden_dim,
                    ffn_gate_dim,
                    label,
                )?
            };
            let ffn_up_w = unsafe {
                alloc_linear_weight_tensor(
                    api,
                    ctx,
                    &mut uploads,
                    store,
                    &layer.ffn_up,
                    hidden_dim,
                    ffn_up_dim,
                    label,
                )?
            };
            let ffn_down_w = unsafe {
                alloc_linear_weight_tensor(
                    api,
                    ctx,
                    &mut uploads,
                    store,
                    &layer.ffn_down,
                    ffn_down_in,
                    hidden_dim,
                    label,
                )?
            };

            let attn_norm_w = match layer.attn_norm.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_f32_vector_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        hidden_dim,
                        label,
                    )?
                }),
                None => None,
            };
            let attn_q_norm_w = match layer.attn_q_norm.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_f32_vector_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        spec.head_dim,
                        label,
                    )?
                }),
                None => None,
            };
            let attn_k_norm_w = match layer.attn_k_norm.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_f32_vector_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        spec.head_dim,
                        label,
                    )?
                }),
                None => None,
            };
            let post_attn_norm_w = match layer.post_attention_norm.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_f32_vector_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        hidden_dim,
                        label,
                    )?
                }),
                None => None,
            };
            let ffn_norm_w = match layer.ffn_norm.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_f32_vector_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        hidden_dim,
                        label,
                    )?
                }),
                None => None,
            };
            let post_ffn_norm_w = match layer.post_ffw_norm.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_f32_vector_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        hidden_dim,
                        label,
                    )?
                }),
                None => None,
            };

            let inp_gate_w = match layer.inp_gate.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_linear_weight_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        hidden_dim,
                        entry.dimensions.get(1).copied().unwrap_or_default() as usize,
                        label,
                    )?
                }),
                None => None,
            };
            let proj_w = match layer.proj.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_linear_weight_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        entry.dimensions.first().copied().unwrap_or_default() as usize,
                        entry.dimensions.get(1).copied().unwrap_or_default() as usize,
                        label,
                    )?
                }),
                None => None,
            };
            let post_norm_w = match layer.post_norm.as_ref() {
                Some(entry) => Some(unsafe {
                    alloc_f32_vector_tensor(
                        api,
                        ctx,
                        &mut uploads,
                        store,
                        entry,
                        hidden_dim,
                        label,
                    )?
                }),
                None => None,
            };

            let layer_scale = if let Some(entry) = layer.layer_output_scale.as_ref() {
                let raw = tensor_f32_bytes(store, entry)?;
                if raw.len() < 4 {
                    bail!(
                        "ggml full head prefill graph `{label}` invalid layer scale tensor `{}`",
                        entry.name
                    );
                }
                Some(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
            } else {
                None
            };

            let apply_rms_norm = |input: *mut GgmlTensor,
                                  weight: Option<*mut GgmlTensor>,
                                  tag: &str|
             -> Result<*mut GgmlTensor> {
                let normed = unsafe { (api.ggml_rms_norm)(ctx, input, 1e-6) };
                if normed.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to build rms_norm for {tag}"
                    );
                }
                if let Some(weight) = weight {
                    let repeated = unsafe { (api.ggml_repeat)(ctx, weight, normed) };
                    if repeated.is_null() {
                        bail!(
                            "ggml full head prefill graph `{label}` failed to repeat weights for {tag}"
                        );
                    }
                    let scaled = unsafe { (api.ggml_mul)(ctx, normed, repeated) };
                    if scaled.is_null() {
                        bail!(
                            "ggml full head prefill graph `{label}` failed to apply weights for {tag}"
                        );
                    }
                    Ok(scaled)
                } else {
                    Ok(normed)
                }
            };

            let gelu_tanh_mul = |gate: *mut GgmlTensor,
                                 other: *mut GgmlTensor,
                                 tag: &str|
             -> Result<*mut GgmlTensor> {
                let square = unsafe { (api.ggml_mul)(ctx, gate, gate) };
                if square.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to square {tag}");
                }
                let cube = unsafe { (api.ggml_mul)(ctx, square, gate) };
                if cube.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to cube {tag}");
                }
                let cube_scaled = unsafe { (api.ggml_scale)(ctx, cube, 0.044715) };
                if cube_scaled.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to scale cube for {tag}");
                }
                let inner_pre = unsafe { (api.ggml_add)(ctx, gate, cube_scaled) };
                if inner_pre.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to add cube term for {tag}"
                    );
                }
                let inner = unsafe {
                    (api.ggml_scale)(ctx, inner_pre, (2.0f32 / std::f32::consts::PI).sqrt())
                };
                if inner.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to scale tanh input for {tag}"
                    );
                }
                let tanh = unsafe { (api.ggml_tanh)(ctx, inner) };
                if tanh.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to apply tanh for {tag}");
                }
                let scalar_one_repeat = unsafe { (api.ggml_repeat)(ctx, scalar_one, tanh) };
                if scalar_one_repeat.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to repeat scalar one for {tag}"
                    );
                }
                let tanh_plus_one = unsafe { (api.ggml_add)(ctx, tanh, scalar_one_repeat) };
                if tanh_plus_one.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to add one for {tag}");
                }
                let gate_half = unsafe { (api.ggml_scale)(ctx, gate, 0.5) };
                if gate_half.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to scale gate for {tag}");
                }
                let gelu = unsafe { (api.ggml_mul)(ctx, gate_half, tanh_plus_one) };
                if gelu.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to build GELU for {tag}");
                }
                let combined = unsafe { (api.ggml_mul)(ctx, gelu, other) };
                if combined.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to multiply pair for {tag}"
                    );
                }
                Ok(combined)
            };

            let mut attn_input = hidden_input;
            if let Some(weight) = attn_norm_w {
                attn_input = apply_rms_norm(hidden_input, Some(weight), "attn_norm")?;
            }

            let q_raw = unsafe { (api.ggml_mul_mat)(ctx, attn_q_w, attn_input) };
            if q_raw.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build Q matmul");
            }
            unsafe { (api.ggml_mul_mat_set_prec)(q_raw, GGML_PREC_F32) };
            let k_raw = if let Some(attn_k_w) = attn_k_w {
                let k_raw = unsafe { (api.ggml_mul_mat)(ctx, attn_k_w, attn_input) };
                if k_raw.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to build K matmul");
                }
                unsafe { (api.ggml_mul_mat_set_prec)(k_raw, GGML_PREC_F32) };
                k_raw
            } else {
                shared_k_input.expect("shared K input allocated")
            };
            let v_raw = if let Some(attn_v_w) = attn_v_w {
                let v_raw = unsafe { (api.ggml_mul_mat)(ctx, attn_v_w, attn_input) };
                if v_raw.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to build V matmul");
                }
                unsafe { (api.ggml_mul_mat_set_prec)(v_raw, GGML_PREC_F32) };
                v_raw
            } else {
                shared_v_input.expect("shared V input allocated")
            };

            let q_3d = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    q_raw,
                    spec.head_dim as i64,
                    spec.n_heads as i64,
                    batch_count as i64,
                )
            };
            let k_3d = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    k_raw,
                    spec.head_dim as i64,
                    spec.n_kv_heads as i64,
                    batch_count as i64,
                )
            };
            let v_3d = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    v_raw,
                    spec.head_dim as i64,
                    spec.n_kv_heads as i64,
                    batch_count as i64,
                )
            };
            if q_3d.is_null() || k_3d.is_null() || v_3d.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to reshape qkv tensors");
            }

            let q_normed = apply_rms_norm(q_3d, attn_q_norm_w, "attn_q_norm")?;
            let k_normed = if spec.uses_shared_kv {
                k_3d
            } else {
                apply_rms_norm(k_3d, attn_k_norm_w, "attn_k_norm")?
            };
            let v_normed =
                if spec.uses_shared_kv { v_3d } else { apply_rms_norm(v_3d, None, "attn_v_norm")? };

            let q_4d = unsafe {
                (api.ggml_reshape_4d)(
                    ctx,
                    q_normed,
                    spec.head_dim as i64,
                    spec.n_heads as i64,
                    batch_count as i64,
                    1,
                )
            };
            let k_4d = unsafe {
                (api.ggml_reshape_4d)(
                    ctx,
                    k_normed,
                    spec.head_dim as i64,
                    spec.n_kv_heads as i64,
                    batch_count as i64,
                    1,
                )
            };
            let v_4d = unsafe {
                (api.ggml_reshape_4d)(
                    ctx,
                    v_normed,
                    spec.head_dim as i64,
                    spec.n_kv_heads as i64,
                    batch_count as i64,
                    1,
                )
            };
            if q_4d.is_null() || k_4d.is_null() || v_4d.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to reshape qkv 4d tensors");
            }
            if std::env::var_os("COMPUTE_GGML_DEBUG_FULL_GRAPH").is_some() {
                eprintln!(
                    "ggml full head prefill `{label}` q4={:?} k4={:?} v4={:?}",
                    unsafe { (*q_4d).ne },
                    unsafe { (*k_4d).ne },
                    unsafe { (*v_4d).ne },
                );
            }

            let q_rope = if spec.proportional_rope && spec.uses_shared_kv {
                let pair_count = spec.rope_rotary_dim / 2;
                let cos_4d = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        proportional_q_cos.expect("proportional Q cos allocated"),
                        pair_count as i64,
                        spec.n_heads as i64,
                        batch_count as i64,
                        1,
                    )
                };
                let sin_4d = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        proportional_q_sin.expect("proportional Q sin allocated"),
                        pair_count as i64,
                        spec.n_heads as i64,
                        batch_count as i64,
                        1,
                    )
                };
                unsafe {
                    build_proportional_neox_rope_tensor(
                        api,
                        ctx,
                        q_4d,
                        cos_4d,
                        sin_4d,
                        spec.head_dim,
                        spec.rope_rotary_dim,
                        label,
                    )?
                }
            } else {
                unsafe {
                    (api.ggml_rope_ext)(
                        ctx,
                        q_4d,
                        positions_input,
                        rope_freq_factors.unwrap_or(std::ptr::null_mut()),
                        spec.rope_rotary_dim as c_int,
                        GGML_ROPE_TYPE_NEOX,
                        0,
                        spec.rope_base_theta,
                        1.0,
                        0.0,
                        1.0,
                        32.0,
                        1.0,
                    )
                }
            };
            let k_rope = if spec.uses_shared_kv {
                k_4d
            } else {
                unsafe {
                    (api.ggml_rope_ext)(
                        ctx,
                        k_4d,
                        positions_input,
                        rope_freq_factors.unwrap_or(std::ptr::null_mut()),
                        spec.rope_rotary_dim as c_int,
                        GGML_ROPE_TYPE_NEOX,
                        0,
                        spec.rope_base_theta,
                        1.0,
                        0.0,
                        1.0,
                        32.0,
                        1.0,
                    )
                }
            };
            if q_rope.is_null() || k_rope.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build Q/K rope ops");
            }
            if std::env::var_os("COMPUTE_GGML_DEBUG_FULL_GRAPH").is_some() {
                eprintln!(
                    "ggml full head prefill `{label}` q_rope={:?} k_rope={:?}",
                    unsafe { (*q_rope).ne },
                    unsafe { (*k_rope).ne },
                );
            }

            let mut k_flash_src = k_rope;
            let mut v_flash_src = v_4d;
            if spec.n_heads != spec.n_kv_heads {
                let head_groups = spec.n_heads / spec.n_kv_heads;
                let repeated_elements = spec
                    .head_dim
                    .checked_mul(spec.n_heads)
                    .and_then(|count| count.checked_mul(batch_count))
                    .ok_or_else(|| {
                        anyhow!("ggml full head prefill graph `{label}` repeat size overflow")
                    })?;
                let k_repeat_src = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        k_rope,
                        spec.head_dim as i64,
                        1,
                        spec.n_kv_heads as i64,
                        batch_count as i64,
                    )
                };
                let v_repeat_src = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        v_4d,
                        spec.head_dim as i64,
                        1,
                        spec.n_kv_heads as i64,
                        batch_count as i64,
                    )
                };
                if k_repeat_src.is_null() || v_repeat_src.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to permute KV repeat source tensors"
                    );
                }
                let k_repeat_base = unsafe {
                    (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_F32, repeated_elements as i64)
                };
                let v_repeat_base = unsafe {
                    (api.ggml_new_tensor_1d)(ctx, GGML_TYPE_F32, repeated_elements as i64)
                };
                if k_repeat_base.is_null() || v_repeat_base.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to allocate repeat targets"
                    );
                }
                let k_repeat_target = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        k_repeat_base,
                        spec.head_dim as i64,
                        head_groups as i64,
                        spec.n_kv_heads as i64,
                        batch_count as i64,
                    )
                };
                let v_repeat_target = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        v_repeat_base,
                        spec.head_dim as i64,
                        head_groups as i64,
                        spec.n_kv_heads as i64,
                        batch_count as i64,
                    )
                };
                if k_repeat_target.is_null() || v_repeat_target.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to reshape repeat targets"
                    );
                }
                if std::env::var_os("COMPUTE_GGML_DEBUG_FULL_GRAPH").is_some() {
                    eprintln!(
                        "ggml full head prefill `{label}` kv-repeat k_src={:?} v_src={:?} k_tgt={:?} v_tgt={:?}",
                        unsafe { (*k_repeat_src).ne },
                        unsafe { (*v_repeat_src).ne },
                        unsafe { (*k_repeat_target).ne },
                        unsafe { (*v_repeat_target).ne },
                    );
                }
                k_flash_src = unsafe { (api.ggml_repeat)(ctx, k_repeat_src, k_repeat_target) };
                v_flash_src = unsafe { (api.ggml_repeat)(ctx, v_repeat_src, v_repeat_target) };
                if k_flash_src.is_null() || v_flash_src.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to expand KV heads");
                }
                k_flash_src = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        k_flash_src,
                        spec.head_dim as i64,
                        spec.n_heads as i64,
                        batch_count as i64,
                        1,
                    )
                };
                v_flash_src = unsafe {
                    (api.ggml_reshape_4d)(
                        ctx,
                        v_flash_src,
                        spec.head_dim as i64,
                        spec.n_heads as i64,
                        batch_count as i64,
                        1,
                    )
                };
                if k_flash_src.is_null() || v_flash_src.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to reshape expanded KV heads"
                    );
                }
            }

            let q_flash = unsafe { (api.ggml_permute)(ctx, q_rope, 0, 2, 1, 3) };
            let k_flash = unsafe { (api.ggml_permute)(ctx, k_flash_src, 0, 2, 1, 3) };
            let v_flash = unsafe { (api.ggml_permute)(ctx, v_flash_src, 0, 2, 1, 3) };
            if q_flash.is_null() || k_flash.is_null() || v_flash.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to permute qkv tensors");
            }
            let q_flash = unsafe { (api.ggml_cont)(ctx, q_flash) };
            let k_flash = unsafe { (api.ggml_cont)(ctx, k_flash) };
            let v_flash = unsafe { (api.ggml_cont)(ctx, v_flash) };
            if q_flash.is_null() || k_flash.is_null() || v_flash.is_null() {
                bail!(
                    "ggml full head prefill graph `{label}` failed to materialize qkv flash tensors"
                );
            }
            let q_shape = unsafe { (*q_flash).ne };
            let k_shape = unsafe { (*k_flash).ne };
            let v_shape = unsafe { (*v_flash).ne };
            if std::env::var_os("COMPUTE_GGML_DEBUG_FULL_GRAPH").is_some() {
                eprintln!(
                    "ggml full head prefill `{label}` q={:?} k={:?} v={:?} mask={:?}",
                    q_shape,
                    k_shape,
                    v_shape,
                    unsafe { (*mask_input).ne }
                );
            }
            if k_shape[1] != v_shape[1] || k_shape[2] != v_shape[2] {
                bail!(
                    "ggml full head prefill graph `{label}` produced mismatched K/V shapes: k={:?} v={:?} q={:?}",
                    k_shape,
                    v_shape,
                    q_shape
                );
            }

            let q_attn = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    q_flash,
                    spec.head_dim as i64,
                    batch_count as i64,
                    spec.n_heads as i64,
                )
            };
            let k_attn = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    k_flash,
                    spec.head_dim as i64,
                    batch_count as i64,
                    spec.n_heads as i64,
                )
            };
            let v_attn = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    v_flash,
                    spec.head_dim as i64,
                    batch_count as i64,
                    spec.n_heads as i64,
                )
            };
            if q_attn.is_null() || k_attn.is_null() || v_attn.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to reshape attention tensors");
            }

            let attn_scores = unsafe { (api.ggml_mul_mat)(ctx, k_attn, q_attn) };
            if attn_scores.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build attention scores");
            }
            unsafe { (api.ggml_mul_mat_set_prec)(attn_scores, GGML_PREC_F32) };
            let mask_repeat_base = unsafe {
                (api.ggml_new_tensor_1d)(
                    ctx,
                    GGML_TYPE_F32,
                    (batch_count * batch_count * spec.n_heads) as i64,
                )
            };
            if mask_repeat_base.is_null() {
                bail!(
                    "ggml full head prefill graph `{label}` failed to allocate attention mask repeat target"
                );
            }
            let mask_repeat_target = unsafe {
                (api.ggml_reshape_3d)(
                    ctx,
                    mask_repeat_base,
                    batch_count as i64,
                    batch_count as i64,
                    spec.n_heads as i64,
                )
            };
            if mask_repeat_target.is_null() {
                bail!(
                    "ggml full head prefill graph `{label}` failed to reshape attention mask repeat target"
                );
            }
            let mask_attn = unsafe { (api.ggml_repeat)(ctx, mask_input, mask_repeat_target) };
            if mask_attn.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to repeat attention mask");
            }
            let attn_scores = unsafe { (api.ggml_add)(ctx, attn_scores, mask_attn) };
            if attn_scores.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to apply attention mask");
            }
            let attn_scores = unsafe { (api.ggml_soft_max)(ctx, attn_scores) };
            if attn_scores.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build attention softmax");
            }
            let v_attn_t = unsafe {
                (api.ggml_reshape_4d)(
                    ctx,
                    v_attn,
                    spec.head_dim as i64,
                    batch_count as i64,
                    spec.n_heads as i64,
                    1,
                )
            };
            if v_attn_t.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to reshape V for context");
            }
            let v_attn_t = unsafe { (api.ggml_permute)(ctx, v_attn_t, 1, 0, 2, 3) };
            if v_attn_t.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to transpose V for context");
            }
            let v_attn_t = unsafe { (api.ggml_cont)(ctx, v_attn_t) };
            if v_attn_t.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to materialize transposed V");
            }

            let attn_ctx = unsafe { (api.ggml_mul_mat)(ctx, v_attn_t, attn_scores) };
            if attn_ctx.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build attention context");
            }
            unsafe { (api.ggml_mul_mat_set_prec)(attn_ctx, GGML_PREC_F32) };
            let attn_ctx = unsafe {
                (api.ggml_reshape_4d)(
                    ctx,
                    attn_ctx,
                    spec.head_dim as i64,
                    batch_count as i64,
                    spec.n_heads as i64,
                    1,
                )
            };
            if attn_ctx.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to reshape attention context");
            }
            let attn_ctx = unsafe { (api.ggml_permute)(ctx, attn_ctx, 0, 2, 1, 3) };
            if attn_ctx.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to permute attention context");
            }
            let attn_ctx = unsafe { (api.ggml_cont)(ctx, attn_ctx) };
            if attn_ctx.is_null() {
                bail!(
                    "ggml full head prefill graph `{label}` failed to materialize attention context"
                );
            }

            let attn_flat =
                unsafe { (api.ggml_cont_2d)(ctx, attn_ctx, q_dim as i64, batch_count as i64) };
            if attn_flat.is_null() {
                bail!(
                    "ggml full head prefill graph `{label}` failed to flatten flash attention output"
                );
            }
            let attn_projected = unsafe { (api.ggml_mul_mat)(ctx, attn_output_w, attn_flat) };
            if attn_projected.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build attn output matmul");
            }
            unsafe { (api.ggml_mul_mat_set_prec)(attn_projected, GGML_PREC_F32) };

            let mut state = if let Some(weight) = post_attn_norm_w {
                apply_rms_norm(attn_projected, Some(weight), "post_attention_norm")?
            } else {
                attn_projected
            };
            state = unsafe { (api.ggml_add)(ctx, state, hidden_input) };
            if state.is_null() {
                bail!(
                    "ggml full head prefill graph `{label}` failed to build post-attention residual"
                );
            }

            let ffn_input = if let Some(weight) = ffn_norm_w {
                apply_rms_norm(state, Some(weight), "ffn_norm")?
            } else {
                state
            };

            let gate = unsafe { (api.ggml_mul_mat)(ctx, ffn_gate_w, ffn_input) };
            let up = unsafe { (api.ggml_mul_mat)(ctx, ffn_up_w, ffn_input) };
            if gate.is_null() || up.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build FFN gate/up matmuls");
            }
            unsafe {
                (api.ggml_mul_mat_set_prec)(gate, GGML_PREC_F32);
                (api.ggml_mul_mat_set_prec)(up, GGML_PREC_F32);
            }
            let gate_activated = gelu_tanh_mul(gate, up, "ffn_gate_up")?;
            let down = unsafe { (api.ggml_mul_mat)(ctx, ffn_down_w, gate_activated) };
            if down.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build FFN down matmul");
            }
            unsafe { (api.ggml_mul_mat_set_prec)(down, GGML_PREC_F32) };

            let state_after_attn = state;
            let mut state = if let Some(weight) = post_ffn_norm_w {
                apply_rms_norm(down, Some(weight), "post_ffn_norm")?
            } else {
                down
            };
            state = unsafe { (api.ggml_add)(ctx, state, state_after_attn) };
            if state.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to build FFN residual");
            }

            if let (Some(inp_gate_w), Some(proj_w), Some(post_norm_w), Some(prompt_aux_input)) =
                (inp_gate_w, proj_w, post_norm_w, prompt_aux_input)
            {
                let residual_after_ffn = state;
                let ple_gate = unsafe { (api.ggml_mul_mat)(ctx, inp_gate_w, residual_after_ffn) };
                if ple_gate.is_null() {
                    bail!(
                        "ggml full head prefill graph `{label}` failed to build PLE inp_gate matmul"
                    );
                }
                unsafe { (api.ggml_mul_mat_set_prec)(ple_gate, GGML_PREC_F32) };
                let ple_gate = gelu_tanh_mul(ple_gate, prompt_aux_input, "ple_inp_gate")?;
                let ple_proj = unsafe { (api.ggml_mul_mat)(ctx, proj_w, ple_gate) };
                if ple_proj.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to build PLE proj matmul");
                }
                unsafe { (api.ggml_mul_mat_set_prec)(ple_proj, GGML_PREC_F32) };
                let ple_proj = apply_rms_norm(ple_proj, Some(post_norm_w), "post_norm")?;
                state = unsafe { (api.ggml_add)(ctx, ple_proj, residual_after_ffn) };
                if state.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to build PLE residual");
                }
            }

            if let Some(scale) = layer_scale {
                state = unsafe { (api.ggml_scale)(ctx, state, scale) };
                if state.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to apply layer scale");
                }
            }

            let hidden_output =
                unsafe { (api.ggml_cont_2d)(ctx, state, hidden_dim as i64, batch_count as i64) };
            if hidden_output.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to flatten hidden output");
            }

            let k_cache_output = if spec.uses_shared_kv {
                None
            } else {
                let output =
                    unsafe { (api.ggml_cont_2d)(ctx, k_rope, k_dim as i64, batch_count as i64) };
                if output.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to flatten K cache");
                }
                Some(output)
            };
            let v_cache_output = if spec.uses_shared_kv {
                None
            } else {
                let output =
                    unsafe { (api.ggml_cont_2d)(ctx, v_normed, k_dim as i64, batch_count as i64) };
                if output.is_null() {
                    bail!("ggml full head prefill graph `{label}` failed to flatten V cache");
                }
                Some(output)
            };

            let graph = unsafe { (api.ggml_new_graph)(ctx) };
            if graph.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to allocate compute graph");
            }
            unsafe {
                (api.ggml_build_forward_expand)(graph, hidden_output);
                if let Some(k_cache_output) = k_cache_output {
                    (api.ggml_build_forward_expand)(graph, k_cache_output);
                }
                if let Some(v_cache_output) = v_cache_output {
                    (api.ggml_build_forward_expand)(graph, v_cache_output);
                }
            }

            let buffer = unsafe { (api.ggml_backend_alloc_ctx_tensors)(ctx, backend) };
            if buffer.is_null() {
                bail!("ggml full head prefill graph `{label}` failed to allocate backend tensors");
            }

            let position_values: Vec<i32> = (0..batch_count)
                .map(|idx| i32::try_from(idx).expect("batch_count fits in i32"))
                .collect();
            uploads.push((positions_input, encode_i32_bytes(&position_values)));
            uploads.push((
                mask_input,
                encode_f32_bytes(&build_causal_mask(batch_count, spec.sliding_window)),
            ));
            uploads.push((scalar_one, encode_f32_bytes(&[1.0])));
            if let Some(rope_freq_factors) = rope_freq_factors {
                uploads.push((
                    rope_freq_factors,
                    encode_f32_bytes(&build_proportional_rope_freq_factors(
                        spec.rope_base_theta,
                        spec.head_dim,
                        spec.rope_rotary_dim,
                    )),
                ));
            }
            if let (Some(proportional_q_cos), Some(proportional_q_sin)) =
                (proportional_q_cos, proportional_q_sin)
            {
                let (cos, sin) = build_proportional_rope_head_tables(
                    spec.rope_base_theta,
                    spec.head_dim,
                    spec.rope_rotary_dim,
                    spec.n_heads,
                    batch_count,
                    0,
                    None,
                )?;
                uploads.push((proportional_q_cos, encode_f32_bytes(&cos)));
                uploads.push((proportional_q_sin, encode_f32_bytes(&sin)));
            }

            for (tensor, raw) in uploads {
                unsafe {
                    (api.ggml_backend_tensor_set)(
                        tensor,
                        raw.as_ptr().cast::<c_void>(),
                        0,
                        raw.len(),
                    );
                }
            }

            Ok(Self {
                api,
                backend,
                ctx,
                buffer,
                graph,
                hidden_input,
                prompt_aux_input,
                shared_k_input,
                shared_v_input,
                hidden_output,
                k_cache_output,
                v_cache_output,
                hidden_dim,
                k_dim,
                ple_dim,
                batch_count,
                layer_index: layer.layer_index,
                uses_shared_kv: spec.uses_shared_kv,
                backend_name,
            })
        })();

        if result.is_err() {
            unsafe {
                (api.ggml_free)(ctx);
                (api.ggml_backend_free)(backend);
            }
        }
        result
    }

    pub fn summary_label(&self) -> String {
        format!(
            "ggml-full-head-prefill backend={} layer={} hidden_dim={} k_dim={} batch_count={} ple_dim={} shared_kv={}",
            self.backend_name,
            self.layer_index,
            self.hidden_dim,
            self.k_dim,
            self.batch_count,
            self.ple_dim.unwrap_or(0),
            self.uses_shared_kv
        )
    }

    pub fn run(
        &mut self,
        hidden_states: &[Vec<f32>],
        prompt_aux: Option<&[Vec<f32>]>,
        shared_attention_cache: Option<(&[Vec<f32>], &[Vec<f32>])>,
    ) -> Result<GgmlFullHeadPrefillLayerRunResult> {
        if hidden_states.len() != self.batch_count {
            bail!(
                "ggml full head prefill input batch mismatch: got {} expected {}",
                hidden_states.len(),
                self.batch_count
            );
        }
        let mut hidden_flat = Vec::with_capacity(self.hidden_dim * self.batch_count);
        for state in hidden_states {
            if state.len() != self.hidden_dim {
                bail!(
                    "ggml full head prefill hidden width mismatch: got {} expected {}",
                    state.len(),
                    self.hidden_dim
                );
            }
            hidden_flat.extend_from_slice(state);
        }
        unsafe {
            (self.api.ggml_backend_tensor_set)(
                self.hidden_input,
                f32_bytes(&hidden_flat).as_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(hidden_flat.as_slice()),
            );
        }

        match (self.prompt_aux_input, self.ple_dim, prompt_aux) {
            (Some(tensor), Some(ple_dim), Some(prompt_aux)) => {
                if prompt_aux.len() != self.batch_count {
                    bail!(
                        "ggml full head prefill prompt-aux batch mismatch: got {} expected {}",
                        prompt_aux.len(),
                        self.batch_count
                    );
                }
                let mut prompt_aux_flat = Vec::with_capacity(ple_dim * self.batch_count);
                for values in prompt_aux {
                    if values.len() != ple_dim {
                        bail!(
                            "ggml full head prefill prompt-aux width mismatch: got {} expected {}",
                            values.len(),
                            ple_dim
                        );
                    }
                    prompt_aux_flat.extend_from_slice(values);
                }
                unsafe {
                    (self.api.ggml_backend_tensor_set)(
                        tensor,
                        f32_bytes(&prompt_aux_flat).as_ptr().cast::<c_void>(),
                        0,
                        std::mem::size_of_val(prompt_aux_flat.as_slice()),
                    );
                }
            }
            (Some(_), Some(_), None) => {
                bail!("ggml full head prefill runtime requires prompt-aux input");
            }
            _ => {}
        }

        match (self.shared_k_input, self.shared_v_input, shared_attention_cache) {
            (
                Some(shared_k_input),
                Some(shared_v_input),
                Some((shared_k_cache, shared_v_cache)),
            ) => {
                if shared_k_cache.len() != self.batch_count
                    || shared_v_cache.len() != self.batch_count
                {
                    bail!(
                        "ggml full head prefill shared-KV batch mismatch: k={} v={} expected={}",
                        shared_k_cache.len(),
                        shared_v_cache.len(),
                        self.batch_count
                    );
                }
                let mut shared_k_flat = Vec::with_capacity(self.k_dim * self.batch_count);
                let mut shared_v_flat = Vec::with_capacity(self.k_dim * self.batch_count);
                for (k, v) in shared_k_cache.iter().zip(shared_v_cache.iter()) {
                    if k.len() != self.k_dim || v.len() != self.k_dim {
                        bail!(
                            "ggml full head prefill shared-KV width mismatch: k={} v={} expected={}",
                            k.len(),
                            v.len(),
                            self.k_dim
                        );
                    }
                    shared_k_flat.extend_from_slice(k);
                    shared_v_flat.extend_from_slice(v);
                }
                unsafe {
                    (self.api.ggml_backend_tensor_set)(
                        shared_k_input,
                        f32_bytes(&shared_k_flat).as_ptr().cast::<c_void>(),
                        0,
                        std::mem::size_of_val(shared_k_flat.as_slice()),
                    );
                    (self.api.ggml_backend_tensor_set)(
                        shared_v_input,
                        f32_bytes(&shared_v_flat).as_ptr().cast::<c_void>(),
                        0,
                        std::mem::size_of_val(shared_v_flat.as_slice()),
                    );
                }
            }
            (Some(_), Some(_), None) => {
                bail!("ggml full head prefill runtime requires shared-KV input");
            }
            (None, None, Some(_)) => {
                bail!("ggml full head prefill runtime received unexpected shared-KV input");
            }
            _ => {}
        }

        let status = unsafe { (self.api.ggml_backend_graph_compute)(self.backend, self.graph) };
        if status != GGML_STATUS_SUCCESS {
            bail!("ggml full head prefill graph compute failed with status {}", status);
        }
        unsafe { (self.api.ggml_backend_synchronize)(self.backend) };

        let mut hidden_flat = vec![0.0f32; self.hidden_dim * self.batch_count];
        unsafe {
            (self.api.ggml_backend_tensor_get)(
                self.hidden_output,
                hidden_flat.as_mut_ptr().cast::<c_void>(),
                0,
                std::mem::size_of_val(hidden_flat.as_slice()),
            );
        }
        let k_cache = if let Some(k_cache_output) = self.k_cache_output {
            let mut k_flat = vec![0.0f32; self.k_dim * self.batch_count];
            unsafe {
                (self.api.ggml_backend_tensor_get)(
                    k_cache_output,
                    k_flat.as_mut_ptr().cast::<c_void>(),
                    0,
                    std::mem::size_of_val(k_flat.as_slice()),
                );
            }
            Some(k_flat.chunks_exact(self.k_dim).map(|chunk| chunk.to_vec()).collect())
        } else {
            None
        };
        let v_cache = if let Some(v_cache_output) = self.v_cache_output {
            let mut v_flat = vec![0.0f32; self.k_dim * self.batch_count];
            unsafe {
                (self.api.ggml_backend_tensor_get)(
                    v_cache_output,
                    v_flat.as_mut_ptr().cast::<c_void>(),
                    0,
                    std::mem::size_of_val(v_flat.as_slice()),
                );
            }
            Some(v_flat.chunks_exact(self.k_dim).map(|chunk| chunk.to_vec()).collect())
        } else {
            None
        };

        Ok(GgmlFullHeadPrefillLayerRunResult {
            hidden_states: hidden_flat
                .chunks_exact(self.hidden_dim)
                .map(|chunk| chunk.to_vec())
                .collect(),
            k_cache,
            v_cache,
        })
    }
}

impl Drop for GgmlLinearGraphRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                (self.api.ggml_backend_buffer_free)(self.buffer);
            }
            if !self.ctx.is_null() {
                (self.api.ggml_free)(self.ctx);
            }
        }
    }
}

impl Drop for GgmlOwnedBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.backend.is_null() {
                (self.api.ggml_backend_free)(self.backend);
            }
        }
    }
}

impl Drop for GgmlGetRowsGraphRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                (self.api.ggml_backend_buffer_free)(self.buffer);
            }
            if !self.backend.is_null() {
                (self.api.ggml_backend_free)(self.backend);
            }
            if !self.ctx.is_null() {
                (self.api.ggml_free)(self.ctx);
            }
        }
    }
}

impl Drop for GgmlPleIngressGraphRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                (self.api.ggml_backend_buffer_free)(self.buffer);
            }
            if !self.backend.is_null() {
                (self.api.ggml_backend_free)(self.backend);
            }
            if !self.ctx.is_null() {
                (self.api.ggml_free)(self.ctx);
            }
        }
    }
}

impl Drop for GgmlRopeGraphRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                (self.api.ggml_backend_buffer_free)(self.buffer);
            }
            if !self.backend.is_null() {
                (self.api.ggml_backend_free)(self.backend);
            }
            if !self.ctx.is_null() {
                (self.api.ggml_free)(self.ctx);
            }
        }
    }
}

impl Drop for GgmlFullHeadPrefillLayerRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                (self.api.ggml_backend_buffer_free)(self.buffer);
            }
            if !self.backend.is_null() {
                (self.api.ggml_backend_free)(self.backend);
            }
            if !self.ctx.is_null() {
                (self.api.ggml_free)(self.ctx);
            }
        }
    }
}

impl GgmlTailLayerRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        operator_plan: &GgmlStageOperatorPlan,
        store: &StageTensorStore,
    ) -> Result<Self> {
        Self::new_for_layer_with_batch_count(runtime, operator_plan, store, 0, 1)
    }

    pub fn new_for_layer(
        runtime: &GgmlRuntimePlan,
        operator_plan: &GgmlStageOperatorPlan,
        store: &StageTensorStore,
        layer_offset: usize,
    ) -> Result<Self> {
        Self::new_for_layer_with_batch_count(runtime, operator_plan, store, layer_offset, 1)
    }

    pub fn new_for_layer_with_batch_count(
        runtime: &GgmlRuntimePlan,
        operator_plan: &GgmlStageOperatorPlan,
        store: &StageTensorStore,
        layer_offset: usize,
        batch_count: usize,
    ) -> Result<Self> {
        let hidden_dim = operator_plan.hidden_dim()?;
        let layer = operator_plan.layers.get(layer_offset).cloned().ok_or_else(|| {
            anyhow!(
                "ggml tail layer runtime requires layer offset {} within role={}",
                layer_offset,
                operator_plan.role
            )
        })?;

        let api = GgmlDynamicApi::shared()?;
        let (backend, backend_name) = api
            .init_backend(runtime.target)
            .with_context(|| format!("initialize ggml backend for {}", runtime.summary_label()))?;
        let owned_backend = GgmlOwnedBackend { api, backend, backend_name };

        let qkv = GgmlLinearGraphRuntime::new(
            api,
            owned_backend.backend,
            hidden_dim,
            batch_count,
            &[layer.attn_q.clone(), layer.attn_k.clone(), layer.attn_v.clone()],
            store,
            "tail-layer-qkv",
        )?;
        let attn_output = GgmlLinearGraphRuntime::new(
            api,
            owned_backend.backend,
            layer.attn_output.dimensions.first().copied().unwrap_or_default() as usize,
            batch_count,
            &[layer.attn_output.clone()],
            store,
            "tail-layer-attn-output",
        )?;
        let gate_up = GgmlLinearGraphRuntime::new(
            api,
            owned_backend.backend,
            hidden_dim,
            batch_count,
            &[layer.ffn_gate.clone(), layer.ffn_up.clone()],
            store,
            "tail-layer-gate-up",
        )?;
        let down = GgmlLinearGraphRuntime::new(
            api,
            owned_backend.backend,
            layer.ffn_down.dimensions.first().copied().unwrap_or_default() as usize,
            batch_count,
            &[layer.ffn_down.clone()],
            store,
            "tail-layer-down",
        )?;
        let inp_gate = match layer.inp_gate.as_ref() {
            Some(entry) => Some(GgmlLinearGraphRuntime::new(
                api,
                owned_backend.backend,
                hidden_dim,
                batch_count,
                std::slice::from_ref(entry),
                store,
                "tail-layer-inp-gate",
            )?),
            None => None,
        };
        let proj = match layer.proj.as_ref() {
            Some(entry) => Some(GgmlLinearGraphRuntime::new(
                api,
                owned_backend.backend,
                entry.dimensions.first().copied().unwrap_or_default() as usize,
                batch_count,
                std::slice::from_ref(entry),
                store,
                "tail-layer-proj",
            )?),
            None => None,
        };

        Ok(Self {
            qkv,
            attn_output,
            gate_up,
            down,
            inp_gate,
            proj,
            layer_index: layer.layer_index,
            hidden_dim,
            batch_count,
            backend: owned_backend,
        })
    }

    pub fn summary_label(&self) -> String {
        format!(
            "ggml-layer-runtime backend={} layer={} hidden_dim={} batch_count={}",
            self.backend.backend_name, self.layer_index, self.hidden_dim, self.batch_count
        )
    }

    pub fn layer_index(&self) -> u32 {
        self.layer_index
    }

    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    pub fn qkv(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut outputs = self.qkv.run(input)?;
        if outputs.len() != 3 {
            bail!("ggml tail layer qkv runtime returned {} outputs", outputs.len());
        }
        let v = outputs.pop().expect("v output present");
        let k = outputs.pop().expect("k output present");
        let q = outputs.pop().expect("q output present");
        Ok((q, k, v))
    }

    pub fn attn_output(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut outputs = self.attn_output.run(input)?;
        Ok(outputs.pop().expect("attn output present"))
    }

    pub fn gate_up(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut outputs = self.gate_up.run(input)?;
        if outputs.len() != 2 {
            bail!("ggml tail layer gate_up runtime returned {} outputs", outputs.len());
        }
        let up = outputs.pop().expect("up output present");
        let gate = outputs.pop().expect("gate output present");
        Ok((gate, up))
    }

    pub fn qkv_many(
        &mut self,
        inputs: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if self.batch_count != inputs.len() {
            bail!(
                "ggml tail layer qkv_many input count mismatch: got {} expected {}",
                inputs.len(),
                self.batch_count
            );
        }
        let mut flat = Vec::with_capacity(self.batch_count * self.hidden_dim);
        for input in inputs {
            if input.len() != self.hidden_dim {
                bail!(
                    "ggml tail layer qkv_many input width mismatch: got {} expected {}",
                    input.len(),
                    self.hidden_dim
                );
            }
            flat.extend_from_slice(input);
        }
        let mut outputs = self.qkv.run_flat(&flat)?;
        if outputs.len() != self.batch_count * 3 {
            bail!(
                "ggml tail layer qkv_many runtime returned {} outputs for batch_count {}",
                outputs.len(),
                self.batch_count
            );
        }
        let v = outputs.split_off(self.batch_count * 2);
        let k = outputs.split_off(self.batch_count);
        let q = outputs;
        Ok((q, k, v))
    }

    pub fn down(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut outputs = self.down.run(input)?;
        Ok(outputs.pop().expect("down output present"))
    }

    pub fn attn_output_many(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if self.batch_count != inputs.len() {
            bail!(
                "ggml tail layer attn_output_many input count mismatch: got {} expected {}",
                inputs.len(),
                self.batch_count
            );
        }
        let input_dim = self.attn_output.input_dim;
        let mut flat = Vec::with_capacity(self.batch_count * input_dim);
        for input in inputs {
            if input.len() != input_dim {
                bail!(
                    "ggml tail layer attn_output_many input width mismatch: got {} expected {}",
                    input.len(),
                    input_dim
                );
            }
            flat.extend_from_slice(input);
        }
        self.attn_output.run_flat(&flat)
    }

    pub fn gate_up_many(&mut self, inputs: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if self.batch_count != inputs.len() {
            bail!(
                "ggml tail layer gate_up_many input count mismatch: got {} expected {}",
                inputs.len(),
                self.batch_count
            );
        }
        let mut flat = Vec::with_capacity(self.batch_count * self.hidden_dim);
        for input in inputs {
            if input.len() != self.hidden_dim {
                bail!(
                    "ggml tail layer gate_up_many input width mismatch: got {} expected {}",
                    input.len(),
                    self.hidden_dim
                );
            }
            flat.extend_from_slice(input);
        }
        let mut outputs = self.gate_up.run_flat(&flat)?;
        if outputs.len() != self.batch_count * 2 {
            bail!(
                "ggml tail layer gate_up_many runtime returned {} outputs for batch_count {}",
                outputs.len(),
                self.batch_count
            );
        }
        let up = outputs.split_off(self.batch_count);
        let gate = outputs;
        Ok((gate, up))
    }

    pub fn down_many(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if self.batch_count != inputs.len() {
            bail!(
                "ggml tail layer down_many input count mismatch: got {} expected {}",
                inputs.len(),
                self.batch_count
            );
        }
        let input_dim = self.down.input_dim;
        let mut flat = Vec::with_capacity(self.batch_count * input_dim);
        for input in inputs {
            if input.len() != input_dim {
                bail!(
                    "ggml tail layer down_many input width mismatch: got {} expected {}",
                    input.len(),
                    input_dim
                );
            }
            flat.extend_from_slice(input);
        }
        self.down.run_flat(&flat)
    }

    pub fn inp_gate(&mut self, input: &[f32]) -> Result<Option<Vec<f32>>> {
        let Some(runtime) = self.inp_gate.as_mut() else {
            return Ok(None);
        };
        let mut outputs = runtime.run(input)?;
        Ok(Some(outputs.pop().expect("inp_gate output present")))
    }

    pub fn inp_gate_many(&mut self, inputs: &[Vec<f32>]) -> Result<Option<Vec<Vec<f32>>>> {
        let Some(runtime) = self.inp_gate.as_mut() else {
            return Ok(None);
        };
        if self.batch_count != inputs.len() {
            bail!(
                "ggml tail layer inp_gate_many input count mismatch: got {} expected {}",
                inputs.len(),
                self.batch_count
            );
        }
        let input_dim = runtime.input_dim;
        let mut flat = Vec::with_capacity(self.batch_count * input_dim);
        for input in inputs {
            if input.len() != input_dim {
                bail!(
                    "ggml tail layer inp_gate_many input width mismatch: got {} expected {}",
                    input.len(),
                    input_dim
                );
            }
            flat.extend_from_slice(input);
        }
        Ok(Some(runtime.run_flat(&flat)?))
    }

    pub fn proj(&mut self, input: &[f32]) -> Result<Option<Vec<f32>>> {
        let Some(runtime) = self.proj.as_mut() else {
            return Ok(None);
        };
        let mut outputs = runtime.run(input)?;
        Ok(Some(outputs.pop().expect("proj output present")))
    }

    pub fn proj_many(&mut self, inputs: &[Vec<f32>]) -> Result<Option<Vec<Vec<f32>>>> {
        let Some(runtime) = self.proj.as_mut() else {
            return Ok(None);
        };
        if self.batch_count != inputs.len() {
            bail!(
                "ggml tail layer proj_many input count mismatch: got {} expected {}",
                inputs.len(),
                self.batch_count
            );
        }
        let input_dim = runtime.input_dim;
        let mut flat = Vec::with_capacity(self.batch_count * input_dim);
        for input in inputs {
            if input.len() != input_dim {
                bail!(
                    "ggml tail layer proj_many input width mismatch: got {} expected {}",
                    input.len(),
                    input_dim
                );
            }
            flat.extend_from_slice(input);
        }
        Ok(Some(runtime.run_flat(&flat)?))
    }
}

impl GgmlTailStackRuntime {
    pub fn new(
        runtime: &GgmlRuntimePlan,
        operator_plan: &GgmlStageOperatorPlan,
        store: &StageTensorStore,
        layer_cap: usize,
    ) -> Result<Self> {
        Self::new_with_batch_count(runtime, operator_plan, store, layer_cap, 1)
    }

    pub fn new_with_batch_count(
        runtime: &GgmlRuntimePlan,
        operator_plan: &GgmlStageOperatorPlan,
        store: &StageTensorStore,
        layer_cap: usize,
        batch_count: usize,
    ) -> Result<Self> {
        if layer_cap == 0 {
            bail!("ggml tail stack runtime requires layer_cap > 0");
        }
        let count = layer_cap.min(operator_plan.layers.len());
        let mut layers = Vec::with_capacity(count);
        for layer_offset in 0..count {
            layers.push(GgmlTailLayerRuntime::new_for_layer_with_batch_count(
                runtime,
                operator_plan,
                store,
                layer_offset,
                batch_count,
            )?);
        }
        let backend_name = layers
            .first()
            .map(|layer| layer.backend.backend_name.clone())
            .unwrap_or_else(|| "unknown".into());
        Ok(Self { layers, backend_name, batch_count })
    }

    pub fn summary_label(&self) -> String {
        let first = self.layers.first().map(|layer| layer.layer_index()).unwrap_or_default();
        let last = self.layers.last().map(|layer| layer.layer_index()).unwrap_or_default();
        format!(
            "ggml-layer-stack backend={} layers={} range={}..={} batch_count={}",
            self.backend_name,
            self.layers.len(),
            first,
            last,
            self.batch_count
        )
    }

    pub fn layers_mut(&mut self) -> &mut [GgmlTailLayerRuntime] {
        &mut self.layers
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn batch_count(&self) -> usize {
        self.batch_count
    }
}

impl Drop for GgmlSampleGraphRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                (self.api.ggml_backend_buffer_free)(self.buffer);
            }
            if !self.backend.is_null() {
                (self.api.ggml_backend_free)(self.backend);
            }
            if !self.ctx.is_null() {
                (self.api.ggml_free)(self.ctx);
            }
        }
    }
}
