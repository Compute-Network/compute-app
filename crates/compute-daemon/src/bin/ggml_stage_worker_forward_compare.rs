use anyhow::{Result, bail};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::GgmlStageExecutorKind;
use compute_daemon::inference::ggml_stage_worker::{
    GgmlStageWorkerHostLaunchSpec, GgmlStageWorkerInitSpec, GgmlStageWorkerRequest,
    GgmlStageWorkerResponse, GgmlStageWorkerSession, GgmlStageWorkerTensorSummary,
    run_stage_worker_session_request, spawn_in_process_stage_worker_session,
};
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::stage_acceleration::StageAccelerationTarget;
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout, stage_tensor_byte_sections};
use std::env;
use std::path::PathBuf;

fn target_from_str(value: &str) -> StageAccelerationTarget {
    match value {
        "cpu" => StageAccelerationTarget::Cpu,
        "cuda" => StageAccelerationTarget::Cuda,
        "vulkan" => StageAccelerationTarget::Vulkan,
        "directml" => StageAccelerationTarget::DirectMl,
        _ => StageAccelerationTarget::Metal,
    }
}

fn executor_from_str(value: &str) -> GgmlStageExecutorKind {
    match value {
        "ggml" => GgmlStageExecutorKind::Ggml,
        _ => GgmlStageExecutorKind::ReferenceCpu,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionMode {
    Host,
    InProcess,
}

fn session_mode_from_str(value: &str) -> SessionMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "inproc" | "in-process" | "local" | "direct" => SessionMode::InProcess,
        _ => SessionMode::Host,
    }
}

fn spawn_session(
    init: &GgmlStageWorkerInitSpec,
    launch: &GgmlStageWorkerHostLaunchSpec,
    mode: SessionMode,
) -> Result<GgmlStageWorkerSession> {
    match mode {
        SessionMode::Host => launch.spawn_persistent_session(),
        SessionMode::InProcess => spawn_in_process_stage_worker_session(init),
    }
}

fn load_reference_backend(
    load_spec: &RealForwardStageLoadSpec,
    model_id: &str,
    debug_layer_cap: Option<usize>,
) -> Result<RealGemmaBackend> {
    let mut backend = RealGemmaBackend::new(&load_spec.index_path);
    backend.set_debug_layer_cap(debug_layer_cap);
    if let Some(vocab_path) = load_spec.vocab_path.as_deref() {
        backend.load_tokenizer(vocab_path, load_spec.vocab_scores_path.as_deref())?;
    }
    backend.load_layout(StageLayout {
        model_id: model_id.to_string(),
        stage_id: load_spec.layout.stage_id.clone(),
        start_layer: load_spec.layout.start_layer,
        end_layer: load_spec.layout.end_layer,
        is_head: load_spec.layout.is_head,
        is_tail: load_spec.layout.is_tail,
    })?;
    Ok(backend)
}

fn summary_from_tensor(tensor: &stage_forward_lab::StageTensor) -> GgmlStageWorkerTensorSummary {
    GgmlStageWorkerTensorSummary::from_tensor(tensor)
}

fn numeric_tolerance() -> Option<f32> {
    env::var("COMPUTE_GGML_NUMERIC_TOLERANCE").ok()?.parse::<f32>().ok()
}

fn hidden_bytes<'a>(tensor: &'a stage_forward_lab::StageTensor) -> &'a [u8] {
    stage_tensor_byte_sections(&tensor.bytes)
        .map(|parts| parts.hidden_bytes)
        .unwrap_or(&tensor.bytes)
}

fn decode_hidden_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("hidden bytes len {} is not divisible by 4", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        bail!("hidden vector len mismatch: {} vs {}", a.len(), b.len());
    }
    Ok(a.iter().zip(b.iter()).map(|(lhs, rhs)| (lhs - rhs).abs()).fold(0.0f32, f32::max))
}

fn summary_matches_except_hidden_hash(
    reference: &GgmlStageWorkerTensorSummary,
    candidate: &GgmlStageWorkerTensorSummary,
) -> bool {
    reference.kind == candidate.kind
        && reference.hidden_dim == candidate.hidden_dim
        && reference.hidden_state_bytes == candidate.hidden_state_bytes
        && reference.aux_bytes == candidate.aux_bytes
        && reference.stage_trace_depth == candidate.stage_trace_depth
        && reference.aux_bytes_hash == candidate.aux_bytes_hash
        && reference.prompt_text == candidate.prompt_text
        && reference.max_tokens == candidate.max_tokens
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn prompt_aux_debug(bytes: &[u8]) -> String {
    if bytes.len() < 20 || &bytes[..4] != b"rsa1" {
        return format!("invalid_aux_bytes={}", bytes.len());
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
        .unwrap_or(0);
    let ple_slice_end = 20usize.saturating_add(ple_bytes).min(bytes.len());
    let ple_hash = stable_hash(&bytes[20..ple_slice_end]);
    let mut suffix_hashes = Vec::new();
    let mut offset = ple_slice_end;
    for _ in 0..prefix_hash_count.min(3) {
        if offset + 8 > bytes.len() {
            break;
        }
        suffix_hashes.push(u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]));
        offset += 8;
    }
    format!(
        "seq_len={} layers={} ple_dim={} prefix_hashes={} ple_hash={} prefix_preview={:?}",
        seq_len, layer_count, ple_dim, prefix_hash_count, ple_hash, suffix_hashes
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let shard_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
        )
    });
    let model_name = args.get(2).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let start_layer = args.get(3).and_then(|value| value.parse::<u32>().ok()).unwrap_or(0);
    let end_layer = args.get(4).and_then(|value| value.parse::<u32>().ok()).unwrap_or(20);
    let stage_role = args.get(5).map(|value| value.as_str()).unwrap_or("head");
    let requested = args.get(6).map(|value| value.as_str()).unwrap_or("metal");
    let suite_mode = args
        .get(7)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let executor = args
        .get(8)
        .map(|value| executor_from_str(value))
        .unwrap_or(GgmlStageExecutorKind::ReferenceCpu);
    let debug_layer_cap = args.get(9).and_then(|value| value.parse::<usize>().ok());
    let session_mode =
        args.get(10).map(|value| session_mode_from_str(value)).unwrap_or(SessionMode::Host);
    let case_filter = args.get(11).cloned();

    let is_head = matches!(stage_role, "head" | "first");
    let is_tail = matches!(stage_role, "tail" | "last");
    let load_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_name.clone(),
        shard_path: shard_path.clone(),
        start_layer,
        end_layer,
        total_layers: end_layer + 1,
        is_first_stage: is_head,
        is_last_stage: is_tail,
        max_batch_size: 16,
        context_length: 8192,
    })?;
    let runtime = detect_ggml_runtime_plan(target_from_str(requested));
    let mut init = GgmlStageWorkerInitSpec::from_load_spec(&load_spec, &runtime, executor);
    init.debug_layer_cap = debug_layer_cap;
    let launch = GgmlStageWorkerHostLaunchSpec::from_init_spec(&init)?;
    let mut session = spawn_session(&init, &launch, session_mode)?;
    println!("=== GGML Stage Worker Forward Compare ===");
    println!("model        : {model_name}");
    println!("shard        : {}", shard_path.display());
    println!("layers       : {start_layer}-{end_layer}");
    println!("stage role   : {stage_role}");
    println!("suite mode   : {}", suite_mode.as_str());
    println!("target       : {requested}");
    println!("executor     : {}", executor.as_str());
    println!(
        "layer cap    : {}",
        debug_layer_cap.map(|cap| cap.to_string()).unwrap_or_else(|| "full".into())
    );
    println!("runtime      : {}", runtime.summary_label());
    println!("launch       : {}", launch.summary_label());
    println!(
        "session      : {}",
        match session_mode {
            SessionMode::Host => session.summary_label(),
            SessionMode::InProcess => format!("{} [inproc]", session.summary_label()),
        }
    );
    if let Some(case_filter) = &case_filter {
        println!("case filter  : {case_filter}");
    }
    println!();
    let numeric_tolerance = numeric_tolerance();
    if let Some(tolerance) = numeric_tolerance {
        println!("numeric tol  : {tolerance}");
        println!();
    }

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        if let Some(case_filter) = &case_filter
            && case.name != case_filter
        {
            continue;
        }
        let reference = load_reference_backend(&load_spec, &model_name, debug_layer_cap)?;
        let token_ids = reference.tokenize_generation_prompt(case.prompt);
        let reference_tensor = reference.begin_token_ids(
            &format!("ggml-forward-{}", case.name),
            &token_ids,
            Some(1),
            0,
        )?;
        let reference_summary = summary_from_tensor(&reference_tensor);
        let response = run_stage_worker_session_request(
            &mut session,
            &GgmlStageWorkerRequest::BeginTokenIdsSummary {
                request_id: format!("ggml-forward-{}", case.name),
                token_ids: token_ids.clone(),
                max_tokens: Some(1),
            },
        )?;
        let candidate_summary = match response {
            GgmlStageWorkerResponse::TensorSummary { summary } => summary,
            other => bail!("unexpected worker response: {:?}", other),
        };
        let mut matched = reference_summary.hidden_contract_matches(&candidate_summary);
        let mut numeric_max_abs = None;
        println!("case         : {}", case.name);
        if !matched {
            let candidate_tensor = match run_stage_worker_session_request(
                &mut session,
                &GgmlStageWorkerRequest::BeginTokenIds {
                    request_id: format!("ggml-forward-{}-debug", case.name),
                    token_ids: token_ids.clone(),
                    max_tokens: Some(1),
                },
            )? {
                GgmlStageWorkerResponse::Tensor { tensor } => tensor,
                other => bail!("unexpected worker response: {:?}", other),
            };
            let reference_aux = stage_tensor_byte_sections(&reference_tensor.bytes)
                .and_then(|parts| parts.aux_bytes)
                .unwrap_or(&[]);
            let candidate_aux = stage_tensor_byte_sections(&candidate_tensor.bytes)
                .and_then(|parts| parts.aux_bytes)
                .unwrap_or(&[]);
            if let Some(tolerance) = numeric_tolerance {
                if summary_matches_except_hidden_hash(&reference_summary, &candidate_summary) {
                    let reference_hidden = decode_hidden_f32(hidden_bytes(&reference_tensor))?;
                    let candidate_hidden = decode_hidden_f32(hidden_bytes(&candidate_tensor))?;
                    let max_abs = max_abs_diff(&reference_hidden, &candidate_hidden)?;
                    numeric_max_abs = Some(max_abs);
                    if max_abs <= tolerance {
                        matched = true;
                    }
                }
            }
            println!(
                "compare      : {}",
                if matched {
                    if numeric_max_abs.is_some() { "PASS_NUMERIC" } else { "PASS" }
                } else {
                    "FAIL"
                }
            );
            println!("reference    : {}", reference_summary.summary_label());
            println!("candidate    : {}", candidate_summary.summary_label());
            if let Some(max_abs) = numeric_max_abs {
                println!("max_abs_diff : {max_abs}");
            }
            println!("reference aux: {}", prompt_aux_debug(reference_aux));
            println!("candidate aux: {}", prompt_aux_debug(candidate_aux));
        } else {
            println!("compare      : PASS");
            println!("reference    : {}", reference_summary.summary_label());
            println!("candidate    : {}", candidate_summary.summary_label());
        }
        failed |= !matched;
        println!();
        let _ = run_stage_worker_session_request(
            &mut session,
            &GgmlStageWorkerRequest::ClearDecodeSession {
                request_id: format!("ggml-forward-{}", case.name),
            },
        );
        let _ = run_stage_worker_session_request(
            &mut session,
            &GgmlStageWorkerRequest::ClearDecodeSession {
                request_id: format!("ggml-forward-{}-debug", case.name),
            },
        );
    }

    session.shutdown();

    if failed {
        bail!("ggml stage worker forward compare failed");
    }

    println!("overall: PASS");
    Ok(())
}
