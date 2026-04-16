use anyhow::{Context, Result, bail};
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
use stage_forward_lab::{
    StageForwardBackend, StageLayout, encode_stage_tensor_bytes, stage_tensor_byte_sections,
};
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

fn collapse_aux_to_last_token(aux_bytes: &[u8]) -> Result<Option<Vec<u8>>> {
    if aux_bytes.len() < 20 {
        return Ok(Some(aux_bytes.to_vec()));
    }
    let seq_len =
        u32::from_le_bytes([aux_bytes[4], aux_bytes[5], aux_bytes[6], aux_bytes[7]]) as usize;
    if seq_len <= 1 {
        return Ok(Some(aux_bytes.to_vec()));
    }
    let layer_count =
        u32::from_le_bytes([aux_bytes[8], aux_bytes[9], aux_bytes[10], aux_bytes[11]]) as usize;
    let ple_dim =
        u32::from_le_bytes([aux_bytes[12], aux_bytes[13], aux_bytes[14], aux_bytes[15]]) as usize;
    let prefix_hash_count =
        u32::from_le_bytes([aux_bytes[16], aux_bytes[17], aux_bytes[18], aux_bytes[19]]) as usize;
    let per_token_bytes = layer_count
        .checked_mul(ple_dim)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("prompt aux dimensions overflow"))?;
    let ple_end = 20usize
        .checked_add(
            seq_len
                .checked_mul(per_token_bytes)
                .ok_or_else(|| anyhow::anyhow!("prompt aux token bytes overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("prompt aux byte length overflow"))?;
    let expected_len = ple_end
        .checked_add(
            prefix_hash_count
                .checked_mul(8)
                .ok_or_else(|| anyhow::anyhow!("prompt aux prefix hashes overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("prompt aux trailing bytes overflow"))?;
    if aux_bytes.len() != expected_len {
        bail!("unexpected prompt aux length {} (expected {})", aux_bytes.len(), expected_len);
    }
    let last_token_start = 20 + (seq_len - 1) * per_token_bytes;
    let last_token_end = last_token_start + per_token_bytes;
    let mut collapsed = Vec::with_capacity(20 + per_token_bytes);
    collapsed.extend_from_slice(&aux_bytes[..4]);
    collapsed.extend_from_slice(&(1u32).to_le_bytes());
    collapsed.extend_from_slice(&(layer_count as u32).to_le_bytes());
    collapsed.extend_from_slice(&(ple_dim as u32).to_le_bytes());
    collapsed.extend_from_slice(&(0u32).to_le_bytes());
    collapsed.extend_from_slice(&aux_bytes[last_token_start..last_token_end]);
    Ok(Some(collapsed))
}

fn collapse_tensor_to_last_token(
    tensor: stage_forward_lab::StageTensor,
) -> Result<stage_forward_lab::StageTensor> {
    let sections = stage_tensor_byte_sections(&tensor.bytes);
    let hidden_bytes = sections.map(|parts| parts.hidden_bytes).unwrap_or(&tensor.bytes);
    let per_token_hidden_bytes =
        tensor.hidden_dim.checked_mul(4).ok_or_else(|| anyhow::anyhow!("hidden bytes overflow"))?;
    if per_token_hidden_bytes == 0 || hidden_bytes.len() <= per_token_hidden_bytes {
        return Ok(tensor);
    }
    let last_hidden = hidden_bytes[hidden_bytes.len() - per_token_hidden_bytes..].to_vec();
    let aux_bytes = sections
        .and_then(|parts| parts.aux_bytes)
        .map(collapse_aux_to_last_token)
        .transpose()?
        .flatten();
    Ok(stage_forward_lab::StageTensor {
        bytes: encode_stage_tensor_bytes(&last_hidden, aux_bytes.as_deref()),
        ..tensor
    })
}

fn truncate_token_ids(token_ids: &[u32]) -> String {
    let mut shown = token_ids.iter().take(8).copied().collect::<Vec<_>>();
    if token_ids.len() > 8 {
        shown.push(u32::MAX);
    }
    let mut rendered = shown
        .into_iter()
        .map(|token| if token == u32::MAX { "...".to_string() } else { token.to_string() })
        .collect::<Vec<_>>()
        .join(",");
    if rendered.is_empty() {
        rendered.push('-');
    }
    rendered
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let head_shard = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
        )
    });
    let tail_shard = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json",
        )
    });
    let model_name = args.get(3).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let requested = args.get(4).map(|value| value.as_str()).unwrap_or("metal");
    let suite_mode = args
        .get(5)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let steps = args.get(6).and_then(|value| value.parse::<usize>().ok()).unwrap_or(4).max(1);
    let executor =
        args.get(7).map(|value| executor_from_str(value)).unwrap_or(GgmlStageExecutorKind::Ggml);
    let debug_layer_cap = args.get(8).and_then(|value| value.parse::<usize>().ok());
    let session_mode =
        args.get(9).map(|value| session_mode_from_str(value)).unwrap_or(SessionMode::Host);
    let case_filter = args.get(10).cloned();

    let head_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_name.clone(),
        shard_path: head_shard.clone(),
        start_layer: 0,
        end_layer: 20,
        total_layers: 42,
        is_first_stage: true,
        is_last_stage: false,
        max_batch_size: 16,
        context_length: 8192,
    })?;
    let tail_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_name.clone(),
        shard_path: tail_shard.clone(),
        start_layer: 21,
        end_layer: 41,
        total_layers: 42,
        is_first_stage: false,
        is_last_stage: true,
        max_batch_size: 16,
        context_length: 8192,
    })?;
    let runtime = detect_ggml_runtime_plan(target_from_str(requested));
    let mut init = GgmlStageWorkerInitSpec::from_load_spec(&head_spec, &runtime, executor);
    init.debug_layer_cap = debug_layer_cap;
    let launch = GgmlStageWorkerHostLaunchSpec::from_init_spec(&init)?;
    let mut session = spawn_session(&init, &launch, session_mode)?;

    println!("=== GGML Stage Worker Head Continuation Compare ===");
    println!("model        : {model_name}");
    println!("head shard   : {}", head_shard.display());
    println!("tail shard   : {}", tail_shard.display());
    println!("suite mode   : {}", suite_mode.as_str());
    println!("steps        : {steps}");
    println!("target       : {requested}");
    println!("executor     : {}", executor.as_str());
    println!(
        "layer cap    : {}",
        debug_layer_cap.map(|cap| cap.to_string()).unwrap_or_else(|| "full".into())
    );
    println!("runtime      : {}", runtime.summary_label());
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

        let request_id = format!("ggml-head-cont-{}", case.name);
        let reference_head = load_reference_backend(&head_spec, &model_name, debug_layer_cap)?;
        let reference_tail = load_reference_backend(&tail_spec, &model_name, None)?;
        let prompt_tokens = reference_head.tokenize_generation_prompt(case.prompt);
        let mut current_token_ids = prompt_tokens.clone();
        let mut full_token_history = prompt_tokens.clone();
        let mut case_failed = false;

        println!("case         : {}", case.name);
        for step in 0..steps {
            let reference_tensor = reference_head.begin_token_ids(
                &request_id,
                &current_token_ids,
                Some(steps as u32),
                0,
            )?;
            let reference_summary = summary_from_tensor(&reference_tensor);
            let fresh_reference = load_reference_backend(&head_spec, &model_name, debug_layer_cap)?
                .begin_token_ids(
                    &format!("{request_id}-fresh-step-{step}"),
                    &full_token_history,
                    Some(steps as u32),
                    0,
                )?;
            let fresh_reference = if current_token_ids.len() == 1 {
                collapse_tensor_to_last_token(fresh_reference)?
            } else {
                fresh_reference
            };
            let fresh_reference_summary = summary_from_tensor(&fresh_reference);
            let reference_fresh_match =
                reference_summary.hidden_contract_matches(&fresh_reference_summary);
            let candidate_tensor = if numeric_tolerance.is_some() {
                match run_stage_worker_session_request(
                    &mut session,
                    &GgmlStageWorkerRequest::BeginTokenIds {
                        request_id: request_id.clone(),
                        token_ids: current_token_ids.clone(),
                        max_tokens: Some(steps as u32),
                    },
                )? {
                    GgmlStageWorkerResponse::Tensor { tensor } => Some(tensor),
                    other => bail!("unexpected worker response: {:?}", other),
                }
            } else {
                None
            };
            let candidate_summary = if let Some(candidate_tensor) = candidate_tensor.as_ref() {
                summary_from_tensor(candidate_tensor)
            } else {
                let response = run_stage_worker_session_request(
                    &mut session,
                    &GgmlStageWorkerRequest::BeginTokenIdsSummary {
                        request_id: request_id.clone(),
                        token_ids: current_token_ids.clone(),
                        max_tokens: Some(steps as u32),
                    },
                )?;
                match response {
                    GgmlStageWorkerResponse::TensorSummary { summary } => summary,
                    other => bail!("unexpected worker response: {:?}", other),
                }
            };
            let mut matched = reference_summary.hidden_contract_matches(&candidate_summary);
            let mut numeric_max_abs = None;
            if !matched
                && numeric_tolerance.is_some()
                && summary_matches_except_hidden_hash(&reference_summary, &candidate_summary)
            {
                let candidate_tensor = candidate_tensor
                    .as_ref()
                    .context("numeric tolerance path requires candidate tensor")?;
                let reference_hidden = decode_hidden_f32(hidden_bytes(&reference_tensor))?;
                let candidate_hidden = decode_hidden_f32(hidden_bytes(candidate_tensor))?;
                let max_abs = max_abs_diff(&reference_hidden, &candidate_hidden)?;
                numeric_max_abs = Some(max_abs);
                if max_abs <= numeric_tolerance.expect("numeric tolerance present") {
                    matched = true;
                }
            }
            case_failed |= !matched;
            failed |= !matched;
            println!(
                "step         : {} tokens(len={}, ids=[{}]) compare={} ref_fresh={}",
                step,
                current_token_ids.len(),
                truncate_token_ids(&current_token_ids),
                if matched {
                    if numeric_max_abs.is_some() { "PASS_NUMERIC" } else { "PASS" }
                } else {
                    "FAIL"
                },
                if reference_fresh_match { "PASS" } else { "FAIL" }
            );
            if !reference_fresh_match {
                println!("ref-current  : {}", reference_summary.summary_label());
                println!("ref-fresh    : {}", fresh_reference_summary.summary_label());
            }
            if !matched {
                println!("reference    : {}", reference_summary.summary_label());
                println!("candidate    : {}", candidate_summary.summary_label());
            } else if let Some(max_abs) = numeric_max_abs {
                println!("max_abs_diff : {}", max_abs);
            }

            let reference_tail_output = reference_tail.continue_forward(reference_tensor)?;
            let reference_sample = reference_tail.sample_tail(reference_tail_output)?;
            let Some(&next_token) = reference_sample.token_ids.first() else {
                break;
            };
            full_token_history.push(next_token);
            current_token_ids = vec![next_token];
        }
        let _ = run_stage_worker_session_request(
            &mut session,
            &GgmlStageWorkerRequest::ClearDecodeSession { request_id: request_id.clone() },
        );
        reference_head.clear_decode_session(&request_id);
        reference_tail.clear_decode_session(&request_id);
        println!("overall      : {}", if case_failed { "FAIL" } else { "PASS" });
        println!();
    }

    session.shutdown();

    if failed {
        bail!("ggml stage worker head continuation compare failed");
    }

    println!("overall: PASS");
    Ok(())
}
