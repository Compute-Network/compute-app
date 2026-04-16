use anyhow::{Result, bail};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::GgmlStageExecutorKind;
use compute_daemon::inference::ggml_stage_worker::{
    GgmlStageWorkerInitSpec, GgmlStageWorkerRequest, GgmlStageWorkerResponse,
    run_stage_worker_session_request, spawn_in_process_stage_worker_session,
};
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::stage_acceleration::StageAccelerationTarget;
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{
    StageForwardBackend, StageLayout, StageTensor, stage_tensor_byte_sections,
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

fn hidden_bytes(tensor: &StageTensor) -> &[u8] {
    stage_tensor_byte_sections(&tensor.bytes)
        .map(|sections| sections.hidden_bytes)
        .unwrap_or(&tensor.bytes)
}

fn decode_hidden_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("hidden byte slice is not a multiple of 4: {}", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        bail!("hidden lengths differ: {} vs {}", a.len(), b.len());
    }
    Ok(a.iter().zip(b.iter()).map(|(left, right)| (left - right).abs()).fold(0.0f32, f32::max))
}

fn tensor_max_abs(reference: &StageTensor, candidate: &StageTensor) -> Result<f32> {
    let reference_hidden = decode_hidden_f32(hidden_bytes(reference))?;
    let candidate_hidden = decode_hidden_f32(hidden_bytes(candidate))?;
    max_abs_diff(&reference_hidden, &candidate_hidden)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let shard_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
        )
    });
    let model_id = args.get(2).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let start_layer = args.get(3).and_then(|value| value.parse::<u32>().ok()).unwrap_or(0);
    let end_layer = args.get(4).and_then(|value| value.parse::<u32>().ok()).unwrap_or(20);
    let stage_role = args.get(5).map(|value| value.as_str()).unwrap_or("head");
    let requested = args.get(6).map(|value| value.as_str()).unwrap_or("metal");
    let suite_mode = args
        .get(7)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let max_tokens = args.get(8).and_then(|value| value.parse::<u32>().ok()).unwrap_or(1);
    let max_cap = args.get(9).and_then(|value| value.parse::<usize>().ok()).unwrap_or(21);
    let case_filter = args.get(10).cloned();

    let is_head = matches!(stage_role, "head" | "first" | "single");
    let is_tail = matches!(stage_role, "tail" | "last" | "single");
    if !is_head {
        bail!("head cap profile requires a head or single stage, got `{stage_role}`");
    }

    let load_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_id.clone(),
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
    let init =
        GgmlStageWorkerInitSpec::from_load_spec(&load_spec, &runtime, GgmlStageExecutorKind::Ggml);

    println!("=== GGML Full-Graph Head Cap Profile ===");
    println!("model        : {model_id}");
    println!("shard        : {}", shard_path.display());
    println!("layers       : {start_layer}-{end_layer}");
    println!("stage role   : {stage_role}");
    println!("suite mode   : {}", suite_mode.as_str());
    println!("target       : {requested}");
    println!("max tokens   : {max_tokens}");
    println!("max cap      : {max_cap}");
    println!("runtime      : {}", runtime.summary_label());
    if let Some(case_filter) = &case_filter {
        println!("case filter  : {case_filter}");
    }
    println!();

    for case in validation_prompt_cases(suite_mode) {
        if let Some(case_filter) = &case_filter
            && case.name != case_filter
        {
            continue;
        }
        let mut reference = load_reference_backend(&load_spec, &model_id, None)?;
        let token_ids = reference.tokenize_generation_prompt(case.prompt);

        println!("case         : {}", case.name);
        println!("prompt toks  : {}", token_ids.len());
        println!("prompt       : {:?}", case.prompt);
        for cap in 1..=max_cap.min((end_layer - start_layer + 1) as usize) {
            reference.set_debug_layer_cap(Some(cap));
            let reference_tensor = reference.begin_token_ids(
                &format!("head-cap-profile-ref-{cap}"),
                &token_ids,
                Some(max_tokens),
                0,
            )?;

            let mut candidate_init = init.clone();
            candidate_init.debug_layer_cap = Some(cap);
            let mut candidate_session = spawn_in_process_stage_worker_session(&candidate_init)?;
            let candidate_tensor = match run_stage_worker_session_request(
                &mut candidate_session,
                &GgmlStageWorkerRequest::BeginTokenIds {
                    request_id: format!("head-cap-profile-candidate-{cap}"),
                    token_ids: token_ids.clone(),
                    max_tokens: Some(max_tokens),
                },
            )? {
                GgmlStageWorkerResponse::Tensor { tensor } => tensor,
                other => bail!("unexpected candidate worker response: {:?}", other),
            };
            candidate_session.shutdown();

            let max_abs = tensor_max_abs(&reference_tensor, &candidate_tensor)?;
            println!("  cap={:<2} layer={} max_abs={}", cap, start_layer + cap as u32 - 1, max_abs);
        }
        println!();
    }

    println!("overall: PASS");
    Ok(())
}
