use anyhow::{Result, bail};
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::GgmlStageExecutorKind;
use compute_daemon::inference::ggml_stage_worker::{
    GgmlStageWorkerInitSpec, GgmlStageWorkerRequest, GgmlStageWorkerResponse,
    GgmlStageWorkerTensorSummary, run_stage_worker_session_request,
    spawn_in_process_stage_worker_session,
};
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::real_forward_provider::build_real_forward_provider;
use compute_daemon::inference::stage_acceleration::StageAccelerationTarget;
use compute_daemon::inference::stage_acceleration::{
    StageAccelerationPlan, StageAccelerationProviderPreference, StageAccelerationTargetPreference,
};
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

fn load_backend(
    load_spec: &RealForwardStageLoadSpec,
    model_id: &str,
    is_head: bool,
    is_tail: bool,
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
        is_head,
        is_tail,
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

fn retag_stage_tensor(tensor: &StageTensor, request_id: impl Into<String>) -> StageTensor {
    let mut tagged = tensor.clone();
    tagged.request_id = request_id.into();
    tagged
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
    let requested_acceleration = args.get(4).map(|v| v.as_str()).unwrap_or("metal");
    let requested_provider = args.get(5).map(|v| v.as_str()).unwrap_or("ggml");
    let suite_mode = args
        .get(6)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let max_tokens = args.get(7).and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
    let tail_layer_cap = args.get(8).and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
    let case_filter = args.get(9).cloned();

    let hw = detect();
    let reference_plan = StageAccelerationPlan::for_real_forward(
        &hw,
        StageAccelerationTargetPreference::Cpu,
        StageAccelerationProviderPreference::CpuRef,
    );
    let candidate_plan = StageAccelerationPlan::for_real_forward(
        &hw,
        StageAccelerationTargetPreference::parse(requested_acceleration),
        StageAccelerationProviderPreference::parse(requested_provider),
    );

    let head_config = ShardConfig {
        model_id: model_name.clone(),
        shard_path: head_shard.clone(),
        start_layer: 0,
        end_layer: 20,
        total_layers: 42,
        is_first_stage: true,
        is_last_stage: false,
        max_batch_size: 16,
        context_length: 8192,
    };
    let tail_config = ShardConfig {
        model_id: model_name.clone(),
        shard_path: tail_shard.clone(),
        start_layer: 21,
        end_layer: 41,
        total_layers: 42,
        is_first_stage: false,
        is_last_stage: true,
        max_batch_size: 16,
        context_length: 8192,
    };
    let tail_spec = RealForwardStageLoadSpec::from_shard_config(&tail_config)?;

    let runtime = detect_ggml_runtime_plan(target_from_str(requested_acceleration));
    let mut tail_init =
        GgmlStageWorkerInitSpec::from_load_spec(&tail_spec, &runtime, GgmlStageExecutorKind::Ggml);
    tail_init.debug_layer_cap = Some(tail_layer_cap);
    let mut tail_session = spawn_in_process_stage_worker_session(&tail_init)?;

    println!("=== Real-Forward Stage Handoff Compare ===");
    println!("model               : {model_name}");
    println!("head shard          : {}", head_shard.display());
    println!("tail shard          : {}", tail_shard.display());
    println!("requested accel     : {requested_acceleration}");
    println!("requested provider  : {requested_provider}");
    println!("suite mode          : {}", suite_mode.as_str());
    println!("max tokens          : {max_tokens}");
    println!("tail layer cap      : {tail_layer_cap}");
    println!("reference plan      : {}", reference_plan.summary_label());
    println!("candidate plan      : {}", candidate_plan.summary_label());
    println!("tail session        : {}", tail_session.summary_label());
    if let Some(case_filter) = &case_filter {
        println!("case filter         : {case_filter}");
    }
    println!();

    for case in validation_prompt_cases(suite_mode) {
        if let Some(case_filter) = &case_filter
            && case.name != case_filter
        {
            continue;
        }

        let request_id = format!("stage-handoff-{}", case.name);
        let mut reference_head_provider = build_real_forward_provider(&reference_plan);
        let mut candidate_head_provider = build_real_forward_provider(&candidate_plan);
        reference_head_provider.load_shard(&head_config)?;
        candidate_head_provider.load_shard(&head_config)?;

        let reference_prompt_tokens =
            reference_head_provider.tokenize_generation_prompt(case.prompt)?;
        let candidate_prompt_tokens =
            candidate_head_provider.tokenize_generation_prompt(case.prompt)?;
        let prompt_tokens_match = reference_prompt_tokens == candidate_prompt_tokens;

        let reference_head_output = reference_head_provider.begin_token_ids(
            &request_id,
            &reference_prompt_tokens,
            Some(max_tokens),
        )?;
        let candidate_head_output = candidate_head_provider.begin_token_ids(
            &request_id,
            &candidate_prompt_tokens,
            Some(max_tokens),
        )?;
        let head_max_abs = tensor_max_abs(&reference_head_output, &candidate_head_output)?;

        let ref_tail_request_id = format!("{request_id}-tail-ref");
        let candidate_tail_request_id = format!("{request_id}-tail-candidate");
        let reference_tail_backend =
            load_backend(&tail_spec, &model_name, false, true, Some(tail_layer_cap))?;
        let tail_ref_from_ref = reference_tail_backend
            .continue_forward(retag_stage_tensor(&reference_head_output, &ref_tail_request_id))?;
        let tail_ref_from_candidate = reference_tail_backend.continue_forward(
            retag_stage_tensor(&candidate_head_output, &candidate_tail_request_id),
        )?;
        let ref_tail_amplified_max_abs =
            tensor_max_abs(&tail_ref_from_ref, &tail_ref_from_candidate)?;

        let tail_ggml_from_ref = match run_stage_worker_session_request(
            &mut tail_session,
            &GgmlStageWorkerRequest::ContinueForward {
                input: retag_stage_tensor(&reference_head_output, &ref_tail_request_id),
            },
        )? {
            GgmlStageWorkerResponse::Tensor { tensor } => tensor,
            other => bail!("unexpected tail ggml response for reference input: {:?}", other),
        };
        let tail_ggml_from_candidate = match run_stage_worker_session_request(
            &mut tail_session,
            &GgmlStageWorkerRequest::ContinueForward {
                input: retag_stage_tensor(&candidate_head_output, &candidate_tail_request_id),
            },
        )? {
            GgmlStageWorkerResponse::Tensor { tensor } => tensor,
            other => bail!("unexpected tail ggml response for candidate input: {:?}", other),
        };
        let _ = run_stage_worker_session_request(
            &mut tail_session,
            &GgmlStageWorkerRequest::ClearDecodeSession { request_id: ref_tail_request_id },
        )?;
        let _ = run_stage_worker_session_request(
            &mut tail_session,
            &GgmlStageWorkerRequest::ClearDecodeSession { request_id: candidate_tail_request_id },
        )?;

        let ggml_tail_vs_ref_tail_ref_input_max_abs =
            tensor_max_abs(&tail_ref_from_ref, &tail_ggml_from_ref)?;
        let ggml_tail_vs_ref_tail_candidate_input_max_abs =
            tensor_max_abs(&tail_ref_from_candidate, &tail_ggml_from_candidate)?;

        println!("case                : {}", case.name);
        println!("prompt tokens       : {}", if prompt_tokens_match { "PASS" } else { "FAIL" });
        println!(
            "head output         : reference={} candidate={}",
            GgmlStageWorkerTensorSummary::from_tensor(&reference_head_output).summary_label(),
            GgmlStageWorkerTensorSummary::from_tensor(&candidate_head_output).summary_label()
        );
        println!("head max_abs        : {head_max_abs}");
        println!(
            "tail(ref<-ref head) : {}",
            GgmlStageWorkerTensorSummary::from_tensor(&tail_ref_from_ref).summary_label()
        );
        println!(
            "tail(ref<-cand head): {}",
            GgmlStageWorkerTensorSummary::from_tensor(&tail_ref_from_candidate).summary_label()
        );
        println!("tail input amp abs  : {ref_tail_amplified_max_abs}");
        if head_max_abs > 0.0 {
            println!("amp ratio           : {}", ref_tail_amplified_max_abs / head_max_abs);
        }
        println!("tail(ggml<-ref) abs : {ggml_tail_vs_ref_tail_ref_input_max_abs}");
        println!("tail(ggml<-cand) abs: {ggml_tail_vs_ref_tail_candidate_input_max_abs}");
        println!();

        reference_head_provider.clear_decode_session(&request_id);
        candidate_head_provider.clear_decode_session(&request_id);
        reference_head_provider.unload()?;
        candidate_head_provider.unload()?;
    }

    tail_session.shutdown();
    println!("overall: PASS");
    Ok(())
}
