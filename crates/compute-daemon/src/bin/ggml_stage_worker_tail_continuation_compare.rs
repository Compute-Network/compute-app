use anyhow::{Result, bail};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::GgmlStageExecutorKind;
use compute_daemon::inference::ggml_stage_worker::{
    GgmlStageWorkerHostLaunchSpec, GgmlStageWorkerInitSpec, GgmlStageWorkerRequest,
    GgmlStageWorkerResponse, GgmlStageWorkerTensorSummary, run_stage_worker_session_request,
};
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::stage_acceleration::StageAccelerationTarget;
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
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
    match value.trim().to_ascii_lowercase().as_str() {
        "ggml" | "ggml-worker" | "ggml_worker" => GgmlStageExecutorKind::Ggml,
        _ => GgmlStageExecutorKind::ReferenceCpu,
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

fn pass_fail(value: bool) -> &'static str {
    if value { "PASS" } else { "FAIL" }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let tail_shard_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json",
        )
    });
    let head_shard_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
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
    let case_filter = args.get(9).cloned();

    let tail_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_name.clone(),
        shard_path: tail_shard_path.clone(),
        start_layer: 21,
        end_layer: 41,
        total_layers: 42,
        is_first_stage: false,
        is_last_stage: true,
        max_batch_size: 16,
        context_length: 8192,
    })?;
    let head_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_name.clone(),
        shard_path: head_shard_path.clone(),
        start_layer: 0,
        end_layer: 20,
        total_layers: 42,
        is_first_stage: true,
        is_last_stage: false,
        max_batch_size: 16,
        context_length: 8192,
    })?;
    let runtime = detect_ggml_runtime_plan(target_from_str(requested));
    let mut init = GgmlStageWorkerInitSpec::from_load_spec(&tail_spec, &runtime, executor);
    init.debug_layer_cap = debug_layer_cap;
    let launch = GgmlStageWorkerHostLaunchSpec::from_init_spec(&init)?;
    let mut session = launch.spawn_persistent_session()?;

    println!("=== GGML Stage Worker Tail Continuation Compare ===");
    println!("model        : {model_name}");
    println!("head shard   : {}", head_shard_path.display());
    println!("tail shard   : {}", tail_shard_path.display());
    println!("suite mode   : {}", suite_mode.as_str());
    println!("steps        : {steps}");
    println!("target       : {requested}");
    println!("executor     : {}", executor.as_str());
    println!(
        "layer cap    : {}",
        debug_layer_cap.map(|cap| cap.to_string()).unwrap_or_else(|| "full".into())
    );
    println!("runtime      : {}", runtime.summary_label());
    println!("session      : {}", session.summary_label());
    if let Some(case_filter) = &case_filter {
        println!("case filter  : {case_filter}");
    }
    println!();

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        if let Some(case_filter) = &case_filter
            && case.name != case_filter
        {
            continue;
        }

        let request_id = format!("ggml-tail-cont-{}", case.name);
        let head_backend = load_backend(&head_spec, &model_name, true, false, None)?;
        let tail_backend = load_backend(&tail_spec, &model_name, false, true, debug_layer_cap)?;
        let prompt_tokens = head_backend.tokenize_generation_prompt(case.prompt);
        let mut head_output =
            head_backend.begin_token_ids(&request_id, &prompt_tokens, Some(steps as u32), 0)?;
        let mut case_failed = false;

        println!("case         : {}", case.name);
        for step in 0..steps {
            let reference_tensor = tail_backend.continue_forward(head_output.clone())?;
            let reference_summary = GgmlStageWorkerTensorSummary::from_tensor(&reference_tensor);
            let response = run_stage_worker_session_request(
                &mut session,
                &GgmlStageWorkerRequest::ContinueForward { input: head_output.clone() },
            )?;
            let candidate_tensor = match response {
                GgmlStageWorkerResponse::Tensor { tensor } => tensor,
                other => bail!("unexpected worker response: {:?}", other),
            };
            let candidate_summary = GgmlStageWorkerTensorSummary::from_tensor(&candidate_tensor);
            let tensor_match = reference_summary.hidden_contract_matches(&candidate_summary);

            let reference_sample = tail_backend.sample_tail(reference_tensor)?;
            let candidate_sample = match run_stage_worker_session_request(
                &mut session,
                &GgmlStageWorkerRequest::SampleTail { input: candidate_tensor },
            )? {
                GgmlStageWorkerResponse::Sample { sample } => sample,
                other => bail!("unexpected sample worker response: {:?}", other),
            };
            let sample_match = reference_sample == candidate_sample;
            let matched = tensor_match && sample_match;
            case_failed |= !matched;
            failed |= !matched;
            println!(
                "step         : {} tensor={} sample={} overall={}",
                step,
                pass_fail(tensor_match),
                pass_fail(sample_match),
                pass_fail(matched)
            );
            if !tensor_match {
                println!("reference    : {}", reference_summary.summary_label());
                println!("candidate    : {}", candidate_summary.summary_label());
            }
            if !sample_match {
                println!("ref sample   : {:?}", reference_sample);
                println!("cand sample  : {:?}", candidate_sample);
            }

            let Some(&next_token) = reference_sample.token_ids.first() else {
                break;
            };
            head_output =
                head_backend.begin_token_ids(&request_id, &[next_token], Some(steps as u32), 0)?;
        }

        let _ = run_stage_worker_session_request(
            &mut session,
            &GgmlStageWorkerRequest::ClearDecodeSession { request_id: request_id.clone() },
        );
        tail_backend.clear_decode_session(&request_id);
        println!("overall      : {}", pass_fail(!case_failed));
        println!();
    }

    session.shutdown();

    if failed {
        bail!("ggml stage worker tail continuation compare failed");
    }

    println!("overall: PASS");
    Ok(())
}
