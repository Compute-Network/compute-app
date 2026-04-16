use anyhow::{Result, bail};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::GgmlStageExecutorKind;
use compute_daemon::inference::ggml_stage_worker::{
    GgmlStageWorkerHostLaunchSpec, GgmlStageWorkerInitSpec, GgmlStageWorkerRequest,
    GgmlStageWorkerResponse, run_stage_worker_request,
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

fn load_backend(
    load_spec: &RealForwardStageLoadSpec,
    model_id: &str,
    is_head: bool,
    is_tail: bool,
) -> Result<RealGemmaBackend> {
    let mut backend = RealGemmaBackend::new(&load_spec.index_path);
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
    let init =
        GgmlStageWorkerInitSpec::from_load_spec(&tail_spec, &runtime, GgmlStageExecutorKind::Ggml);
    let launch = GgmlStageWorkerHostLaunchSpec::from_init_spec(&init)?;

    println!("=== GGML Stage Worker Sample Compare ===");
    println!("model        : {model_name}");
    println!("head shard   : {}", head_shard_path.display());
    println!("tail shard   : {}", tail_shard_path.display());
    println!("suite mode   : {}", suite_mode.as_str());
    println!("target       : {requested}");
    println!("runtime      : {}", runtime.summary_label());
    println!("launch       : {}", launch.summary_label());
    println!();

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        let head_backend = load_backend(&head_spec, &model_name, true, false)?;
        let tail_backend = load_backend(&tail_spec, &model_name, false, true)?;
        let prompt_tokens = head_backend.tokenize_generation_prompt(case.prompt);
        let request_id = format!("ggml-sample-{}", case.name);
        let tail_input = head_backend.begin_token_ids(&request_id, &prompt_tokens, Some(1), 0)?;
        let tail_output = tail_backend.continue_forward(tail_input.clone())?;
        let reference_sample = tail_backend.sample_tail(tail_output.clone())?;
        let response = run_stage_worker_request(
            &launch,
            &GgmlStageWorkerRequest::SampleTail { input: tail_output },
        )?;
        let candidate_sample = match response {
            GgmlStageWorkerResponse::Sample { sample } => sample,
            other => bail!("unexpected worker response: {:?}", other),
        };
        let matched = reference_sample == candidate_sample;
        failed |= !matched;
        println!("case         : {}", case.name);
        println!("compare      : {}", if matched { "PASS" } else { "FAIL" });
        println!(
            "reference    : text={:?} token_ids={:?} completion_tokens={}",
            reference_sample.text, reference_sample.token_ids, reference_sample.completion_tokens
        );
        println!(
            "candidate    : text={:?} token_ids={:?} completion_tokens={}",
            candidate_sample.text, candidate_sample.token_ids, candidate_sample.completion_tokens
        );
        println!();
    }

    if failed {
        bail!("ggml stage worker sample compare failed");
    }
    println!("overall: PASS");
    Ok(())
}
