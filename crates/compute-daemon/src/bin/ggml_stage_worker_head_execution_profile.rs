use anyhow::{Result, bail};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::{
    GgmlHeadExecutionProfile, GgmlStageExecutorKind, build_ggml_stage_executor,
};
use compute_daemon::inference::ggml_stage_worker::GgmlStageWorkerInitSpec;
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::stage_acceleration::StageAccelerationTarget;
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
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

fn avg_us(total_us: u64, iterations: u32) -> f64 {
    total_us as f64 / iterations.max(1) as f64
}

fn print_profile(label: &str, profile: &GgmlHeadExecutionProfile) {
    println!(
        "{label:<18} avg_total_us={:>10.1} avg_ingress_us={:>10.1} avg_payload_us={:>10.1} payload={} hidden={} aux={} layers={}",
        profile.avg_total_us(),
        profile.avg_ingress_us(),
        profile.avg_payload_encode_us(),
        profile.payload_bytes,
        profile.hidden_state_bytes,
        profile.aux_bytes,
        profile.effective_layer_cap,
    );
    for layer in &profile.layers {
        println!(
            "{:<18} layer {:>2} avg_us={:>10.1} attn_cpu={:>9.1} attn_mm={:>9.1} ffn_cpu={:>9.1} ffn_mm={:>9.1} ple={:>9.1}",
            "",
            layer.layer_index,
            avg_us(layer.total_us, profile.iterations),
            avg_us(layer.attention_cpu_us, profile.iterations),
            avg_us(layer.attention_matmul_us, profile.iterations),
            avg_us(layer.ffn_cpu_us, profile.iterations),
            avg_us(layer.ffn_matmul_us, profile.iterations),
            avg_us(layer.ple_us, profile.iterations),
        );
    }
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
    let executor =
        args.get(8).map(|value| executor_from_str(value)).unwrap_or(GgmlStageExecutorKind::Ggml);
    let debug_layer_cap = args.get(9).and_then(|value| value.parse::<usize>().ok());
    let iterations = args.get(10).and_then(|value| value.parse::<u32>().ok()).unwrap_or(1);
    let case_filter = args.get(11).cloned();

    let is_head = matches!(stage_role, "head" | "first" | "single");
    let is_tail = matches!(stage_role, "tail" | "last" | "single");
    if !is_head {
        bail!("head execution profile requires a head or single stage, got `{stage_role}`");
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
    let mut init = GgmlStageWorkerInitSpec::from_load_spec(&load_spec, &runtime, executor);
    init.debug_layer_cap = debug_layer_cap;
    let mut profile_executor = build_ggml_stage_executor(&init)?;

    println!("=== GGML Stage Worker Head Execution Profile ===");
    println!("model        : {model_id}");
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
    println!("iterations   : {}", iterations.max(1));
    println!("runtime      : {}", runtime.summary_label());
    println!("plan         : {}", profile_executor.plan().summary_label());
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
        let token_ids = profile_executor.tokenize_generation_prompt(case.prompt)?;
        let _ = profile_executor.profile_begin_token_ids_execution(&token_ids, Some(1), 1)?;
        let profile =
            profile_executor.profile_begin_token_ids_execution(&token_ids, Some(1), iterations)?;

        println!("case         : {}", case.name);
        println!("prompt toks  : {}", token_ids.len());
        println!("prompt       : {:?}", case.prompt);
        print_profile("profile", &profile);
        println!();
    }

    Ok(())
}
