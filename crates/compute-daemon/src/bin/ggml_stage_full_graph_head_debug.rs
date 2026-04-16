use anyhow::{Result, bail};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::{
    GgmlStageExecutorKind, build_ggml_stage_executor, debug_full_graph_head_layers,
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
    let case_filter = args.get(9).cloned();

    let is_head = matches!(stage_role, "head" | "first" | "single");
    let is_tail = matches!(stage_role, "tail" | "last" | "single");
    if !is_head {
        bail!("full-graph head debug requires a head or single stage, got `{stage_role}`");
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

    let mut tokenizer_init = init.clone();
    tokenizer_init.requested_executor = GgmlStageExecutorKind::ReferenceCpu;
    let mut tokenizer = build_ggml_stage_executor(&tokenizer_init)?;

    println!("=== GGML Full-Graph Head Debug ===");
    println!("model        : {model_id}");
    println!("shard        : {}", shard_path.display());
    println!("layers       : {start_layer}-{end_layer}");
    println!("stage role   : {stage_role}");
    println!("suite mode   : {}", suite_mode.as_str());
    println!("target       : {requested}");
    println!("max tokens   : {max_tokens}");
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
        let token_ids = tokenizer.tokenize_generation_prompt(case.prompt)?;
        let rows = debug_full_graph_head_layers(&init, &token_ids, Some(max_tokens))?;

        println!("case         : {}", case.name);
        println!("prompt toks  : {}", token_ids.len());
        println!("prompt       : {:?}", case.prompt);
        for row in rows {
            println!(
                "  layer={} local_max_abs={} cumulative_max_abs={}",
                row.layer_index, row.local_output_max_abs, row.cumulative_output_max_abs
            );
            println!("    runtime  : {}", row.runtime);
        }
        println!();
    }

    println!("overall: PASS");
    Ok(())
}
