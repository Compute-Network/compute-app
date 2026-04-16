use anyhow::{Result, bail};
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::{
    GgmlStageExecutorKind, debug_proportional_shared_kv_layer,
};
use compute_daemon::inference::ggml_stage_worker::GgmlStageWorkerInitSpec;
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::real_forward_provider::build_real_forward_provider;
use compute_daemon::inference::stage_acceleration::{
    StageAccelerationPlan, StageAccelerationProviderPreference, StageAccelerationTarget,
    StageAccelerationTargetPreference,
};
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use stage_forward_lab::{StageTensor, stage_tensor_byte_sections};
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

fn parse_layers(value: Option<&String>) -> Result<Vec<u32>> {
    let value = value.map(String::as_str).unwrap_or("29,35,41");
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|err| anyhow::anyhow!("invalid layer `{value}` in layer list: {err}"))
        })
        .collect()
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
    let layers = parse_layers(args.get(8))?;
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
    let tail_init =
        GgmlStageWorkerInitSpec::from_load_spec(&tail_spec, &runtime, GgmlStageExecutorKind::Ggml);

    println!("=== GGML Proportional Shared-KV Debug ===");
    println!("model               : {model_name}");
    println!("head shard          : {}", head_shard.display());
    println!("tail shard          : {}", tail_shard.display());
    println!("requested accel     : {requested_acceleration}");
    println!("requested provider  : {requested_provider}");
    println!("suite mode          : {}", suite_mode.as_str());
    println!("max tokens          : {max_tokens}");
    println!("layers              : {:?}", layers);
    println!("reference plan      : {}", reference_plan.summary_label());
    println!("candidate plan      : {}", candidate_plan.summary_label());
    println!("tail runtime        : {}", runtime.summary_label());
    println!("tail init           : {}", tail_init.summary_label());
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

        let request_id = format!("prop-shared-kv-{}", case.name);
        let mut reference_head_provider = build_real_forward_provider(&reference_plan);
        let mut candidate_head_provider = build_real_forward_provider(&candidate_plan);
        reference_head_provider.load_shard(&head_config)?;
        candidate_head_provider.load_shard(&head_config)?;

        let reference_prompt_tokens =
            reference_head_provider.tokenize_generation_prompt(case.prompt)?;
        let candidate_prompt_tokens =
            candidate_head_provider.tokenize_generation_prompt(case.prompt)?;
        if reference_prompt_tokens != candidate_prompt_tokens {
            bail!("prompt token mismatch on case {}", case.name);
        }

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

        println!("case                : {}", case.name);
        println!(
            "prompt head max_abs : {}",
            tensor_max_abs(&reference_head_output, &candidate_head_output)?
        );

        for &layer in &layers {
            let debug = debug_proportional_shared_kv_layer(
                &tail_init,
                &reference_head_output,
                &candidate_head_output,
                layer,
            )?;
            println!(
                "  layer={} shared_kv={} tokens={} input_max_abs={} q_rope(ref_vs_cand)={} q_input(cand_vs_ggml)={} q_rope(cand_vs_ggml)={} attn_out={} layer_out={} layer_out(cand_vs_ggml)={}",
                debug.layer_index,
                debug.shared_kv_source_layer,
                debug.token_count,
                debug.input_max_abs,
                debug.q_rope_ref_candidate_max_abs,
                debug.q_input_ggml_max_abs,
                debug.q_rope_ggml_candidate_max_abs,
                debug.attn_out_max_abs,
                debug.layer_out_max_abs,
                debug.layer_out_ggml_candidate_max_abs,
            );
            println!("    rope runtime    : {}", debug.rope_runtime);
            println!("    full runtime    : {}", debug.full_runtime);
        }
        println!();

        reference_head_provider.clear_decode_session(&request_id);
        candidate_head_provider.clear_decode_session(&request_id);
        reference_head_provider.unload()?;
        candidate_head_provider.unload()?;
    }

    println!("overall: PASS");
    Ok(())
}
