use anyhow::{Result, bail};
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::real_forward_provider::build_real_forward_provider;
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

fn shared_kv_source_for_layer(model_id: &str, hidden_dim: usize, layer_index: u32) -> Option<u32> {
    let gemma4_e4b = model_id.contains("gemma-4-e4b") && hidden_dim == 2560;
    if !gemma4_e4b || layer_index < 24 {
        return None;
    }
    if layer_index % 6 == 5 { Some(23) } else { Some(22) }
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
    let max_tail_layer_cap = args.get(8).and_then(|v| v.parse::<usize>().ok()).unwrap_or(21);
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

    println!("=== Real-Forward Stage Handoff Profile ===");
    println!("model               : {model_name}");
    println!("head shard          : {}", head_shard.display());
    println!("tail shard          : {}", tail_shard.display());
    println!("requested accel     : {requested_acceleration}");
    println!("requested provider  : {requested_provider}");
    println!("suite mode          : {}", suite_mode.as_str());
    println!("max tokens          : {max_tokens}");
    println!("max tail layer cap  : {max_tail_layer_cap}");
    println!("reference plan      : {}", reference_plan.summary_label());
    println!("candidate plan      : {}", candidate_plan.summary_label());
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

        let request_id = format!("stage-handoff-profile-{}", case.name);
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
        let head_max_abs = tensor_max_abs(&reference_head_output, &candidate_head_output)?;

        println!("case                : {}", case.name);
        println!("head max_abs        : {head_max_abs}");
        println!("tail caps:");

        let mut previous_abs = head_max_abs;
        let tail_stage_layers = (tail_config.end_layer - tail_config.start_layer + 1) as usize;
        for layer_cap in 1..=max_tail_layer_cap.min(tail_stage_layers) {
            let reference_tail_backend =
                load_backend(&tail_spec, &model_name, false, true, Some(layer_cap))?;
            let ref_tail_request_id = format!("{request_id}-tail-ref-cap-{layer_cap}");
            let candidate_tail_request_id = format!("{request_id}-tail-candidate-cap-{layer_cap}");
            let tail_ref_from_ref = reference_tail_backend.continue_forward(retag_stage_tensor(
                &reference_head_output,
                &ref_tail_request_id,
            ))?;
            let tail_ref_from_candidate = reference_tail_backend.continue_forward(
                retag_stage_tensor(&candidate_head_output, &candidate_tail_request_id),
            )?;
            let tail_max_abs = tensor_max_abs(&tail_ref_from_ref, &tail_ref_from_candidate)?;
            let global_layer = tail_config.start_layer + layer_cap as u32 - 1;
            let shared_kv = shared_kv_source_for_layer(
                &model_name,
                reference_head_output.hidden_dim,
                global_layer,
            );
            let amp_vs_head = if head_max_abs > 0.0 { tail_max_abs / head_max_abs } else { 0.0 };
            let growth_vs_prev = if previous_abs > 0.0 { tail_max_abs / previous_abs } else { 0.0 };
            println!(
                "  cap={:<2} layer={} shared_kv={:<4} tail_max_abs={:<12} amp_vs_head={:<10} growth_vs_prev={}",
                layer_cap,
                global_layer,
                shared_kv.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string()),
                tail_max_abs,
                amp_vs_head,
                growth_vs_prev
            );
            previous_abs = tail_max_abs;
        }
        println!();
    }

    println!("overall: PASS");
    Ok(())
}
