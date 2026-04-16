use anyhow::{Result, bail};
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::real_forward_provider::build_real_forward_provider;
use compute_daemon::inference::stage_acceleration::{
    StageAccelerationPlan, StageAccelerationProviderPreference, StageAccelerationTargetPreference,
};
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use std::env;
use std::path::PathBuf;

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
    let requested_acceleration = args.get(6).map(|value| value.as_str()).unwrap_or("metal");
    let requested_provider = args.get(7).map(|value| value.as_str()).unwrap_or("ggml");
    let suite_mode = args
        .get(8)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);

    let is_head = matches!(stage_role, "head" | "first");
    let is_tail = matches!(stage_role, "tail" | "last");
    let config = ShardConfig {
        model_id: model_name.clone(),
        shard_path: shard_path.clone(),
        start_layer,
        end_layer,
        total_layers: end_layer + 1,
        is_first_stage: is_head,
        is_last_stage: is_tail,
        max_batch_size: 16,
        context_length: 8192,
    };

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

    let mut reference_provider = build_real_forward_provider(&reference_plan);
    let mut candidate_provider = build_real_forward_provider(&candidate_plan);
    reference_provider.load_shard(&config)?;
    let candidate_load = candidate_provider.load_shard(&config).err();

    println!("=== Real-Forward Provider Metadata Compare ===");
    println!("model               : {model_name}");
    println!("shard               : {}", shard_path.display());
    println!("layers              : {start_layer}-{end_layer}");
    println!("stage role          : {stage_role}");
    println!("suite mode          : {}", suite_mode.as_str());
    println!("reference provider  : {}", reference_provider.provider_name());
    println!("candidate provider  : {}", candidate_provider.provider_name());
    println!("reference plan      : {}", reference_plan.summary_label());
    println!("candidate plan      : {}", candidate_plan.summary_label());
    println!(
        "candidate load      : {}",
        candidate_load.as_ref().map(|err| err.to_string()).unwrap_or_else(|| "ok".into())
    );
    println!();

    let mut failed = false;
    let eos_reference = reference_provider.eos_token_id();
    let eos_candidate = candidate_provider.eos_token_id();
    let eos_match = eos_reference == eos_candidate;
    failed |= !eos_match;
    println!(
        "eos                 : reference={:?} candidate={:?} match={}",
        eos_reference,
        eos_candidate,
        pass_fail(eos_match)
    );

    for case in validation_prompt_cases(suite_mode) {
        println!("case                : {}", case.name);

        let reference_prompt_tokens = reference_provider.tokenize_generation_prompt(case.prompt)?;
        let candidate_prompt_tokens = candidate_provider.tokenize_generation_prompt(case.prompt)?;
        let prompt_match = reference_prompt_tokens == candidate_prompt_tokens;
        failed |= !prompt_match;
        println!(
            "generation prompt   : match={} len={} first={:?}",
            pass_fail(prompt_match),
            candidate_prompt_tokens.len(),
            candidate_prompt_tokens.iter().take(8).copied().collect::<Vec<_>>()
        );

        let reference_text_tokens = reference_provider.tokenize_text(case.prompt)?;
        let candidate_text_tokens = candidate_provider.tokenize_text(case.prompt)?;
        let text_match = reference_text_tokens == candidate_text_tokens;
        failed |= !text_match;
        println!(
            "raw text tokens     : match={} len={} first={:?}",
            pass_fail(text_match),
            candidate_text_tokens.len(),
            candidate_text_tokens.iter().take(8).copied().collect::<Vec<_>>()
        );

        let decode_probe = reference_prompt_tokens.iter().take(8).copied().collect::<Vec<_>>();
        let reference_decoded = reference_provider.decode_token_ids(&decode_probe)?;
        let candidate_decoded = candidate_provider.decode_token_ids(&decode_probe)?;
        let decode_match = reference_decoded == candidate_decoded;
        failed |= !decode_match;
        println!(
            "decode probe        : match={} tokens={:?} text={:?}",
            pass_fail(decode_match),
            decode_probe,
            candidate_decoded
        );
        println!();
    }

    reference_provider.unload()?;
    candidate_provider.unload()?;

    if failed {
        bail!("real-forward provider metadata compare failed");
    }

    println!("overall: PASS");
    Ok(())
}

fn pass_fail(value: bool) -> &'static str {
    if value { "PASS" } else { "FAIL" }
}
