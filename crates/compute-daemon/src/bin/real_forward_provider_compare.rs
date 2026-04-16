use anyhow::{Result, bail};
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::real_forward_provider::build_real_forward_provider;
use compute_daemon::inference::stage_acceleration::{
    StageAccelerationPlan, StageAccelerationProviderPreference, StageAccelerationTargetPreference,
};
use compute_daemon::real_chain::RealStageArtifactSpec;
use stage_forward_lab::prompt_suite::{
    ValidationPromptSuiteMode, expectation_matches, validation_prompt_cases,
};
use stage_forward_lab::prompting::{GemmaPromptMode, format_gemma_prompt};
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout, StageSample, StageTensor};
use std::env;
use std::path::{Path, PathBuf};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let default_specs = [
        "../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json@0-20",
        "../compute-backend/out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json@21-41",
    ]
    .join(",");
    let stage_specs_arg = args.get(1).cloned().unwrap_or(default_specs);
    let vocab_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("../compute-backend/out/gemma-e4b-2stage/packed-stage-1/vocab.json")
    });
    let prompt_mode =
        args.get(3).and_then(|value| GemmaPromptMode::parse(value)).unwrap_or_default();
    let suite_mode = args
        .get(4)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let requested_acceleration = args.get(5).map(|v| v.as_str()).unwrap_or("auto");
    let requested_provider = args.get(6).map(|v| v.as_str()).unwrap_or("auto");
    let model_name = args.get(7).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let case_filter = args.get(8).cloned();

    let stage_specs = parse_stage_specs(&stage_specs_arg)?;
    let total_layers = stage_specs.last().map(|spec| spec.end_layer + 1).unwrap_or(0);
    let local_backends = load_backend_chain(&stage_specs, &vocab_path, &model_name)?;

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

    println!("=== Real-Forward Provider Compare ===");
    println!("model               : {model_name}");
    println!("vocab               : {}", vocab_path.display());
    println!("prompt mode         : {}", prompt_mode.as_str());
    println!("suite mode          : {}", suite_mode.as_str());
    println!("requested accel     : {requested_acceleration}");
    println!("requested provider  : {requested_provider}");
    println!("reference plan      : {}", reference_plan.summary_label());
    println!("candidate plan      : {}", candidate_plan.summary_label());
    if let Some(case_filter) = &case_filter {
        println!("case filter         : {case_filter}");
    }
    for (index, spec) in stage_specs.iter().enumerate() {
        println!(
            "stage               : {index} => {}-{} @ {}",
            spec.start_layer,
            spec.end_layer,
            spec.artifact_path.display()
        );
    }
    println!();

    let mut failed = false;
    for (stage_index, spec) in stage_specs.iter().enumerate() {
        let is_first = stage_index == 0;
        let is_last = stage_index + 1 == stage_specs.len();
        let config = ShardConfig {
            model_id: model_name.clone(),
            shard_path: spec.artifact_path.clone(),
            start_layer: spec.start_layer,
            end_layer: spec.end_layer,
            total_layers,
            is_first_stage: is_first,
            is_last_stage: is_last,
            max_batch_size: 16,
            context_length: 8192,
        };

        println!(
            "=== Stage {} ({}-{}, role={}) ===",
            stage_index,
            spec.start_layer,
            spec.end_layer,
            stage_role(is_first, is_last)
        );

        let mut reference_provider = build_real_forward_provider(&reference_plan);
        let mut candidate_provider = build_real_forward_provider(&candidate_plan);
        reference_provider.load_shard(&config)?;
        candidate_provider.load_shard(&config)?;

        let capability_match =
            reference_provider.capabilities() == candidate_provider.capabilities();
        println!(
            "providers           : reference={} candidate={} capabilities={}",
            reference_provider.provider_name(),
            candidate_provider.provider_name(),
            pass_fail(capability_match)
        );
        failed |= !capability_match;

        for case in validation_prompt_cases(suite_mode) {
            if let Some(case_filter) = &case_filter
                && case.name != case_filter
            {
                continue;
            }

            println!("case                : {}", case.name);
            let formatted_prompt = format_gemma_prompt(prompt_mode, case.prompt);
            let request_id = format!("provider-compare-s{}-{}", stage_index, case.name);
            let compare_max_tokens = 1u32;

            if is_first {
                let reference_tokens = reference_provider.tokenize_text(&formatted_prompt)?;
                let candidate_tokens = candidate_provider.tokenize_text(&formatted_prompt)?;
                let token_match = reference_tokens == candidate_tokens;
                let eos_match =
                    reference_provider.eos_token_id() == candidate_provider.eos_token_id();

                if is_last {
                    let reference_activation = reference_provider.begin_token_ids(
                        &request_id,
                        &reference_tokens,
                        Some(compare_max_tokens),
                    )?;
                    let candidate_activation = candidate_provider.begin_token_ids(
                        &request_id,
                        &candidate_tokens,
                        Some(compare_max_tokens),
                    )?;
                    let activation_match = reference_activation == candidate_activation;
                    let reference_sample = reference_provider.sample_tail(reference_activation)?;
                    let candidate_sample = candidate_provider.sample_tail(candidate_activation)?;
                    let sample_match = reference_sample == candidate_sample;
                    let quality_match =
                        expectation_matches(case.first_token_expectation, &candidate_sample.text);
                    let ok = token_match
                        && eos_match
                        && activation_match
                        && sample_match
                        && quality_match;
                    failed |= !ok;
                    println!(
                        "compare             : tokens={} eos={} activation={} sample={} quality={} overall={}",
                        pass_fail(token_match),
                        pass_fail(eos_match),
                        pass_fail(activation_match),
                        pass_fail(sample_match),
                        pass_fail(quality_match),
                        pass_fail(ok)
                    );
                    println!(
                        "sample              : reference={:?} candidate={:?}",
                        sample_summary(&reference_sample),
                        sample_summary(&candidate_sample)
                    );
                } else {
                    let reference_output = reference_provider.begin_token_ids(
                        &request_id,
                        &reference_tokens,
                        Some(compare_max_tokens),
                    )?;
                    let candidate_output = candidate_provider.begin_token_ids(
                        &request_id,
                        &candidate_tokens,
                        Some(compare_max_tokens),
                    )?;
                    let tensor_match = reference_output == candidate_output;
                    let ok = token_match && eos_match && tensor_match;
                    failed |= !ok;
                    println!(
                        "compare             : tokens={} eos={} tensor={} overall={}",
                        pass_fail(token_match),
                        pass_fail(eos_match),
                        pass_fail(tensor_match),
                        pass_fail(ok)
                    );
                    println!(
                        "tensor              : reference={} candidate={}",
                        tensor_summary(&reference_output),
                        tensor_summary(&candidate_output)
                    );
                }
            } else {
                let upstream_input = build_stage_input(
                    &local_backends,
                    stage_index,
                    &formatted_prompt,
                    compare_max_tokens,
                    &request_id,
                )?;

                if is_last {
                    let reference_output =
                        reference_provider.continue_forward(upstream_input.clone())?;
                    let candidate_output = candidate_provider.continue_forward(upstream_input)?;
                    let tensor_match = reference_output == candidate_output;
                    let reference_sample = reference_provider.sample_tail(reference_output)?;
                    let candidate_sample = candidate_provider.sample_tail(candidate_output)?;
                    let sample_match = reference_sample == candidate_sample;
                    let quality_match =
                        expectation_matches(case.first_token_expectation, &candidate_sample.text);
                    let ok = tensor_match && sample_match && quality_match;
                    failed |= !ok;
                    println!(
                        "compare             : tensor={} sample={} quality={} overall={}",
                        pass_fail(tensor_match),
                        pass_fail(sample_match),
                        pass_fail(quality_match),
                        pass_fail(ok)
                    );
                    println!(
                        "sample              : reference={:?} candidate={:?}",
                        sample_summary(&reference_sample),
                        sample_summary(&candidate_sample)
                    );
                } else {
                    let reference_output =
                        reference_provider.continue_forward(upstream_input.clone())?;
                    let candidate_output = candidate_provider.continue_forward(upstream_input)?;
                    let tensor_match = reference_output == candidate_output;
                    failed |= !tensor_match;
                    println!(
                        "compare             : tensor={} overall={}",
                        pass_fail(tensor_match),
                        pass_fail(tensor_match)
                    );
                    println!(
                        "tensor              : reference={} candidate={}",
                        tensor_summary(&reference_output),
                        tensor_summary(&candidate_output)
                    );
                }
            }
        }

        reference_provider.unload()?;
        candidate_provider.unload()?;
        println!();
    }

    if failed {
        bail!("real-forward provider compare failed");
    }

    println!("overall: PASS");
    Ok(())
}

fn parse_stage_specs(value: &str) -> Result<Vec<RealStageArtifactSpec>> {
    let specs = value
        .split(',')
        .filter_map(|entry| {
            let trimmed = entry.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .map(parse_stage_spec)
        .collect::<Result<Vec<_>>>()?;
    if specs.is_empty() {
        bail!("Expected at least one stage spec in path@start-end form");
    }
    Ok(specs)
}

fn parse_stage_spec(value: &str) -> Result<RealStageArtifactSpec> {
    let (path, layers) = value
        .rsplit_once('@')
        .ok_or_else(|| anyhow::anyhow!("Stage spec must be path@start-end: {value}"))?;
    let (start_layer, end_layer) = layers
        .split_once('-')
        .ok_or_else(|| anyhow::anyhow!("Stage spec layer range must be start-end: {value}"))?;
    Ok(RealStageArtifactSpec {
        artifact_path: PathBuf::from(path),
        start_layer: start_layer.parse()?,
        end_layer: end_layer.parse()?,
    })
}

fn load_backend_chain(
    specs: &[RealStageArtifactSpec],
    vocab_path: &Path,
    model_name: &str,
) -> Result<Vec<RealGemmaBackend>> {
    let scores_path =
        vocab_path.parent().map(|parent| parent.join("vocab_scores.json")).unwrap_or_else(|| {
            PathBuf::from("../compute-backend/out/gemma-e4b-2stage/vocab_scores.json")
        });
    let mut backends = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let mut backend = RealGemmaBackend::new(&spec.artifact_path);
        if vocab_path.exists() {
            let sp = if scores_path.exists() { Some(scores_path.as_path()) } else { None };
            backend.load_tokenizer(vocab_path, sp)?;
        }
        backend.load_layout(StageLayout {
            model_id: model_name.into(),
            stage_id: format!("stage-{}-{}", spec.start_layer, spec.end_layer),
            start_layer: spec.start_layer,
            end_layer: spec.end_layer,
            is_head: index == 0,
            is_tail: index + 1 == specs.len(),
        })?;
        backends.push(backend);
    }
    Ok(backends)
}

fn build_stage_input(
    backends: &[RealGemmaBackend],
    stage_index: usize,
    formatted_prompt: &str,
    max_tokens: u32,
    request_id: &str,
) -> Result<StageTensor> {
    if backends.is_empty() {
        bail!("Need at least one backend");
    }
    if stage_index == 0 {
        bail!("Stage input build is only needed for downstream stages");
    }

    let token_ids = backends[0].tokenize_text(formatted_prompt);
    let mut activation =
        backends[0].begin_token_ids(request_id, &token_ids, Some(max_tokens), 0)?;
    for backend in &backends[1..stage_index] {
        activation = backend.continue_forward(activation)?;
    }
    Ok(activation)
}

fn stage_role(is_first: bool, is_last: bool) -> &'static str {
    match (is_first, is_last) {
        (true, true) => "head+tail",
        (true, false) => "head",
        (false, true) => "tail",
        (false, false) => "middle",
    }
}

fn tensor_summary(tensor: &StageTensor) -> String {
    format!(
        "kind={:?} hidden_dim={} bytes={} trace={}",
        tensor.kind,
        tensor.hidden_dim,
        tensor.bytes.len(),
        tensor.stage_trace.join(" -> ")
    )
}

fn sample_summary(sample: &StageSample) -> String {
    format!(
        "finish_tokens={} text={:?} token_ids={:?}",
        sample.completion_tokens, sample.text, sample.token_ids
    )
}

fn pass_fail(value: bool) -> &'static str {
    if value { "PASS" } else { "FAIL" }
}
