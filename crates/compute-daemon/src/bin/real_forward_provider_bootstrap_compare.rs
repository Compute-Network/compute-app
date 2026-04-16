use anyhow::{Result, bail};
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_stage_worker::GgmlStageWorkerTensorSummary;
use compute_daemon::inference::real_forward_provider::RealForwardProviderCapabilities;
use compute_daemon::inference::real_forward_provider::build_real_forward_provider;
use compute_daemon::inference::stage_acceleration::{
    StageAccelerationPlan, StageAccelerationProviderPreference, StageAccelerationTargetPreference,
};
use compute_daemon::real_chain::RealStageArtifactSpec;
use stage_forward_lab::prompt_suite::{
    ValidationPromptSuiteMode, expectation_matches, validation_prompt_cases,
};
use stage_forward_lab::prompting::GemmaPromptMode;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{
    StageForwardBackend, StageLayout, StageSample, StageTensor, stage_tensor_byte_sections,
};
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
    let _prompt_mode =
        args.get(3).and_then(|value| GemmaPromptMode::parse(value)).unwrap_or_default();
    let suite_mode = args
        .get(4)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let requested_acceleration = args.get(5).map(|v| v.as_str()).unwrap_or("metal");
    let requested_provider = args.get(6).map(|v| v.as_str()).unwrap_or("ggml");
    let model_name = args.get(7).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let case_filter = args.get(8).cloned();

    let stage_specs = parse_stage_specs(&stage_specs_arg)?;
    let total_layers = stage_specs.last().map(|spec| spec.end_layer + 1).unwrap_or(0);
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

    println!("=== Real-Forward Provider Bootstrap Compare ===");
    println!("model               : {model_name}");
    println!("vocab               : {}", vocab_path.display());
    println!("prompt mode         : provider tokenize_generation_prompt");
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
    if let Some(tolerance) = numeric_tolerance() {
        println!("numeric tol         : {tolerance}");
    }
    if let Some(tolerance) = head_numeric_tolerance()
        && Some(tolerance) != numeric_tolerance()
    {
        println!("head numeric tol    : {tolerance}");
    }
    if let Some(tolerance) = tail_numeric_tolerance()
        && Some(tolerance) != numeric_tolerance()
    {
        println!("tail numeric tol    : {tolerance}");
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

        let stage_reference_provider = build_real_forward_provider(&reference_plan);
        let mut stage_candidate_provider = build_real_forward_provider(&candidate_plan);
        let candidate_load = stage_candidate_provider.load_shard(&config).err();

        let expected_caps = expected_bootstrap_capabilities(is_first, is_last);
        let capability_match = stage_candidate_provider.capabilities() == expected_caps;
        failed |= !capability_match;

        println!(
            "providers           : reference={} candidate={} capabilities={}",
            stage_reference_provider.provider_name(),
            stage_candidate_provider.provider_name(),
            pass_fail(capability_match)
        );
        println!(
            "candidate load      : {}",
            candidate_load.as_ref().map(|err| err.to_string()).unwrap_or_else(|| "ok".into())
        );

        for case in validation_prompt_cases(suite_mode) {
            if let Some(case_filter) = &case_filter
                && case.name != case_filter
            {
                continue;
            }

            println!("case                : {}", case.name);
            let request_id = format!("provider-bootstrap-s{}-{}", stage_index, case.name);
            let compare_max_tokens = 1u32;
            let mut reference_provider = build_real_forward_provider(&reference_plan);
            let mut candidate_provider = build_real_forward_provider(&candidate_plan);
            reference_provider.load_shard(&config)?;
            let _ = candidate_provider.load_shard(&config).err();

            if is_first {
                let reference_tokens =
                    reference_provider.tokenize_generation_prompt(case.prompt)?;
                let candidate_tokens =
                    candidate_provider.tokenize_generation_prompt(case.prompt)?;
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
                    let (activation_match, activation_max_abs) = tensor_contract_match(
                        &reference_activation,
                        &candidate_activation,
                        head_numeric_tolerance(),
                    )?;
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
                    if let Some(max_abs) = activation_max_abs {
                        println!("activation max_abs  : {max_abs}");
                    }
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
                    let (tensor_match, tensor_max_abs) = tensor_contract_match(
                        &reference_output,
                        &candidate_output,
                        head_numeric_tolerance(),
                    )?;
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
                    if let Some(max_abs) = tensor_max_abs {
                        println!("tensor max_abs      : {max_abs}");
                    }
                }
            } else {
                let upstream_input = build_stage_input(
                    &stage_specs,
                    &vocab_path,
                    &model_name,
                    stage_index,
                    case.prompt,
                    compare_max_tokens,
                    &request_id,
                )?;

                let reference_output =
                    reference_provider.continue_forward(upstream_input.clone())?;
                let candidate_output = candidate_provider.continue_forward(upstream_input)?;
                let (tensor_match, tensor_max_abs) = tensor_contract_match(
                    &reference_output,
                    &candidate_output,
                    tail_numeric_tolerance(),
                )?;

                if is_last {
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
                    if let Some(max_abs) = tensor_max_abs {
                        println!("tensor max_abs      : {max_abs}");
                    }
                } else {
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
                    if let Some(max_abs) = tensor_max_abs {
                        println!("tensor max_abs      : {max_abs}");
                    }
                }
            }

            reference_provider.clear_decode_session(&request_id);
            candidate_provider.clear_decode_session(&request_id);
            reference_provider.unload()?;
            candidate_provider.unload()?;
        }

        stage_candidate_provider.unload()?;
        println!();
    }

    if failed {
        bail!("real-forward provider bootstrap compare failed");
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
    specs: &[RealStageArtifactSpec],
    vocab_path: &Path,
    model_name: &str,
    stage_index: usize,
    prompt: &str,
    max_tokens: u32,
    request_id: &str,
) -> Result<StageTensor> {
    let backends = load_backend_chain(specs, vocab_path, model_name)?;
    if backends.is_empty() {
        bail!("Need at least one backend");
    }
    if stage_index == 0 {
        bail!("Stage input build is only needed for downstream stages");
    }

    let token_ids = backends[0].tokenize_generation_prompt(prompt);
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

fn expected_bootstrap_capabilities(
    is_first: bool,
    is_last: bool,
) -> RealForwardProviderCapabilities {
    RealForwardProviderCapabilities {
        hidden_state_ingress: !is_first,
        hidden_state_egress: !is_last,
        token_id_prompt_ingress: is_first,
        tail_sampling: is_last,
        per_stage_decode_sessions: true,
    }
}

fn parse_tolerance_env(name: &str) -> Option<f32> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn numeric_tolerance() -> Option<f32> {
    parse_tolerance_env("COMPUTE_GGML_NUMERIC_TOLERANCE")
}

fn head_numeric_tolerance() -> Option<f32> {
    parse_tolerance_env("COMPUTE_GGML_HEAD_NUMERIC_TOLERANCE").or_else(numeric_tolerance)
}

fn tail_numeric_tolerance() -> Option<f32> {
    parse_tolerance_env("COMPUTE_GGML_TAIL_NUMERIC_TOLERANCE").or_else(numeric_tolerance)
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

fn summary_matches_except_hidden_hash(
    reference: &GgmlStageWorkerTensorSummary,
    candidate: &GgmlStageWorkerTensorSummary,
) -> bool {
    reference.kind == candidate.kind
        && reference.hidden_dim == candidate.hidden_dim
        && reference.hidden_state_bytes == candidate.hidden_state_bytes
        && reference.aux_bytes == candidate.aux_bytes
        && reference.stage_trace_depth == candidate.stage_trace_depth
        && reference.aux_bytes_hash == candidate.aux_bytes_hash
        && reference.prompt_text == candidate.prompt_text
        && reference.max_tokens == candidate.max_tokens
}

fn tensor_contract_match(
    reference: &StageTensor,
    candidate: &StageTensor,
    tolerance: Option<f32>,
) -> Result<(bool, Option<f32>)> {
    let reference_summary = GgmlStageWorkerTensorSummary::from_tensor(reference);
    let candidate_summary = GgmlStageWorkerTensorSummary::from_tensor(candidate);
    if reference_summary.hidden_contract_matches(&candidate_summary) {
        return Ok((true, None));
    }
    if let Some(tolerance) = tolerance
        && summary_matches_except_hidden_hash(&reference_summary, &candidate_summary)
    {
        let reference_hidden = decode_hidden_f32(hidden_bytes(reference))?;
        let candidate_hidden = decode_hidden_f32(hidden_bytes(candidate))?;
        let max_abs = max_abs_diff(&reference_hidden, &candidate_hidden)?;
        if max_abs <= tolerance {
            return Ok((true, Some(max_abs)));
        }
        return Ok((false, Some(max_abs)));
    }
    Ok((false, None))
}

fn tensor_summary(tensor: &StageTensor) -> String {
    let summary = GgmlStageWorkerTensorSummary::from_tensor(tensor);
    format!(
        "{} total_bytes={} trace={}",
        summary.summary_label(),
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
