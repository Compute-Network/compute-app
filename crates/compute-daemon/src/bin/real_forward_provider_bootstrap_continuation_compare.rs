use anyhow::{Result, bail};
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_stage_worker::GgmlStageWorkerTensorSummary;
use compute_daemon::inference::real_forward_provider::build_real_forward_provider;
use compute_daemon::inference::stage_acceleration::{
    StageAccelerationPlan, StageAccelerationProviderPreference, StageAccelerationTargetPreference,
};
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use stage_forward_lab::{StageSample, StageTensor, stage_tensor_byte_sections};
use std::env;
use std::path::PathBuf;

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
    let steps = args.get(7).and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).max(1);
    let case_filter = args.get(8).cloned();

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

    let mut candidate_head = build_real_forward_provider(&candidate_plan);
    let mut candidate_tail = build_real_forward_provider(&candidate_plan);
    let candidate_head_load = candidate_head.load_shard(&head_config).err();
    let candidate_tail_load = candidate_tail.load_shard(&tail_config).err();

    println!("=== Real-Forward Provider Bootstrap Continuation Compare ===");
    println!("model               : {model_name}");
    println!("head shard          : {}", head_shard.display());
    println!("tail shard          : {}", tail_shard.display());
    println!("requested accel     : {requested_acceleration}");
    println!("requested provider  : {requested_provider}");
    println!("suite mode          : {}", suite_mode.as_str());
    println!("steps               : {steps}");
    println!("reference plan      : {}", reference_plan.summary_label());
    println!("candidate plan      : {}", candidate_plan.summary_label());
    println!(
        "candidate head load : {}",
        candidate_head_load.as_ref().map(|e| e.to_string()).unwrap_or_else(|| "ok".into())
    );
    println!(
        "candidate tail load : {}",
        candidate_tail_load.as_ref().map(|e| e.to_string()).unwrap_or_else(|| "ok".into())
    );
    if let Some(case_filter) = &case_filter {
        println!("case filter         : {case_filter}");
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
    if let Some(tolerance) = prompt_tail_numeric_tolerance()
        && Some(tolerance) != tail_numeric_tolerance()
    {
        println!("prompt tail tol     : {tolerance}");
    }
    println!();

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        if let Some(case_filter) = &case_filter
            && case.name != case_filter
        {
            continue;
        }

        let request_id = format!("provider-bootstrap-cont-{}", case.name);
        let mut reference_head = build_real_forward_provider(&reference_plan);
        let mut reference_tail = build_real_forward_provider(&reference_plan);
        reference_head.load_shard(&head_config)?;
        reference_tail.load_shard(&tail_config)?;
        let reference_prompt_tokens = reference_head.tokenize_generation_prompt(case.prompt)?;
        let candidate_prompt_tokens = candidate_head.tokenize_generation_prompt(case.prompt)?;
        let token_match = reference_prompt_tokens == candidate_prompt_tokens;
        failed |= !token_match;

        let mut reference_head_output = reference_head.begin_token_ids(
            &request_id,
            &reference_prompt_tokens,
            Some(steps as u32),
        )?;
        let mut candidate_head_output = candidate_head.begin_token_ids(
            &request_id,
            &candidate_prompt_tokens,
            Some(steps as u32),
        )?;
        let (initial_head_match, initial_head_max_abs) = tensor_contract_match(
            &reference_head_output,
            &candidate_head_output,
            head_numeric_tolerance(),
        )?;
        let mut head_match = initial_head_match;
        let initial_reference_head_summary =
            GgmlStageWorkerTensorSummary::from_tensor(&reference_head_output);
        let initial_candidate_head_summary =
            GgmlStageWorkerTensorSummary::from_tensor(&candidate_head_output);

        let mut reference_samples = Vec::new();
        let mut candidate_samples = Vec::new();
        let mut tail_match = true;
        let mut sample_match = true;
        let mut step_diagnostics = Vec::new();

        if !head_match {
            step_diagnostics.push(format!(
                "step=prompt head reference={} candidate={}",
                initial_reference_head_summary.summary_label(),
                initial_candidate_head_summary.summary_label()
            ));
            if let Some(max_abs) = initial_head_max_abs {
                step_diagnostics.push(format!("step=prompt head_max_abs={max_abs}"));
            }
        } else if let Some(max_abs) = initial_head_max_abs {
            step_diagnostics.push(format!("step=prompt head_max_abs={max_abs}"));
        }

        for step in 0..steps {
            let reference_tail_output = reference_tail.continue_forward(reference_head_output)?;
            let candidate_tail_output = candidate_tail.continue_forward(candidate_head_output)?;
            let (current_tail_match, current_tail_max_abs) = tensor_contract_match(
                &reference_tail_output,
                &candidate_tail_output,
                if step == 0 { prompt_tail_numeric_tolerance() } else { tail_numeric_tolerance() },
            )?;
            tail_match &= current_tail_match;
            if !current_tail_match {
                step_diagnostics.push(format!(
                    "step={step} tail reference={} candidate={}",
                    GgmlStageWorkerTensorSummary::from_tensor(&reference_tail_output)
                        .summary_label(),
                    GgmlStageWorkerTensorSummary::from_tensor(&candidate_tail_output)
                        .summary_label()
                ));
                if let Some(max_abs) = current_tail_max_abs {
                    step_diagnostics.push(format!("step={step} tail_max_abs={max_abs}"));
                }
            } else if let Some(max_abs) = current_tail_max_abs {
                step_diagnostics.push(format!("step={step} tail_max_abs={max_abs}"));
            }

            let reference_sample = reference_tail.sample_tail(reference_tail_output)?;
            let candidate_sample = candidate_tail.sample_tail(candidate_tail_output)?;
            sample_match &= reference_sample == candidate_sample;
            reference_samples.push(reference_sample.clone());
            candidate_samples.push(candidate_sample.clone());

            let Some(&next_token_id) = reference_sample.token_ids.first() else {
                break;
            };

            reference_head_output = reference_head.begin_token_ids(
                &request_id,
                &[next_token_id],
                Some(steps as u32),
            )?;
            candidate_head_output = candidate_head.begin_token_ids(
                &request_id,
                &[next_token_id],
                Some(steps as u32),
            )?;
            let (current_head_match, current_head_max_abs) = tensor_contract_match(
                &reference_head_output,
                &candidate_head_output,
                head_numeric_tolerance(),
            )?;
            head_match &= current_head_match;
            if !current_head_match {
                step_diagnostics.push(format!(
                    "step={} head token={} reference={} candidate={}",
                    step + 1,
                    next_token_id,
                    GgmlStageWorkerTensorSummary::from_tensor(&reference_head_output)
                        .summary_label(),
                    GgmlStageWorkerTensorSummary::from_tensor(&candidate_head_output)
                        .summary_label()
                ));
                if let Some(max_abs) = current_head_max_abs {
                    step_diagnostics.push(format!(
                        "step={} head token={} max_abs={max_abs}",
                        step + 1,
                        next_token_id
                    ));
                }
            } else if let Some(max_abs) = current_head_max_abs {
                step_diagnostics.push(format!(
                    "step={} head token={} max_abs={max_abs}",
                    step + 1,
                    next_token_id
                ));
            }
        }

        let reference_ids: Vec<u32> =
            reference_samples.iter().flat_map(|sample| sample.token_ids.clone()).collect();
        let candidate_ids: Vec<u32> =
            candidate_samples.iter().flat_map(|sample| sample.token_ids.clone()).collect();
        let reference_text = samples_text(&reference_samples);
        let candidate_text = samples_text(&candidate_samples);
        let sequence_match = reference_ids == candidate_ids && reference_text == candidate_text;
        let ok = token_match && head_match && tail_match && sample_match && sequence_match;
        failed |= !ok;

        println!("case                : {}", case.name);
        println!(
            "compare             : tokens={} head={} tail={} sample={} sequence={} overall={}",
            pass_fail(token_match),
            pass_fail(head_match),
            pass_fail(tail_match),
            pass_fail(sample_match),
            pass_fail(sequence_match),
            pass_fail(ok)
        );
        println!("reference sequence  : ids={reference_ids:?} text={reference_text:?}");
        println!("candidate sequence  : ids={candidate_ids:?} text={candidate_text:?}");
        for diagnostic in &step_diagnostics {
            println!("diagnostic          : {diagnostic}");
        }
        println!();

        reference_head.clear_decode_session(&request_id);
        reference_tail.clear_decode_session(&request_id);
        candidate_head.clear_decode_session(&request_id);
        candidate_tail.clear_decode_session(&request_id);
        reference_head.unload()?;
        reference_tail.unload()?;
    }

    candidate_head.unload()?;
    candidate_tail.unload()?;

    if failed {
        bail!("real-forward provider bootstrap continuation compare failed");
    }

    println!("overall: PASS");
    Ok(())
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

fn prompt_tail_numeric_tolerance() -> Option<f32> {
    parse_tolerance_env("COMPUTE_GGML_PROMPT_TAIL_NUMERIC_TOLERANCE")
        .or_else(tail_numeric_tolerance)
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

fn samples_text(samples: &[StageSample]) -> String {
    samples.iter().map(|sample| sample.text.as_str()).collect()
}

fn pass_fail(value: bool) -> &'static str {
    if value { "PASS" } else { "FAIL" }
}
