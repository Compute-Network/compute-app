use anyhow::{Result, bail};
use compute_daemon::config::Config;
use compute_daemon::hardware::HardwareInfo;
use compute_daemon::real_chain::{
    RealStageArtifactSpec, prepare_temp_stage_root, restore_stage_root,
};
use compute_daemon::stage_runtime::{StagePrototypeRuntimeProfile, start_stage_prototype_chain};
use stage_forward_lab::prompt_suite::{
    ValidationPromptSuiteMode, expectation_matches, validation_prompt_cases,
};
use stage_forward_lab::prompting::GemmaPromptMode;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
struct LocalRun {
    prompt_tokens: usize,
    finish_reason: String,
    text: String,
    token_ids: Vec<u32>,
    ttft_ms: u128,
    total_ms: u128,
}

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
    let model_name = args.get(5).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let (case_filter, stage_acceleration, stage_acceleration_provider) = match args.len() {
        0..=6 => (None, "metal".to_string(), "ggml".to_string()),
        7 => (
            args.get(6).cloned().filter(|value| !value.is_empty()),
            "metal".to_string(),
            "ggml".to_string(),
        ),
        8 => (None, args[6].clone(), args[7].clone()),
        _ => (
            args.get(6).cloned().filter(|value| !value.is_empty()),
            args.get(7).cloned().unwrap_or_else(|| "metal".to_string()),
            args.get(8).cloned().unwrap_or_else(|| "ggml".to_string()),
        ),
    };

    let stage_specs = parse_stage_specs(&stage_specs_arg)?;
    let stage_ranges =
        stage_specs.iter().map(|spec| (spec.start_layer, spec.end_layer)).collect::<Vec<_>>();

    println!("=== Real Stage Chain Compare ===");
    println!("model         : {}", model_name);
    println!("vocab         : {}", vocab_path.display());
    println!("prompt mode   : {}", prompt_mode.as_str());
    println!("suite mode    : {}", suite_mode.as_str());
    println!("accel target  : {}", stage_acceleration);
    println!("accel provider: {}", stage_acceleration_provider);
    if let Some(case_filter) = &case_filter {
        println!("case filter   : {}", case_filter);
    }
    for (index, spec) in stage_specs.iter().enumerate() {
        println!(
            "stage         : {} => {}-{} @ {}",
            index,
            spec.start_layer,
            spec.end_layer,
            spec.artifact_path.display()
        );
    }
    println!();

    let local_backends = load_backend_chain(&stage_specs, &vocab_path, &model_name)?;
    let temp_stage_root = prepare_temp_stage_root(&model_name, &stage_specs)?;
    let previous_stage_root = env::var_os("COMPUTE_STAGE_ROOT");
    unsafe {
        env::set_var("COMPUTE_STAGE_ROOT", &temp_stage_root);
    }

    let mut config = Config::default();
    config.experimental.stage_backend = "real_forward".to_string();
    config.experimental.stage_acceleration = stage_acceleration.clone();
    config.experimental.stage_acceleration_provider = stage_acceleration_provider.clone();
    let hw = HardwareInfo::empty();
    let mut handles = start_stage_prototype_chain(
        &config,
        &hw,
        "real-stage-chain-compare",
        &model_name,
        &stage_ranges,
    )
    .await?;
    let head_client = handles[0].client();

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        if let Some(case_filter) = &case_filter {
            if case.name != case_filter {
                continue;
            }
        }
        println!("=== Case: {} ===", case.name);
        println!("phase         : local");
        let local = run_local_chain(
            &local_backends,
            case.prompt,
            case.max_tokens,
            case.stop_sequences,
            prompt_mode,
            &format!("local-{}", case.name),
        )?;

        println!("phase         : runtime");
        let live_start = Instant::now();
        let live = head_client
            .complete_prompt(
                format!("live-{}", case.name),
                case.prompt.to_string(),
                Some(case.max_tokens),
                case.stop_sequences.iter().map(|stop| stop.to_string()).collect(),
            )
            .await?;
        let live_wall_ms = live_start.elapsed().as_millis();

        let text_match = live.content == local.text;
        let finish_match = live.finish_reason == local.finish_reason;
        let tokens_match = live.completion_token_ids == local.token_ids;
        let prompt_tokens_match = live.prompt_tokens as usize == local.prompt_tokens;
        let local_quality_match = expectation_matches(case.continuation_expectation, &local.text);
        let runtime_quality_match =
            expectation_matches(case.continuation_expectation, &live.content);
        let ok = text_match
            && finish_match
            && tokens_match
            && prompt_tokens_match
            && local_quality_match
            && runtime_quality_match;
        failed |= !ok;

        println!("prompt        : {:?}", case.prompt);
        println!(
            "local         : ttft={}ms total={}ms finish={} prompt_toks={} text={:?} token_ids={:?}",
            local.ttft_ms,
            local.total_ms,
            local.finish_reason,
            local.prompt_tokens,
            local.text,
            local.token_ids
        );
        println!(
            "runtime chain : ttft={}ms total={}ms wall={}ms finish={} prompt_toks={} cont_tok_s={:.2} total_tok_s={:.2} text={:?} token_ids={:?}",
            live.ttft_ms,
            live.total_ms,
            live_wall_ms,
            live.finish_reason,
            live.prompt_tokens,
            continuation_tok_s(live.ttft_ms, live.total_ms, live.completion_tokens),
            total_tok_s(live.total_ms, live.completion_tokens),
            live.content,
            live.completion_token_ids
        );
        println!("runtime profile: {}", format_runtime_profile(&live.runtime_profile));
        println!(
            "compare       : text={} finish={} token_ids={} prompt_toks={} local_quality={} runtime_quality={} overall={}",
            pass_fail(text_match),
            pass_fail(finish_match),
            pass_fail(tokens_match),
            pass_fail(prompt_tokens_match),
            pass_fail(local_quality_match),
            pass_fail(runtime_quality_match),
            pass_fail(ok)
        );
        println!();
    }

    while let Some(handle) = handles.pop() {
        handle.stop().await;
    }
    restore_stage_root(previous_stage_root);
    let _ = fs::remove_dir_all(&temp_stage_root);

    if failed {
        bail!("real stage chain compare failed");
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

fn run_local_chain(
    backends: &[RealGemmaBackend],
    prompt: &str,
    max_tokens: u32,
    stop_sequences: &[&str],
    prompt_mode: GemmaPromptMode,
    request_id: &str,
) -> Result<LocalRun> {
    if backends.is_empty() {
        bail!("Need at least one backend for local chain");
    }
    let head = &backends[0];
    let tail = backends.last().unwrap();
    let eos_token_id = tail.eos_token_id().or_else(|| head.eos_token_id());
    let mut prompt_token_ids = head.tokenize_prompt_mode(prompt, prompt_mode);
    let prompt_tokens = prompt_token_ids.len();
    let mut generated_token_ids = Vec::with_capacity(max_tokens as usize);
    let mut finish_reason = "length".to_string();
    let total_start = Instant::now();
    let mut ttft_ms = 0u128;
    let mut text = String::new();

    for step in 0..max_tokens as usize {
        let step_start = Instant::now();
        let step_token_ids = if step == 0 {
            prompt_token_ids.clone()
        } else {
            vec![*generated_token_ids.last().unwrap()]
        };
        let mut stage_output = head.begin_token_ids(request_id, &step_token_ids, Some(1), 0)?;
        for backend in &backends[1..] {
            stage_output = backend.continue_forward(stage_output)?;
        }
        let sample = tail.sample_tail(stage_output)?;
        let step_ms = step_start.elapsed().as_millis();

        let Some(&next_token_id) = sample.token_ids.first() else {
            break;
        };
        if ttft_ms == 0 {
            ttft_ms = step_ms;
        }
        if eos_token_id == Some(next_token_id) {
            finish_reason = "stop".to_string();
            break;
        }

        generated_token_ids.push(next_token_id);
        prompt_token_ids.push(next_token_id);
        text = tail.decode_token_ids(&generated_token_ids);
        if let Some(trimmed) = trim_at_stop_sequence(&text, stop_sequences) {
            text = trimmed;
            finish_reason = "stop".to_string();
            break;
        }
    }

    let total_ms = total_start.elapsed().as_millis();
    if text.is_empty() && !generated_token_ids.is_empty() {
        text = tail.decode_token_ids(&generated_token_ids);
    }
    for backend in backends {
        backend.clear_decode_session(request_id);
    }

    Ok(LocalRun {
        prompt_tokens,
        finish_reason,
        text,
        token_ids: generated_token_ids,
        ttft_ms,
        total_ms,
    })
}

fn trim_at_stop_sequence(text: &str, stop_sequences: &[&str]) -> Option<String> {
    let stop_at = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()?;
    Some(text[..stop_at].to_string())
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

fn total_tok_s(total_ms: u128, completion_tokens: u32) -> f64 {
    if total_ms == 0 || completion_tokens == 0 {
        return 0.0;
    }
    completion_tokens as f64 / (total_ms as f64 / 1000.0)
}

fn continuation_tok_s(ttft_ms: u128, total_ms: u128, completion_tokens: u32) -> f64 {
    if completion_tokens <= 1 {
        return total_tok_s(total_ms, completion_tokens);
    }
    let continuation_ms = total_ms.saturating_sub(ttft_ms);
    if continuation_ms == 0 {
        return 0.0;
    }
    (completion_tokens - 1) as f64 / (continuation_ms as f64 / 1000.0)
}

fn format_runtime_profile(profile: &StagePrototypeRuntimeProfile) -> String {
    format!(
        "steps={} tokenize={}ms connect={}ms assign={}ms head={}ms down_wait={}ms tail_engine={}ms tail_total={}ms detok={}ms clear={}ms first(head={}ms down_wait={}ms tail_engine={}ms tail_total={}ms)",
        profile.steps,
        profile.tokenize_ms,
        profile.connect_ms,
        profile.assign_ms,
        profile.head_engine_ms,
        profile.downstream_wait_ms,
        profile.downstream_stage_engine_ms,
        profile.downstream_stage_total_ms,
        profile.detokenize_ms,
        profile.clear_decode_session_ms,
        profile.first_head_engine_ms,
        profile.first_downstream_wait_ms,
        profile.first_downstream_stage_engine_ms,
        profile.first_downstream_stage_total_ms,
    )
}
