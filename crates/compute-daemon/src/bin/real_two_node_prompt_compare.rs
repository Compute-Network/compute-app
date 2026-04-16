use anyhow::{Result, bail};
use compute_daemon::config::Config;
use compute_daemon::hardware::HardwareInfo;
use compute_daemon::stage_runtime::{StagePrototypeSpec, start_stage_prototype_with_bind_addr};
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use stage_forward_lab::prompting::GemmaPromptMode;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::fs;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    let stage1_index = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
        )
    });
    let stage2_index = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json",
        )
    });
    let vocab_path = args.get(3).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("../compute-backend/out/gemma-e4b-2stage/packed-stage-1/vocab.json")
    });
    let prompt_mode =
        args.get(4).and_then(|value| GemmaPromptMode::parse(value)).unwrap_or_default();
    let suite_mode = args
        .get(5)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let remote_tail_addr = args.get(6).cloned();
    let stage_acceleration = args.get(7).cloned().unwrap_or_else(|| "metal".to_string());
    let stage_acceleration_provider = args.get(8).cloned().unwrap_or_else(|| "ggml".to_string());

    println!("=== Real Two-Node Prompt Compare ===");
    println!("stage 1 index: {}", stage1_index.display());
    println!("stage 2 index: {}", stage2_index.display());
    println!("vocab        : {}", vocab_path.display());
    println!("prompt mode  : {}", prompt_mode.as_str());
    println!("suite mode   : {}", suite_mode.as_str());
    println!("remote tail  : {:?}", remote_tail_addr);
    println!("accel target : {}", stage_acceleration);
    println!("accel provider: {}", stage_acceleration_provider);
    println!();

    let stage1_dir = stage_dir_from_index(&stage1_index)?;
    let stage2_dir = stage_dir_from_index(&stage2_index)?;
    let (head, tail) = load_backend_pair(&stage1_index, &stage2_index, &vocab_path)?;

    let stage_root = prepare_temp_stage_root(&stage1_dir, &stage2_dir)?;
    let previous_stage_root = env::var_os("COMPUTE_STAGE_ROOT");
    // This process owns the loopback harness for its lifetime; set the stage root
    // before the runtimes start and restore it after both have shut down.
    unsafe {
        env::set_var("COMPUTE_STAGE_ROOT", &stage_root);
    }

    let mut config = Config::default();
    config.experimental.stage_backend = "real_forward".to_string();
    config.experimental.stage_acceleration = stage_acceleration.clone();
    config.experimental.stage_acceleration_provider = stage_acceleration_provider.clone();
    let hw = HardwareInfo::empty();

    let tail_handle = if remote_tail_addr.is_none() {
        Some(
            start_stage_prototype_with_bind_addr(
                &config,
                &hw,
                StagePrototypeSpec {
                    pipeline_id: "prompt-compare".into(),
                    model_name: "gemma-4-e4b-q4".into(),
                    shard_id: "tail".into(),
                    start_layer: 21,
                    end_layer: 41,
                    stage_index: 1,
                    total_stages: 2,
                    upstream_addr: None,
                    downstream_addr: None,
                },
                SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            )
            .await?,
        )
    } else {
        None
    };
    let downstream_addr = remote_tail_addr
        .clone()
        .unwrap_or_else(|| tail_handle.as_ref().unwrap().listen_addr().to_string());

    let head_handle = start_stage_prototype_with_bind_addr(
        &config,
        &hw,
        StagePrototypeSpec {
            pipeline_id: "prompt-compare".into(),
            model_name: "gemma-4-e4b-q4".into(),
            shard_id: "head".into(),
            start_layer: 0,
            end_layer: 20,
            stage_index: 0,
            total_stages: 2,
            upstream_addr: None,
            downstream_addr: Some(downstream_addr),
        },
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
    )
    .await?;
    let head_client = head_handle.client();

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        let local = run_local(
            &head,
            &tail,
            case.prompt,
            case.max_tokens,
            case.stop_sequences,
            prompt_mode,
            &format!("local-{}", case.name),
        )?;

        let start = Instant::now();
        let live = head_client
            .complete_prompt(
                format!("live-{}", case.name),
                case.prompt.to_string(),
                Some(case.max_tokens),
                case.stop_sequences.iter().map(|stop| stop.to_string()).collect(),
            )
            .await?;
        let live_wall_ms = start.elapsed().as_millis();

        let text_match = live.content == local.text;
        let finish_match = live.finish_reason == local.finish_reason;
        let tokens_match = live.completion_token_ids == local.token_ids;
        let prompt_tokens_match = live.prompt_tokens as usize == local.prompt_tokens;
        let ok = text_match && finish_match && tokens_match && prompt_tokens_match;
        failed |= !ok;

        println!("=== Case: {} ===", case.name);
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
            "two-node live : ttft={}ms total={}ms wall={}ms finish={} prompt_toks={} cont_tok_s={:.2} total_tok_s={:.2} text={:?} token_ids={:?}",
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
        println!(
            "compare       : text={} finish={} token_ids={} prompt_toks={} overall={}",
            pass_fail(text_match),
            pass_fail(finish_match),
            pass_fail(tokens_match),
            pass_fail(prompt_tokens_match),
            pass_fail(ok)
        );
        println!();
    }

    head_handle.stop().await;
    if let Some(tail_handle) = tail_handle {
        tail_handle.stop().await;
    }
    restore_stage_root(previous_stage_root);
    let _ = fs::remove_dir_all(&stage_root);

    if failed {
        bail!("two-node prompt compare failed");
    }

    println!("overall: PASS");
    Ok(())
}

fn load_backend_pair(
    stage1_index: &Path,
    stage2_index: &Path,
    vocab_path: &Path,
) -> Result<(RealGemmaBackend, RealGemmaBackend)> {
    let scores_path =
        vocab_path.parent().map(|parent| parent.join("vocab_scores.json")).unwrap_or_else(|| {
            PathBuf::from("../compute-backend/out/gemma-e4b-2stage/vocab_scores.json")
        });

    let mut head = RealGemmaBackend::new(stage1_index);
    if vocab_path.exists() {
        let sp = if scores_path.exists() { Some(scores_path.as_path()) } else { None };
        head.load_tokenizer(vocab_path, sp)?;
    }
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;

    let mut tail = RealGemmaBackend::new(stage2_index);
    if vocab_path.exists() {
        let sp = if scores_path.exists() { Some(scores_path.as_path()) } else { None };
        tail.load_tokenizer(vocab_path, sp)?;
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;

    Ok((head, tail))
}

fn run_local(
    head: &RealGemmaBackend,
    tail: &RealGemmaBackend,
    prompt: &str,
    max_tokens: u32,
    stop_sequences: &[&str],
    prompt_mode: GemmaPromptMode,
    request_id: &str,
) -> Result<LocalRun> {
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
        let step_token_ids: Vec<u32> = if step == 0 {
            prompt_token_ids.clone()
        } else {
            vec![*generated_token_ids.last().unwrap()]
        };
        let head_output = head.begin_token_ids(request_id, &step_token_ids, Some(1), 0)?;
        let tail_output = tail.continue_forward(head_output)?;
        let sample = tail.sample_tail(tail_output)?;
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

    head.clear_decode_session(request_id);
    tail.clear_decode_session(request_id);

    Ok(LocalRun {
        prompt_tokens,
        finish_reason,
        text,
        token_ids: generated_token_ids,
        ttft_ms,
        total_ms,
    })
}

fn stage_dir_from_index(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.canonicalize()?);
    }
    path.parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot resolve parent of {}", path.display()))?
        .canonicalize()
        .map_err(Into::into)
}

fn prepare_temp_stage_root(stage1_dir: &Path, stage2_dir: &Path) -> Result<PathBuf> {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let root = env::temp_dir()
        .join(format!("compute-stage-root-{}-{unique}", std::process::id()))
        .join("stages");
    let model_root = root.join("gemma-4-e4b-q4");
    fs::create_dir_all(&model_root)?;
    mirror_dir(stage1_dir, &model_root.join("packed-stage-0-20"))?;
    mirror_dir(stage2_dir, &model_root.join("packed-stage-21-41"))?;
    Ok(root)
}

fn restore_stage_root(previous: Option<std::ffi::OsString>) {
    if let Some(previous) = previous {
        unsafe {
            env::set_var("COMPUTE_STAGE_ROOT", previous);
        }
    } else {
        unsafe {
            env::remove_var("COMPUTE_STAGE_ROOT");
        }
    }
}

fn mirror_dir(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }

    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(src, dst).is_ok() {
            return Ok(());
        }
    }

    copy_dir_all(src, dst)
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
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
