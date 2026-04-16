use anyhow::{Result, bail};
use compute_daemon::config::Config;
use compute_daemon::hardware::HardwareInfo;
use compute_daemon::prototype_chain::request_hosted_prototype_head;
use compute_daemon::stage_runtime::start_stage_prototype_chain;
use compute_network::models::ModelCatalog;
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use std::env;
use std::net::SocketAddr;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_name = args.get(1).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let stage_count = args.get(2).and_then(|value| value.parse::<u32>().ok()).unwrap_or(3);
    let suite_mode = args
        .get(3)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let stage_addrs =
        args.get(4).map(|value| parse_stage_addrs(value)).transpose()?.unwrap_or_default();
    if stage_addrs.is_empty() {
        bail!("Expected comma-separated hosted stage addresses");
    }
    if stage_count == 0 {
        bail!("Stage count must be at least 1");
    }
    if stage_addrs.len() != stage_count as usize {
        bail!(
            "Hosted stage address count {} did not match requested stage count {}",
            stage_addrs.len(),
            stage_count
        );
    }

    let model = ModelCatalog::default_catalog()
        .find(&model_name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Unknown model in catalog: {model_name}"))?;
    let ranges = model
        .shard_for_stages(stage_count)
        .into_iter()
        .map(|shard| (shard.start_layer, shard.end_layer))
        .collect::<Vec<_>>();

    println!("=== Prototype Stage Chain Compare ===");
    println!("model         : {}", model.id);
    println!("stages        : {}", stage_count);
    println!("suite mode    : {}", suite_mode.as_str());
    for (index, ((start_layer, end_layer), addr)) in
        ranges.iter().zip(stage_addrs.iter()).enumerate()
    {
        println!("stage         : {} => {}-{} @ {}", index, start_layer, end_layer, addr);
    }
    println!();

    let mut config = Config::default();
    config.experimental.stage_backend = "prototype".to_string();
    let hw = HardwareInfo::empty();
    let mut local_handles =
        start_stage_prototype_chain(&config, &hw, "prototype-chain-compare", &model.id, &ranges)
            .await?;
    let local_head = local_handles[0].client();
    let external_head = stage_addrs[0];

    let mut failed = false;
    for case in validation_prompt_cases(suite_mode) {
        let stop_sequences =
            case.stop_sequences.iter().map(|stop| stop.to_string()).collect::<Vec<_>>();
        let local_start = Instant::now();
        let local = local_head
            .complete_prompt(
                format!("local-{}", case.name),
                case.prompt.to_string(),
                Some(case.max_tokens),
                stop_sequences.clone(),
            )
            .await?;
        let local_ms = local_start.elapsed().as_millis();

        let external = request_hosted_prototype_head(
            external_head,
            case.prompt,
            Some(case.max_tokens),
            &stop_sequences,
            2048,
        )
        .await?;

        let text_match = external.content == local.content;
        let finish_match = external.finish_reason == local.finish_reason;
        let tokens_match = external.completion_token_ids == local.completion_token_ids;
        let prompt_tokens_match = external.prompt_tokens == local.prompt_tokens as usize;
        let ok = text_match && finish_match && tokens_match && prompt_tokens_match;
        failed |= !ok;

        println!("=== Case: {} ===", case.name);
        println!("prompt        : {:?}", case.prompt);
        println!(
            "local         : total={}ms finish={} prompt_toks={} text={:?} token_ids={:?}",
            local_ms,
            local.finish_reason,
            local.prompt_tokens,
            local.content,
            local.completion_token_ids
        );
        println!(
            "external      : total={}ms finish={} prompt_toks={} text={:?} token_ids={:?}",
            external.elapsed_ms,
            external.finish_reason,
            external.prompt_tokens,
            external.content,
            external.completion_token_ids
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

    while let Some(handle) = local_handles.pop() {
        handle.stop().await;
    }

    if failed {
        bail!("prototype chain compare failed");
    }
    println!("overall: PASS");
    Ok(())
}

fn parse_stage_addrs(value: &str) -> Result<Vec<SocketAddr>> {
    value
        .split(',')
        .filter_map(|entry| {
            let trimmed = entry.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .map(str::parse)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn pass_fail(value: bool) -> &'static str {
    if value { "PASS" } else { "FAIL" }
}
