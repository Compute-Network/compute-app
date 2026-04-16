use anyhow::{Context, Result};
use compute_daemon::config::Config;
use compute_daemon::hardware::HardwareInfo;
use compute_daemon::stage_runtime::start_stage_prototype_chain;
use compute_network::models::ModelCatalog;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_name = args.get(1).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let stage_count = args.get(2).and_then(|value| value.parse::<u32>().ok()).unwrap_or(3);
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Hello".to_string());
    let max_tokens = args.get(4).and_then(|value| value.parse::<u32>().ok());
    let stage_backend = args.get(5).cloned().unwrap_or_else(|| "prototype".to_string());

    let model = ModelCatalog::default_catalog()
        .find(&model_name)
        .cloned()
        .with_context(|| format!("Unknown model in catalog: {model_name}"))?;
    if stage_count == 0 {
        anyhow::bail!("Stage count must be at least 1");
    }
    let ranges = model
        .shard_for_stages(stage_count)
        .into_iter()
        .map(|shard| (shard.start_layer, shard.end_layer))
        .collect::<Vec<_>>();

    println!("=== Prototype Stage Chain Roundtrip ===");
    println!("model         : {}", model.id);
    println!("stages        : {}", stage_count);
    println!("prompt        : {:?}", prompt);
    println!("max tokens    : {:?}", max_tokens);
    println!("backend       : {}", stage_backend);
    println!();
    for (index, (start_layer, end_layer)) in ranges.iter().enumerate() {
        println!("planned stage : {} => {}-{}", index, start_layer, end_layer);
    }
    println!();

    let mut config = Config::default();
    config.experimental.stage_backend = stage_backend;
    config.experimental.stage_mode_enabled = true;
    let hw = HardwareInfo::empty();
    let mut handles =
        start_stage_prototype_chain(&config, &hw, "prototype-chain-roundtrip", &model.id, &ranges)
            .await?;

    for (index, handle) in handles.iter().enumerate() {
        println!("live stage    : {} => {} @ {}", index, ranges[index].0, handle.listen_addr());
    }
    println!();

    let result = handles[0]
        .client()
        .complete_prompt("prototype-chain-roundtrip".into(), prompt, max_tokens, Vec::new())
        .await?;

    println!("finish reason : {}", result.finish_reason);
    println!("prompt tokens : {}", result.prompt_tokens);
    println!("completion tok: {}", result.completion_tokens);
    println!("token ids     : {:?}", result.completion_token_ids);
    println!("content       : {:?}", result.content);

    while let Some(handle) = handles.pop() {
        handle.stop().await;
    }

    Ok(())
}
