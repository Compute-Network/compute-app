use anyhow::Result;
use compute_daemon::prototype_chain::request_hosted_prototype_head;
use std::env;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let head_addr: SocketAddr = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or_else(|| "127.0.0.1:9090".parse().expect("valid default head addr"));
    let prompt = args.get(2).cloned().unwrap_or_else(|| "Hello".to_string());
    let max_tokens = args.get(3).and_then(|value| value.parse::<u32>().ok());
    let stop_sequences = args
        .get(4)
        .map(|value| {
            value
                .split(',')
                .filter_map(|entry| {
                    let trimmed = entry.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let hidden_dim = args.get(5).and_then(|value| value.parse::<usize>().ok()).unwrap_or(2048);

    println!("=== Prototype Stage Chain Client ===");
    println!("head addr     : {}", head_addr);
    println!("prompt        : {:?}", prompt);
    println!("max tokens    : {:?}", max_tokens);
    println!("stop seqs     : {:?}", stop_sequences);
    println!("hidden dim    : {}", hidden_dim);

    let result =
        request_hosted_prototype_head(head_addr, &prompt, max_tokens, &stop_sequences, hidden_dim)
            .await?;
    println!("elapsed ms    : {}", result.elapsed_ms);
    println!("prompt tokens : {}", result.prompt_tokens);
    println!("completion tok: {}", result.completion_token_ids.len());
    println!("finish reason : {}", result.finish_reason);
    println!("token ids     : {:?}", result.completion_token_ids);
    println!("content       : {:?}", result.content);
    Ok(())
}
