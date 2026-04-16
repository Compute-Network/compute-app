use anyhow::{Result, bail};
use compute_daemon::inference::llama_stage_gateway::LlamaStageGatewayRelayClient;
use llama_stage_backend::{ManagedGatewayStack, greedy_single_node_completion, resolve_model_arg};
use std::env;
use std::path::PathBuf;

fn default_prompts() -> Vec<String> {
    vec![
        "The capital of France is".to_string(),
        "The opposite of hot is".to_string(),
        "Continue: 1, 2, 3,".to_string(),
    ]
}

fn parse_args() -> Result<(PathBuf, u32, bool, Option<String>, Vec<String>)> {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);
    let mut max_tokens = 4u32;
    let mut reconnect_after_prompt = false;
    let mut gateway_addr = None;

    while idx < args.len() {
        match args[idx].as_str() {
            "--max-tokens" => {
                if let Some(raw) = args.get(idx + 1) {
                    if let Ok(parsed) = raw.parse::<u32>() {
                        max_tokens = parsed.max(1);
                    }
                }
                idx += 2;
            }
            "--reconnect-after-prompt" => {
                reconnect_after_prompt = true;
                idx += 1;
            }
            "--gateway" => {
                let Some(addr) = args.get(idx + 1) else {
                    bail!("--gateway requires an address");
                };
                gateway_addr = Some(addr.clone());
                idx += 2;
            }
            _ => break,
        }
    }

    let prompts = if args.len() > idx { args[idx..].to_vec() } else { default_prompts() };
    Ok((model_path, max_tokens, reconnect_after_prompt, gateway_addr, prompts))
}

#[tokio::main]
async fn main() -> Result<()> {
    let (model_path, max_tokens, reconnect_after_prompt, gateway_addr, prompts) = parse_args()?;
    let mut managed_gateway_stack = None;

    let gateway_addr = if let Some(addr) = gateway_addr {
        addr
    } else {
        let stack = ManagedGatewayStack::spawn_local(model_path.clone(), reconnect_after_prompt)?;
        let addr = stack.gateway_addr().to_string();
        managed_gateway_stack = Some(stack);
        addr
    };

    let client = LlamaStageGatewayRelayClient::connect(&gateway_addr)?;
    let mut all_match = true;

    for (idx, prompt) in prompts.iter().enumerate() {
        let baseline = greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let request_body = serde_json::json!({
            "model": "gemma-4-E4B-it-Q4_K_M.gguf",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "stream": false,
        });
        let completion =
            client.complete_chat_request(format!("relay-smoke-{idx}"), &request_body).await?;
        let result = completion.result;
        let response_body = completion.body;

        let match_text = result.content == baseline.text;
        let match_tokens = result.completion.token_ids == baseline.token_ids;
        let match_completion_tokens = result.completion_tokens == baseline.completion_tokens;
        let response_text =
            response_body["choices"][0]["message"]["content"].as_str().unwrap_or_default();
        let response_completion_tokens =
            response_body["usage"]["completion_tokens"].as_u64().unwrap_or_default() as u32;
        let response_total_tokens =
            response_body["usage"]["total_tokens"].as_u64().unwrap_or_default() as u32;
        let response_gateway_flag = response_body["llama_stage_gateway"].as_bool().unwrap_or(false);
        let response_stage_flag = response_body["prototype_stage_mode"].as_bool().unwrap_or(true);
        let response_has_timings = response_body["gateway_timings"].is_object();
        let response_shape_ok = response_text == result.content
            && response_completion_tokens == result.completion_tokens
            && response_total_tokens == result.total_tokens
            && response_gateway_flag
            && !response_stage_flag
            && response_has_timings;
        let case_match = match_text && match_tokens && match_completion_tokens && response_shape_ok;
        all_match &= case_match;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("baseline_text={:?}", baseline.text);
        println!("relay_text={:?}", result.content);
        println!("baseline_token_ids={:?}", baseline.token_ids);
        println!("relay_token_ids={:?}", result.completion.token_ids);
        println!("baseline_tokens={}", baseline.completion_tokens);
        println!("relay_tokens={}", result.completion_tokens);
        println!("response_finish_reason={:?}", response_body["choices"][0]["finish_reason"]);
        println!("response_prompt_tokens={}", response_body["usage"]["prompt_tokens"]);
        println!("response_completion_tokens={}", response_body["usage"]["completion_tokens"]);
        println!("head_prefill_ms={}", result.completion.timings.head_prefill_ms);
        println!("head_decode_ms={}", result.completion.timings.head_decode_ms);
        println!("tail_decode_ms={}", result.completion.timings.tail_decode_ms);
        println!("ttft_ms={}", result.completion.timings.ttft_ms);
        println!("total_ms={}", result.completion.timings.total_ms);
        println!("response_shape_ok={response_shape_ok}");
        println!("match={case_match}");
        println!();
    }

    drop(managed_gateway_stack);

    if all_match {
        println!("overall=PASS");
        Ok(())
    } else {
        bail!("relay smoke probe did not match baseline")
    }
}
