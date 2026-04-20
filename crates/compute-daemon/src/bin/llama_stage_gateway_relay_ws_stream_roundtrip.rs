use anyhow::{Context, Result, bail};
use compute_daemon::config::Config;
use compute_daemon::relay::{AssignmentPush, RelayClient};
use compute_daemon::stage_runtime::StagePrototypeClient;
use futures_util::{SinkExt, StreamExt};
use llama_stage_backend::{ManagedGatewayStack, resolve_model_arg};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

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

    let listener = TcpListener::bind("127.0.0.1:0").await.context("bind mock ws addr")?;
    let ws_addr = listener.local_addr().context("mock ws local addr")?;
    let prompts_for_server = prompts.clone();

    let server_task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.context("accept relay websocket")?;
        let mut ws = accept_async(stream).await.context("accept websocket handshake")?;

        loop {
            match ws.next().await {
                Some(Ok(Message::Text(text))) => {
                    let value: serde_json::Value =
                        serde_json::from_str(&text).context("parse identify message")?;
                    if value.get("type").and_then(|v| v.as_str()) == Some("identify") {
                        break;
                    }
                }
                Some(Ok(Message::Ping(data))) => {
                    ws.send(Message::Pong(data)).await.context("pong identify")?;
                }
                Some(Ok(_)) => {}
                Some(Err(err)) => return Err(anyhow::anyhow!("relay websocket read error: {err}")),
                None => bail!("relay disconnected before identify"),
            }
        }

        ws.send(Message::Text(json!({"type":"identified"}).to_string()))
            .await
            .context("send identified")?;

        let mut results = Vec::new();
        for (idx, prompt) in prompts_for_server.iter().enumerate() {
            let request = json!({
                "id": format!("ws-stream-{idx}"),
                "type": "inference_request",
                "method": "POST",
                "path": "/v1/chat/completions",
                "body": {
                    "model": "gemma-4-E4B-it-Q4_K_M.gguf",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": true,
                }
            });
            ws.send(Message::Text(request.to_string()))
                .await
                .with_context(|| format!("send streaming request {idx}"))?;

            let mut streamed_text = String::new();
            let mut saw_done = false;
            let mut saw_finish_reason = false;
            let final_response = loop {
                match ws.next().await {
                    Some(Ok(Message::Text(text))) => {
                        let value: serde_json::Value =
                            serde_json::from_str(&text).context("parse relay message")?;
                        match value.get("type").and_then(|v| v.as_str()) {
                            Some("inference_stream_chunk") => {
                                let chunk =
                                    value.get("chunk").and_then(|v| v.as_str()).unwrap_or_default();
                                for line in chunk.lines() {
                                    let Some(data) = line.strip_prefix("data: ") else {
                                        continue;
                                    };
                                    if data == "[DONE]" {
                                        saw_done = true;
                                        continue;
                                    }
                                    let chunk_json: serde_json::Value = serde_json::from_str(data)
                                        .context("parse sse chunk json")?;
                                    if let Some(content) =
                                        chunk_json["choices"][0]["delta"]["content"].as_str()
                                    {
                                        streamed_text.push_str(content);
                                    }
                                    if !chunk_json["choices"][0]["finish_reason"].is_null() {
                                        saw_finish_reason = true;
                                    }
                                }
                            }
                            Some("inference_response") => break value,
                            _ => {}
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        ws.send(Message::Pong(data)).await.context("pong streaming request")?;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        return Err(anyhow::anyhow!("relay websocket read error: {err}"));
                    }
                    None => bail!("relay disconnected before streaming response"),
                }
            };

            results.push((streamed_text, saw_done, saw_finish_reason, final_response));
        }

        let _ = ws.close(None).await;
        Ok::<_, anyhow::Error>(results)
    });

    let mut config = Config::default();
    config.network.orchestrator_url = format!("http://{ws_addr}");
    config.wallet.public_address = "wallet-test".to_string();
    config.wallet.node_id = "node-test".to_string();
    config.wallet.node_token = "token-test".to_string();
    config.experimental.stage_mode_enabled = true;
    config.experimental.stage_backend = "llama-stage-gateway".to_string();
    config.experimental.stage_gateway_addr = gateway_addr.clone();

    let shutdown = Arc::new(AtomicBool::new(false));
    let (assignment_tx, _assignment_rx) = mpsc::channel::<AssignmentPush>(8);
    let (_ws_outbound_tx, ws_outbound_rx) = mpsc::channel::<String>(16);
    let stage_client: Arc<tokio::sync::Mutex<Option<StagePrototypeClient>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let gateway_client = Arc::new(tokio::sync::Mutex::new(Some(
        compute_daemon::inference::llama_stage_gateway::LlamaStageGatewayRelayClient::connect(
            &gateway_addr,
        )?,
    )));

    let relay = RelayClient::new(
        &config,
        shutdown.clone(),
        assignment_tx,
        ws_outbound_rx,
        stage_client,
        gateway_client,
        std::sync::Arc::new(std::sync::atomic::AtomicU16::new(0)),
    );

    let relay_task = tokio::spawn(async move { relay.run().await });
    let results = server_task.await.context("mock orchestrator join")??;
    shutdown.store(true, Ordering::Relaxed);
    let _ = relay_task.await.context("relay task join")??;

    let mut all_match = true;
    for (idx, prompt) in prompts.iter().enumerate() {
        let baseline =
            llama_stage_backend::greedy_single_node_completion(&model_path, prompt, max_tokens)?;
        let (streamed_text, saw_done, saw_finish_reason, response) =
            results.get(idx).context("missing streaming result")?;
        let body = response.get("body").context("streaming response missing body")?;

        let completion_tokens =
            body["usage"]["completion_tokens"].as_u64().unwrap_or_default() as u32;
        let gateway_flag = body["llama_stage_gateway"].as_bool().unwrap_or(false);
        let stage_flag = body["prototype_stage_mode"].as_bool().unwrap_or(true);
        let has_timings = body["gateway_timings"].is_object();
        let status = response.get("status").and_then(|v| v.as_u64()).unwrap_or_default();

        let case_match = status == 200
            && streamed_text == &baseline.text
            && completion_tokens == baseline.completion_tokens
            && *saw_done
            && *saw_finish_reason
            && gateway_flag
            && !stage_flag
            && has_timings;
        all_match &= case_match;

        println!("case={idx}");
        println!("prompt={prompt:?}");
        println!("baseline_text={:?}", baseline.text);
        println!("streamed_text={streamed_text:?}");
        println!("baseline_token_ids={:?}", baseline.token_ids);
        println!("response_prompt_tokens={}", body["usage"]["prompt_tokens"]);
        println!("response_completion_tokens={completion_tokens}");
        println!("head_prefill_ms={}", body["gateway_timings"]["head_prefill_ms"]);
        println!("head_decode_ms={}", body["gateway_timings"]["head_decode_ms"]);
        println!("tail_decode_ms={}", body["gateway_timings"]["tail_decode_ms"]);
        println!("ttft_ms={}", body["gateway_timings"]["ttft_ms"]);
        println!("total_ms={}", body["gateway_timings"]["total_ms"]);
        println!("saw_finish_reason={saw_finish_reason}");
        println!("saw_done={saw_done}");
        println!("match={case_match}");
        println!();
    }

    drop(managed_gateway_stack);

    if all_match {
        println!("overall=PASS");
        Ok(())
    } else {
        bail!("websocket streaming relay roundtrip did not match baseline")
    }
}
