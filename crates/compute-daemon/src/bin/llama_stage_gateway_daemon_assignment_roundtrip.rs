use anyhow::{Context, Result, bail};
use compute_daemon::config::Config;
use compute_daemon::runtime::DaemonRuntime;
use futures_util::{SinkExt, StreamExt};
use llama_stage_backend::{
    ManagedGatewayLaunchSpec, ManagedGatewayStack, greedy_single_node_completion, resolve_model_arg,
};
use serde_json::json;
use std::env;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::time::{Duration, timeout};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

fn default_prompt() -> String {
    "The capital of France is".to_string()
}

fn reserve_loopback_addr() -> Result<String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").context("bind loopback port")?;
    let addr = listener.local_addr().context("loopback local addr")?;
    Ok(addr.to_string())
}

struct ParsedArgs {
    model_path: PathBuf,
    max_tokens: u32,
    reconnect_after_prompt: bool,
    stream: bool,
    gateway_addr: Option<String>,
    delay_gateway_start_ms: u64,
    gateway_connect_timeout_ms: u64,
    gateway_retry_window_ms: u64,
    gateway_retry_interval_ms: u64,
    gateway_startup_grace_ms: u64,
    mock_gateway_error: Option<MockGatewayError>,
    expect_node_error: Option<String>,
    prompt: String,
}

#[derive(Clone, Copy, Debug)]
enum MockGatewayError {
    ProtocolMismatch,
    ModelMismatch,
    UnusableGateway,
}

impl MockGatewayError {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "protocol-mismatch" => Ok(Self::ProtocolMismatch),
            "model-mismatch" => Ok(Self::ModelMismatch),
            "unusable-gateway" => Ok(Self::UnusableGateway),
            other => bail!("unknown --mock-gateway-error scenario: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ProtocolMismatch => "protocol-mismatch",
            Self::ModelMismatch => "model-mismatch",
            Self::UnusableGateway => "unusable-gateway",
        }
    }
}

enum ServerOutcome {
    Success {
        identify: serde_json::Value,
        node_ready: serde_json::Value,
        streamed_text: String,
        saw_done: bool,
        saw_finish_reason: bool,
        response: serde_json::Value,
    },
    NodeError {
        identify: serde_json::Value,
        node_error: serde_json::Value,
    },
}

fn parse_args() -> Result<ParsedArgs> {
    let args: Vec<String> = env::args().collect();
    let (model_path, mut idx) = resolve_model_arg(&args);
    let mut max_tokens = 4u32;
    let mut reconnect_after_prompt = false;
    let mut stream = false;
    let mut gateway_addr = None;
    let mut delay_gateway_start_ms = 0u64;
    let mut gateway_connect_timeout_ms = 2000u64;
    let mut gateway_retry_window_ms = 30000u64;
    let mut gateway_retry_interval_ms = 250u64;
    let mut gateway_startup_grace_ms = 0u64;
    let mut mock_gateway_error = None;
    let mut expect_node_error = None;

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
            "--stream" => {
                stream = true;
                idx += 1;
            }
            "--gateway" => {
                let Some(addr) = args.get(idx + 1) else {
                    bail!("--gateway requires an address");
                };
                gateway_addr = Some(addr.clone());
                idx += 2;
            }
            "--delay-gateway-start-ms" => {
                let Some(raw) = args.get(idx + 1) else {
                    bail!("--delay-gateway-start-ms requires a duration");
                };
                delay_gateway_start_ms = raw.parse::<u64>().context("parse delay ms")?;
                idx += 2;
            }
            "--gateway-connect-timeout-ms" => {
                let Some(raw) = args.get(idx + 1) else {
                    bail!("--gateway-connect-timeout-ms requires a duration");
                };
                gateway_connect_timeout_ms =
                    raw.parse::<u64>().context("parse gateway connect timeout ms")?;
                idx += 2;
            }
            "--gateway-retry-window-ms" => {
                let Some(raw) = args.get(idx + 1) else {
                    bail!("--gateway-retry-window-ms requires a duration");
                };
                gateway_retry_window_ms =
                    raw.parse::<u64>().context("parse gateway retry window ms")?;
                idx += 2;
            }
            "--gateway-retry-interval-ms" => {
                let Some(raw) = args.get(idx + 1) else {
                    bail!("--gateway-retry-interval-ms requires a duration");
                };
                gateway_retry_interval_ms =
                    raw.parse::<u64>().context("parse gateway retry interval ms")?;
                idx += 2;
            }
            "--gateway-startup-grace-ms" => {
                let Some(raw) = args.get(idx + 1) else {
                    bail!("--gateway-startup-grace-ms requires a duration");
                };
                gateway_startup_grace_ms =
                    raw.parse::<u64>().context("parse gateway startup grace ms")?;
                idx += 2;
            }
            "--mock-gateway-error" => {
                let Some(raw) = args.get(idx + 1) else {
                    bail!("--mock-gateway-error requires a scenario");
                };
                mock_gateway_error = Some(MockGatewayError::parse(raw)?);
                idx += 2;
            }
            "--expect-node-error" => {
                let Some(raw) = args.get(idx + 1) else {
                    bail!("--expect-node-error requires a substring");
                };
                expect_node_error = Some(raw.clone());
                idx += 2;
            }
            _ => break,
        }
    }

    if gateway_addr.is_some() && delay_gateway_start_ms > 0 {
        bail!("--gateway and --delay-gateway-start-ms are mutually exclusive");
    }
    if gateway_addr.is_some() && mock_gateway_error.is_some() {
        bail!("--gateway and --mock-gateway-error are mutually exclusive");
    }
    if delay_gateway_start_ms > 0 && mock_gateway_error.is_some() {
        bail!("--delay-gateway-start-ms and --mock-gateway-error are mutually exclusive");
    }

    let prompt = args.get(idx).cloned().unwrap_or_else(default_prompt);
    Ok(ParsedArgs {
        model_path,
        max_tokens: max_tokens.max(1),
        reconnect_after_prompt,
        stream,
        gateway_addr,
        delay_gateway_start_ms,
        gateway_connect_timeout_ms: gateway_connect_timeout_ms.max(1),
        gateway_retry_window_ms: gateway_retry_window_ms.max(1),
        gateway_retry_interval_ms: gateway_retry_interval_ms.max(1),
        gateway_startup_grace_ms,
        mock_gateway_error,
        expect_node_error,
        prompt,
    })
}

fn spawn_mock_gateway_process(scenario: MockGatewayError) -> Result<(String, Child)> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .context("resolve workspace root")?;
    let debug_bin = workspace_root.join("target/debug/llama_stage_gateway_mock_node");
    let release_bin = workspace_root.join("target/release/llama_stage_gateway_mock_node");

    let mut command = if debug_bin.exists() {
        let mut cmd = Command::new(debug_bin);
        cmd.arg("--bind").arg("127.0.0.1:0");
        cmd.arg("--scenario").arg(scenario.as_str());
        cmd
    } else if release_bin.exists() {
        let mut cmd = Command::new(release_bin);
        cmd.arg("--bind").arg("127.0.0.1:0");
        cmd.arg("--scenario").arg(scenario.as_str());
        cmd
    } else {
        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("-q")
            .arg("-p")
            .arg("compute-daemon")
            .arg("--bin")
            .arg("llama_stage_gateway_mock_node")
            .arg("--")
            .arg("--bind")
            .arg("127.0.0.1:0")
            .arg("--scenario")
            .arg(scenario.as_str());
        cmd
    };

    command.stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = command.spawn().context("spawn mock gateway process")?;
    let stderr = child.stderr.take().context("mock gateway stderr unavailable")?;
    let reader = BufReader::new(stderr);
    for line in reader.lines() {
        let line = line.context("read mock gateway stderr")?;
        if let Some(addr) = line.strip_prefix("listening=") {
            return Ok((addr.trim().to_string(), child));
        }
    }
    bail!("mock gateway did not report a listening address")
}

#[tokio::main]
async fn main() -> Result<()> {
    let ParsedArgs {
        model_path,
        max_tokens,
        reconnect_after_prompt,
        stream,
        gateway_addr,
        delay_gateway_start_ms,
        gateway_connect_timeout_ms,
        gateway_retry_window_ms,
        gateway_retry_interval_ms,
        gateway_startup_grace_ms,
        mock_gateway_error,
        expect_node_error,
        prompt,
    } = parse_args()?;
    let mut managed_gateway_stack = None;
    let mut delayed_shutdown = None;
    let mut delayed_stack_task = None;
    let mut mock_gateway_child = None;

    let gateway_addr = if let Some(addr) = gateway_addr {
        addr
    } else if let Some(mock_gateway_error) = mock_gateway_error {
        let (gateway_bind, child) = spawn_mock_gateway_process(mock_gateway_error)?;
        mock_gateway_child = Some(child);
        gateway_bind
    } else if delay_gateway_start_ms > 0 {
        let head_bind = reserve_loopback_addr()?;
        let tail_bind = reserve_loopback_addr()?;
        let gateway_bind = reserve_loopback_addr()?;
        let launch_spec = ManagedGatewayLaunchSpec {
            head_bind: Some(head_bind),
            tail_bind: Some(tail_bind),
            gateway_bind: Some(gateway_bind.clone()),
            ..ManagedGatewayLaunchSpec::default()
        };
        let model_path_for_task = model_path.clone();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        delayed_shutdown = Some(shutdown_tx);
        delayed_stack_task = Some(tokio::spawn(async move {
            let mut shutdown_rx = shutdown_rx;
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(delay_gateway_start_ms)) => {}
                _ = &mut shutdown_rx => {
                    return Ok::<_, anyhow::Error>(());
                }
            }
            let stack = ManagedGatewayStack::spawn_local_with_spec(
                model_path_for_task,
                reconnect_after_prompt,
                &launch_spec,
            )
            .context("spawn delayed gateway stack")?;
            let _ = shutdown_rx.await;
            drop(stack);
            Ok::<_, anyhow::Error>(())
        }));
        gateway_bind
    } else {
        let stack = ManagedGatewayStack::spawn_local(model_path.clone(), reconnect_after_prompt)?;
        let addr = stack.gateway_addr().to_string();
        managed_gateway_stack = Some(stack);
        addr
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.context("bind mock ws addr")?;
    let ws_addr = listener.local_addr().context("mock ws local addr")?;
    let prompt_for_server = prompt.clone();
    let expect_node_error_for_server = expect_node_error.clone();

    let server_task = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.context("accept relay websocket")?;
        let mut ws = accept_async(socket).await.context("accept websocket handshake")?;

        let identify = loop {
            match timeout(Duration::from_secs(30), ws.next()).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    let value: serde_json::Value =
                        serde_json::from_str(&text).context("parse identify message")?;
                    if value.get("type").and_then(|v| v.as_str()) == Some("identify") {
                        break value;
                    }
                }
                Ok(Some(Ok(Message::Ping(data)))) => {
                    ws.send(Message::Pong(data)).await.context("pong identify")?;
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(err))) => {
                    return Err(anyhow::anyhow!("relay websocket read error: {err}"));
                }
                Ok(None) => bail!("relay disconnected before identify"),
                Err(_) => bail!("timed out waiting for identify"),
            }
        };

        ws.send(Message::Text(json!({"type":"identified"}).to_string()))
            .await
            .context("send identified")?;

        let assignment = json!({
            "type": "assignment",
            "pipeline_id": "gateway-pipeline",
            "model_name": "gemma-4-e4b-q4",
            "stage": 0,
            "total_stages": 1,
            "assignment_mode": "solo"
        });
        ws.send(Message::Text(assignment.to_string())).await.context("send assignment")?;

        let assignment_result = loop {
            match timeout(Duration::from_secs(45), ws.next()).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    let value: serde_json::Value =
                        serde_json::from_str(&text).context("parse post-assignment message")?;
                    match value.get("type").and_then(|v| v.as_str()) {
                        Some("node_ready") => {
                            if expect_node_error_for_server.is_some() {
                                bail!("received node_ready while expecting node_error: {value}");
                            }
                            break value;
                        }
                        Some("node_error") => {
                            if expect_node_error_for_server.is_some() {
                                let _ = ws.close(None).await;
                                return Ok::<_, anyhow::Error>(ServerOutcome::NodeError {
                                    identify,
                                    node_error: value,
                                });
                            }
                            bail!("received node_error: {value}");
                        }
                        _ => {}
                    }
                }
                Ok(Some(Ok(Message::Ping(data)))) => {
                    ws.send(Message::Pong(data)).await.context("pong node_ready")?;
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(err))) => {
                    return Err(anyhow::anyhow!("relay websocket read error: {err}"));
                }
                Ok(None) => bail!("relay disconnected before node_ready"),
                Err(_) => bail!("timed out waiting for node_ready"),
            }
        };

        let request = json!({
            "id": "assignment-roundtrip-0",
            "type": "inference_request",
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": "gemma-4-E4B-it-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": prompt_for_server}],
                "max_tokens": max_tokens,
                "stream": stream,
            }
        });
        ws.send(Message::Text(request.to_string())).await.context("send inference request")?;

        let mut streamed_text = String::new();
        let mut saw_done = false;
        let mut saw_finish_reason = false;
        let response = loop {
            match timeout(Duration::from_secs(30), ws.next()).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    let value: serde_json::Value =
                        serde_json::from_str(&text).context("parse inference response")?;
                    match value.get("type").and_then(|v| v.as_str()) {
                        Some("inference_stream_chunk") if stream => {
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
                                let chunk_json: serde_json::Value =
                                    serde_json::from_str(data).context("parse sse chunk json")?;
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
                Ok(Some(Ok(Message::Ping(data)))) => {
                    ws.send(Message::Pong(data)).await.context("pong inference response")?;
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(err))) => {
                    return Err(anyhow::anyhow!("relay websocket read error: {err}"));
                }
                Ok(None) => bail!("relay disconnected before inference response"),
                Err(_) => bail!("timed out waiting for inference response"),
            }
        };

        let _ = ws.close(None).await;
        Ok::<_, anyhow::Error>(ServerOutcome::Success {
            identify,
            node_ready: assignment_result,
            streamed_text,
            saw_done,
            saw_finish_reason,
            response,
        })
    });

    let mut config = Config::default();
    config.network.orchestrator_url = format!("http://{ws_addr}");
    config.wallet.public_address = "wallet-test".to_string();
    config.wallet.node_id = "node-test".to_string();
    config.wallet.node_token = "token-test".to_string();
    config.experimental.stage_mode_enabled = true;
    config.experimental.stage_backend = "llama-stage-gateway".to_string();
    config.experimental.stage_gateway_addr = gateway_addr;
    config.experimental.stage_gateway_connect_timeout_ms = gateway_connect_timeout_ms;
    config.experimental.stage_gateway_retry_window_ms = gateway_retry_window_ms;
    config.experimental.stage_gateway_retry_interval_ms = gateway_retry_interval_ms;
    config.experimental.stage_gateway_startup_grace_ms = gateway_startup_grace_ms;

    let runtime = Arc::new(DaemonRuntime::new(config));
    let runtime_task = {
        let runtime = runtime.clone();
        tokio::spawn(async move { runtime.run().await })
    };

    let outcome = server_task.await.context("mock orchestrator join")??;
    runtime.shutdown();
    let _ = timeout(Duration::from_secs(5), runtime_task)
        .await
        .context("timed out waiting for daemon shutdown")?
        .context("daemon runtime failed")?;
    drop(managed_gateway_stack);
    if let Some(shutdown_tx) = delayed_shutdown.take() {
        let _ = shutdown_tx.send(());
    }
    if let Some(task) = delayed_stack_task.take() {
        let _ = timeout(Duration::from_secs(5), task)
            .await
            .context("timed out waiting for delayed gateway stack shutdown")?
            .context("delayed gateway stack task failed")?;
    }
    if let Some(mut child) = mock_gateway_child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }

    if let Some(expected_error) = expect_node_error {
        let ServerOutcome::NodeError { identify, node_error } = outcome else {
            bail!("expected node_error outcome");
        };
        let identify_node = identify.get("node_id").and_then(|v| v.as_str()).unwrap_or_default();
        let node_error_model =
            node_error.get("model_name").and_then(|v| v.as_str()).unwrap_or_default();
        let error_text = node_error.get("error").and_then(|v| v.as_str()).unwrap_or_default();
        let ok = identify_node == "node-test"
            && node_error_model == "gemma-4-e4b-q4"
            && error_text.contains(&expected_error);

        println!("identify_node={identify_node}");
        println!("node_error_model={node_error_model}");
        println!("node_error={error_text:?}");
        println!("expected_node_error_substring={expected_error:?}");
        println!("gateway_connect_timeout_ms={gateway_connect_timeout_ms}");
        println!("gateway_retry_window_ms={gateway_retry_window_ms}");
        println!("gateway_retry_interval_ms={gateway_retry_interval_ms}");
        println!("gateway_startup_grace_ms={gateway_startup_grace_ms}");
        println!("delay_gateway_start_ms={delay_gateway_start_ms}");
        println!("match={ok}");

        if ok {
            println!("overall=PASS");
            return Ok(());
        }
        bail!("daemon assignment roundtrip node_error did not match expectation");
    }

    let ServerOutcome::Success {
        identify,
        node_ready,
        streamed_text,
        saw_done,
        saw_finish_reason,
        response,
    } = outcome
    else {
        bail!("expected successful assignment outcome");
    };

    let baseline = greedy_single_node_completion(&model_path, &prompt, max_tokens)?;
    let identify_node = identify.get("node_id").and_then(|v| v.as_str()).unwrap_or_default();
    let node_ready_model =
        node_ready.get("model_name").and_then(|v| v.as_str()).unwrap_or_default();
    let body = response.get("body").context("response missing body")?;
    let response_text = body["choices"][0]["message"]["content"].as_str().unwrap_or_default();
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or_default() as u32;
    let gateway_flag = body["llama_stage_gateway"].as_bool().unwrap_or(false);
    let stage_flag = body["prototype_stage_mode"].as_bool().unwrap_or(true);
    let has_timings = body["gateway_timings"].is_object();
    let status = response.get("status").and_then(|v| v.as_u64()).unwrap_or_default();
    let response_text_ok = if stream {
        response_text.is_empty() || response_text == baseline.text
    } else {
        response_text == baseline.text
    };
    let content_ok =
        if stream { streamed_text == baseline.text } else { response_text == baseline.text };
    let stream_ok = if stream { saw_done && saw_finish_reason } else { true };

    let ok = identify_node == "node-test"
        && node_ready_model == "gemma-4-e4b-q4"
        && status == 200
        && response_text_ok
        && content_ok
        && completion_tokens == baseline.completion_tokens
        && gateway_flag
        && !stage_flag
        && has_timings
        && stream_ok;

    println!("identify_node={identify_node}");
    println!("node_ready_model={node_ready_model}");
    println!("prompt={prompt:?}");
    println!("baseline_text={:?}", baseline.text);
    println!("response_text={response_text:?}");
    if stream {
        println!("streamed_text={streamed_text:?}");
        println!("saw_finish_reason={saw_finish_reason}");
        println!("saw_done={saw_done}");
    }
    println!("baseline_token_ids={:?}", baseline.token_ids);
    println!("response_completion_tokens={completion_tokens}");
    println!("head_prefill_ms={}", body["gateway_timings"]["head_prefill_ms"]);
    println!("head_decode_ms={}", body["gateway_timings"]["head_decode_ms"]);
    println!("tail_decode_ms={}", body["gateway_timings"]["tail_decode_ms"]);
    println!("ttft_ms={}", body["gateway_timings"]["ttft_ms"]);
    println!("total_ms={}", body["gateway_timings"]["total_ms"]);
    println!("gateway_connect_timeout_ms={gateway_connect_timeout_ms}");
    println!("gateway_retry_window_ms={gateway_retry_window_ms}");
    println!("gateway_retry_interval_ms={gateway_retry_interval_ms}");
    println!("gateway_startup_grace_ms={gateway_startup_grace_ms}");
    println!("delay_gateway_start_ms={delay_gateway_start_ms}");
    println!("stream={stream}");
    println!("match={ok}");

    if ok {
        println!("overall=PASS");
        Ok(())
    } else {
        bail!("daemon assignment roundtrip did not match expectation")
    }
}
