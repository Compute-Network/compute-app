use anyhow::{Context, Result};
use llama_stage_backend::{GatewayServiceClient, GatewayStep, RemoteStageCompletion};
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct GatewayCompletionResult {
    pub model_name: String,
    pub content: String,
    pub finish_reason: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub completion: RemoteStageCompletion,
}

#[derive(Debug, Clone)]
pub struct GatewayChatCompletion {
    pub body: serde_json::Value,
    pub result: GatewayCompletionResult,
}

#[derive(Debug, Clone)]
pub struct GatewayStreamingSession {
    pub request_id: String,
    model_name: String,
    prompt_tokens: u32,
    max_tokens: u32,
    stop_sequences: Vec<String>,
    seen_text: String,
}

#[derive(Debug, Clone)]
pub struct GatewayStreamingProgress {
    pub chunks: Vec<String>,
    pub completion: Option<GatewayChatCompletion>,
}

struct GatewayClientState {
    addr: String,
    connect_timeout: Option<Duration>,
    client: Option<GatewayServiceClient>,
}

#[derive(Clone)]
pub struct LlamaStageGatewayRelayClient {
    inner: Arc<Mutex<GatewayClientState>>,
}

impl LlamaStageGatewayRelayClient {
    pub fn connect(addr: impl Into<String>) -> Result<Self> {
        Self::connect_with_timeout(addr, None)
    }

    pub fn connect_with_timeout(
        addr: impl Into<String>,
        connect_timeout: Option<Duration>,
    ) -> Result<Self> {
        let addr = addr.into();
        let client = connect_gateway_service_client(&addr, connect_timeout)
            .with_context(|| format!("connecting to stage gateway at {addr}"))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(GatewayClientState {
                addr,
                connect_timeout,
                client: Some(client),
            })),
        })
    }

    pub fn addr(&self) -> String {
        self.inner.lock().expect("gateway client mutex poisoned").addr.clone()
    }

    pub fn model_name(&self) -> Result<String> {
        let mut state = self.inner.lock().expect("gateway client mutex poisoned");
        Self::with_client(&mut state, |client| Ok(client.info().head_info.model_id.clone()))
    }

    fn with_client<T>(
        state: &mut GatewayClientState,
        mut op: impl FnMut(&mut GatewayServiceClient) -> Result<T>,
    ) -> Result<T> {
        if state.client.is_none() {
            state.client = Some(
                connect_gateway_service_client(&state.addr, state.connect_timeout)
                    .with_context(|| format!("connecting to stage gateway at {}", state.addr))?,
            );
        }

        match op(state.client.as_mut().expect("gateway client initialized")) {
            Ok(value) => Ok(value),
            Err(first_err) => {
                state.client = Some(
                    connect_gateway_service_client(&state.addr, state.connect_timeout)
                        .with_context(|| {
                            format!("reconnecting to stage gateway at {}", state.addr)
                        })?,
                );
                op(state.client.as_mut().expect("gateway client reinitialized"))
                    .with_context(|| format!("gateway operation retry failed after: {first_err}"))
            }
        }
    }

    pub async fn complete_prompt(
        &self,
        request_id: String,
        prompt: String,
        max_tokens: Option<u32>,
        stop_sequences: Vec<String>,
    ) -> Result<GatewayCompletionResult> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut state = inner.lock().expect("gateway client mutex poisoned");
            let prompt_tokens = Self::with_client(&mut state, |client| client.tokenize(&prompt))?;
            let completion = Self::with_client(&mut state, |client| {
                client.complete(request_id.clone(), prompt.clone(), max_tokens.unwrap_or(96))
            })?;

            let mut content = completion.text.clone();
            let mut finish_reason = if completion.completion_tokens >= max_tokens.unwrap_or(96) {
                "length".to_string()
            } else {
                "stop".to_string()
            };

            if let Some(stop) = stop_sequences
                .iter()
                .filter(|value| !value.is_empty())
                .find_map(|stop| content.find(stop).map(|idx| (stop, idx)))
            {
                content.truncate(stop.1);
                finish_reason = "stop".to_string();
            }

            Ok(GatewayCompletionResult {
                model_name: state
                    .client
                    .as_ref()
                    .expect("gateway client available")
                    .info()
                    .head_info
                    .model_id
                    .clone(),
                content,
                finish_reason,
                prompt_tokens: prompt_tokens.len() as u32,
                completion_tokens: completion.completion_tokens,
                total_tokens: prompt_tokens.len() as u32 + completion.completion_tokens,
                completion,
            })
        })
        .await
        .context("stage gateway completion task failed")?
    }

    pub async fn complete_chat_request(
        &self,
        request_id: String,
        body: &serde_json::Value,
    ) -> Result<GatewayChatCompletion> {
        let prompt = extract_request_prompt(body);
        let stop_sequences = extract_stop_sequences(body);
        let max_tokens = extract_max_tokens(body);
        let result =
            self.complete_prompt(request_id.clone(), prompt, max_tokens, stop_sequences).await?;
        let body = build_chat_completion_body(&request_id, &result);
        Ok(GatewayChatCompletion { body, result })
    }

    pub async fn begin_chat_stream(
        &self,
        request_id: String,
        body: &serde_json::Value,
    ) -> Result<GatewayStreamingSession> {
        let prompt = extract_request_prompt(body);
        let stop_sequences = extract_stop_sequences(body);
        let max_tokens = extract_max_tokens(body).unwrap_or(96);
        let inner = self.inner.clone();
        let request_id_clone = request_id.clone();

        tokio::task::spawn_blocking(move || {
            let mut state = inner.lock().expect("gateway client mutex poisoned");
            let prompt_tokens = Self::with_client(&mut state, |client| client.tokenize(&prompt))?;
            let model_name = state
                .client
                .as_ref()
                .expect("gateway client available")
                .info()
                .head_info
                .model_id
                .clone();
            Self::with_client(&mut state, |client| {
                client.begin_completion(request_id_clone.clone(), prompt.clone(), max_tokens)
            })?;

            Ok(GatewayStreamingSession {
                request_id,
                model_name,
                prompt_tokens: prompt_tokens.len() as u32,
                max_tokens,
                stop_sequences,
                seen_text: String::new(),
            })
        })
        .await
        .context("stage gateway begin stream task failed")?
    }

    pub async fn step_chat_stream(
        &self,
        session: &mut GatewayStreamingSession,
    ) -> Result<GatewayStreamingProgress> {
        let inner = self.inner.clone();
        let request_id = session.request_id.clone();
        let step = tokio::task::spawn_blocking(move || {
            let mut state = inner.lock().expect("gateway client mutex poisoned");
            Self::with_client(&mut state, |client| client.step_completion(request_id.clone()))
        })
        .await
        .context("stage gateway stream step task failed")??;

        match step {
            GatewayStep::Token { text, .. } => {
                let delta = text
                    .strip_prefix(&session.seen_text)
                    .map(str::to_string)
                    .unwrap_or_else(|| text.clone());
                session.seen_text = text;
                Ok(GatewayStreamingProgress {
                    chunks: if delta.is_empty() {
                        Vec::new()
                    } else {
                        vec![build_chat_completion_chunk(
                            &session.request_id,
                            &session.model_name,
                            Some(&delta),
                            None,
                        )]
                    },
                    completion: None,
                })
            }
            GatewayStep::Complete { completion, .. } => {
                let delta = completion
                    .text
                    .strip_prefix(&session.seen_text)
                    .map(str::to_string)
                    .unwrap_or_else(|| completion.text.clone());
                session.seen_text = completion.text.clone();
                let result = finalize_gateway_completion(
                    &session.model_name,
                    session.prompt_tokens,
                    session.max_tokens,
                    &session.stop_sequences,
                    completion,
                );
                let body = build_chat_completion_body(&session.request_id, &result);

                let mut chunks = Vec::new();
                if !delta.is_empty() {
                    chunks.push(build_chat_completion_chunk(
                        &session.request_id,
                        &session.model_name,
                        Some(&delta),
                        None,
                    ));
                }
                chunks.push(build_chat_completion_chunk(
                    &session.request_id,
                    &session.model_name,
                    None,
                    Some(&result.finish_reason),
                ));
                chunks.push("data: [DONE]\n\n".to_string());

                Ok(GatewayStreamingProgress {
                    chunks,
                    completion: Some(GatewayChatCompletion { body, result }),
                })
            }
        }
    }

    pub async fn clear_stream(&self, request_id: impl Into<String>) -> Result<()> {
        let inner = self.inner.clone();
        let request_id = request_id.into();
        tokio::task::spawn_blocking(move || {
            let mut state = inner.lock().expect("gateway client mutex poisoned");
            Self::with_client(&mut state, |client| client.clear_completion(request_id.clone()))
        })
        .await
        .context("stage gateway clear stream task failed")?
    }
}

fn connect_gateway_service_client(
    addr: &str,
    connect_timeout: Option<Duration>,
) -> Result<GatewayServiceClient> {
    match connect_timeout {
        Some(connect_timeout) => {
            GatewayServiceClient::connect_with_timeout(addr, Some(connect_timeout))
        }
        None => GatewayServiceClient::connect(addr),
    }
}

pub fn extract_request_prompt(body: &serde_json::Value) -> String {
    // Legacy completion-style API: pass a pre-formed prompt through untouched
    // so callers can opt out of the chat template if they want to drive the
    // tokenizer directly.
    if let Some(prompt) = body.get("prompt").and_then(|value| value.as_str()) {
        return prompt.to_string();
    }

    if let Some(messages) = body.get("messages").and_then(|value| value.as_array()) {
        // Chat-style API: format with the model's chat template before
        // tokenization. Without this, the head tokenizes raw user text and
        // the model treats it as a continuation — which is why short prompts
        // like `say "hi"` loop ("hi" and then "hi" again, and then...) while
        // long prompts coincidentally work because the instruction itself
        // disambiguates. Currently only Gemma runs through the split path;
        // other families fall back to the legacy last-user behaviour.
        let model = body.get("model").and_then(|value| value.as_str()).unwrap_or("");
        if is_gemma_model(model) {
            return format_gemma_chat(messages);
        }

        for message in messages.iter().rev() {
            let role = message.get("role").and_then(|value| value.as_str()).unwrap_or("user");
            if role != "user" {
                continue;
            }

            let text = message_content_text(message);
            if !text.is_empty() {
                return text;
            }
        }

        let combined = messages
            .iter()
            .filter_map(|message| {
                let role = message.get("role").and_then(|value| value.as_str()).unwrap_or("user");
                let content = message.get("content")?;
                if let Some(text) = content.as_str() {
                    Some(format!("{role}: {text}"))
                } else if let Some(parts) = content.as_array() {
                    let text = parts
                        .iter()
                        .filter_map(|part| {
                            if part.get("type").and_then(|value| value.as_str()) == Some("text") {
                                part.get("text")
                                    .and_then(|value| value.as_str())
                                    .map(str::to_string)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    if text.is_empty() { None } else { Some(format!("{role}: {text}")) }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !combined.is_empty() {
            return combined;
        }
    }

    String::new()
}

fn is_gemma_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("gemma")
}

fn message_content_text(message: &serde_json::Value) -> String {
    let Some(content) = message.get("content") else {
        return String::new();
    };
    if let Some(text) = content.as_str() {
        return text.trim().to_string();
    }
    if let Some(parts) = content.as_array() {
        return parts
            .iter()
            .filter_map(|part| {
                if part.get("type").and_then(|value| value.as_str()) == Some("text") {
                    part.get("text").and_then(|value| value.as_str()).map(str::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
    }
    String::new()
}

/// Format chat messages with Gemma's chat template so the model sees proper
/// turn markers. Gemma does not have a native `system` role, so system
/// messages are prepended to the first user turn (matching the HF tokenizer
/// config's fallback behaviour). BOS is added automatically by llama.cpp's
/// tokenize() when `add_special=true`, so we start directly with
/// `<start_of_turn>user`.
fn format_gemma_chat(messages: &[serde_json::Value]) -> String {
    let mut out = String::new();
    let mut system_prefix = String::new();
    let mut first_user_emitted = false;

    for message in messages {
        let role = message.get("role").and_then(|value| value.as_str()).unwrap_or("user");
        let content = message_content_text(message);
        if content.is_empty() {
            continue;
        }

        if role == "system" {
            if !system_prefix.is_empty() {
                system_prefix.push_str("\n\n");
            }
            system_prefix.push_str(&content);
            continue;
        }

        let gemma_role = if role == "assistant" { "model" } else { "user" };

        let body = if gemma_role == "user" && !first_user_emitted && !system_prefix.is_empty() {
            let merged = format!("{system_prefix}\n\n{content}");
            system_prefix.clear();
            merged
        } else {
            content
        };

        if gemma_role == "user" {
            first_user_emitted = true;
        }

        out.push_str("<start_of_turn>");
        out.push_str(gemma_role);
        out.push('\n');
        out.push_str(&body);
        out.push_str("<end_of_turn>\n");
    }

    // Bare system-only input: emit it as a user turn so the model has
    // something to respond to.
    if out.is_empty() && !system_prefix.is_empty() {
        out.push_str("<start_of_turn>user\n");
        out.push_str(&system_prefix);
        out.push_str("<end_of_turn>\n");
    }

    // Generation prompt tells the model it's its turn to speak.
    out.push_str("<start_of_turn>model\n");
    out
}

pub fn extract_stop_sequences(body: &serde_json::Value) -> Vec<String> {
    match body.get("stop") {
        Some(serde_json::Value::String(stop)) if !stop.is_empty() => vec![stop.clone()],
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_str())
            .filter(|stop| !stop.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

pub fn extract_max_tokens(body: &serde_json::Value) -> Option<u32> {
    body.get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(|value| value.as_u64())
        .map(|value| value as u32)
}

fn build_chat_completion_body(
    request_id: &str,
    result: &GatewayCompletionResult,
) -> serde_json::Value {
    json!({
        "id": request_id,
        "object": "chat.completion",
        "model": result.model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result.content,
            },
            "finish_reason": result.finish_reason
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
        "prototype_stage_mode": false,
        "llama_stage_gateway": true,
        "gateway_timings": {
            "head_prefill_ms": result.completion.timings.head_prefill_ms,
            "head_decode_ms": result.completion.timings.head_decode_ms,
            "tail_decode_ms": result.completion.timings.tail_decode_ms,
            "sample_ms": result.completion.timings.sample_ms,
            "transfer_bytes": result.completion.timings.transfer_bytes,
            "ttft_ms": result.completion.timings.ttft_ms,
            "total_ms": result.completion.timings.total_ms,
            "spec_active": result.completion.timings.spec_active,
        }
    })
}

fn finalize_gateway_completion(
    model_name: &str,
    prompt_tokens: u32,
    max_tokens: u32,
    stop_sequences: &[String],
    completion: RemoteStageCompletion,
) -> GatewayCompletionResult {
    let mut content = completion.text.clone();
    let mut finish_reason = if completion.completion_tokens >= max_tokens {
        "length".to_string()
    } else {
        "stop".to_string()
    };

    if let Some((_, idx)) = stop_sequences
        .iter()
        .filter(|value| !value.is_empty())
        .find_map(|stop| content.find(stop).map(|idx| (stop, idx)))
    {
        content.truncate(idx);
        finish_reason = "stop".to_string();
    }

    GatewayCompletionResult {
        model_name: model_name.to_string(),
        content,
        finish_reason,
        prompt_tokens,
        completion_tokens: completion.completion_tokens,
        total_tokens: prompt_tokens + completion.completion_tokens,
        completion,
    }
}

fn build_chat_completion_chunk(
    request_id: &str,
    model_name: &str,
    delta: Option<&str>,
    finish_reason: Option<&str>,
) -> String {
    let chunk = json!({
        "id": request_id,
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": match delta {
                Some(content) => json!({ "content": content }),
                None => json!({}),
            },
            "finish_reason": finish_reason
        }]
    });
    format!("data: {}\n\n", chunk)
}

#[cfg(test)]
mod tests {
    use super::{extract_max_tokens, extract_request_prompt, extract_stop_sequences};

    #[test]
    fn extract_request_prompt_prefers_prompt_field() {
        let body = serde_json::json!({
            "prompt": "hello from prompt",
            "messages": [
                {"role": "user", "content": "ignored"}
            ]
        });
        assert_eq!(extract_request_prompt(&body), "hello from prompt");
    }

    #[test]
    fn extract_request_prompt_flattens_chat_messages() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Say hi"}
            ]
        });
        assert_eq!(extract_request_prompt(&body), "Say hi");
    }

    #[test]
    fn extract_request_prompt_flattens_text_parts() {
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part one"},
                        {"type": "image_url", "image_url": {"url": "ignored"}},
                        {"type": "text", "text": "Part two"}
                    ]
                }
            ]
        });
        assert_eq!(extract_request_prompt(&body), "Part one Part two");
    }

    #[test]
    fn extract_request_prompt_prefers_last_user_message() {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "First request"},
                {"role": "assistant", "content": "Interim answer"},
                {"role": "user", "content": "Second request"}
            ]
        });
        assert_eq!(extract_request_prompt(&body), "Second request");
    }

    #[test]
    fn extract_request_prompt_applies_gemma_chat_template_for_short_prompt() {
        // Without the template, short prompts like `Say "hi"` get tokenized as
        // raw text and Gemma continues them in a loop instead of answering.
        // The template wraps the turn in `<start_of_turn>user` markers and
        // closes with `<start_of_turn>model` so the model knows it's replying.
        let body = serde_json::json!({
            "model": "gemma-4-e4b-q4",
            "messages": [
                {"role": "user", "content": "Say \"hi\""}
            ]
        });
        assert_eq!(
            extract_request_prompt(&body),
            "<start_of_turn>user\nSay \"hi\"<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn extract_request_prompt_merges_system_into_first_user_for_gemma() {
        // Gemma has no native system role; HF fallback folds it into the first
        // user turn separated by a blank line.
        let body = serde_json::json!({
            "model": "gemma-4-e4b-q4",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Say hi"}
            ]
        });
        assert_eq!(
            extract_request_prompt(&body),
            "<start_of_turn>user\nYou are terse.\n\nSay hi<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn extract_request_prompt_renders_gemma_multi_turn_dialogue() {
        let body = serde_json::json!({
            "model": "gemma-4-e4b-q4",
            "messages": [
                {"role": "user", "content": "Who won?"},
                {"role": "assistant", "content": "Nobody."},
                {"role": "user", "content": "Why?"}
            ]
        });
        assert_eq!(
            extract_request_prompt(&body),
            "<start_of_turn>user\nWho won?<end_of_turn>\n\
             <start_of_turn>model\nNobody.<end_of_turn>\n\
             <start_of_turn>user\nWhy?<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn extract_request_prompt_gemma_preserves_raw_prompt_field() {
        // When the caller passes `prompt`, they've already formatted it —
        // don't double-wrap.
        let body = serde_json::json!({
            "model": "gemma-4-e4b-q4",
            "prompt": "<start_of_turn>user\npre-baked<end_of_turn>\n<start_of_turn>model\n",
        });
        assert_eq!(
            extract_request_prompt(&body),
            "<start_of_turn>user\npre-baked<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn extract_stop_sequences_supports_string_and_array() {
        let single = serde_json::json!({ "stop": "END" });
        assert_eq!(extract_stop_sequences(&single), vec!["END".to_string()]);

        let many = serde_json::json!({ "stop": ["A", "", "B", 3] });
        assert_eq!(extract_stop_sequences(&many), vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn extract_max_tokens_supports_legacy_and_openai_names() {
        let legacy = serde_json::json!({ "max_tokens": 12 });
        assert_eq!(extract_max_tokens(&legacy), Some(12));

        let openai = serde_json::json!({ "max_completion_tokens": 34 });
        assert_eq!(extract_max_tokens(&openai), Some(34));
    }
}
