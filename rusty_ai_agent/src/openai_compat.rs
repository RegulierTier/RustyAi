//! OpenAI-compatible **Chat Completions** HTTP client implementing [`LlmBackend`].
//!
//! Works with OpenAI’s API and many proxies (Ollama `/v1`, LM Studio, Azure with adjusted `base_url`).
//! Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` (default `https://api.openai.com/v1`).
//!
//! **Streaming:** [`OpenAiCompatBackend::complete_stream`] reads an SSE (`text/event-stream`) body
//! incrementally, invokes a callback per text delta, and aggregates streamed `tool_calls` (OpenAI-style
//! deltas by `index`). [`complete_stream_text`](OpenAiCompatBackend::complete_stream_text) is an alias.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};

use reqwest::blocking::Client;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::llm_backend::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, LlmBackend, ModelToolCall,
};
use crate::tool_parse::parse_json_arguments_loose;
use crate::LlmError;

/// Endpoint and credentials for [`OpenAiCompatBackend`].
#[derive(Clone, Debug)]
pub struct OpenAiChatConfig {
    /// e.g. `https://api.openai.com/v1` or `http://localhost:11434/v1` (no trailing slash).
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAiChatConfig {
    /// Default OpenAI cloud URL and model `gpt-4o-mini` (empty API key — set `api_key` or use [`from_env`](Self::from_env)).
    pub fn openai(model: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: model.into(),
        }
    }

    /// Local [Ollama](https://ollama.com) OpenAI-compatible endpoint, no bearer token.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self {
            base_url: "http://127.0.0.1:11434/v1".into(),
            api_key: String::new(),
            model: model.into(),
        }
    }

    /// `OPENAI_API_KEY` required; `OPENAI_BASE_URL` overrides `base_url` when set.
    pub fn from_env(model: impl Into<String>) -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| LlmError::msg("environment variable OPENAI_API_KEY is not set"))?;
        let mut c = Self::openai(model);
        c.api_key = api_key;
        if let Ok(u) = std::env::var("OPENAI_BASE_URL") {
            c.base_url = u;
        }
        Ok(c)
    }
}

/// HTTP client for `POST …/chat/completions` (OpenAI-compatible JSON).
pub struct OpenAiCompatBackend {
    client: Client,
    config: OpenAiChatConfig,
}

impl OpenAiCompatBackend {
    pub fn new(config: OpenAiChatConfig) -> Result<Self, LlmError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| LlmError::msg(format!("http client: {e}")))?;
        Ok(Self { client, config })
    }

    pub fn from_env(model: impl Into<String>) -> Result<Self, LlmError> {
        Self::new(OpenAiChatConfig::from_env(model)?)
    }

    pub fn config(&self) -> &OpenAiChatConfig {
        &self.config
    }

    /// Chat Completions with `stream: true`: SSE lesen, Text-Deltas per Callback, optionale
    /// `tool_calls`-Fragmente pro Chunk zusammenführen (wie OpenAI `delta.tool_calls[index]`).
    pub fn complete_stream<F>(
        &self,
        request: CompletionRequest,
        mut on_delta: F,
    ) -> Result<CompletionResponse, LlmError>
    where
        F: FnMut(&str),
    {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let body = build_request_body(&self.config.model, &request, true)?;
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");
        if !self.config.api_key.is_empty() {
            req = req.bearer_auth(&self.config.api_key);
        }
        let resp = req
            .json(&body)
            .send()
            .map_err(|e| LlmError::msg(format!("HTTP request failed: {e}")))?;
        let status = resp.status();
        if status != StatusCode::OK {
            let text = resp
                .text()
                .map_err(|e| LlmError::msg(format!("reading error body: {e}")))?;
            return Err(LlmError::msg(format!(
                "HTTP {}: {}",
                status.as_u16(),
                truncate_body(&text)
            )));
        }
        let reader = BufReader::new(resp);
        let (content, finish_reason, tool_calls) = parse_sse_stream(reader, &mut on_delta)?;
        let has_text = !content.is_empty();
        let message = if has_text || !tool_calls.is_empty() {
            Some(ChatMessage {
                role: ChatRole::Assistant,
                content,
            })
        } else {
            None
        };
        Ok(CompletionResponse {
            message,
            tool_calls,
            finish_reason,
        })
    }

    /// Alias für [`complete_stream`] (älterer Name).
    pub fn complete_stream_text<F>(
        &self,
        request: CompletionRequest,
        on_delta: F,
    ) -> Result<CompletionResponse, LlmError>
    where
        F: FnMut(&str),
    {
        self.complete_stream(request, on_delta)
    }
}

impl LlmBackend for OpenAiCompatBackend {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let body = build_request_body(&self.config.model, &request, false)?;
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");
        if !self.config.api_key.is_empty() {
            req = req.bearer_auth(&self.config.api_key);
        }
        let resp = req
            .json(&body)
            .send()
            .map_err(|e| LlmError::msg(format!("HTTP request failed: {e}")))?;
        let status = resp.status();
        let text = resp
            .text()
            .map_err(|e| LlmError::msg(format!("reading response body: {e}")))?;
        if status != StatusCode::OK {
            return Err(LlmError::msg(format!(
                "HTTP {}: {}",
                status.as_u16(),
                truncate_body(&text)
            )));
        }
        let parsed: ChatCompletionApiResponse = serde_json::from_str(&text).map_err(|e| {
            LlmError::msg(format!(
                "invalid JSON response: {e}; body: {}",
                truncate_body(&text)
            ))
        })?;
        parse_choice(parsed)
    }
}

fn truncate_body(s: &str) -> String {
    const MAX: usize = 2000;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}… (truncated)", &s[..MAX])
    }
}

#[derive(Serialize)]
struct ChatCompletionRequestBody<'a> {
    model: &'a str,
    messages: Vec<ChatMessageApi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolApi>,
    #[serde(skip_serializing_if = "stream_is_false")]
    stream: bool,
}

fn stream_is_false(b: &bool) -> bool {
    !*b
}

#[derive(Serialize)]
struct ChatMessageApi {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ToolApi {
    #[serde(rename = "type")]
    ty: String,
    function: ToolFunctionApi,
}

#[derive(Serialize)]
struct ToolFunctionApi {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

fn build_request_body<'a>(
    model: &'a str,
    request: &'a CompletionRequest,
    stream: bool,
) -> Result<ChatCompletionRequestBody<'a>, LlmError> {
    let messages: Vec<ChatMessageApi> = request
        .messages
        .iter()
        .map(|m| ChatMessageApi {
            role: role_str(&m.role).to_string(),
            content: m.content.clone(),
        })
        .collect();
    let tools: Vec<ToolApi> = request
        .tools
        .iter()
        .map(|t| ToolApi {
            ty: "function".into(),
            function: ToolFunctionApi {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t
                    .parameters
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} })),
            },
        })
        .collect();
    Ok(ChatCompletionRequestBody {
        model,
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        stop: request.stop_sequences.clone(),
        tools,
        stream,
    })
}

fn role_str(r: &ChatRole) -> &'static str {
    match r {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
    }
}

#[derive(Deserialize)]
struct ChatCompletionApiResponse {
    choices: Vec<ApiChoice>,
}

#[derive(Deserialize)]
struct ApiChoice {
    message: ApiMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Deserialize)]
struct ApiToolCall {
    id: String,
    function: ApiFunctionCall,
}

#[derive(Deserialize)]
struct ApiFunctionCall {
    name: String,
    arguments: String,
}

fn parse_choice(resp: ChatCompletionApiResponse) -> Result<CompletionResponse, LlmError> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| LlmError::msg("empty choices[] in chat completion response"))?;
    let tool_calls: Vec<ModelToolCall> = choice
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| {
            let arguments: serde_json::Value =
                parse_json_arguments_loose(&tc.function.arguments).unwrap_or_else(|_| {
                    serde_json::Value::String(tc.function.arguments.clone())
                });
            ModelToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments,
            }
        })
        .collect();

    let has_text = choice
        .message
        .content
        .as_ref()
        .is_some_and(|s| !s.is_empty());
    let message = if has_text || !tool_calls.is_empty() {
        Some(ChatMessage {
            role: ChatRole::Assistant,
            content: choice.message.content.unwrap_or_default(),
        })
    } else {
        None
    };

    Ok(CompletionResponse {
        message,
        tool_calls,
        finish_reason: choice.finish_reason,
    })
}

#[derive(Deserialize)]
struct SseChunk {
    choices: Option<Vec<StreamChoice>>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Option<StreamDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ApiToolCallDelta>>,
}

#[derive(Deserialize)]
struct ApiToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ApiFunctionCallDeltaStream>,
}

#[derive(Deserialize)]
struct ApiFunctionCallDeltaStream {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn apply_tool_delta(partial: &mut BTreeMap<usize, PartialToolCall>, tc: &ApiToolCallDelta) {
    let entry = partial.entry(tc.index).or_default();
    if let Some(ref id) = tc.id {
        if !id.is_empty() {
            entry.id.clone_from(id);
        }
    }
    if let Some(ref f) = tc.function {
        if let Some(ref n) = f.name {
            if !n.is_empty() {
                entry.name.clone_from(n);
            }
        }
        if let Some(ref a) = f.arguments {
            entry.arguments.push_str(a);
        }
    }
}

fn finalize_partial_tool_calls(
    partial: BTreeMap<usize, PartialToolCall>,
) -> Result<Vec<ModelToolCall>, LlmError> {
    let mut out = Vec::new();
    for (idx, p) in partial {
        if p.name.is_empty() {
            return Err(LlmError::msg(
                "streamed tool_calls: missing function name after aggregation",
            ));
        }
        let id = if p.id.is_empty() {
            format!("stream_call_{idx}")
        } else {
            p.id
        };
        let arguments = parse_json_arguments_loose(&p.arguments).unwrap_or_else(|_| {
            serde_json::Value::String(p.arguments.clone())
        });
        out.push(ModelToolCall {
            id,
            name: p.name,
            arguments,
        });
    }
    Ok(out)
}

fn parse_sse_stream<R: BufRead>(
    reader: R,
    on_delta: &mut impl FnMut(&str),
) -> Result<(String, Option<String>, Vec<ModelToolCall>), LlmError> {
    let mut buf_reader = reader;
    let mut line = String::new();
    let mut accumulated = String::new();
    let mut last_finish: Option<String> = None;
    let mut partial_tools: BTreeMap<usize, PartialToolCall> = BTreeMap::new();

    loop {
        line.clear();
        let n = buf_reader
            .read_line(&mut line)
            .map_err(|e| LlmError::msg(format!("reading SSE: {e}")))?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let payload = if let Some(rest) = trimmed.strip_prefix("data:") {
            rest.trim_start()
        } else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        let chunk: serde_json::Result<SseChunk> = serde_json::from_str(payload);
        let chunk = match chunk {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Some(choices) = chunk.choices {
            for c in choices {
                if let Some(delta) = c.delta {
                    if let Some(ref text) = delta.content {
                        if !text.is_empty() {
                            accumulated.push_str(text);
                            on_delta(text);
                        }
                    }
                    if let Some(ref tcs) = delta.tool_calls {
                        for tc in tcs {
                            apply_tool_delta(&mut partial_tools, tc);
                        }
                    }
                }
                if let Some(fr) = c.finish_reason.filter(|s| !s.is_empty()) {
                    last_finish = Some(fr);
                }
            }
        }
    }

    let tool_calls = finalize_partial_tool_calls(partial_tools)?;
    Ok((accumulated, last_finish, tool_calls))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_response_with_tool() {
        let json = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"src/lib.rs\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let api: ChatCompletionApiResponse = serde_json::from_str(json).unwrap();
        let out = parse_choice(api).unwrap();
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "read_file");
        assert!(out.message.is_some());
    }

    #[test]
    fn parse_loose_fenced_tool_arguments() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "```json\n{\"path\":\"x.rs\"}\n```"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let api: super::ChatCompletionApiResponse = serde_json::from_value(body).unwrap();
        let out = parse_choice(api).unwrap();
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "read_file");
        assert_eq!(out.tool_calls[0].arguments["path"], "x.rs");
    }

    #[test]
    fn parse_sse_stream_accumulates() {
        use std::io::Cursor;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n\
                   data: [DONE]\n";
        let mut buf = Cursor::new(sse);
        let mut deltas = Vec::new();
        let (full, fr, tools) =
            super::parse_sse_stream(&mut buf, &mut |d| deltas.push(d.to_string())).unwrap();
        assert_eq!(full, "Hi!");
        assert_eq!(fr.as_deref(), Some("stop"));
        assert!(tools.is_empty());
        assert_eq!(deltas, vec!["Hi".to_string(), "!".to_string()]);
    }

    #[test]
    fn parse_sse_stream_tool_deltas() {
        use std::io::Cursor;
        let sse = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"read_file"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"x.rs\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
"#;
        let mut buf = Cursor::new(sse);
        let mut deltas = Vec::new();
        let (full, fr, tools) =
            super::parse_sse_stream(&mut buf, &mut |d| deltas.push(d.to_string())).unwrap();
        assert_eq!(full, "");
        assert_eq!(fr.as_deref(), Some("tool_calls"));
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].arguments["path"], "x.rs");
        assert!(deltas.is_empty());
    }
}
