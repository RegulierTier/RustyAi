//! [`LlmBackend`]: minimal sync completion API (Phase 0).
//!
//! Implementations can wrap HTTP clients, local inference, or [`rusty_ai_llm`](https://docs.rs/rusty_ai_llm)
//! in a separate crate.

use serde::{Deserialize, Serialize};

use crate::error::LlmError;

/// Role in a chat transcript (OpenAI-style naming).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// One message in the conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Optional tool schema hint for the model (subset of JSON Schema; embed as JSON value).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the `arguments` object (e.g. `properties` for `read_file`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Request to a language model: messages + optional tools + sampling caps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Sequences that should truncate generation (OpenAI-style `stop`; multiple strings).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
}

/// One tool call emitted by the model (before orchestrator maps to [`crate::ToolInvocation`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Token-Nutzung wie von OpenAI-kompatiblen APIs geliefert (optional).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CompletionUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

/// Response from the model: assistant text and/or tool calls to execute.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ModelToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Gesetzt z. B. von [`crate::OpenAiCompatBackend`] (Feature `http`), wenn die API `usage` liefert.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

/// Pluggable LLM (sync). Async wrappers can be added in the application crate.
pub trait LlmBackend {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoBackend;

    impl LlmBackend for EchoBackend {
        fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            let last = request
                .messages
                .last()
                .cloned()
                .unwrap_or(ChatMessage {
                    role: ChatRole::User,
                    content: String::new(),
                });
            Ok(CompletionResponse {
                message: Some(ChatMessage {
                    role: ChatRole::Assistant,
                    content: last.content,
                }),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
                usage: None,
            })
        }
    }

    #[test]
    fn echo_backend_roundtrip() {
        let b = EchoBackend;
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hi".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
        };
        let res = b.complete(req).unwrap();
        assert_eq!(
            res.message.unwrap().content,
            "hi"
        );
    }
}
