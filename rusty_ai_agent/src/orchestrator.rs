//! Mehrturnige Orchestrierung: bei fehlerhaften Tool-JSON erneut [`LlmBackend::complete`] aufrufen,
//! nachdem eine User-Nachricht mit [`tool_parse_retry_instruction`](crate::tool_parse_retry_instruction) angehängt wurde.

use crate::llm_backend::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, LlmBackend,
};
use crate::telemetry::LocalTelemetry;
use crate::tool_parse::{
    tool_invocations_try_each, tool_parse_retry_instruction, ToolInvocationParseItem,
};
use crate::{LlmError, ToolInvocation};

/// Ruft [`LlmBackend::complete`] wiederholt auf, bis alle `tool_calls` zu [`ToolInvocation`]s
/// werden oder `max_complete_calls` erreicht ist.
///
/// - Keine `tool_calls` in der Antwort: `(resp, vec![])`.
/// - Alle parsebar: `(resp, invocations)`.
/// - Teilweise oder vollständig fehlgeschlagen: Assistenten- und User-Nachricht anhängen, erneut aufrufen.
///
/// **Hinweis:** Der Verlauf ist bewusst einfach gehalten (Text + serialisierte `tool_calls` im
/// Assistenten-String). Produktive Clients sollten das Nachrichtenformat eures Backends nutzen
/// (z. B. OpenAI-Tool-Nachrichten).
///
/// `telemetry`: optional [`LocalTelemetry`] — bei jedem Parse-Retry wird ein interner Zähler erhöht (siehe [`LocalTelemetry::snapshot`](crate::LocalTelemetry::snapshot)).
pub fn complete_with_tool_parse_retries(
    backend: &impl LlmBackend,
    mut request: CompletionRequest,
    max_complete_calls: usize,
    telemetry: Option<&LocalTelemetry>,
) -> Result<(CompletionResponse, Vec<ToolInvocation>), LlmError> {
    if max_complete_calls == 0 {
        return Err(LlmError::msg("max_complete_calls must be >= 1"));
    }
    let mut completed = 0usize;
    loop {
        completed += 1;
        let resp = backend.complete(request.clone())?;

        if resp.tool_calls.is_empty() {
            return Ok((resp, vec![]));
        }

        let (ok, failed) = tool_invocations_try_each(&resp.tool_calls);
        if failed.is_empty() {
            return Ok((resp, ok));
        }

        if completed >= max_complete_calls {
            return Err(LlmError::msg(format!(
                "tool parse retries exhausted after {} complete() call(s): {}",
                max_complete_calls,
                tool_parse_retry_instruction(&failed)
            )));
        }

        append_retry_messages(&mut request, &resp, &failed);
        if let Some(t) = telemetry {
            t.record_tool_parse_retry_turn();
        }
    }
}

fn append_retry_messages(
    request: &mut CompletionRequest,
    resp: &CompletionResponse,
    failed: &[ToolInvocationParseItem],
) {
    let mut assistant = String::new();
    if let Some(m) = &resp.message {
        if !m.content.is_empty() {
            assistant.push_str(&m.content);
        }
    }
    if !resp.tool_calls.is_empty() {
        if !assistant.is_empty() {
            assistant.push('\n');
        }
        assistant.push_str("[model tool_calls] ");
        assistant.push_str(
            &serde_json::to_string(&resp.tool_calls).unwrap_or_else(|_| "{}".to_string()),
        );
    }
    request.messages.push(ChatMessage {
        role: ChatRole::Assistant,
        content: assistant,
    });
    request.messages.push(ChatMessage {
        role: ChatRole::User,
        content: tool_parse_retry_instruction(failed),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_backend::{CompletionResponse, LlmBackend, ModelToolCall};
    use crate::names;
    use serde_json::json;
    use std::cell::Cell;

    struct FakeLlmBackend {
        scripted: Vec<CompletionResponse>,
        next: Cell<usize>,
    }

    impl FakeLlmBackend {
        fn new(scripted: Vec<CompletionResponse>) -> Self {
            Self {
                scripted,
                next: Cell::new(0),
            }
        }
    }

    impl LlmBackend for FakeLlmBackend {
        fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            let i = self.next.get();
            self.next.set(i + 1);
            self.scripted
                .get(i)
                .cloned()
                .ok_or_else(|| LlmError::msg("FakeLlmBackend: no more scripted responses"))
        }
    }

    #[test]
    fn second_round_succeeds() {
        let backend = FakeLlmBackend::new(vec![
            CompletionResponse {
                message: None,
                tool_calls: vec![ModelToolCall {
                    id: "1".into(),
                    name: "unknown_x".into(),
                    arguments: json!({}),
                }],
                finish_reason: Some("tool_calls".into()),
            },
            CompletionResponse {
                message: None,
                tool_calls: vec![ModelToolCall {
                    id: "2".into(),
                    name: names::READ_FILE.into(),
                    arguments: json!({ "path": "rusty_ai_agent/src/lib.rs" }),
                }],
                finish_reason: Some("tool_calls".into()),
            },
        ]);
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "read".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
        };
        let out = complete_with_tool_parse_retries(&backend, req, 2, None).unwrap();
        assert_eq!(out.1.len(), 1);
    }

    #[test]
    fn records_retry_turns_in_telemetry() {
        use crate::LocalTelemetry;
        use std::sync::Arc;

        let tel = Arc::new(LocalTelemetry::new());
        let backend = FakeLlmBackend::new(vec![
            CompletionResponse {
                message: None,
                tool_calls: vec![ModelToolCall {
                    id: "1".into(),
                    name: "unknown_x".into(),
                    arguments: json!({}),
                }],
                finish_reason: Some("tool_calls".into()),
            },
            CompletionResponse {
                message: None,
                tool_calls: vec![ModelToolCall {
                    id: "2".into(),
                    name: names::READ_FILE.into(),
                    arguments: json!({ "path": "rusty_ai_agent/src/lib.rs" }),
                }],
                finish_reason: Some("tool_calls".into()),
            },
        ]);
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "read".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
        };
        let _ = complete_with_tool_parse_retries(&backend, req, 2, Some(tel.as_ref())).unwrap();
        assert_eq!(tel.snapshot().tool_parse_retry_turns, 1);
    }

    #[test]
    fn exhausts_max_rounds() {
        let backend = FakeLlmBackend::new(vec![CompletionResponse {
            message: None,
            tool_calls: vec![ModelToolCall {
                id: "1".into(),
                name: "bad".into(),
                arguments: json!({}),
            }],
            finish_reason: None,
        }]);
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "x".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
        };
        assert!(complete_with_tool_parse_retries(&backend, req, 1, None).is_err());
    }
}
