//! [`TimedBackend`] + [`LocalTelemetry`] (Latenz, Aufrufe, `cargo check`-Zähler).
//!
//! `cargo run -p rusty_ai_agent --example telemetry_demo`

use std::sync::Arc;

use rusty_ai_agent::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, LlmBackend, LlmError,
    LocalTelemetry, TimedBackend,
};

struct EchoBackend;

impl LlmBackend for EchoBackend {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let last = request.messages.last().cloned().unwrap_or(ChatMessage {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tel = Arc::new(LocalTelemetry::new());
    let backend = TimedBackend::new(EchoBackend, Arc::clone(&tel));

    let req = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
        }],
        tools: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop_sequences: vec![],
    };

    backend.complete(req)?;

    tel.record_cargo_check(true);
    tel.record_cargo_check(false);

    println!("{}", serde_json::to_string_pretty(&tel.snapshot())?);
    Ok(())
}
