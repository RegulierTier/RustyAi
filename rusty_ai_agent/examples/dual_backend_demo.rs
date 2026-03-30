//! [`FallbackBackend`]: primär scheitert, sekundär antwortet (z. B. Cloud → lokal).
//!
//! `cargo run -p rusty_ai_agent --example dual_backend_demo`

use std::cell::Cell;

use rusty_ai_agent::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, FallbackBackend, LlmBackend,
    LlmError,
};

struct ScriptedBackend {
    out: Cell<Option<Result<CompletionResponse, LlmError>>>,
}

impl ScriptedBackend {
    fn ok(content: &str) -> Self {
        Self {
            out: Cell::new(Some(Ok(CompletionResponse {
                message: Some(ChatMessage {
                    role: ChatRole::Assistant,
                    content: content.into(),
                }),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
                usage: None,
            }))),
        }
    }

    fn err() -> Self {
        Self {
            out: Cell::new(Some(Err(LlmError::msg("simulated primary failure")))),
        }
    }
}

impl LlmBackend for ScriptedBackend {
    fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.out.take().unwrap()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fb = FallbackBackend::new(ScriptedBackend::err(), ScriptedBackend::ok("from fallback"));

    let req = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "ping".into(),
        }],
        tools: vec![],
        max_tokens: None,
        temperature: None,
        stop_sequences: vec![],
    };

    let r = fb.complete(req)?;
    println!(
        "{}",
        r.message
            .as_ref()
            .map(|m| m.content.as_str())
            .unwrap_or("(no message)")
    );
    Ok(())
}
