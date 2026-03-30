//! Demo: [`complete_with_tool_parse_retries`] — Fake-LLM liefert zuerst ein **unbekanntes** Tool, dann `read_file`.
//!
//! `cargo run -p rusty_ai_agent --example agent_retry_demo`

use std::cell::Cell;

use rusty_ai_agent::{
    complete_with_tool_parse_retries, names, ChatMessage, ChatRole, CompletionRequest,
    CompletionResponse, LlmBackend, LlmError, ModelToolCall,
};
use serde_json::json;

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
        println!("--- [fake complete] round {} ---", i + 1);
        self.scripted
            .get(i)
            .cloned()
            .ok_or_else(|| LlmError::msg("FakeLlmBackend: no more scripted responses"))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = FakeLlmBackend::new(vec![
        CompletionResponse {
            message: None,
            tool_calls: vec![ModelToolCall {
                id: "1".into(),
                name: "typo_read_file".into(),
                arguments: json!({ "path": "x" }),
            }],
            finish_reason: Some("tool_calls".into()),
            usage: None,
        },
        CompletionResponse {
            message: None,
            tool_calls: vec![ModelToolCall {
                id: "2".into(),
                name: names::READ_FILE.into(),
                arguments: json!({ "path": "rusty_ai_agent/src/lib.rs" }),
            }],
            finish_reason: Some("tool_calls".into()),
            usage: None,
        },
    ]);

    let req = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "Show me lib.rs".into(),
        }],
        tools: vec![],
        max_tokens: None,
        temperature: None,
        stop_sequences: vec![],
    };

    let (last_resp, invs) = complete_with_tool_parse_retries(&backend, req, 2, None)?;
    println!("Parsed {} tool invocation(s)", invs.len());
    println!("finish_reason: {:?}", last_resp.finish_reason);
    println!("{invs:?}");
    Ok(())
}
