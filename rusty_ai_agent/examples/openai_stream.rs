//! Streaming chat completion (`stream: true`) via [`OpenAiCompatBackend::complete_stream`].
//!
//! OpenAI cloud:
//! `set OPENAI_API_KEY=...` then  
//! `cargo run -p rusty_ai_agent --example openai_stream --features http`
//!
//! Ollama (no key):
//! `cargo run -p rusty_ai_agent --example openai_stream --features http -- --ollama`

use std::io::Write;

use rusty_ai_agent::{
    ChatMessage, ChatRole, CompletionRequest, OpenAiChatConfig, OpenAiCompatBackend,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = std::env::args().any(|a| a == "--ollama");
    let backend = if ollama {
        let mut c = OpenAiChatConfig::ollama(
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into()),
        );
        if let Ok(u) = std::env::var("OLLAMA_BASE_URL") {
            c.base_url = u;
        }
        OpenAiCompatBackend::new(c)?
    } else {
        OpenAiCompatBackend::from_env(
            std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
        )?
    };

    let req = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "Reply with exactly: OK".into(),
        }],
        tools: vec![],
        max_tokens: Some(16),
        temperature: Some(0.0),
        top_p: None,
        stop_sequences: vec![],
    };

    print!("assistant: ");
    std::io::stdout().flush()?;

    let resp = backend.complete_stream(req, |delta| {
        print!("{delta}");
        let _ = std::io::stdout().flush();
    })?;

    println!();
    println!("finish_reason: {:?}", resp.finish_reason);
    if !resp.tool_calls.is_empty() {
        println!("tool_calls: {}", resp.tool_calls.len());
    }
    Ok(())
}
