//! [`FallbackBackend`]: primär scheitert, Fallback ist lokales [`MiniGptTinyBackend`] (ohne Netz).
//!
//! Lädt nach Möglichkeit das Repo-Checkpoint unter `rusty_ai/assets/mini_local/` (relativ zum
//! Workspace). Fehlt der Ordner, wird ein **Zufallsmodell** verwendet — die Ausgabe ist dann
//! erwartungsgemäß **kein** lesbarer Text (nur Demo-Fallback).
//!
//! `cargo run -p rusty_ai_agent --example tiny_llm_fallback_demo --features minigpt`

use std::cell::Cell;
use std::path::PathBuf;

use rusty_ai_agent::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, FallbackBackend, LlmBackend,
    LlmError, MiniGptTinyBackend,
};
use rusty_ai_llm::{MiniGpt, MiniGptConfig};

struct ScriptedBackend {
    out: Cell<Option<Result<CompletionResponse, LlmError>>>,
}

impl ScriptedBackend {
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
    let repo_mini = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rusty_ai/assets/mini_local");
    let tiny = if repo_mini.join("config.json").is_file() && repo_mini.join("model.safetensors").is_file() {
        eprintln!("Using checkpoint: {}", repo_mini.display());
        MiniGptTinyBackend::from_checkpoint_dir(&repo_mini)?
    } else {
        eprintln!(
            "No checkpoint at {} — using random weights (output will look like garbage).",
            repo_mini.display()
        );
        let mut seed = 42u32;
        let model = MiniGpt::random(MiniGptConfig::micro_local(), &mut seed)?;
        MiniGptTinyBackend::with_defaults(model, 99)
    };

    let fb = FallbackBackend::new(ScriptedBackend::err(), tiny);

    let req = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "ping ".into(),
        }],
        tools: vec![],
        max_tokens: Some(24),
        temperature: Some(0.85),
        top_p: None,
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
