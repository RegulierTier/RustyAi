//! Phase-0 demo: fake LLM backend → tool calls → allowlist → dry-run **or** real FS / subprocess (`--features real-exec -- --real`).
//!
//! Dry-run (default):
//! `cargo run -p rusty_ai_agent --example agent_demo`
//!
//! Real `read_file` / `cargo` (only with feature):
//! `cargo run -p rusty_ai_agent --example agent_demo --features real-exec -- --real`

use rusty_ai_agent::{
    names, AllowlistPolicy, ChatMessage, ChatRole, CompletionRequest, CompletionResponse,
    LlmBackend, LlmError, ModelToolCall, ToolInvocation,
};

#[cfg(feature = "real-exec")]
use rusty_ai_agent::RealExecutor;

/// Returns scripted responses (simulates an API).
struct FakeLlmBackend {
    scripted: Vec<CompletionResponse>,
    next: std::cell::Cell<usize>,
}

impl FakeLlmBackend {
    fn new(scripted: Vec<CompletionResponse>) -> Self {
        Self {
            scripted,
            next: std::cell::Cell::new(0),
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

trait Dispatch {
    fn go(&self, inv: &ToolInvocation) -> Result<(), String>;
}

struct DryDispatch;

impl Dispatch for DryDispatch {
    fn go(&self, inv: &ToolInvocation) -> Result<(), String> {
        println!(
            "[dry-run] would execute: {}",
            serde_json::to_string(inv).unwrap_or_else(|_| format!("{inv:?}"))
        );
        Ok(())
    }
}

#[cfg(feature = "real-exec")]
struct RealDispatch {
    ex: RealExecutor,
    policy: AllowlistPolicy,
}

#[cfg(feature = "real-exec")]
impl Dispatch for RealDispatch {
    fn go(&self, inv: &ToolInvocation) -> Result<(), String> {
        let r = self
            .ex
            .execute(&self.policy, inv)
            .map_err(|e| e.to_string())?;
        println!(
            "[real] ok={} exit={:?} stdout_chars={} stderr_chars={}",
            r.ok,
            r.exit_code,
            r.stdout.len(),
            r.stderr.len()
        );
        Ok(())
    }
}

fn run_turn(
    backend: &dyn LlmBackend,
    policy: &AllowlistPolicy,
    disp: &dyn Dispatch,
    req: CompletionRequest,
) -> Result<(), String> {
    let resp = backend.complete(req).map_err(|e| e.to_string())?;
    for call in &resp.tool_calls {
        let inv = ToolInvocation::from_model_call(call).map_err(|e| e.to_string())?;
        policy.validate(&inv)?;
        disp.go(&inv)?;
    }
    if let Some(msg) = &resp.message {
        println!("[assistant] {}", msg.content);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let use_real = std::env::args().any(|a| a == "--real");
    #[cfg(not(feature = "real-exec"))]
    if use_real {
        eprintln!(
            "Rebuild with: cargo run -p rusty_ai_agent --example agent_demo --features real-exec -- --real"
        );
        std::process::exit(1);
    }

    let policy = AllowlistPolicy::new(
        vec![
            "rusty_ai_agent".to_string(),
            "examples".to_string(),
            ".".to_string(),
        ],
        vec!["cargo".to_string()],
    );

    #[cfg(feature = "real-exec")]
    let disp: Box<dyn Dispatch> = if use_real {
        Box::new(RealDispatch {
            ex: RealExecutor::new(std::env::current_dir()?)?,
            policy: policy.clone(),
        })
    } else {
        Box::new(DryDispatch)
    };
    #[cfg(not(feature = "real-exec"))]
    let disp: Box<dyn Dispatch> = Box::new(DryDispatch);

    // --- Demo 1: allowlisted read_file ---
    let backend_ok = FakeLlmBackend::new(vec![CompletionResponse {
        message: None,
        tool_calls: vec![ModelToolCall {
            id: "call_1".into(),
            name: names::READ_FILE.into(),
            arguments: serde_json::json!({ "path": "rusty_ai_agent/src/lib.rs" }),
        }],
        finish_reason: Some("tool_calls".into()),
        usage: None,
    }]);

    println!("=== Demo 1: allowlisted read_file ===");
    let req = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "Read rusty_ai_agent/src/lib.rs".into(),
        }],
        tools: vec![],
        max_tokens: None,
        temperature: None,
        stop_sequences: vec![],
    };
    run_turn(&backend_ok, &policy, disp.as_ref(), req)?;
    println!();

    // --- Demo 2: path with `..` → rejected in executor / policy ---
    let backend_escape = FakeLlmBackend::new(vec![CompletionResponse {
        message: None,
        tool_calls: vec![ModelToolCall {
            id: "call_2".into(),
            name: names::READ_FILE.into(),
            arguments: serde_json::json!({ "path": "../secrets" }),
        }],
        finish_reason: Some("tool_calls".into()),
        usage: None,
    }]);

    println!("=== Demo 2: rejected path (parent dir) ===");
    let req2 = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "bad".into(),
        }],
        tools: vec![],
        max_tokens: None,
        temperature: None,
        stop_sequences: vec![],
    };
    match run_turn(&backend_escape, &policy, disp.as_ref(), req2) {
        Ok(()) => {}
        Err(e) => println!("[blocked] {e}"),
    }
    println!();

    // --- Demo 3: run_cmd cargo check ---
    let backend_cargo = FakeLlmBackend::new(vec![CompletionResponse {
        message: None,
        tool_calls: vec![ModelToolCall {
            id: "call_3".into(),
            name: names::RUN_CMD.into(),
            arguments: serde_json::json!({
                "argv": ["cargo", "check", "-p", "rusty_ai_agent"],
                "cwd": "."
            }),
        }],
        finish_reason: Some("tool_calls".into()),
        usage: None,
    }]);

    println!("=== Demo 3: cargo check (dry-run or real) ===");
    let req3 = CompletionRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "check the crate".into(),
        }],
        tools: vec![],
        max_tokens: None,
        temperature: None,
        stop_sequences: vec![],
    };
    run_turn(&backend_cargo, &policy, disp.as_ref(), req3)?;

    Ok(())
}
