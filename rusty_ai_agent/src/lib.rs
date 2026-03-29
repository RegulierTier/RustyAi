//! Agent protocol for **Phase 0** (IDE roadmap): [`LlmBackend`], chat/completion types, and
//! [`ToolInvocation`] JSON. See the crate `README.md` and `docs/ARCHITEKTUR_IDE_ROADMAP_B.md` in the workspace.
//!
//! Optional feature **`real-exec`**: `RealExecutor` runs [`ToolInvocation`] via `std::fs` / `std::process`
//! after [`AllowlistPolicy`] checks. Optional **`http`**: OpenAI-kompatibles HTTP (`OpenAiCompatBackend`, u. a. SSE `complete_stream`; siehe README).
//! Helpers: [`tool_invocations_try_each`], [`tool_parse_retry_instruction`], [`format_replace_preview`], [`complete_with_tool_parse_retries`],
//! [`FallbackBackend`] (primär + Fallback), [`LocalTelemetry`] / [`TimedBackend`] (lokale Metriken).
//! Phase 2: Diagnosen ([`UnifiedDiagnostic`], [`parse_cargo_json_stream`]), Prompts ([`PromptKind`], [`render_embedded`]), Tests ([`CargoTestInvocation`]).

mod diff_preview;
pub mod diagnostics;
mod error;
mod fallback_backend;
mod llm_backend;
mod policy;
mod telemetry;
mod tool_parse;
mod tools;

mod orchestrator;
mod prompts;
mod cargo_test;

#[cfg(feature = "real-exec")]
mod executor;

#[cfg(feature = "http")]
mod openai_compat;

pub use error::{LlmError, ToolInvocationParseError};
pub use llm_backend::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, LlmBackend, ModelToolCall,
    ToolDefinition,
};
pub use fallback_backend::FallbackBackend;
pub use orchestrator::complete_with_tool_parse_retries;
pub use diff_preview::{format_replace_preview, truncate_middle};
pub use diagnostics::{
    format_for_prompt, merge_diagnostics, parse_cargo_json_line, parse_cargo_json_stream,
    parse_lsp_diagnostic_json, DiagnosticSeverity, DiagnosticSource, Position, Range,
    UnifiedDiagnostic,
};
pub use prompts::{
    load_embedded, load_from_dir, render_embedded, render_template, PromptKind, RenderedPrompt,
    EMBEDDED_PROMPT_VERSION,
};
pub use cargo_test::{CargoTestArgvError, CargoTestInvocation};
pub use telemetry::{LocalTelemetry, TelemetrySnapshot, TimedBackend};
pub use policy::AllowlistPolicy;
pub use tool_parse::{
    parse_json_arguments_loose, tool_invocations_from_model_calls, tool_invocations_try_each,
    tool_parse_retry_instruction, ToolInvocationParseItem,
};
pub use tools::{names, ToolExecutionResult, ToolInvocation};

#[cfg(feature = "real-exec")]
pub use executor::{ExecutorError, RealExecutor};

#[cfg(feature = "http")]
pub use openai_compat::{OpenAiChatConfig, OpenAiCompatBackend};
