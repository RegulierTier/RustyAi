//! Agent protocol for **Phase 0** (IDE roadmap): [`LlmBackend`], chat/completion types, and
//! [`ToolInvocation`] JSON. See the crate `README.md` and `docs/ARCHITEKTUR_IDE_ROADMAP_B.md` in the workspace.
//!
//! Optional feature **`real-exec`**: `RealExecutor` runs [`ToolInvocation`] via `std::fs` / `std::process`
//! after [`AllowlistPolicy`] checks. Optional **`http`**: OpenAI-kompatibles HTTP (`OpenAiCompatBackend`, u. a. SSE `complete_stream`; siehe README).
//! Helpers: [`tool_invocations_try_each`], [`tool_parse_retry_instruction`], [`format_replace_preview`], [`complete_with_tool_parse_retries`],
//! [`FallbackBackend`] (primär + Fallback), [`LocalTelemetry`] / [`TimedBackend`] (lokale Metriken).
//! Phase 2: Diagnosen ([`UnifiedDiagnostic`], [`parse_cargo_json_stream`]), Prompts ([`PromptKind`], [`render_embedded`]), Tests ([`CargoTestInvocation`]).
//! Phase 3: [`PolicyCatalog`] / `RUSTY_AI_AGENT_POLICY`, [`BatchReport`], [`BudgetLlmBackend`], [`CompletionUsage`] (HTTP).

mod core;
pub mod tools;
pub mod policy;
pub mod telemetry;
pub mod feedback;
pub mod batch;
mod execution;
#[cfg(feature = "http")]
pub mod http;

pub use core::error::{LlmError, ToolInvocationParseError};
pub use core::llm_backend::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, CompletionUsage, LlmBackend,
    ModelToolCall, ToolDefinition,
};
pub use core::llm_backend;

pub use tools::{
    format_replace_preview, names, parse_json_arguments_loose, tool_invocations_from_model_calls,
    tool_invocations_try_each, tool_parse_retry_instruction, truncate_middle, truncate_utf8_prefix,
    ToolInvocation, ToolInvocationParseItem, ToolExecutionResult,
};
pub use policy::{AllowlistPolicy, PolicyCatalog, ENV_POLICY};
pub use telemetry::{LocalTelemetry, TelemetrySnapshot, TimedBackend};
pub use execution::complete_with_tool_parse_retries;
pub use execution::FallbackBackend;
pub use batch::{BatchReport, BatchStepKind, BatchStepRecord, BudgetLlmBackend};

pub use feedback::cargo_test as cargo_test;
pub use feedback::diagnostics as diagnostics;
pub use feedback::prompts as prompts;

pub use cargo_test::{CargoTestArgvError, CargoTestInvocation};
pub use diagnostics::{
    format_for_prompt, merge_diagnostics, parse_cargo_json_line, parse_cargo_json_stream,
    parse_lsp_diagnostic_json, DiagnosticSeverity, DiagnosticSource, Position, Range,
    UnifiedDiagnostic,
};
pub use prompts::{
    load_embedded, load_from_dir, render_embedded, render_template, PromptKind, RenderedPrompt,
    EMBEDDED_PROMPT_VERSION,
};

#[cfg(feature = "real-exec")]
pub use execution::{ExecutorError, RealExecutor};

#[cfg(feature = "http")]
pub use http::{OpenAiChatConfig, OpenAiCompatBackend};
