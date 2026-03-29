//! Agent protocol for **Phase 0** (IDE roadmap): [`LlmBackend`], chat/completion types, and
//! [`ToolInvocation`] JSON. See the crate `README.md` and `docs/ARCHITEKTUR_IDE_ROADMAP_B.md` in the workspace.
//!
//! Execution of tools (filesystem, subprocess) is **not** implemented here — only types and parsing.

mod error;
mod llm_backend;
mod tools;

pub use error::{LlmError, ToolInvocationParseError};
pub use llm_backend::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, LlmBackend, ModelToolCall,
    ToolDefinition,
};
pub use tools::{names, ToolExecutionResult, ToolInvocation};
