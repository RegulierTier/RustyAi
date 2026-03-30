//! Errors from parsing tool payloads and generic LLM backend failures.

/// Failed to map a model tool call to [`crate::ToolInvocation`].
#[derive(Debug, thiserror::Error)]
pub enum ToolInvocationParseError {
    #[error("unknown tool name: {0}")]
    UnknownTool(String),
    #[error("invalid arguments for tool {name}: {source}")]
    InvalidArguments {
        name: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("missing field `{field}` for tool {tool}")]
    MissingField { tool: String, field: &'static str },
    #[error("argv must be a non-empty JSON array of strings")]
    EmptyArgv,
    #[error("argv[{index}]: expected JSON string")]
    ArgvNotString { index: usize },
    #[error("search_replace: old_string must not be empty")]
    EmptyOldString,
}

/// Backend-agnostic completion failure (network, quota, model error, …).
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("completion failed: {0}")]
    Message(String),
    #[error(transparent)]
    ToolParse(#[from] ToolInvocationParseError),
}

impl LlmError {
    pub fn msg(s: impl Into<String>) -> Self {
        LlmError::Message(s.into())
    }
}
