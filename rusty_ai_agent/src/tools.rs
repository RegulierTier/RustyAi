//! JSON-serializable tool invocations and execution results (Phase 0).
//!
//! Wire format for a single invocation uses an **externally tagged** enum compatible with
//! [`serde_json`](https://docs.rs/serde_json): `{"tool":"read_file","path":"..."}`.
//! See [`schemas/tool_invocation.json`](../../schemas/tool_invocation.json).

use serde::{Deserialize, Serialize};

use crate::error::ToolInvocationParseError;
use crate::ModelToolCall;

/// Stable names for tools (use in prompts and OpenAI-style `function.name`).
pub mod names {
    pub const READ_FILE: &str = "read_file";
    pub const WRITE_FILE: &str = "write_file";
    pub const RUN_CMD: &str = "run_cmd";
}

/// What the orchestrator should run after the model selects a tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
pub enum ToolInvocation {
    /// Read UTF-8 text from a path relative to the workspace root (policy enforced by executor).
    ReadFile { path: String },
    /// Overwrite or create a file (executor may require confirmation in UI).
    WriteFile { path: String, content: String },
    /// Run a command: first element is the binary; remainder are arguments. Optional working directory.
    RunCmd {
        argv: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },
}

impl ToolInvocation {
    /// Build from an OpenAI-style tool call: `name` + JSON object `arguments`.
    pub fn from_model_call(call: &ModelToolCall) -> Result<Self, ToolInvocationParseError> {
        let v = &call.arguments;
        match call.name.as_str() {
            names::READ_FILE => {
                let path = parse_string_field(v, names::READ_FILE, "path")?;
                Ok(ToolInvocation::ReadFile { path })
            }
            names::WRITE_FILE => {
                let path = parse_string_field(v, names::WRITE_FILE, "path")?;
                let content = parse_string_field(v, names::WRITE_FILE, "content")?;
                Ok(ToolInvocation::WriteFile { path, content })
            }
            names::RUN_CMD => {
                let argv: Vec<String> = parse_argv(v)?;
                let cwd = parse_optional_string(v, "cwd");
                Ok(ToolInvocation::RunCmd { argv, cwd })
            }
            other => Err(ToolInvocationParseError::UnknownTool(other.to_string())),
        }
    }
}

fn parse_string_field(
    v: &serde_json::Value,
    tool: &str,
    field: &'static str,
) -> Result<String, ToolInvocationParseError> {
    v.get(field)
        .and_then(|x| x.as_str())
        .map(str::to_owned)
        .ok_or_else(|| ToolInvocationParseError::MissingField {
            tool: tool.to_string(),
            field,
        })
}

fn parse_optional_string(v: &serde_json::Value, field: &str) -> Option<String> {
    v.get(field).and_then(|x| x.as_str()).map(str::to_owned)
}

fn parse_argv(v: &serde_json::Value) -> Result<Vec<String>, ToolInvocationParseError> {
    let arr = v
        .get("argv")
        .and_then(|x| x.as_array())
        .ok_or_else(|| ToolInvocationParseError::MissingField {
            tool: names::RUN_CMD.to_string(),
            field: "argv",
        })?;
    let mut out = Vec::with_capacity(arr.len());
    for (index, item) in arr.iter().enumerate() {
        let s = item.as_str().ok_or(ToolInvocationParseError::ArgvNotString { index })?;
        out.push(s.to_string());
    }
    if out.is_empty() {
        return Err(ToolInvocationParseError::EmptyArgv);
    }
    Ok(out)
}

/// Result of executing a [`ToolInvocation`] (stdout/stderr or error text).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stdout: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stderr: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

impl ToolExecutionResult {
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            ok: true,
            stdout: stdout.into(),
            stderr: String::new(),
            exit_code: Some(0),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            stdout: String::new(),
            stderr: message.into(),
            exit_code: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_invocation_roundtrip_json() {
        let t = ToolInvocation::ReadFile {
            path: "src/lib.rs".into(),
        };
        let j = serde_json::to_string(&t).unwrap();
        let back: ToolInvocation = serde_json::from_str(&j).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn from_model_read_file() {
        let call = ModelToolCall {
            id: "1".into(),
            name: names::READ_FILE.into(),
            arguments: serde_json::json!({ "path": "a.rs" }),
        };
        let inv = ToolInvocation::from_model_call(&call).unwrap();
        assert_eq!(
            inv,
            ToolInvocation::ReadFile {
                path: "a.rs".into()
            }
        );
    }
}
