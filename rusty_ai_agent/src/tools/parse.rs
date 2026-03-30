//! Map many [`ModelToolCall`]s to [`ToolInvocation`]s (Phase 1: robust batch parsing).

use std::fmt::Write;

use crate::core::error::ToolInvocationParseError;
use crate::core::llm_backend::ModelToolCall;
use super::invocation::ToolInvocation;

/// Ein fehlgeschlagener Einzelaufruf aus [`tool_invocations_try_each`].
#[derive(Debug)]
pub struct ToolInvocationParseItem {
    pub index: usize,
    pub call: ModelToolCall,
    pub error: ToolInvocationParseError,
}

/// Parst jeden Aufruf einzeln; sammelt Erfolge und Fehler (Orchestrierung: Retry mit Hinweis).
pub fn tool_invocations_try_each(
    calls: &[ModelToolCall],
) -> (Vec<ToolInvocation>, Vec<ToolInvocationParseItem>) {
    let mut ok = Vec::new();
    let mut failed = Vec::new();
    for (index, call) in calls.iter().enumerate() {
        match ToolInvocation::from_model_call(call) {
            Ok(inv) => ok.push(inv),
            Err(error) => failed.push(ToolInvocationParseItem {
                index,
                call: call.clone(),
                error,
            }),
        }
    }
    (ok, failed)
}

/// Text für eine **Folgenachricht** an das Modell, damit fehlerhafte Tool-JSON repariert werden.
pub fn tool_parse_retry_instruction(failed: &[ToolInvocationParseItem]) -> String {
    let mut s = String::from(
        "The following tool calls could not be parsed. Fix the JSON arguments and reply with the same tool names.\n\n",
    );
    for f in failed {
        let args = serde_json::to_string(&f.call.arguments).unwrap_or_else(|_| "{}".into());
        let _ = writeln!(
            &mut s,
            "- index {}: tool `{}` — error: {}; current arguments: {}",
            f.index, f.call.name, f.error, args
        );
    }
    s
}

/// Parse JSON from a model `function.arguments` string, tolerating surrounding whitespace and
/// optional fenced markdown blocks (`` ```json ... ``` ``).
pub fn parse_json_arguments_loose(raw: &str) -> Result<serde_json::Value, serde_json::Error> {
    let t = raw.trim();
    let json_slice = if let Some(rest) = t.strip_prefix("```") {
        let rest = rest.trim_start();
        let rest = rest.strip_prefix("json").unwrap_or(rest).trim_start();
        let end = rest.rfind("```").unwrap_or(rest.len());
        rest[..end].trim()
    } else {
        t
    };
    serde_json::from_str(json_slice)
}

/// Converts every call in order; stops at the first parse error.
pub fn tool_invocations_from_model_calls(
    calls: &[ModelToolCall],
) -> Result<Vec<ToolInvocation>, ToolInvocationParseError> {
    calls
        .iter()
        .map(ToolInvocation::from_model_call)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::names;
    use serde_json::json;

    #[test]
    fn two_calls_ok() {
        let calls = vec![
            ModelToolCall {
                id: "a".into(),
                name: names::READ_FILE.into(),
                arguments: json!({ "path": "rusty_ai_agent/src/lib.rs" }),
            },
            ModelToolCall {
                id: "b".into(),
                name: names::RUN_CMD.into(),
                arguments: json!({ "argv": ["cargo", "--version"] }),
            },
        ];
        let inv = tool_invocations_from_model_calls(&calls).unwrap();
        assert_eq!(inv.len(), 2);
    }

    #[test]
    fn loose_parse_strips_fence() {
        let raw = r#"```json
{"path": "a.rs"}
```"#;
        let v = parse_json_arguments_loose(raw).unwrap();
        assert_eq!(v["path"], "a.rs");
    }

    #[test]
    fn first_error_stops() {
        let calls = vec![
            ModelToolCall {
                id: "a".into(),
                name: "unknown_tool".into(),
                arguments: json!({}),
            },
        ];
        assert!(tool_invocations_from_model_calls(&calls).is_err());
    }

    #[test]
    fn try_each_partitions() {
        let calls = vec![
            ModelToolCall {
                id: "a".into(),
                name: names::READ_FILE.into(),
                arguments: json!({ "path": "p" }),
            },
            ModelToolCall {
                id: "b".into(),
                name: "bad".into(),
                arguments: json!({}),
            },
        ];
        let (ok, failed) = tool_invocations_try_each(&calls);
        assert_eq!(ok.len(), 1);
        assert_eq!(failed.len(), 1);
        assert!(tool_parse_retry_instruction(&failed).contains("bad"));
    }
}
