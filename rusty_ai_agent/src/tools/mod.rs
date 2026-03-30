//! Tool-Protokoll: Aufrufe, Parsing, Diff-Vorschau.
mod diff_preview;
mod invocation;
mod parse;

pub use diff_preview::{format_replace_preview, truncate_middle, truncate_utf8_prefix};
pub use invocation::{names, ToolExecutionResult, ToolInvocation};
pub use parse::{
    parse_json_arguments_loose, tool_invocations_from_model_calls, tool_invocations_try_each,
    tool_parse_retry_instruction, ToolInvocationParseItem,
};
