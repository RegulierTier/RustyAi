//! Orchestrierung, Fallback-Backend, optional echter Executor.
#[cfg(feature = "real-exec")]
mod executor;
mod fallback_backend;
mod orchestrator;

#[cfg(feature = "real-exec")]
pub use executor::{ExecutorError, RealExecutor};
pub use fallback_backend::FallbackBackend;
pub use orchestrator::complete_with_tool_parse_retries;
