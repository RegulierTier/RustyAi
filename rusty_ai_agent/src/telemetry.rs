//! **Lokale** Sitzungsmetriken (kein Netzwerk, keine Cloud): Aufrufe, Latenz, optional Ergebnisse von `cargo check`-ähnlichen Schritten.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

use crate::llm_backend::{CompletionRequest, CompletionResponse, LlmBackend};
use crate::LlmError;

/// Thread-sichere Zähler für eine Agent-Session (nur Prozess-lokal).
#[derive(Debug, Default)]
pub struct LocalTelemetry {
    complete_calls: AtomicU64,
    total_latency_ms: AtomicU64,
    /// Zusätzliche Turns durch Tool-Parse-Retry ([`crate::complete_with_tool_parse_retries`]).
    tool_parse_retry_turns: AtomicU64,
    /// Anzahl gemeldeter `cargo check` (oder ähnlicher) Läufe.
    cargo_check_runs: AtomicU64,
    /// Anzahl **erfolgreicher** gemeldeter Läufe (Exit ok).
    cargo_check_ok: AtomicU64,
}

impl LocalTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Nach einem `run_cmd` mit `cargo check` o. Ä. aufrufen (Orchestrierung/Executor).
    pub fn record_cargo_check(&self, ok: bool) {
        self.cargo_check_runs.fetch_add(1, Ordering::Relaxed);
        if ok {
            self.cargo_check_ok.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_tool_parse_retry_turn(&self) {
        self.tool_parse_retry_turns
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        let c = self.complete_calls.load(Ordering::Relaxed);
        let ms = self.total_latency_ms.load(Ordering::Relaxed);
        TelemetrySnapshot {
            complete_calls: c,
            total_latency_ms: ms,
            avg_latency_ms: if c == 0 {
                0.0
            } else {
                ms as f64 / c as f64
            },
            tool_parse_retry_turns: self.tool_parse_retry_turns.load(Ordering::Relaxed),
            cargo_check_runs: self.cargo_check_runs.load(Ordering::Relaxed),
            cargo_check_ok: self.cargo_check_ok.load(Ordering::Relaxed),
        }
    }
}

/// Lesbare Momentaufnahme (z. B. Log am Session-Ende).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TelemetrySnapshot {
    pub complete_calls: u64,
    pub total_latency_ms: u64,
    pub avg_latency_ms: f64,
    pub tool_parse_retry_turns: u64,
    pub cargo_check_runs: u64,
    pub cargo_check_ok: u64,
}

/// [`LlmBackend`]-Wrapper: misst [`LlmBackend::complete`] und schreibt in [`LocalTelemetry`].
#[derive(Debug)]
pub struct TimedBackend<B> {
    pub inner: B,
    pub telemetry: std::sync::Arc<LocalTelemetry>,
}

impl<B> TimedBackend<B> {
    pub fn new(inner: B, telemetry: std::sync::Arc<LocalTelemetry>) -> Self {
        Self { inner, telemetry }
    }
}

impl<B: LlmBackend> LlmBackend for TimedBackend<B> {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let start = Instant::now();
        let out = self.inner.complete(request);
        let ms = start.elapsed().as_millis() as u64;
        self.telemetry
            .complete_calls
            .fetch_add(1, Ordering::Relaxed);
        self.telemetry
            .total_latency_ms
            .fetch_add(ms, Ordering::Relaxed);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_backend::{ChatMessage, ChatRole, LlmBackend};
    use std::sync::Arc;

    struct EchoBackend;

    impl LlmBackend for EchoBackend {
        fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            let last = request.messages.last().cloned().unwrap_or(ChatMessage {
                role: ChatRole::User,
                content: String::new(),
            });
            Ok(CompletionResponse {
                message: Some(ChatMessage {
                    role: ChatRole::Assistant,
                    content: last.content,
                }),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
            })
        }
    }

    #[test]
    fn timed_counts_and_latency() {
        let tel = Arc::new(LocalTelemetry::new());
        let b = TimedBackend::new(EchoBackend, Arc::clone(&tel));
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "x".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
        };
        b.complete(req).unwrap();
        let s = tel.snapshot();
        assert_eq!(s.complete_calls, 1);
        assert!(s.total_latency_ms <= 10_000);
    }

    #[test]
    fn cargo_check_record() {
        let tel = LocalTelemetry::new();
        tel.record_cargo_check(true);
        tel.record_cargo_check(false);
        let s = tel.snapshot();
        assert_eq!(s.cargo_check_runs, 2);
        assert_eq!(s.cargo_check_ok, 1);
    }
}
