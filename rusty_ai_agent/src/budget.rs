//! Begrenzung von [`LlmBackend`]-Aufrufen und kumulativen Tokens (Phase 3, Kosten-/Last-Schutz).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::llm_backend::{CompletionRequest, CompletionResponse, CompletionUsage, LlmBackend};
use crate::LlmError;

/// Wrapper: erzwingt `max_complete_calls` und optional `max_total_tokens` (aus [`CompletionUsage`]).
/// Token-Zähler in [`crate::LocalTelemetry`] erreicht man z. B. mit [`crate::TimedBackend`] um dieses Backend.
#[derive(Debug)]
pub struct BudgetLlmBackend<B> {
    pub inner: B,
    pub max_complete_calls: Option<u64>,
    pub max_total_tokens: Option<u64>,
    complete_calls: AtomicU64,
    cumulative_tokens: AtomicU64,
}

impl<B> BudgetLlmBackend<B> {
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            max_complete_calls: None,
            max_total_tokens: None,
            complete_calls: AtomicU64::new(0),
            cumulative_tokens: AtomicU64::new(0),
        }
    }

    pub fn with_limits(
        inner: B,
        max_complete_calls: Option<u64>,
        max_total_tokens: Option<u64>,
    ) -> Self {
        Self {
            inner,
            max_complete_calls,
            max_total_tokens,
            complete_calls: AtomicU64::new(0),
            cumulative_tokens: AtomicU64::new(0),
        }
    }

    fn tokens_from_usage(u: &CompletionUsage) -> u64 {
        if let Some(t) = u.total_tokens {
            return t as u64;
        }
        let p = u.prompt_tokens.unwrap_or(0) as u64;
        let c = u.completion_tokens.unwrap_or(0) as u64;
        p + c
    }
}

impl<B: LlmBackend> LlmBackend for BudgetLlmBackend<B> {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let prev_calls = self.complete_calls.load(Ordering::Relaxed);
        if let Some(max) = self.max_complete_calls {
            if prev_calls >= max {
                return Err(LlmError::msg(format!(
                    "budget: max_complete_calls ({max}) reached"
                )));
            }
        }

        let prev_tokens = self.cumulative_tokens.load(Ordering::Relaxed);
        if let Some(max_t) = self.max_total_tokens {
            if prev_tokens >= max_t {
                return Err(LlmError::msg(format!(
                    "budget: max_total_tokens ({max_t}) already reached"
                )));
            }
        }

        let resp = self.inner.complete(request)?;

        let add = resp
            .usage
            .as_ref()
            .map(Self::tokens_from_usage)
            .unwrap_or(0);

        if let Some(max_t) = self.max_total_tokens {
            let new_total = prev_tokens.saturating_add(add);
            if new_total > max_t {
                return Err(LlmError::msg(format!(
                    "budget: would exceed max_total_tokens ({max_t}); last usage ~{add} tokens"
                )));
            }
        }

        self.complete_calls.fetch_add(1, Ordering::Relaxed);
        self.cumulative_tokens
            .fetch_add(add, Ordering::Relaxed);

        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_backend::{ChatMessage, ChatRole};

    struct OnceUsageBackend;

    impl LlmBackend for OnceUsageBackend {
        fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                message: Some(ChatMessage {
                    role: ChatRole::Assistant,
                    content: "x".into(),
                }),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
                usage: Some(CompletionUsage {
                    prompt_tokens: Some(10),
                    completion_tokens: Some(5),
                    total_tokens: Some(15),
                }),
            })
        }
    }

    #[test]
    fn blocks_second_call_when_max_1() {
        let b = BudgetLlmBackend::with_limits(OnceUsageBackend, Some(1), None);
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "a".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
        };
        assert!(b.complete(req.clone()).is_ok());
        assert!(b.complete(req).is_err());
    }

    #[test]
    fn blocks_when_tokens_exceed() {
        let b = BudgetLlmBackend::with_limits(OnceUsageBackend, None, Some(10));
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "a".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
        };
        assert!(b.complete(req).is_err());
    }
}
