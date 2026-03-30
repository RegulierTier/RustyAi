//! Zwei [`LlmBackend`]-Implementierungen kombinieren: zuerst **primär** (z. B. Cloud-API), bei Fehler **Fallback** (z. B. lokaler Ollama-Server).

use crate::llm_backend::{CompletionRequest, CompletionResponse, LlmBackend};
use crate::LlmError;

/// Ruft zuerst `primary` auf; nur wenn das fehlschlägt, `fallback` (gleiche [`CompletionRequest`]-Kopie).
///
/// Typisches Muster: OpenAI-kompatibles Remote-Endpoint + lokales `http://127.0.0.1:11434/v1` (`OpenAiCompatBackend` mit Feature **`http`**, je eigener `OpenAiChatConfig`).
#[derive(Clone, Debug)]
pub struct FallbackBackend<A, B> {
    pub primary: A,
    pub fallback: B,
}

impl<A, B> FallbackBackend<A, B> {
    pub fn new(primary: A, fallback: B) -> Self {
        Self { primary, fallback }
    }
}

impl<A: LlmBackend, B: LlmBackend> LlmBackend for FallbackBackend<A, B> {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        match self.primary.complete(request.clone()) {
            Ok(r) => Ok(r),
            Err(e1) => match self.fallback.complete(request) {
                Ok(r) => Ok(r),
                Err(e2) => Err(LlmError::msg(format!(
                    "primary failed ({e1}); fallback failed ({e2})"
                ))),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_backend::{ChatMessage, ChatRole, LlmBackend};
    use std::cell::Cell;

    struct ScriptedBackend {
        out: Cell<Option<Result<CompletionResponse, LlmError>>>,
    }

    impl ScriptedBackend {
        fn ok() -> Self {
            Self {
                out: Cell::new(Some(Ok(CompletionResponse {
                    message: Some(ChatMessage {
                        role: ChatRole::Assistant,
                        content: "p".into(),
                    }),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                    usage: None,
                }))),
            }
        }

        fn err() -> Self {
            Self {
                out: Cell::new(Some(Err(LlmError::msg("network")))),
            }
        }
    }

    impl LlmBackend for ScriptedBackend {
        fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            self.out.take().unwrap()
        }
    }

    #[test]
    fn uses_fallback_when_primary_fails() {
        let fb = FallbackBackend::new(ScriptedBackend::err(), ScriptedBackend::ok());
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hi".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: vec![],
        };
        let r = fb.complete(req).unwrap();
        assert_eq!(r.message.unwrap().content, "p");
    }

    #[test]
    fn both_fail() {
        let fb = FallbackBackend::new(ScriptedBackend::err(), ScriptedBackend::err());
        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hi".into(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: vec![],
        };
        assert!(fb.complete(req).is_err());
    }
}
