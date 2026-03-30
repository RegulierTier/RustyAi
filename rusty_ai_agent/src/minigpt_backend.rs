//! [`MiniGptTinyBackend`]: [`LlmBackend`]-Implementierung für [`rusty_ai_llm::MiniGpt`] (Byte-Tokenizer).
//!
//! Feature **`minigpt`** muss aktiv sein. Keine Tool-Calls; Anfragen mit nicht-leeren `tools` schlagen fehl.
//!
//! **Sampling:** `temperature` und `top_p` in [`CompletionRequest`](crate::llm_backend::CompletionRequest)
//! sind optional; bei `None` gelten die Backend-Defaults ([`MiniGptTinyBackend::with_defaults`] /
//! [`MiniGptTinyBackend::new`]).
//!
//! **Stop-Sequenzen:** Nicht-leere Einträge in `stop_sequences` (siehe [`CompletionRequest`](crate::llm_backend::CompletionRequest)) werden als **Substring**
//! auf dem bisher generierten **Assistant-Text** geprüft (nur neue Tokens, dekodiert mit
//! [`ByteTokenizer`](rusty_ai_llm::ByteTokenizer)). Kein Token-Grenzen-Stop wie bei BPE-APIs; mehrbyte-UTF-8-Zeichen können an
//! Grenzen zwischen generierten Bytes liegen.
//!
//! **[`CompletionUsage`](crate::llm_backend::CompletionUsage):** `prompt_tokens` = Byte-Tokenizer-Länge
//! des Prompts; `completion_tokens` = **Anzahl gesampelter neuer Token-IDs** (Länge der internen
//! `new_ids`-Liste). Das kann größer sein als die Byte-Länge des ausgegebenen `content`, wenn eine
//! Stop-Sequenz den sichtbaren Text kürzt, bevor alle gesampelten Bytes im String erscheinen.
//! `total_tokens` = Summe aus beiden (OpenAI-kompatibles Feldlayout, geschätzte Byte-„Token“-Zählung).

use std::cell::{Cell, RefCell};
use std::path::Path;

use rusty_ai_llm::{
    generate_from_ids_with_callback, load_minigpt_checkpoint, ByteTokenizer, MiniGpt,
};

use crate::llm_backend::{
    ChatMessage, ChatRole, CompletionRequest, CompletionResponse, CompletionUsage, LlmBackend,
};
use crate::LlmError;

const DEFAULT_MAX_NEW_TOKENS: usize = 64;
const DEFAULT_TEMPERATURE: f32 = 0.9;
const DEFAULT_TOP_P: f32 = 0.95;

/// Lokales „Tiny-LLM“ über [`MiniGpt`] und inkrementeller Generierung (inkl. Stop-Sequenzen).
pub struct MiniGptTinyBackend {
    model: MiniGpt,
    default_temperature: f32,
    default_top_p: f32,
    default_max_new_tokens: usize,
    seed: Cell<u32>,
}

impl MiniGptTinyBackend {
    /// Erstellt ein Backend mit Sampling-Defaults (0.9 / 0.95 / 64 neue Tokens) und RNG-Startwert.
    pub fn new(
        model: MiniGpt,
        default_temperature: f32,
        default_top_p: f32,
        default_max_new_tokens: usize,
        initial_seed: u32,
    ) -> Self {
        Self {
            model,
            default_temperature,
            default_top_p,
            default_max_new_tokens,
            seed: Cell::new(initial_seed),
        }
    }

    /// Bequemer Konstruktor mit üblichen Defaults (`micro_local`-Training kompatibel).
    pub fn with_defaults(model: MiniGpt, initial_seed: u32) -> Self {
        Self::new(
            model,
            DEFAULT_TEMPERATURE,
            DEFAULT_TOP_P,
            DEFAULT_MAX_NEW_TOKENS,
            initial_seed,
        )
    }

    /// Lädt `config.json` + `model.safetensors` aus einem Verzeichnis (RustyAi-Checkpoint-Format).
    pub fn from_checkpoint_dir(dir: impl AsRef<Path>) -> Result<Self, LlmError> {
        let model = load_minigpt_checkpoint(dir).map_err(|e| LlmError::msg(e.to_string()))?;
        Ok(Self::with_defaults(model, 42))
    }

    fn prompt_from_messages(messages: &[ChatMessage]) -> Result<String, LlmError> {
        if messages.is_empty() {
            return Err(LlmError::msg(
                "MiniGptTinyBackend: CompletionRequest.messages must not be empty",
            ));
        }
        let prompt = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if prompt.is_empty() {
            return Err(LlmError::msg(
                "MiniGptTinyBackend: all message contents are empty",
            ));
        }
        Ok(prompt)
    }
}

/// Kürzt `assistant` vor dem **frühesten** Vorkommen einer nicht-leeren Stop-Sequenz.
fn truncate_at_earliest_stop(assistant: &str, stops: &[String]) -> (String, bool) {
    let mut best: Option<usize> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(p) = assistant.find(s.as_str()) {
            best = Some(match best {
                None => p,
                Some(b) => b.min(p),
            });
        }
    }
    match best {
        Some(p) => (assistant[..p].to_string(), true),
        None => (assistant.to_string(), false),
    }
}

impl LlmBackend for MiniGptTinyBackend {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        if !request.tools.is_empty() {
            return Err(LlmError::msg(
                "MiniGptTinyBackend: tool definitions are not supported (empty tools required)",
            ));
        }

        let prompt = Self::prompt_from_messages(&request.messages)?;
        let prompt_ids = ByteTokenizer::encode(&prompt);
        let prompt_len = prompt_ids.len();
        let prompt_u32 = prompt_len as u32;

        let max_new = request
            .max_tokens
            .map(|n| n as usize)
            .unwrap_or(self.default_max_new_tokens);
        let temperature = request
            .temperature
            .unwrap_or(self.default_temperature);
        let top_p = request.top_p.unwrap_or(self.default_top_p);

        let stops: Vec<String> = request
            .stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();

        if max_new == 0 {
            return Ok(CompletionResponse {
                message: Some(ChatMessage {
                    role: ChatRole::Assistant,
                    content: String::new(),
                }),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
                usage: Some(CompletionUsage {
                    prompt_tokens: Some(prompt_u32),
                    completion_tokens: Some(0),
                    total_tokens: Some(prompt_u32),
                }),
            });
        }

        let mut seed = self.seed.get();
        let cut = RefCell::new(None::<String>);
        let mut new_ids: Vec<usize> = Vec::new();

        let full_ids = generate_from_ids_with_callback(
            &self.model,
            &prompt_ids,
            max_new,
            temperature,
            top_p,
            &mut seed,
            |id| {
                new_ids.push(id);
                if stops.is_empty() {
                    return true;
                }
                let assistant_so_far = ByteTokenizer::decode(&new_ids);
                let (trimmed, hit) = truncate_at_earliest_stop(&assistant_so_far, &stops);
                if hit {
                    *cut.borrow_mut() = Some(trimmed);
                    false
                } else {
                    true
                }
            },
        )
        .map_err(|e| LlmError::msg(format!("MiniGptTinyBackend: generation failed: {e}")))?;
        self.seed.set(seed);

        let taken = cut.into_inner();
        let hit_stop = taken.is_some();
        let content = taken.unwrap_or_else(|| {
            let new_ids_slice = full_ids.get(prompt_len..).unwrap_or(&[]);
            ByteTokenizer::decode(new_ids_slice)
        });

        // Gezählt werden gesampelte Byte-Token-IDs (Budget/Telemetry), nicht die Länge von `content`
        // nach Stop-Kürzung.
        let completion_token_count = new_ids.len() as u32;

        // Ohne Stop-Treffer läuft `generate_from_ids_with_callback` immer bis `max_new` (kein früher Abbruch).
        let finish_reason = if hit_stop { "stop" } else { "length" };

        Ok(CompletionResponse {
            message: Some(ChatMessage {
                role: ChatRole::Assistant,
                content,
            }),
            tool_calls: vec![],
            finish_reason: Some(finish_reason.into()),
            usage: Some(CompletionUsage {
                prompt_tokens: Some(prompt_u32),
                completion_tokens: Some(completion_token_count),
                total_tokens: Some(prompt_u32.saturating_add(completion_token_count)),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_backend::{CompletionRequest, LlmBackend};
    use rusty_ai_llm::{MiniGptConfig, MiniGpt};

    #[test]
    fn truncate_at_earliest_stop_basic() {
        let (s, hit) = truncate_at_earliest_stop("abcSTOPtail", &["STOP".into()]);
        assert!(hit);
        assert_eq!(s, "abc");
    }

    #[test]
    fn tiny_backend_completes_with_non_empty_assistant_text() {
        let mut seed = 7u32;
        let model = MiniGpt::random(MiniGptConfig::micro_local(), &mut seed).expect("random");
        let backend = MiniGptTinyBackend::with_defaults(model, 12345);

        let req = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello ".into(),
            }],
            tools: vec![],
            max_tokens: Some(16),
            temperature: Some(0.9),
            top_p: None,
            stop_sequences: vec![],
        };

        let resp = backend.complete(req).expect("complete");
        let text = resp
            .message
            .as_ref()
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(!text.is_empty(), "expected non-empty assistant content");

        let u = resp.usage.expect("usage");
        assert_eq!(u.prompt_tokens, Some(ByteTokenizer::encode("hello ").len() as u32));
        assert!(u.completion_tokens.unwrap_or(0) > 0);
        assert_eq!(
            u.total_tokens,
            Some(u.prompt_tokens.unwrap_or(0) + u.completion_tokens.unwrap_or(0))
        );
    }

    /// `top_p: None` muss dem Default `DEFAULT_TOP_P` entsprechen (gleiches Ergebnis bei gleichem Seed).
    #[test]
    fn top_p_none_matches_explicit_default() {
        let mk_req = |top_p: Option<f32>| CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello ".into(),
            }],
            tools: vec![],
            max_tokens: Some(8),
            temperature: Some(0.9),
            top_p,
            stop_sequences: vec![],
        };

        let mut seed = 7u32;
        let model_a = MiniGpt::random(MiniGptConfig::micro_local(), &mut seed).expect("random");
        let mut seed = 7u32;
        let model_b = MiniGpt::random(MiniGptConfig::micro_local(), &mut seed).expect("random");

        let a = MiniGptTinyBackend::with_defaults(model_a, 999);
        let ra = a.complete(mk_req(None)).expect("complete");

        let b = MiniGptTinyBackend::with_defaults(model_b, 999);
        let rb = b.complete(mk_req(Some(DEFAULT_TOP_P))).expect("complete");

        assert_eq!(
            ra.message.as_ref().map(|m| m.content.as_str()),
            rb.message.as_ref().map(|m| m.content.as_str())
        );
    }

    /// Greedy Sampling: erste generierte Zeichen als Stop → kurzer Assist-Text, `completion_tokens` ≥ 1.
    #[test]
    fn stop_sequence_trims_and_sets_usage() {
        let mut seed = 7u32;
        let model_first = MiniGpt::random(MiniGptConfig::micro_local(), &mut seed).expect("random");
        let mut seed = 7u32;
        let model_stop = MiniGpt::random(MiniGptConfig::micro_local(), &mut seed).expect("random");

        let base = CompletionRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "x".into(),
            }],
            tools: vec![],
            max_tokens: Some(16),
            temperature: Some(1e-7),
            top_p: Some(1.0),
            stop_sequences: vec![],
        };

        let backend = MiniGptTinyBackend::with_defaults(model_first, 4242);
        let first = backend.complete(base.clone()).expect("complete");
        let text = first
            .message
            .as_ref()
            .map(|m| m.content.as_str())
            .unwrap_or("");
        assert!(!text.is_empty(), "need non-empty greedy output for stop test");

        let stop_ch = text
            .chars()
            .next()
            .expect("non-empty string has first char")
            .to_string();

        let with_stop = CompletionRequest {
            stop_sequences: vec![stop_ch],
            ..base
        };
        let backend2 = MiniGptTinyBackend::with_defaults(model_stop, 4242);
        let second = backend2.complete(with_stop).expect("complete");

        assert_eq!(
            second.finish_reason.as_deref(),
            Some("stop"),
            "expected early stop"
        );
        let u = second.usage.expect("usage");
        assert!(
            u.completion_tokens.unwrap_or(0) >= 1,
            "at least one sampled token before substring match"
        );
        assert_eq!(
            u.total_tokens,
            Some(u.prompt_tokens.unwrap_or(0) + u.completion_tokens.unwrap_or(0))
        );
    }
}
