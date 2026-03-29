//! GPT-2–compatible BPE using Hugging Face [`tokenizers`](https://crates.io/crates/tokenizers) (Rust).
//! Load `tokenizer.json` from a Hugging Face model directory (same layout as `transformers`).

use std::path::Path;

use tokenizers::Tokenizer;

use crate::generate::generate_from_ids;
use crate::model::MiniGpt;

/// Wraps HF [`Tokenizer`] for OpenAI/HF GPT-2–style byte-level BPE.
pub struct Gpt2Tokenizer {
    inner: Tokenizer,
}

/// Errors loading or using the BPE tokenizer.
#[derive(Debug)]
pub enum Gpt2TokenizerError {
    Io(std::io::Error),
    Tokenizer(String),
}

impl std::fmt::Display for Gpt2TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gpt2TokenizerError::Io(e) => write!(f, "{e}"),
            Gpt2TokenizerError::Tokenizer(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Gpt2TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Gpt2TokenizerError::Io(e) => Some(e),
            Gpt2TokenizerError::Tokenizer(_) => None,
        }
    }
}

impl From<std::io::Error> for Gpt2TokenizerError {
    fn from(e: std::io::Error) -> Self {
        Gpt2TokenizerError::Io(e)
    }
}

impl Gpt2Tokenizer {
    /// Load from a `tokenizer.json` path (Hugging Face format).
    ///
    /// File-not-found and other I/O errors surface as [`Gpt2TokenizerError::Io`]; invalid JSON or
    /// tokenizer schema issues as [`Gpt2TokenizerError::Tokenizer`].
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Gpt2TokenizerError> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let inner = Tokenizer::from_bytes(content.as_bytes())
            .map_err(|e| Gpt2TokenizerError::Tokenizer(format!("parse tokenizer.json: {e}")))?;
        Ok(Self { inner })
    }

    /// Load `tokenizer.json` inside a model directory (e.g. cloned GPT-2 repo folder).
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, Gpt2TokenizerError> {
        let p = dir.as_ref().join("tokenizer.json");
        Self::from_file(p)
    }

    /// Vocabulary size (including added tokens when applicable). Should match `MiniGptConfig::vocab_size`.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Text → token ids (`usize` indices into the model embedding table).
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, Gpt2TokenizerError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Gpt2TokenizerError::Tokenizer(format!("encode: {e}")))?;
        Ok(enc.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Token ids → text (`skip_special_tokens: true` matches common HF decode).
    pub fn decode(&self, ids: &[usize]) -> Result<String, Gpt2TokenizerError> {
        let u32s: Vec<u32> = ids
            .iter()
            .map(|&x| (x.min(u32::MAX as usize)) as u32)
            .collect();
        self.inner
            .decode(&u32s, true)
            .map_err(|e| Gpt2TokenizerError::Tokenizer(format!("decode: {e}")))
    }
}

/// Encode error vs. tensor forward/sampling error when running full GPT-2 text generation.
#[derive(Debug)]
pub enum Gpt2PipelineError {
    Tokenizer(Gpt2TokenizerError),
    Model(rusty_ai_core::TensorError),
}

impl std::fmt::Display for Gpt2PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gpt2PipelineError::Tokenizer(e) => write!(f, "{e}"),
            Gpt2PipelineError::Model(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Gpt2PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Gpt2PipelineError::Tokenizer(e) => Some(e),
            Gpt2PipelineError::Model(e) => Some(e),
        }
    }
}

impl From<Gpt2TokenizerError> for Gpt2PipelineError {
    fn from(e: Gpt2TokenizerError) -> Self {
        Gpt2PipelineError::Tokenizer(e)
    }
}

impl From<rusty_ai_core::TensorError> for Gpt2PipelineError {
    fn from(e: rusty_ai_core::TensorError) -> Self {
        Gpt2PipelineError::Model(e)
    }
}

/// Autoregressive text generation with BPE: `encode(prompt)` → [`generate_from_ids`] → `decode`.
pub fn generate_gpt2_text(
    model: &MiniGpt,
    tok: &Gpt2Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: &mut u32,
) -> Result<String, Gpt2PipelineError> {
    let ids = tok.encode(prompt)?;
    if max_tokens == 0 {
        return Ok(tok.decode(&ids)?);
    }
    let out = generate_from_ids(model, &ids, max_tokens, temperature, top_p, seed)?;
    Ok(tok.decode(&out)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Optional integration check: `RUSTY_AI_TEST_TOKENIZER` = path to `tokenizer.json` (e.g. HF GPT-2 checkout).
    #[test]
    #[ignore = "set RUSTY_AI_TEST_TOKENIZER to tokenizer.json path; cargo test --features gpt2-bpe -- --ignored"]
    fn gpt2_tokenizer_roundtrip_from_env_path() {
        let path = std::env::var("RUSTY_AI_TEST_TOKENIZER").expect("RUSTY_AI_TEST_TOKENIZER");
        let tok = Gpt2Tokenizer::from_file(&path).expect("load tokenizer");
        let ids = tok.encode("Hello").expect("encode");
        assert!(!ids.is_empty());
        let s = tok.decode(&ids).expect("decode");
        assert!(s.contains("Hello"));
    }
}
