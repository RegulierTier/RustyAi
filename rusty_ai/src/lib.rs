//! RustyAi — umbrella crate re-exporting workspace libraries and common types.
//!
//! Submodules mirror dependency crates: [`core`], [`autograd`], [`nn`], [`ml`], [`llm`]
//! (see `Cargo.toml` members: `rusty_ai_core`, `rusty_ai_autograd`, etc.).
//!
//! TODO: higher-level “agent” or HTTP client crates are intentionally out of scope here.

pub use rusty_ai_autograd as autograd;
pub use rusty_ai_core as core;
pub use rusty_ai_llm as llm;
pub use rusty_ai_ml as ml;
pub use rusty_ai_nn as nn;

#[cfg(feature = "candle")]
pub use rusty_ai_backend_candle as candle;

#[cfg(feature = "hf-hub")]
pub use rusty_ai_llm::load_minigpt_from_hf;

#[cfg(feature = "gpt2-bpe")]
pub use rusty_ai_llm::{generate_gpt2_text, Gpt2PipelineError, Gpt2Tokenizer, Gpt2TokenizerError};

pub use rusty_ai_autograd::{backward, no_grad, set_grad_enabled, Variable};
pub use rusty_ai_core::{add, matmul, softmax, Tensor, TensorError};
pub use rusty_ai_llm::{
    generate, generate_from_ids, generate_from_ids_with_callback, load_minigpt_checkpoint,
    load_minigpt_from_gpt2_safetensors, save_minigpt_checkpoint, ByteTokenizer, KvCache, MiniGpt,
    MiniGptConfig, TrainableMiniGpt,
};
pub use rusty_ai_ml::{Adam, Sgd};
pub use rusty_ai_nn::Linear;

#[cfg(test)]
mod planned_unimplemented_markers {
    #[allow(dead_code)]
    fn _llm_backend_trait_stub() {
        unimplemented!("TODO: pluggable remote LLM backend trait for IDE-style apps");
    }
}
