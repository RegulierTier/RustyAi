//! RustyAi — umbrella crate re-exporting workspace libraries and common types.
//!
//! Submodules mirror dependency crates: [`core`], [`autograd`], [`nn`], [`ml`], [`llm`]
//! (see `Cargo.toml` members: `rusty_ai_core`, `rusty_ai_autograd`, etc.).

pub use rusty_ai_autograd as autograd;
pub use rusty_ai_core as core;
pub use rusty_ai_llm as llm;
pub use rusty_ai_ml as ml;
pub use rusty_ai_nn as nn;

#[cfg(feature = "candle")]
pub use rusty_ai_backend_candle as candle;

#[cfg(feature = "hf-hub")]
pub use rusty_ai_llm::load_minigpt_from_hf;

pub use rusty_ai_autograd::{backward, no_grad, set_grad_enabled, Variable};
pub use rusty_ai_core::{add, matmul, softmax, Tensor, TensorError};
pub use rusty_ai_llm::{
    generate,
    load_minigpt_checkpoint,
    load_minigpt_from_gpt2_safetensors,
    save_minigpt_checkpoint,
    ByteTokenizer,
    KvCache,
    MiniGpt,
    MiniGptConfig,
    TrainableMiniGpt,
};
pub use rusty_ai_ml::{Adam, Sgd};
pub use rusty_ai_nn::Linear;
