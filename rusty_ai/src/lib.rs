//! RustyAi — umbrella crate re-exporting workspace libraries and common types.
//!
//! Submodules mirror dependency crates: [`core`], [`autograd`], [`nn`], [`ml`], [`llm`]
//! (see `Cargo.toml` members: `rusty_ai_core`, `rusty_ai_autograd`, etc.).

pub use rusty_ai_autograd as autograd;
pub use rusty_ai_core as core;
pub use rusty_ai_llm as llm;
pub use rusty_ai_ml as ml;
pub use rusty_ai_nn as nn;

pub use rusty_ai_autograd::{backward, no_grad, set_grad_enabled, Variable};
pub use rusty_ai_core::{add, matmul, softmax, Tensor, TensorError};
pub use rusty_ai_llm::{generate, ByteTokenizer, MiniGpt, MiniGptConfig};
pub use rusty_ai_ml::{Adam, Sgd};
pub use rusty_ai_nn::Linear;
