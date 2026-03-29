//! Decoder-only Transformer, byte-level tokenization, and text generation (CPU tensors).
//!
//! Inference uses plain [`rusty_ai_core::Tensor`] weights ([`MiniGpt`]). Training uses
//! [`TrainableMiniGpt`] with [`rusty_ai_autograd::Variable`] (same forward numerically).

mod attention;
mod attention_var;
mod generate;
mod heads;
mod kv_cache;
mod linear_tensor;
mod model;
mod tokenizer;
mod trainable;

pub use attention::{attention_single_query, causal_attention};
pub use attention_var::causal_attention_var;
pub use generate::{generate, sample_token};
pub use heads::{merge_heads, split_heads};
pub use kv_cache::{KvCache, LayerKv};
pub use linear_tensor::linear_3d;
pub use model::{DecoderBlock, MiniGpt, MiniGptConfig};
pub use tokenizer::ByteTokenizer;
pub use trainable::{DecoderBlockTrainable, TrainableMiniGpt};
