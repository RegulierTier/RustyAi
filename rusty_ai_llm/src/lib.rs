//! Decoder-only Transformer, byte-level tokenization, and text generation (CPU tensors).
//!
//! This crate is **inference-oriented**: weights are plain [`rusty_ai_core::Tensor`] values;
//! there is no autograd training loop for `MiniGpt` in the workspace.

mod attention;
mod generate;
mod heads;
mod kv_cache;
mod linear_tensor;
mod model;
mod tokenizer;

pub use attention::{attention_single_query, causal_attention};
pub use generate::{generate, sample_token};
pub use heads::{merge_heads, split_heads};
pub use kv_cache::{KvCache, LayerKv};
pub use linear_tensor::linear_3d;
pub use model::{DecoderBlock, MiniGpt, MiniGptConfig};
pub use tokenizer::ByteTokenizer;
