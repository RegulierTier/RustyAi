//! Byte-Tokenizer und optional GPT-2-BPE (Feature `gpt2-bpe`).

mod byte;
#[cfg(feature = "gpt2-bpe")]
mod gpt2_bpe;

pub use byte::ByteTokenizer;
#[cfg(feature = "gpt2-bpe")]
pub use gpt2_bpe::*;
