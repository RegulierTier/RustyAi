//! Decoder-only Transformer, byte-level tokenization, and text generation.
//! Core weights use [`rusty_ai_core::Tensor`] on CPU; see `checkpoint` and `gpt2_hf` for safetensors / Hub.
//!
//! With feature **`gpt2-bpe`**, `Gpt2Tokenizer` loads Hugging Face `tokenizer.json` for GPT-2–compatible BPE.
//!
//! Inference uses plain [`rusty_ai_core::Tensor`] weights ([`MiniGpt`]). Training uses
//! [`TrainableMiniGpt`] with [`rusty_ai_autograd::Variable`] (same forward numerically).

mod attention;
mod attention_var;
mod checkpoint;
mod generate;
mod gpt2_hf;
#[cfg(feature = "gpt2-bpe")]
mod gpt2_tokenizer;
mod heads;
mod kv_cache;
mod linear_tensor;
mod model;
mod tokenizer;
mod trainable;

pub use attention::{attention_single_query, causal_attention};
pub use attention_var::causal_attention_var;
pub use checkpoint::{
    load_minigpt_checkpoint, mini_gpt_from_state_dict, minigpt_to_safetensors_bytes,
    save_minigpt_checkpoint, state_dict, tensor_from_safetensors_view, CheckpointError,
    MiniGptConfigFile,
};
pub use generate::{generate, generate_from_ids, sample_token};
pub use gpt2_hf::{
    gpt2_state_dict_to_minigpt, load_minigpt_from_gpt2_safetensors, Gpt2MappingError,
};
pub use heads::{merge_heads, split_heads};
pub use kv_cache::{KvCache, LayerKv};
pub use linear_tensor::linear_3d;
pub use model::{DecoderBlock, MiniGpt, MiniGptConfig};
pub use tokenizer::ByteTokenizer;
pub use trainable::{DecoderBlockTrainable, TrainableMiniGpt};

#[cfg(feature = "gpt2-bpe")]
pub use gpt2_tokenizer::{
    generate_gpt2_text, Gpt2PipelineError, Gpt2Tokenizer, Gpt2TokenizerError,
};

#[cfg(feature = "hf-hub")]
pub use checkpoint::load_minigpt_from_hf;
