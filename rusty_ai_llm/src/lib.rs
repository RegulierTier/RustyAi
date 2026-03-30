//! Decoder-only Transformer, byte-level tokenization, and text generation.
//! Core weights use [`rusty_ai_core::Tensor`] on CPU; see `checkpoint` and `gpt2_hf` for safetensors / Hub.
//!
//! With feature **`gpt2-bpe`**, `Gpt2Tokenizer` loads Hugging Face `tokenizer.json` for GPT-2–compatible BPE.
//!
//! Inference uses plain [`rusty_ai_core::Tensor`] weights ([`MiniGpt`]). Training uses
//! [`TrainableMiniGpt`] with [`rusty_ai_autograd::Variable`] (same forward numerically).
//! **FIM:** [`forward_fim`](MiniGpt::forward_fim), [`fim_middle_prediction_positions`],
//! [`generate_fim_middle_from_ids`], and
//! [`Variable::cross_entropy_next_token_subset`](rusty_ai_autograd::Variable::cross_entropy_next_token_subset).
//!
//! TODO: public “agent/IDE” hooks (stop sequences, tool protocol) live outside this crate for now.

mod attention;
mod attention_var;
mod checkpoint;
mod fim;
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

pub use attention::{
    attention_single_query, attention_with_additive_mask, causal_attention, causal_attention_windowed,
};
pub use attention_var::{causal_attention_var, fim_attention_var};
pub use checkpoint::{
    load_minigpt_checkpoint, load_minigpt_checkpoint_bytes, mini_gpt_from_state_dict,
    minigpt_to_safetensors_bytes, save_minigpt_checkpoint, state_dict,
    tensor_from_safetensors_view, CheckpointError, MiniGptConfigFile,
};
pub use fim::{
    fim_additive_mask, fim_allowed, fim_middle_prediction_positions, fim_next_logit_timestep,
};
pub use generate::{
    generate, generate_fim_middle_from_ids, generate_from_ids, generate_from_ids_with_callback,
    sample_token,
};
pub use gpt2_hf::{
    gpt2_state_dict_to_minigpt, load_minigpt_from_gpt2_safetensors, Gpt2MappingError,
};
pub use heads::{merge_heads, split_heads};
pub use kv_cache::{slice_along_seq, truncate_last_along_seq, KvCache, LayerKv};
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
