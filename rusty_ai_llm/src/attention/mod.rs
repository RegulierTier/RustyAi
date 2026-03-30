//! Kausale / FIM-Attention (Tensor und [`rusty_ai_autograd::Variable`]).

mod causal;
mod variable;

pub use causal::{
    attention_single_query, attention_with_additive_mask, causal_attention, causal_attention_windowed,
};
pub use variable::{causal_attention_var, fim_attention_var};
