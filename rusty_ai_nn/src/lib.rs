//! Neural network building blocks: layers and activation functions on [`rusty_ai_core::Tensor`]
//! and [`rusty_ai_autograd::Variable`].

mod activation;
mod init;
mod linear;

pub use activation::{gelu, layer_norm};
pub use init::{glorot_uniform, uniform, zeros_bias};
pub use linear::Linear;
