//! Decoder-Stack ([`MiniGpt`]), Multi-Head-Layouts, Linear-3D, trainierbare Variante.

mod heads;
mod linear;
mod minigpt;
mod trainable;

pub use heads::{merge_heads, split_heads};
pub use linear::linear_3d;
pub use minigpt::{DecoderBlock, MiniGpt, MiniGptConfig};
pub use trainable::{DecoderBlockTrainable, TrainableMiniGpt};
