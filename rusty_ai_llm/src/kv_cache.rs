//! Optional KV-cache placeholder for faster autoregressive decoding (not wired in `generate` yet).

use rusty_ai_core::Tensor;

/// Per-layer storage for past keys and values; shape convention when used:
/// `(batch * heads, past_len, d_head)`.
///
/// The default [`crate::generate::generate`] path recomputes the full sequence each step
/// (simpler, slower). Extend this struct to concatenate new `K`/`V` per step.
#[derive(Clone, Debug, Default)]
pub struct LayerKv {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerKv {
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}
