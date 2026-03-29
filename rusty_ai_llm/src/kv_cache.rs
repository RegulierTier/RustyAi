//! KV-cache storage for autoregressive decoding: past keys/values per layer.
//!
//! TODO: paged / windowed KV for very long contexts (reduce memory growth).

use rusty_ai_core::{Tensor, TensorError};

/// Per-layer storage for past keys and values; shape `(batch * heads, past_len, d_head)`.
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

/// Full-model KV cache: one [`LayerKv`] per decoder block.
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    pub layers: Vec<LayerKv>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![LayerKv::default(); n_layers],
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

/// Concatenate `a` and `b` along sequence axis (dim 1). Both must be rank-3 with matching
/// `(batch_heads, _, d_head)`.
pub(crate) fn concat_along_seq(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let sa = a.shape();
    let sb = b.shape();
    if sa.len() != 3 || sb.len() != 3 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: sa.to_vec(),
                to: vec![],
            },
        ));
    }
    let bh = sa[0];
    let s1 = sa[1];
    let dh = sa[2];
    if sb[0] != bh || sb[2] != dh {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: sb.to_vec(),
                to: sa.to_vec(),
            },
        ));
    }
    let s2 = sb[1];
    let da = a.data();
    let db = b.data();
    let mut out = vec![0.0f32; bh * (s1 + s2) * dh];
    for bh_i in 0..bh {
        let base_a = bh_i * s1 * dh;
        let base_b = bh_i * s2 * dh;
        let base_o = bh_i * (s1 + s2) * dh;
        out[base_o..base_o + s1 * dh].copy_from_slice(&da[base_a..base_a + s1 * dh]);
        out[base_o + s1 * dh..base_o + (s1 + s2) * dh]
            .copy_from_slice(&db[base_b..base_b + s2 * dh]);
    }
    Tensor::from_vec(out, vec![bh, s1 + s2, dh])
}
