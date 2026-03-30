//! KV-cache storage for autoregressive decoding: past keys/values per layer.
//!
//! Mit [`truncate_last_along_seq`] bleibt die gespeicherte Sequenzlänge bei konfiguriertem
//! **Sliding-Window** begrenzt (`MiniGptConfig::attention_window`).

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

/// Behält nur die letzten `max_len` Zeitschritte auf der Sequenzachse (Dimension 1), `[bh, seq, dh]`.
pub fn truncate_last_along_seq(t: &Tensor, max_len: usize) -> Result<Tensor, TensorError> {
    let s = t.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
    }
    let seq = s[1];
    if seq <= max_len {
        return Ok(t.clone());
    }
    let start = seq - max_len;
    slice_along_seq(t, start, seq)
}

/// Teilsequenz `t[.., start:end, ..]` (exklusives `end`), Shape `[bh, end-start, dh]`.
pub fn slice_along_seq(t: &Tensor, start: usize, end: usize) -> Result<Tensor, TensorError> {
    let s = t.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
    }
    let seq = s[1];
    let bh = s[0];
    let dh = s[2];
    if start > end || end > seq {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: vec![start, end],
                to: vec![seq],
            },
        ));
    }
    let len = end - start;
    if len == 0 {
        return Err(TensorError::EmptyTensor);
    }
    let data = t.data();
    let mut out = vec![0.0f32; bh * len * dh];
    for b in 0..bh {
        for j in 0..len {
            let src_j = start + j;
            let dst_off = b * len * dh + j * dh;
            let src_off = b * seq * dh + src_j * dh;
            out[dst_off..dst_off + dh].copy_from_slice(&data[src_off..src_off + dh]);
        }
    }
    Tensor::from_vec(out, vec![bh, len, dh])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_suffix() {
        let bh = 1usize;
        let seq = 5usize;
        let dh = 2usize;
        let data: Vec<f32> = (0..bh * seq * dh).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, vec![bh, seq, dh]).unwrap();
        let u = truncate_last_along_seq(&t, 2).unwrap();
        assert_eq!(u.shape(), &[bh, 2, dh]);
        assert_eq!(u.data(), &[6.0, 7.0, 8.0, 9.0]);
    }
}
