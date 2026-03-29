//! Reshape helpers for multi-head attention: `(batch, seq, d_model)` ↔ `(batch * heads, seq, d_head)`.

use rusty_ai_core::{Tensor, TensorError};

/// Splits the last dimension `d_model = n_heads * d_head` into separate head batches.
///
/// Output layout: `[batch0_h0, batch0_h1, …, batch1_h0, …]` along the first dimension.
pub fn split_heads(x: &Tensor, heads: usize, d_head: usize) -> Result<Tensor, TensorError> {
    let s = x.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(rusty_ai_core::ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![],
        }));
    }
    let batch = s[0];
    let seq = s[1];
    let d = s[2];
    if d != heads * d_head {
        return Err(TensorError::Shape(rusty_ai_core::ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![batch, seq, heads * d_head],
        }));
    }
    let mut out = vec![0.0f32; batch * heads * seq * d_head];
    let data = x.data();
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..heads {
                for dh in 0..d_head {
                    let src = data[b * seq * d + t * d + h * d_head + dh];
                    let dst_b = b * heads + h;
                    out[dst_b * seq * d_head + t * d_head + dh] = src;
                }
            }
        }
    }
    Tensor::from_vec(out, vec![batch * heads, seq, d_head])
}

/// Inverse of [`split_heads`]: merges head dimension back into `d_model`.
pub fn merge_heads(x: &Tensor, batch: usize, heads: usize) -> Result<Tensor, TensorError> {
    let s = x.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(rusty_ai_core::ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![],
        }));
    }
    let bh = s[0];
    let seq = s[1];
    let d_head = s[2];
    if bh != batch * heads {
        return Err(TensorError::Shape(rusty_ai_core::ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![],
        }));
    }
    let d = heads * d_head;
    let mut out = vec![0.0f32; batch * seq * d];
    let data = x.data();
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..heads {
                for dh in 0..d_head {
                    let src = data[(b * heads + h) * seq * d_head + t * d_head + dh];
                    out[b * seq * d + t * d + h * d_head + dh] = src;
                }
            }
        }
    }
    Tensor::from_vec(out, vec![batch, seq, d])
}
