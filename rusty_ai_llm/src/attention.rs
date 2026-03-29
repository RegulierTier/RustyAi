//! Scaled dot-product attention with a causal (lower-triangular) mask.

use rusty_ai_core::{matmul, mul, softmax, transpose_batched_last2, Tensor, TensorError};

/// Multi-head attention core: `softmax(mask(Q K^T / sqrt(d_head))) V`.
///
/// Inputs are already split and merged: `q`, `k`, `v` each shape `(batch * n_heads, seq, d_head)`.
/// The causal mask sets `scores[i,j] = -1e9` where `j > i` so position `i` cannot attend
/// to future tokens.
pub fn causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_head: usize,
) -> Result<Tensor, TensorError> {
    let s = q.shape();
    if s.len() != 3 || k.shape() != s || v.shape() != s {
        return Err(TensorError::Shape(rusty_ai_core::ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![],
        }));
    }
    let bh = s[0];
    let seq = s[1];
    let scale = 1.0f32 / (d_head as f32).sqrt();

    let kt = transpose_batched_last2(k)?;
    let mut scores = matmul(q, &kt)?;
    let sc = mul(&scores, &Tensor::scalar(scale))?;
    scores = sc;

    apply_causal_mask(&mut scores, bh, seq)?;

    let attn = softmax(&scores)?;
    matmul(&attn, v)
}

/// In-place: upper triangle (strictly future positions) set to large negative.
fn apply_causal_mask(scores: &mut Tensor, bh: usize, seq: usize) -> Result<(), TensorError> {
    let d = scores.data_mut();
    let stride = seq * seq;
    for b in 0..bh {
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    d[b * stride + i * seq + j] = -1e9f32;
                }
            }
        }
    }
    Ok(())
}
