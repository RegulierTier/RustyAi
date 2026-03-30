//! Scaled dot-product attention with a causal (lower-triangular) mask.
//!
//! FIXME: \(O(L^2)\) memory/time — no flash/sliding-window attention.

use rusty_ai_core::{add, matmul, mul, softmax, transpose_batched_last2, Tensor, TensorError};

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
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
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

/// Wie [`causal_attention`], aber mit additiver Maske auf den Scores (`mask[i,j]` zu jedem `scores[b,i,j]`).
///
/// `mask` hat Shape `(seq, seq)` (z. B. [`crate::fim::fim_additive_mask`]) und wird pro Batch-Zeile `b` wiederholt.
pub fn attention_with_additive_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_head: usize,
    mask: &Tensor,
) -> Result<Tensor, TensorError> {
    let s = q.shape();
    if s.len() != 3 || k.shape() != s || v.shape() != s {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
    }
    let bh = s[0];
    let seq = s[1];
    let ms = mask.shape();
    if ms != [seq, seq] {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: ms.to_vec(),
                to: vec![seq, seq],
            },
        ));
    }
    let scale = 1.0f32 / (d_head as f32).sqrt();

    let kt = transpose_batched_last2(k)?;
    let mut scores = matmul(q, &kt)?;
    let sc = mul(&scores, &Tensor::scalar(scale))?;
    scores = sc;

    let mask_b = broadcast_mask_rows(mask, bh)?;
    scores = add(&scores, &mask_b)?;

    let attn = softmax(&scores)?;
    matmul(&attn, v)
}

fn broadcast_mask_rows(mask: &Tensor, bh: usize) -> Result<Tensor, TensorError> {
    let s = mask.shape();
    if s.len() != 2 || s[0] != s[1] {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![0, 0],
            },
        ));
    }
    let seq = s[0];
    let md = mask.data();
    let mut v = vec![0.0f32; bh * seq * seq];
    for b in 0..bh {
        for i in 0..seq {
            for j in 0..seq {
                v[b * seq * seq + i * seq + j] = md[i * seq + j];
            }
        }
    }
    Tensor::from_vec(v, vec![bh, seq, seq])
}

/// Scaled dot-product attention when **Q** has a single position per head batch row
/// (`(batch * heads, 1, d_head)`) and **K**, **V** hold the full key/value history
/// `(batch * heads, T, d_head)`. No extra mask: all keys are valid for this query.
///
/// Used for autoregressive decoding with a KV-cache (new token only).
pub fn attention_single_query(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_head: usize,
) -> Result<Tensor, TensorError> {
    let sq = q.shape();
    let sk = k.shape();
    let sv = v.shape();
    if sq.len() != 3 || sk.len() != 3 || sv.len() != 3 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: sq.to_vec(),
                to: vec![],
            },
        ));
    }
    if sq[1] != 1 || sk != sv || sq[0] != sk[0] || sq[2] != sk[2] {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: sq.to_vec(),
                to: sk.to_vec(),
            },
        ));
    }
    let scale = 1.0f32 / (d_head as f32).sqrt();

    let kt = transpose_batched_last2(k)?;
    let mut scores = matmul(q, &kt)?;
    let sc = mul(&scores, &Tensor::scalar(scale))?;
    scores = sc;

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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fim::fim_additive_mask;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn additive_mask_all_prefix_matches_causal() {
        let bh = 1usize;
        let t = 5usize;
        let dh = 4usize;
        let mut qd = vec![0.0f32; bh * t * dh];
        let mut kd = vec![0.0f32; bh * t * dh];
        let mut vd = vec![0.0f32; bh * t * dh];
        for i in 0..qd.len() {
            qd[i] = (i as f32) * 0.01 - 0.5;
            kd[i] = (i as f32) * 0.02 + 0.1;
            vd[i] = (i as f32) * -0.015;
        }
        let q = Tensor::from_vec(qd, vec![bh, t, dh]).unwrap();
        let k = Tensor::from_vec(kd, vec![bh, t, dh]).unwrap();
        let v = Tensor::from_vec(vd, vec![bh, t, dh]).unwrap();

        let causal = causal_attention(&q, &k, &v, dh).unwrap();
        let mask = fim_additive_mask(t, t, 0).unwrap();
        let masked = attention_with_additive_mask(&q, &k, &v, dh, &mask).unwrap();
        let d = max_abs_diff(causal.data(), masked.data());
        assert!(d < 1e-5, "max abs diff {}", d);
    }

    #[test]
    fn single_query_matches_last_row_of_causal() {
        let bh = 2usize;
        let t = 5usize;
        let dh = 4usize;
        let mut qd = vec![0.0f32; bh * t * dh];
        let mut kd = vec![0.0f32; bh * t * dh];
        let mut vd = vec![0.0f32; bh * t * dh];
        for i in 0..qd.len() {
            qd[i] = (i as f32) * 0.01 - 0.5;
            kd[i] = (i as f32) * 0.02 + 0.1;
            vd[i] = (i as f32) * -0.015;
        }
        let q = Tensor::from_vec(qd, vec![bh, t, dh]).unwrap();
        let k = Tensor::from_vec(kd, vec![bh, t, dh]).unwrap();
        let v = Tensor::from_vec(vd, vec![bh, t, dh]).unwrap();

        let full = causal_attention(&q, &k, &v, dh).unwrap();
        let q_last = {
            let mut sl = vec![0.0f32; bh * dh];
            for b in 0..bh {
                for d in 0..dh {
                    sl[b * dh + d] = q.data()[b * t * dh + (t - 1) * dh + d];
                }
            }
            Tensor::from_vec(sl, vec![bh, 1, dh]).unwrap()
        };
        let one = attention_single_query(&q_last, &k, &v, dh).unwrap();

        let full_last = {
            let mut out = vec![0.0f32; bh * dh];
            for b in 0..bh {
                for d in 0..dh {
                    out[b * dh + d] = full.data()[b * t * dh + (t - 1) * dh + d];
                }
            }
            out
        };
        assert_eq!(one.data(), full_last.as_slice());
    }
}
