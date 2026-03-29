//! Head split/merge (same layout as `rusty_ai_llm::heads`) for autograd.

use rusty_ai_core::{Tensor, TensorError};

/// Same as `rusty_ai_llm::split_heads` on raw tensors.
pub fn split_heads_tensor(x: &Tensor, heads: usize, d_head: usize) -> Result<Tensor, TensorError> {
    let s = x.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
    }
    let batch = s[0];
    let seq = s[1];
    let d = s[2];
    if d != heads * d_head {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![batch, seq, heads * d_head],
            },
        ));
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

/// Same as `rusty_ai_llm::merge_heads`.
pub fn merge_heads_tensor(x: &Tensor, batch: usize, heads: usize) -> Result<Tensor, TensorError> {
    let s = x.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
    }
    let bh = s[0];
    let seq = s[1];
    let d_head = s[2];
    if bh != batch * heads {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_merge_roundtrip() {
        let x = Tensor::from_vec(vec![1.0f32; 2 * 3 * 8], vec![2, 3, 8]).unwrap();
        let s = split_heads_tensor(&x, 2, 4).unwrap();
        let m = merge_heads_tensor(&s, 2, 2).unwrap();
        assert_eq!(m.data(), x.data());
    }
}
