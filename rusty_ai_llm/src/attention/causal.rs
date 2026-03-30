//! Scaled dot-product attention mit kausaler Maske und optional **Sliding-Window**.
//!
//! Kausaler Forward ohne FIM-Maske nutzt eine **zeilenweise** Auswertung (keine volle `L×L`-Score-Matrix),
//! Peak-Speicher grob **O(batch·L·d_head)** statt **O(batch·L²)**. FIM / additive Masken bleiben bei
//! der klassischen `matmul`-Form ([`attention_with_additive_mask`]).

use rusty_ai_core::{add, matmul, mul, softmax, transpose_batched_last2, Tensor, TensorError};

use crate::cache::slice_along_seq;

fn softmax_last_row_1d(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = x.iter().map(|&xi| (xi - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s <= 0.0 || !s.is_finite() {
        let n = exps.len();
        return vec![1.0f32 / n as f32; n];
    }
    for e in &mut exps {
        *e /= s;
    }
    exps
}

/// Multi-head attention core: kausal; intern zeilenweise ohne volle Score-Matrix.
///
/// Entspricht [`causal_attention_windowed`] mit `window == None`.
pub fn causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_head: usize,
) -> Result<Tensor, TensorError> {
    causal_attention_windowed(q, k, v, d_head, None)
}

/// Kausale Attention mit optionalem **Sliding-Window** der Länge `window`: Position `i` sieht nur
/// Keys `j` mit `max(0, i + 1 - window) <= j <= i`. `None` = volles linkes Fenster (wie klassisch kausal).
pub fn causal_attention_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_head: usize,
    window: Option<usize>,
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
    if let Some(w) = window {
        if w == 0 {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![w],
                    to: vec![1],
                },
            ));
        }
    }
    let bh = s[0];
    let seq = s[1];
    let dh = s[2];
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let qd = q.data();
    let kd = k.data();
    let vd = v.data();
    let mut out = vec![0.0f32; bh * seq * dh];
    for b in 0..bh {
        for i in 0..seq {
            let j_start = match window {
                None => 0usize,
                Some(w) => i.saturating_sub(w.saturating_sub(1)),
            };
            let j_end = i + 1;
            let win = j_end - j_start;
            let mut scores = vec![0.0f32; win];
            for (jj, j) in (j_start..j_end).enumerate() {
                let mut dot = 0.0f32;
                for d in 0..dh {
                    dot += qd[b * seq * dh + i * dh + d] * kd[b * seq * dh + j * dh + d];
                }
                scores[jj] = dot * scale;
            }
            let attn = softmax_last_row_1d(&scores);
            for d in 0..dh {
                let mut acc = 0.0f32;
                for (jj, j) in (j_start..j_end).enumerate() {
                    acc += attn[jj] * vd[b * seq * dh + j * dh + d];
                }
                out[b * seq * dh + i * dh + d] = acc;
            }
        }
    }
    Tensor::from_vec(out, vec![bh, seq, dh])
}

/// Wie [`causal_attention`], aber mit additiver Maske auf den Scores (`mask[i,j]` zu jedem `scores[b,i,j]`).
///
/// `mask` hat Shape `(seq, seq)` (z. B. [`crate::inference::fim::fim_additive_mask`]) und wird pro Batch-Zeile `b` wiederholt.
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
/// (`(batch * heads, 1, d_head)`) und **K**, **V** die Key-/Value-Historie tragen.
///
/// Mit `window == Some(w)` werden nur die letzten `w` Zeitschritte von K/V genutzt (Sliding-Window-Decode).
pub fn attention_single_query(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_head: usize,
    window: Option<usize>,
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
    let (k2, v2) = match window {
        None => (k.clone(), v.clone()),
        Some(w) if w == 0 => {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![w],
                    to: vec![1],
                },
            ));
        }
        Some(w) => {
            let tl = sk[1];
            let keep = tl.min(w);
            if keep == 0 {
                return Err(TensorError::EmptyTensor);
            }
            let start = tl - keep;
            (
                slice_along_seq(k, start, tl)?,
                slice_along_seq(v, start, tl)?,
            )
        }
    };
    let scale = 1.0f32 / (d_head as f32).sqrt();

    let kt = transpose_batched_last2(&k2)?;
    let mut scores = matmul(q, &kt)?;
    let sc = mul(&scores, &Tensor::scalar(scale))?;
    scores = sc;

    let attn = softmax(&scores)?;
    matmul(&attn, &v2)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::inference::fim::fim_additive_mask;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn causal_attention_dense(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        d_head: usize,
    ) -> Result<Tensor, TensorError> {
        let s = q.shape();
        let bh = s[0];
        let seq = s[1];
        let scale = 1.0f32 / (d_head as f32).sqrt();
        let kt = transpose_batched_last2(k)?;
        let mut scores = matmul(q, &kt)?;
        let sc = mul(&scores, &Tensor::scalar(scale))?;
        scores = sc;
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
        let attn = softmax(&scores)?;
        matmul(&attn, v)
    }

    #[test]
    fn causal_rowwise_matches_dense_reference() {
        let bh = 2usize;
        let t = 6usize;
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
        let dense = causal_attention_dense(&q, &k, &v, dh).unwrap();
        let row = causal_attention(&q, &k, &v, dh).unwrap();
        let d = max_abs_diff(dense.data(), row.data());
        assert!(d < 1e-4, "max abs diff {}", d);
    }

    #[test]
    fn sliding_window_large_equals_causal() {
        let bh = 1usize;
        let t = 8usize;
        let dh = 4usize;
        let mut qd = vec![0.0f32; bh * t * dh];
        let mut kd = vec![0.0f32; bh * t * dh];
        let mut vd = vec![0.0f32; bh * t * dh];
        for i in 0..qd.len() {
            qd[i] = (i as f32) * 0.03 - 0.1;
            kd[i] = (i as f32) * -0.02 + 0.2;
            vd[i] = (i as f32) * 0.01;
        }
        let q = Tensor::from_vec(qd, vec![bh, t, dh]).unwrap();
        let k = Tensor::from_vec(kd, vec![bh, t, dh]).unwrap();
        let v = Tensor::from_vec(vd, vec![bh, t, dh]).unwrap();
        let full = causal_attention(&q, &k, &v, dh).unwrap();
        let win = causal_attention_windowed(&q, &k, &v, dh, Some(t)).unwrap();
        let d = max_abs_diff(full.data(), win.data());
        assert!(d < 1e-4, "max abs diff {}", d);
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
        assert!(d < 1e-4, "max abs diff {}", d);
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
        let one = attention_single_query(&q_last, &k, &v, dh, None).unwrap();

        let full_last = {
            let mut out = vec![0.0f32; bh * dh];
            for b in 0..bh {
                for d in 0..dh {
                    out[b * dh + d] = full.data()[b * t * dh + (t - 1) * dh + d];
                }
            }
            out
        };
        let diff: f32 = one
            .data()
            .iter()
            .zip(full_last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Rowwise vs. vollständiger Softmax-Pfad kann minimal abweichen (≈1e-6).
        assert!(diff < 1e-4, "max abs diff {}", diff);
    }
}
