//! Fill-in-the-middle (FIM): Sequenzlayout `[prefix][middle][suffix]`.
//!
//! Pro Query-Position `i` und Key `j` gilt: erlaubt, wenn das Modell an Position `j` „sehen“ darf.
//! **Prefix** nur kausal; **Mitte** sieht Präfix, kausal die eigene Region und den **Suffix**; **Suffix**
//! sieht Präfix, gesamte Mitte und kausal den eigenen Rest.
//!
//! Token-IDs: drei Segmente als flache `usize`-Liste; keine neuen Vokabel-IDs nötig — Spezialmarker
//! (z. B. `<|fim_*|>`) sind optional und nur in Doku/HF-BPE relevant.

use rusty_ai_core::{Tensor, TensorError};

const MASKED: f32 = -1e9f32;

/// Ob Position `j` von Query `i` unter FIM sichtbar ist (`seq_len = prefix + middle + suffix`).
pub fn fim_allowed(
    i: usize,
    j: usize,
    prefix_len: usize,
    middle_len: usize,
    seq_len: usize,
) -> bool {
    let suffix_start = prefix_len + middle_len;
    if seq_len < suffix_start {
        return false;
    }
    if i < prefix_len {
        j <= i && j < prefix_len
    } else if i < suffix_start {
        // Mitte: Präfix + voller Suffix + kausale Mitte — äquivalent zu
        // `j < prefix_len || j >= suffix_start || (j >= prefix_len && j <= i)`.
        j < prefix_len || j >= suffix_start || j <= i
    } else {
        // Suffix: Präfix + ganze Mitte + kausal im Suffix (äquivalent zu getrennten Bereichen).
        j < prefix_len || j < suffix_start || j <= i
    }
}

/// Additive Maske `(seq_len, seq_len)`: `0` erlaubt, `MASKED` verboten (für Scores vor Softmax).
pub fn fim_additive_mask(
    seq_len: usize,
    prefix_len: usize,
    middle_len: usize,
) -> Result<Tensor, TensorError> {
    let suffix_len = seq_len.saturating_sub(prefix_len + middle_len);
    if prefix_len + middle_len + suffix_len != seq_len {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: vec![prefix_len, middle_len, seq_len],
                to: vec![seq_len],
            },
        ));
    }
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if !fim_allowed(i, j, prefix_len, middle_len, seq_len) {
                data[i * seq_len + j] = MASKED;
            }
        }
    }
    Tensor::from_vec(data, vec![seq_len, seq_len])
}

/// Logit-Indizes `t` (0..seq-1) für Next-Token-Loss, die **nur** die Token in der **Mitte** vorhersagen
/// (Ziel: `targets[t+1]` mit `t+1` in `[prefix_len, prefix_len + middle_len)`).
pub fn fim_middle_prediction_positions(
    prefix_len: usize,
    middle_len: usize,
    seq_len: usize,
) -> Vec<usize> {
    if middle_len == 0 || seq_len < prefix_len + middle_len {
        return Vec::new();
    }
    let last_t = prefix_len + middle_len - 2;
    if last_t < prefix_len.saturating_sub(1) {
        return Vec::new();
    }
    let start = prefix_len.saturating_sub(1);
    (start..=last_t).filter(|&t| t + 1 < seq_len).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fim_mask_shape_and_prefix_causal() {
        let seq = 6usize;
        let p = 2usize;
        let m = 2usize;
        let mask = fim_additive_mask(seq, p, m).unwrap();
        assert_eq!(mask.shape(), &[seq, seq]);
        // prefix: position 0 cannot see 1
        assert!(mask.data()[1] < -1.0f32);
        // middle: position 2 can see suffix 4..5
        assert!(mask.data()[2 * seq + 4] > -1.0f32);
    }

    #[test]
    fn middle_positions_match_span() {
        let v = fim_middle_prediction_positions(2, 2, 6);
        assert_eq!(v, vec![1, 2]);
    }
}
