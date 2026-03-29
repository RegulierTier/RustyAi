//! Broadcasting and index utilities for row-major tensors.
//!
//! Broadcasting follows NumPy semantics: dimensions are aligned on the right, missing
//! dimensions are treated as length 1, and each output dimension must equal the input
//! dimension or one side must be 1 (which is then repeated).
//!
//! FIXME: broadcast cost is generic Python-style loops — not optimized for huge tensors.

use crate::error::ShapeError;

/// Computes the output shape when broadcasting `a` and `b` together.
///
/// Returns an error if any dimension pair is neither equal nor has a `1` to broadcast.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for (i, out_slot) in out.iter_mut().enumerate() {
        // Index into the padded (virtual) shapes: left-pad shorter shape with 1s.
        let ia = a.len().saturating_sub(rank - i);
        let ib = b.len().saturating_sub(rank - i);
        let da = if ia < a.len() { a[ia] } else { 1 };
        let db = if ib < b.len() { b[ib] } else { 1 };

        let o = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(ShapeError::IncompatibleBroadcast {
                    left: a.to_vec(),
                    right: b.to_vec(),
                })
            }
        };
        *out_slot = o;
    }
    Ok(out)
}

/// Left-pads `shape` with leading `1` dimensions so the result has length `rank`.
///
/// Used when aligning a tensor's shape with a higher-rank broadcast output.
pub fn pad_left_ones(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut v = vec![1usize; rank];
    let skip = rank.saturating_sub(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        v[skip + i] = d;
    }
    v
}

/// Converts multi-dimensional coordinates to a flat index in **row-major (C) order**:
/// the last index varies fastest (stride 1 along the last dimension).
pub fn ravel_index(coords: &[usize], shape: &[usize]) -> usize {
    debug_assert_eq!(coords.len(), shape.len());
    let mut idx = 0usize;
    for (i, &coord_i) in coords.iter().enumerate() {
        // Stride for dimension i = product of all following dimension sizes.
        let mut stride = 1usize;
        for &d in shape.iter().skip(i + 1) {
            stride *= d;
        }
        idx += coord_i * stride;
    }
    idx
}

/// Maps output coordinates in a broadcast result back to coordinates in the original tensor.
///
/// Where the original tensor had size `1` in a dimension, index `0` is used (virtual repeat).
pub fn broadcast_pick_coords(out_coords: &[usize], orig_shape: &[usize]) -> Vec<usize> {
    let rank = out_coords.len();
    let padded = pad_left_ones(orig_shape, rank);
    out_coords
        .iter()
        .zip(padded.iter())
        .map(|(&c, &d)| if d == 1 { 0 } else { c })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_same() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_row() {
        assert_eq!(broadcast_shapes(&[4, 1], &[1, 5]).unwrap(), vec![4, 5]);
    }
}
