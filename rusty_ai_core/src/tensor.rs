use crate::dtype::DType;
use crate::error::{ShapeError, TensorError};
use crate::shape::{broadcast_pick_coords, broadcast_shapes, ravel_index};

/// A dense tensor: contiguous `f32` buffer with a shape vector, stored in **row-major (C)** order.
///
/// The linear layout is: index `[i0, i1, ..., i_{n-1}]` maps to
/// `i0 * s1 * s2 * ... + i1 * s2 * ... + ... + i_{n-1}` where `sk` are dimension sizes.
/// Broadcasting for binary ops is implemented explicitly in [`Tensor::broadcast_binary`].
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    /// Returns the declared element type (currently always [`DType::F32`]).
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Shape dimensions from slowest-varying (outer) to fastest-varying (inner).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Immutable view of the raw buffer (length equals product of shape).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable view of the raw buffer; caller must preserve shape invariants.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Total number of elements (`product` of all dimensions; `1` for a scalar).
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Fills a tensor with zeros. Empty shape yields an empty buffer.
    pub fn zeros(shape: &[usize], dtype: DType) -> Result<Self, TensorError> {
        let n: usize = shape.iter().product();
        Ok(Self {
            data: vec![0.0; n],
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Builds a tensor from owned data; returns an error if `data.len()` ≠ product of `shape`.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(TensorError::Shape(ShapeError::InvalidReshape {
                from: vec![data.len()],
                to: shape.clone(),
            }));
        }
        Ok(Self {
            data,
            shape,
            dtype: DType::F32,
        })
    }

    /// A 0-D tensor holding a single scalar (shape `[]`).
    pub fn scalar(x: f32) -> Self {
        Self {
            data: vec![x],
            shape: vec![],
            dtype: DType::F32,
        }
    }

    /// Returns a new tensor with the same data but a different shape (total element count unchanged).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let n: usize = new_shape.iter().product();
        if n != self.numel() {
            return Err(TensorError::Shape(ShapeError::InvalidReshape {
                from: self.shape.clone(),
                to: new_shape.to_vec(),
            }));
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype,
        })
    }

    /// Elementwise binary operation with NumPy-style broadcasting.
    ///
    /// For each output flat index, decodes coordinates, maps both operands' coordinates
    /// (with broadcast rules), and applies `op`.
    pub fn broadcast_binary<F>(&self, other: &Self, op: F) -> Result<Self, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        let out_shape = broadcast_shapes(self.shape(), other.shape())?;
        let rank = out_shape.len();
        let n: usize = out_shape.iter().product();
        let mut out_data = vec![0.0f32; n];

        let mut coords = vec![0usize; rank];
        for flat in 0..n {
            // Decode flat index to output coordinates (last dim fastest).
            let mut rem = flat;
            for i in (0..rank).rev() {
                let d = out_shape[i];
                coords[i] = rem % d;
                rem /= d;
            }

            let ca = broadcast_pick_coords(&coords, self.shape());
            let ia = ravel_index(&ca, &crate::shape::pad_left_ones(self.shape(), rank));
            let cb = broadcast_pick_coords(&coords, other.shape());
            let ib = ravel_index(&cb, &crate::shape::pad_left_ones(other.shape(), rank));

            out_data[flat] = op(self.data[ia], other.data[ib]);
        }

        Ok(Self {
            data: out_data,
            shape: out_shape,
            dtype: DType::F32,
        })
    }
}
