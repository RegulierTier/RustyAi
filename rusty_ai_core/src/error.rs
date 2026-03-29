use std::fmt;

/// Errors arising from incompatible tensor shapes or invalid reshape/broadcast operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeError {
    /// Shapes cannot be broadcast together (NumPy-style rules failed).
    IncompatibleBroadcast {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    /// `matmul` requires inner dimensions to match (2D: `A[m,k] @ B[k,n]`).
    MatmulIncompatible {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    /// `reshape` would change the total number of elements.
    InvalidReshape {
        from: Vec<usize>,
        to: Vec<usize>,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::IncompatibleBroadcast { left, right } => {
                write!(f, "incompatible shapes for broadcast: {left:?} vs {right:?}")
            }
            ShapeError::MatmulIncompatible { left, right } => {
                write!(f, "matmul: incompatible shapes {left:?} x {right:?}")
            }
            ShapeError::InvalidReshape { from, to } => {
                write!(f, "reshape: cannot reshape {from:?} to {to:?} (element count mismatch)")
            }
        }
    }
}

impl std::error::Error for ShapeError {}

/// Top-level error type for tensor operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Shape/broadcast/reshape violation.
    Shape(ShapeError),
    /// Tensor has zero elements where at least one was required.
    EmptyTensor,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::Shape(e) => write!(f, "{e}"),
            TensorError::EmptyTensor => write!(f, "tensor has no elements"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<ShapeError> for TensorError {
    fn from(e: ShapeError) -> Self {
        TensorError::Shape(e)
    }
}
