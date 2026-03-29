//! Optimizers, batching, and light preprocessing for training loops.

mod dataset;
mod optimizer;

pub use dataset::{normalize_columns, BatchIter};
pub use optimizer::{Adam, Sgd};
