//! Optimizers, batching, and light preprocessing for training loops.
//!
//! FIXME: no distributed / multi-process data loading.

mod dataset;
mod optimizer;

pub use dataset::{normalize_columns, BatchIter};
pub use optimizer::{Adam, Sgd};

#[cfg(test)]
mod planned_unimplemented_markers {
    #[allow(dead_code)]
    fn _prefetch_batch_stub() {
        unimplemented!("TODO: background batch prefetch / overlap with compute");
    }
}
