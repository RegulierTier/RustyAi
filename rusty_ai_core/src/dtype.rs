/// Element type for tensor storage.
///
/// The framework currently uses [`DType::F32`] everywhere; this enum exists so future
/// `f64` or integer types can be added without a breaking API redesign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE-754 single precision (default for training and inference in this workspace).
    F32,
}
