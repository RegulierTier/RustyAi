//! Thread-local flag: when disabled, [`crate::Variable`] constructors do not record `Op`s.

use std::cell::Cell;

thread_local! {
    /// When `false`, new [`super::Variable`] nodes are created as detached leaves (no graph).
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether gradient tracking is enabled on this thread (default: `true`).
pub fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Globally (per-thread) enable or disable graph construction.
pub fn set_grad_enabled(v: bool) {
    GRAD_ENABLED.with(|g| g.set(v));
}

/// Runs `f` with gradient tracking turned **off**, then restores the previous flag.
///
/// Typical use: inference passes where parameters are not updated and graphs are wasteful.
pub fn no_grad<F: FnOnce() -> R, R>(f: F) -> R {
    GRAD_ENABLED.with(|g| {
        let prev = g.get();
        g.set(false);
        let out = f();
        g.set(prev);
        out
    })
}
