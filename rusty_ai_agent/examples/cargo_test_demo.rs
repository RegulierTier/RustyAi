//! [`CargoTestInvocation`]: sicheres `argv` für `cargo test -p … -- filter`.
//!
//! `cargo run -p rusty_ai_agent --example cargo_test_demo`

use rusty_ai_agent::CargoTestInvocation;

fn main() {
    let inv = CargoTestInvocation::new(
        Some("rusty_ai_agent"),
        &["diagnostics::"],
        false,
    )
    .expect("valid");
    println!("{}", inv.argv().join(" "));
}
