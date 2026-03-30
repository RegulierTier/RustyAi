//! Beispiel: [`BatchReport`] als JSON (Phase 3, CI-/Batch-Artefakt).
//!
//! `cargo run -p rusty_ai_agent --example batch_report_demo`

use rusty_ai_agent::{BatchReport, BatchStepKind, BatchStepRecord};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = BatchReport::from_steps(
        Some("ci".into()),
        std::env::var("GITHUB_SHA").ok(),
        vec![
            BatchStepRecord {
                kind: BatchStepKind::Llm,
                label: "plan migration".into(),
                ok: true,
                detail: None,
            },
            BatchStepRecord {
                kind: BatchStepKind::Tool,
                label: "read_file src/lib.rs".into(),
                ok: true,
                detail: None,
            },
            BatchStepRecord {
                kind: BatchStepKind::Check,
                label: "cargo check -p rusty_ai_agent".into(),
                ok: true,
                detail: Some("Finished `dev` profile".into()),
            },
        ],
    );

    println!("{}", report.to_json_string()?);
    println!("\n--- markdown ---\n{}", report.to_markdown_summary());
    Ok(())
}
