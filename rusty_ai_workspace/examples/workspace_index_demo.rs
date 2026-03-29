//! Baut einen kleinen Index unter dem aktuellen Verzeichnis und sucht nach einem String.
//!
//! `cargo run -p rusty_ai_workspace --example workspace_index_demo`

use rusty_ai_workspace::{IndexConfig, WorkspaceIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IndexConfig {
        root: std::env::current_dir()?,
        max_chunk_lines: 32,
        overlap_lines: 4,
        ..Default::default()
    };

    let index = WorkspaceIndex::build(&config)?;
    println!("chunks: {}", index.chunks().len());

    let hits = index.search_substring("WorkspaceIndex", true);
    println!("hits for 'WorkspaceIndex': {}", hits.len());
    for h in hits.iter().take(3) {
        println!(
            "{}:{}-{} ({} chars)",
            h.path.display(),
            h.start_line,
            h.end_line,
            h.text.len()
        );
    }
    Ok(())
}
