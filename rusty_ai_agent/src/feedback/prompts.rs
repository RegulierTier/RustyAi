//! Versionierte System-Prompt-Vorlagen (Phase 2): Analyse, Migration, Fix.
//!
//! Dateien unter `prompts/v1/` im Crate — Platzhalter `{{workspace_root}}`, `{{task}}`.

use std::fs;
use std::path::Path;

/// Eingebettete Vorlagenversion (Ordnername unter `prompts/`).
pub const EMBEDDED_PROMPT_VERSION: &str = "v1";

/// Art der Aufgabe (wählt die Markdown-Vorlage).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PromptKind {
    Analyze,
    Migrate,
    Fix,
}

impl PromptKind {
    fn file_name(self) -> &'static str {
        match self {
            PromptKind::Analyze => "analyze.md",
            PromptKind::Migrate => "migrate.md",
            PromptKind::Fix => "fix.md",
        }
    }
}

/// Geladene Vorlage mit Ersetzungen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub kind: PromptKind,
    pub content: String,
}

/// Eingebettete Vorlagen aus dem Crate (Release-Build ohne externe Dateien).
pub fn load_embedded(kind: PromptKind) -> &'static str {
    match kind {
        PromptKind::Analyze => {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/prompts/v1/analyze.md"
            ))
        }
        PromptKind::Migrate => {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/prompts/v1/migrate.md"
            ))
        }
        PromptKind::Fix => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/prompts/v1/fix.md")),
    }
}

/// Liest eine Vorlage aus `base/prompts/{version}/{kind}.md`.
pub fn load_from_dir(
    base: &Path,
    version: &str,
    kind: PromptKind,
) -> Result<String, std::io::Error> {
    let p = base.join("prompts").join(version).join(kind.file_name());
    fs::read_to_string(p)
}

/// Ersetzt `{{workspace_root}}` und `{{task}}` (einfach, keine Schleifen).
pub fn render_template(template: &str, workspace_root: &str, task: &str) -> String {
    template
        .replace("{{workspace_root}}", workspace_root)
        .replace("{{task}}", task)
}

/// Lädt eingebettete Vorlage und rendert Platzhalter.
pub fn render_embedded(kind: PromptKind, workspace_root: &str, task: &str) -> RenderedPrompt {
    let raw = load_embedded(kind);
    let content = render_template(raw, workspace_root, task);
    RenderedPrompt { kind, content }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_non_empty() {
        let s = load_embedded(PromptKind::Analyze);
        assert!(s.contains("{{workspace_root}}") || s.contains("analysis"));
    }

    #[test]
    fn render_replaces() {
        let t = render_template("root={{workspace_root}} t={{task}}", "/w", "x");
        assert_eq!(t, "root=/w t=x");
    }
}
