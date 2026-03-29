//! Human-readable **before/after** snippets for patches (orchestrator logs, CLI, IDE hints).
//!
//! Kein externes Diff-Crate — nur Formatierung für kleine Ausschnitte.

/// Markdown-ähnlicher Block mit Pfad und OLD/NEW (für `search_replace` oder manuelle Review).
pub fn format_replace_preview(path: &str, old_str: &str, new_str: &str) -> String {
    format!(
        "### {}\n\n**Entfernen:**\n```text\n{}\n```\n\n**Einfügen:**\n```text\n{}\n```\n",
        path,
        truncate_middle(old_str, 8000),
        truncate_middle(new_str, 8000)
    )
}

/// Kürzt lange Strings in der Mitte, damit Logs lesbar bleiben.
/// Schneidet nur an UTF-8-Zeichengrenzen (keine Panik bei Mehrbyte-Zeichen).
pub fn truncate_middle(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(3) / 2;
    if keep == 0 {
        return "…".to_string();
    }

    let prefix_end = prefix_end_within(s, keep);
    let suffix_start = suffix_start_within(s, keep);

    if prefix_end >= suffix_start {
        return "…".to_string();
    }
    format!("{}…{}", &s[..prefix_end], &s[suffix_start..])
}

fn prefix_end_within(s: &str, max_bytes: usize) -> usize {
    let mut end = 0usize;
    let mut b = 0usize;
    for (i, ch) in s.char_indices() {
        let cl = ch.len_utf8();
        if b + cl > max_bytes {
            break;
        }
        end = i + cl;
        b += cl;
    }
    end
}

fn suffix_start_within(s: &str, max_bytes: usize) -> usize {
    let mut start = s.len();
    let mut b = 0usize;
    for (i, ch) in s.char_indices().rev() {
        let cl = ch.len_utf8();
        if b + cl > max_bytes {
            break;
        }
        start = i;
        b += cl;
    }
    start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_contains_paths_and_snippets() {
        let t = format_replace_preview("src/lib.rs", "fn old()", "fn new()");
        assert!(t.contains("src/lib.rs"));
        assert!(t.contains("fn old()"));
        assert!(t.contains("fn new()"));
    }

    #[test]
    fn truncate_long() {
        let s = "a".repeat(100);
        let t = truncate_middle(&s, 20);
        assert!(t.contains('…'));
        assert!(t.len() <= 23);
    }

    #[test]
    fn truncate_middle_unicode_euro() {
        let s: String = "€".repeat(200);
        let t = truncate_middle(&s, 24);
        assert!(t.contains('…'));
        assert!(t.chars().all(|c| c == '€' || c == '…'));
    }

    #[test]
    fn truncate_middle_emoji() {
        let s = "😀".repeat(80);
        let t = truncate_middle(&s, 20);
        assert!(t.contains('…'));
    }
}
