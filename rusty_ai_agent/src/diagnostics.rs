//! Einheitliche Compiler- und LSP-Diagnosen für Feedback-Schleifen (Phase 2).
//!
//! - **Cargo / rustc:** Zeilenweise JSON von `cargo check --message-format=json` parsen.
//! - **LSP:** Subset von `Diagnostic` + optional `PublishDiagnosticsParams`-Batch (JSON).
//!
//! Kein eingebetteter Language-Server — IDE/Plugin kann exportierte JSON einspeisen.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Herkunft der Diagnose (für Prompt und Telemetrie).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticSource {
    Cargo,
    Lsp,
}

/// Schweregrad (vereinheitlicht rustc + LSP).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
    /// Unbekannt / nicht zuordenbar.
    Unknown,
}

/// Eine Zeile/Spalte (0-basiert, LSP-kompatibel).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// Halb-offener Bereich im Text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

/// `(start_line, start_char, end_line, end_char)` für Dedupe.
type RangeTuple = (u32, u32, u32, u32);

/// Schlüssel für [`merge_diagnostics`] (Pfad, optionale Position, Nachricht).
type DedupeKey = (String, Option<RangeTuple>, String);

/// Diagnose unabhängig von Cargo- oder LSP-Format.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnifiedDiagnostic {
    /// Workspace-relativer oder absoluter Pfad (wie von Cargo/LSP geliefert).
    pub path: PathBuf,
    pub range: Option<Range>,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub source: DiagnosticSource,
    /// Optionaler rustc-Code, z. B. `E0425`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl UnifiedDiagnostic {
    fn dedupe_key(&self) -> DedupeKey {
        let r = self.range.map(|x| {
            (
                x.start.line,
                x.start.character,
                x.end.line,
                x.end.character,
            )
        });
        (
            self.path.to_string_lossy().into_owned(),
            r,
            self.message.clone(),
        )
    }
}

fn rustc_level_to_severity(level: &str) -> DiagnosticSeverity {
    match level {
        "error" | "error: internal compiler error" => DiagnosticSeverity::Error,
        "warning" => DiagnosticSeverity::Warning,
        "note" | "help" => DiagnosticSeverity::Hint,
        _ => DiagnosticSeverity::Unknown,
    }
}

fn lsp_severity_to_diag(s: Option<i32>) -> DiagnosticSeverity {
    match s {
        Some(1) => DiagnosticSeverity::Error,
        Some(2) => DiagnosticSeverity::Warning,
        Some(3) => DiagnosticSeverity::Information,
        Some(4) => DiagnosticSeverity::Hint,
        _ => DiagnosticSeverity::Unknown,
    }
}

// --- Cargo / rustc JSON (message-format=json) ---

#[derive(Debug, Deserialize)]
struct CargoJsonLine {
    reason: Option<String>,
    message: Option<RustcMessage>,
}

#[derive(Debug, Deserialize)]
struct RustcMessage {
    message: String,
    code: Option<RustcCode>,
    level: String,
    spans: Vec<RustcSpan>,
}

#[derive(Debug, Deserialize)]
struct RustcCode {
    code: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RustcSpan {
    file_name: String,
    line_start: usize,
    line_end: usize,
    column_start: usize,
    column_end: usize,
    is_primary: bool,
}

/// Parst **eine** Zeile `cargo`/`rustc` JSON-Ausgabe. Nicht jede Zeile ist eine Diagnose
/// (`build-finished`, `compiler-artifact`, …) — dann `None`.
pub fn parse_cargo_json_line(line: &str) -> Option<UnifiedDiagnostic> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let v: CargoJsonLine = serde_json::from_str(trimmed).ok()?;
    if v.reason.as_deref() != Some("compiler-message") {
        return None;
    }
    let msg = v.message?;
    if msg.spans.is_empty() {
        let path = PathBuf::from("unknown");
        return Some(UnifiedDiagnostic {
            path,
            range: None,
            severity: rustc_level_to_severity(&msg.level),
            message: msg.message,
            source: DiagnosticSource::Cargo,
            code: msg.code.and_then(|c| c.code),
        });
    }
    let primary = msg
        .spans
        .iter()
        .find(|s| s.is_primary)
        .or_else(|| msg.spans.first())?;
    let path = PathBuf::from(&primary.file_name);
    let range = Some(Range {
        start: Position {
            line: primary.line_start.saturating_sub(1) as u32,
            character: primary.column_start.saturating_sub(1) as u32,
        },
        end: Position {
            line: primary.line_end.saturating_sub(1) as u32,
            character: primary.column_end.saturating_sub(1) as u32,
        },
    });
    Some(UnifiedDiagnostic {
        path,
        range,
        severity: rustc_level_to_severity(&msg.level),
        message: msg.message,
        source: DiagnosticSource::Cargo,
        code: msg.code.and_then(|c| c.code),
    })
}

/// Parst alle Zeilen (typisch: stdout von `cargo check --message-format=json`).
pub fn parse_cargo_json_stream(text: &str) -> Vec<UnifiedDiagnostic> {
    text.lines()
        .filter_map(parse_cargo_json_line)
        .collect()
}

// --- LSP subset ---

#[derive(Debug, Deserialize)]
struct LspPosition {
    line: u32,
    character: u32,
}

#[derive(Debug, Deserialize)]
struct LspRange {
    start: LspPosition,
    end: LspPosition,
}

#[derive(Debug, Deserialize)]
struct LspDiagnosticRaw {
    range: LspRange,
    #[serde(default)]
    severity: Option<i32>,
    message: String,
    #[serde(default)]
    code: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct PublishDiagnosticsParams {
    #[serde(default)]
    uri: Option<String>,
    diagnostics: Vec<LspDiagnosticRaw>,
}

fn uri_to_path(uri: &str) -> PathBuf {
    const PREFIX: &str = "file://";
    if let Some(rest) = uri.strip_prefix(PREFIX) {
        // `file:///C:/proj/...` → `/C:/proj/...` — führendes `/` vor Laufwerksbuchstabe entfernen.
        let trimmed = rest.trim_start_matches('/');
        if trimmed.len() >= 2 && trimmed.as_bytes()[1] == b':' {
            return PathBuf::from(trimmed);
        }
        return PathBuf::from(rest);
    }
    PathBuf::from(uri)
}

fn code_to_string(code: &Option<serde_json::Value>) -> Option<String> {
    code.as_ref().and_then(|v| {
        if let Some(s) = v.as_str() {
            return Some(s.to_string());
        }
        v.get("value").and_then(|x| x.as_str()).map(String::from)
    })
}

/// Eine einzelne LSP-`Diagnostic`-JSON (Subset).
pub fn parse_lsp_diagnostic_json(path_hint: Option<&Path>, raw: &str) -> Result<Vec<UnifiedDiagnostic>, serde_json::Error> {
    if let Ok(batch) = serde_json::from_str::<PublishDiagnosticsParams>(raw) {
        let base = batch
            .uri
            .as_deref()
            .map(uri_to_path)
            .or_else(|| path_hint.map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));
        return Ok(batch
            .diagnostics
            .into_iter()
            .map(|d| UnifiedDiagnostic {
                path: base.clone(),
                range: Some(Range {
                    start: Position {
                        line: d.range.start.line,
                        character: d.range.start.character,
                    },
                    end: Position {
                        line: d.range.end.line,
                        character: d.range.end.character,
                    },
                }),
                severity: lsp_severity_to_diag(d.severity),
                message: d.message,
                source: DiagnosticSource::Lsp,
                code: code_to_string(&d.code),
            })
            .collect());
    }
    let d: LspDiagnosticRaw = serde_json::from_str(raw)?;
    let path = path_hint
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    Ok(vec![UnifiedDiagnostic {
        path,
        range: Some(Range {
            start: Position {
                line: d.range.start.line,
                character: d.range.start.character,
            },
            end: Position {
                line: d.range.end.line,
                character: d.range.end.character,
            },
        }),
        severity: lsp_severity_to_diag(d.severity),
        message: d.message,
        source: DiagnosticSource::Lsp,
        code: code_to_string(&d.code),
    }])
}

/// Mehrere Diagnose-Listen zusammenführen, Duplikate entfernen (Pfad + Range + Nachricht).
pub fn merge_diagnostics(lists: &[Vec<UnifiedDiagnostic>]) -> Vec<UnifiedDiagnostic> {
    let mut seen: HashSet<DedupeKey> = HashSet::new();
    let mut out = Vec::new();
    for list in lists {
        for d in list {
            let key = d.dedupe_key();
            if seen.insert(key) {
                out.push(d.clone());
            }
        }
    }
    out
}

/// Kompakter Text für System- oder User-Nachrichten (LLM-Kontext).
pub fn format_for_prompt(diagnostics: &[UnifiedDiagnostic]) -> String {
    if diagnostics.is_empty() {
        return String::new();
    }
    let mut s = String::from("Diagnostics:\n");
    for d in diagnostics {
        let loc = match &d.range {
            Some(r) => format!(
                "{}:{}:{}",
                d.path.display(),
                r.start.line + 1,
                r.start.character + 1
            ),
            None => d.path.display().to_string(),
        };
        let sev = format!("{:?}", d.severity).to_lowercase();
        let src = format!("{:?}", d.source).to_lowercase();
        s.push_str(&format!(
            "- [{}] {} ({}) {}\n",
            sev, loc, src, d.message
        ));
        if let Some(ref c) = d.code {
            s.push_str(&format!("  code: {}\n", c));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_line_compiler_message() {
        let line = r#"{"reason":"compiler-message","package_id":"path+file:///tmp/p#0.1.0","message":{"message":"cannot find value `x` in this scope","code":{"code":"E0425","explanation":null},"level":"error","spans":[{"file_name":"src/lib.rs","byte_start":0,"byte_end":1,"line_start":3,"line_end":3,"column_start":9,"column_end":10,"is_primary":true,"text":[],"label":null,"suggested_replacement":null,"suggestion_applicability":null,"expansion":null}]}}"#;
        let d = parse_cargo_json_line(line).expect("parse");
        assert_eq!(d.severity, DiagnosticSeverity::Error);
        assert_eq!(d.source, DiagnosticSource::Cargo);
        assert!(d.path.to_string_lossy().contains("lib.rs"));
        assert_eq!(d.code.as_deref(), Some("E0425"));
    }

    fn merge_lists() -> Vec<UnifiedDiagnostic> {
        let a = vec![UnifiedDiagnostic {
            path: PathBuf::from("a.rs"),
            range: Some(Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 1,
                },
            }),
            severity: DiagnosticSeverity::Error,
            message: "m".into(),
            source: DiagnosticSource::Cargo,
            code: None,
        }];
        let b = vec![a[0].clone()];
        merge_diagnostics(&[a, b])
    }

    #[test]
    fn merge_dedupes() {
        let m = merge_lists();
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn lsp_batch() {
        let raw = r#"{"uri":"file:///C:/proj/src/x.rs","diagnostics":[{"range":{"start":{"line":1,"character":2},"end":{"line":1,"character":5}},"severity":1,"message":"oops"}]}"#;
        let v = parse_lsp_diagnostic_json(None, raw).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].source, DiagnosticSource::Lsp);
        assert_eq!(v[0].severity, DiagnosticSeverity::Error);
    }

    #[test]
    fn format_nonempty() {
        let d = UnifiedDiagnostic {
            path: PathBuf::from("f.rs"),
            range: Some(Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 1,
                },
            }),
            severity: DiagnosticSeverity::Warning,
            message: "w".into(),
            source: DiagnosticSource::Lsp,
            code: Some("W1".into()),
        };
        let t = format_for_prompt(&[d]);
        assert!(t.contains("Diagnostics:"));
        assert!(t.contains("w"));
        assert!(t.contains("W1"));
    }
}
