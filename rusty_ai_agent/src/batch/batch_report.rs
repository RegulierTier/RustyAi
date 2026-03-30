//! Strukturierter **Batch-/CI-Report** ohne UI (Phase 3): Schritte, Erfolg, Kurzlogs — als JSON serialisierbar.

use serde::{Deserialize, Serialize};

/// Art eines protokollierten Schritts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStepKind {
    Llm,
    Tool,
    Check,
}

/// Ein Eintrag im Lauf (z. B. ein Tool oder ein Check).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchStepRecord {
    pub kind: BatchStepKind,
    pub label: String,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Gesamtbericht für einen nicht-interaktiven Lauf (Nightly, CI).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchReport {
    /// Aktive Policy, z. B. `ci` (siehe [`crate::PolicyCatalog`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    pub steps: Vec<BatchStepRecord>,
    /// `true`, wenn alle Schritte `ok` sind.
    pub overall_ok: bool,
}

impl BatchReport {
    pub fn from_steps(
        policy_name: Option<String>,
        git_sha: Option<String>,
        steps: Vec<BatchStepRecord>,
    ) -> Self {
        let overall_ok = steps.iter().all(|s| s.ok);
        Self {
            policy_name,
            git_sha,
            steps,
            overall_ok,
        }
    }

    /// JSON für Artefakte (`target/…`, CI-Upload).
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Minimaler Markdown-Text für Menschenlesbarkeit.
    pub fn to_markdown_summary(&self) -> String {
        let mut s = String::from("# Batch report\n\n");
        if let Some(ref p) = self.policy_name {
            s.push_str(&format!("- **policy:** {p}\n"));
        }
        if let Some(ref g) = self.git_sha {
            s.push_str(&format!("- **git:** {g}\n"));
        }
        s.push_str(&format!(
            "- **overall:** {}\n\n",
            if self.overall_ok { "ok" } else { "failed" }
        ));
        for step in &self.steps {
            let icon = if step.ok { "ok" } else { "FAIL" };
            s.push_str(&format!("- [{icon}] {:?} — {}\n", step.kind, step.label));
            if let Some(ref d) = step.detail {
                s.push_str(&format!(
                    "  ```\n  {}\n  ```\n",
                    d.lines().take(5).collect::<Vec<_>>().join("\n  ")
                ));
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overall_ok() {
        let r = BatchReport::from_steps(
            Some("ci".into()),
            None,
            vec![BatchStepRecord {
                kind: BatchStepKind::Check,
                label: "cargo check".into(),
                ok: true,
                detail: None,
            }],
        );
        assert!(r.overall_ok);
        assert!(r.to_json_string().unwrap().contains("ci"));
    }
}
