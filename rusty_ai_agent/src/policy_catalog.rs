//! Benannte [`AllowlistPolicy`]-Einträge und Auswahl per Umgebungsvariable (Phase 3).

use std::collections::HashMap;
use std::env;

use serde::Deserialize;

use crate::AllowlistPolicy;

/// Umgebungsvariable für die aktive Policy: `dev`, `ci`, oder ein benutzerdefinierter Name aus einem geladenen Katalog.
pub const ENV_POLICY: &str = "RUSTY_AI_AGENT_POLICY";

#[derive(Debug, Deserialize)]
struct PolicyCatalogFile {
    #[serde(default = "default_policy_name")]
    default: String,
    #[serde(default)]
    policies: HashMap<String, AllowlistPolicy>,
}

fn default_policy_name() -> String {
    "dev".into()
}

/// Mehrere benannte Policies; genau eine ist „aktiv“.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyCatalog {
    policies: HashMap<String, AllowlistPolicy>,
    default_name: String,
    active_name: String,
}

impl PolicyCatalog {
    /// `dev` und `ci` Presets; aktiv = `RUSTY_AI_AGENT_POLICY` oder `dev`.
    pub fn with_builtin_presets() -> Self {
        let mut policies = HashMap::new();
        policies.insert("dev".into(), AllowlistPolicy::preset_dev());
        policies.insert("ci".into(), AllowlistPolicy::preset_ci());
        let default_name: String = "dev".into();
        let active_name = env::var(ENV_POLICY).unwrap_or_else(|_| default_name.clone());
        Self {
            policies,
            default_name,
            active_name,
        }
    }

    /// Builtin `dev`/`ci` plus Einträge aus JSON (überschreibt gleiche Namen). Aktiv: Env, sonst `default` aus der Datei, sonst `dev`.
    pub fn from_json_merging_builtin(json: &str) -> Result<Self, String> {
        let file: PolicyCatalogFile =
            serde_json::from_str(json).map_err(|e| format!("policy catalog JSON: {e}"))?;
        let mut policies = HashMap::new();
        policies.insert("dev".into(), AllowlistPolicy::preset_dev());
        policies.insert("ci".into(), AllowlistPolicy::preset_ci());
        for (k, v) in file.policies {
            policies.insert(k, v);
        }
        let default_name = if file.default.is_empty() {
            "dev".into()
        } else {
            file.default.clone()
        };
        let active_name = env::var(ENV_POLICY).unwrap_or_else(|_| default_name.clone());
        Ok(Self {
            policies,
            default_name,
            active_name,
        })
    }

    /// Aktiver Policy-Name.
    pub fn active_name(&self) -> &str {
        &self.active_name
    }

    /// Vorgabe-Name (für Dokumentation / Fallback).
    pub fn default_name(&self) -> &str {
        &self.default_name
    }

    /// Aktive Policy; Fehler wenn unbekannter Name.
    pub fn active_policy(&self) -> Result<&AllowlistPolicy, String> {
        self.get(&self.active_name)
    }

    /// Policy nach Namen.
    pub fn get(&self, name: &str) -> Result<&AllowlistPolicy, String> {
        self.policies.get(name).ok_or_else(|| {
            format!(
                "unknown policy {name:?}; known: {:?}",
                self.policies.keys().collect::<Vec<_>>()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_dev_ci() {
        let c = PolicyCatalog::with_builtin_presets();
        assert!(c.get("dev").is_ok());
        assert!(c.get("ci").is_ok());
        assert!(c.active_policy().is_ok());
    }

    #[test]
    fn json_adds_strict() {
        let j = r#"{"default":"ci","policies":{"strict":{"path_prefixes":["x"],"allowed_bins":["cargo"]}}}"#;
        let c = PolicyCatalog::from_json_merging_builtin(j).unwrap();
        assert!(c.get("strict").is_ok());
    }
}
