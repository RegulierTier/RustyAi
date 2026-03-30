//! Sichere `argv`-Erzeugung für `cargo test` (Phase 2): Paket, Filter, `--exact` — ohne Shell.

/// Argumente für [`crate::ToolInvocation::RunCmd`] mit `argv[0] == "cargo"`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CargoTestInvocation {
    argv: Vec<String>,
}

impl CargoTestInvocation {
    /// Baut `cargo test …` mit optionaler Paketauswahl und Testfiltern (nach `--`).
    ///
    /// - `package`: `Some("crate_name")` → `-p crate_name`
    /// - `filters`: Strings nach dem `--`-Separator (Rust-Testnamenfilter)
    /// - `exact`: setzt `--exact` vor den Filtern
    ///
    /// **Validierung:** Paketnamen und Filter dürfen kein `;`, keine Zeilenumbrüche und kein `--` enthalten
    /// (verhindert naive Injektion über Modell-Output).
    pub fn new(
        package: Option<&str>,
        filters: &[&str],
        exact: bool,
    ) -> Result<Self, CargoTestArgvError> {
        if let Some(p) = package {
            validate_token(p, "package")?;
        }
        for f in filters {
            validate_token(f, "filter")?;
        }

        let mut argv = vec!["cargo".to_string(), "test".to_string()];
        if let Some(p) = package {
            argv.push("-p".into());
            argv.push(p.to_string());
        }
        if exact {
            argv.push("--exact".into());
        }
        if !filters.is_empty() {
            argv.push("--".into());
            for f in filters {
                argv.push((*f).to_string());
            }
        }
        Ok(Self { argv })
    }

    /// Fertiges `argv` (erstes Element `"cargo"`).
    pub fn argv(&self) -> &[String] {
        &self.argv
    }

    /// Besitzübertragung als `Vec<String>`.
    pub fn into_argv(self) -> Vec<String> {
        self.argv
    }
}

/// Ungültiges Token für [`CargoTestInvocation::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CargoTestArgvError(pub String);

impl std::fmt::Display for CargoTestArgvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CargoTestArgvError {}

fn validate_token(s: &str, ctx: &str) -> Result<(), CargoTestArgvError> {
    if s.is_empty() {
        return Err(CargoTestArgvError(format!("{ctx}: empty string")));
    }
    if s.contains(';') || s.contains('\n') || s.contains('\r') {
        return Err(CargoTestArgvError(format!(
            "{ctx}: forbidden character in {s:?}"
        )));
    }
    if s.contains("--") {
        return Err(CargoTestArgvError(format!(
            "{ctx}: must not contain `--` (got {s:?})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_and_filter() {
        let i = CargoTestInvocation::new(Some("rusty_ai_agent"), &["diagnostics::"], false).unwrap();
        assert_eq!(
            i.argv(),
            &[
                "cargo".to_string(),
                "test".to_string(),
                "-p".to_string(),
                "rusty_ai_agent".to_string(),
                "--".to_string(),
                "diagnostics::".to_string(),
            ]
        );
    }

    #[test]
    fn rejects_double_dash_in_filter() {
        assert!(CargoTestInvocation::new(None, &["a--b"], false).is_err());
    }
}
