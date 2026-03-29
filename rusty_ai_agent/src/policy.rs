//! Path/binary allowlist for tool execution (Phase 0).

use std::collections::HashSet;
use std::path::{Component, Path};

use crate::ToolInvocation;

/// Rejects `..`, restricts relative paths to optional prefixes, and allowlists `run_cmd` binaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowlistPolicy {
    path_prefixes: Vec<String>,
    allowed_bins: HashSet<String>,
}

impl AllowlistPolicy {
    pub fn new(
        path_prefixes: impl IntoIterator<Item = impl Into<String>>,
        allowed_bins: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            path_prefixes: path_prefixes.into_iter().map(Into::into).collect(),
            allowed_bins: allowed_bins.into_iter().map(Into::into).collect(),
        }
    }

    pub fn validate(&self, inv: &ToolInvocation) -> Result<(), String> {
        match inv {
            ToolInvocation::ReadFile { path }
            | ToolInvocation::WriteFile { path, .. }
            | ToolInvocation::SearchReplace { path, .. } => {
                self.check_path(path)
            }
            ToolInvocation::RunCmd { argv, cwd } => {
                let bin = argv.first().map(String::as_str).unwrap_or("");
                if !self.allowed_bins.contains(bin) {
                    return Err(format!("binary not allowlisted: {bin:?}"));
                }
                if let Some(d) = cwd {
                    self.check_path(d)?;
                }
                Ok(())
            }
        }
    }

    fn check_path(&self, path: &str) -> Result<(), String> {
        if Path::new(path)
            .components()
            .any(|c| c == Component::ParentDir)
        {
            return Err("path must not contain `..`".into());
        }
        if !self.path_prefixes.is_empty() {
            let ok = self.path_prefixes.iter().any(|prefix| {
                path == prefix.as_str() || path.starts_with(&format!("{prefix}/"))
            });
            if !ok {
                return Err(format!(
                    "path must start with one of {:?}, got {path:?}",
                    self.path_prefixes
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_parent_dir() {
        let p = AllowlistPolicy::new(["src"], Vec::<&str>::new());
        let inv = ToolInvocation::ReadFile {
            path: "../x".into(),
        };
        assert!(p.validate(&inv).is_err());
    }
}
