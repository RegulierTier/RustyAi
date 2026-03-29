//! Real filesystem and subprocess execution (feature **`real-exec`** only).

use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use crate::policy::AllowlistPolicy;
use crate::ToolExecutionResult;
use crate::ToolInvocation;

/// Failure while resolving paths or running commands.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("path escapes workspace root")]
    PathEscape,
    #[error("policy: {0}")]
    Policy(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("search_replace: old_string not found in file")]
    OldStringNotFound,
}

/// Executes [`ToolInvocation`] under a workspace root after [`AllowlistPolicy::validate`].
#[derive(Clone, Debug)]
pub struct RealExecutor {
    root: PathBuf,
}

impl RealExecutor {
    /// `root` is canonicalized; all relative tool paths are resolved under it.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, ExecutorError> {
        let root = root.as_ref().canonicalize().map_err(ExecutorError::Io)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve `rel` under [`Self::root`] without allowing `..` or absolute roots to escape.
    pub fn join_under_root(&self, rel: &str) -> Result<PathBuf, ExecutorError> {
        join_under_root(&self.root, rel)
    }

    /// Run after policy validation.
    pub fn execute(
        &self,
        policy: &AllowlistPolicy,
        inv: &ToolInvocation,
    ) -> Result<ToolExecutionResult, ExecutorError> {
        policy.validate(inv).map_err(ExecutorError::Policy)?;
        match inv {
            ToolInvocation::ReadFile { path } => {
                let full = self.join_under_root(path)?;
                let text = std::fs::read_to_string(&full)?;
                Ok(ToolExecutionResult::success(text))
            }
            ToolInvocation::WriteFile { path, content } => {
                let full = self.join_under_root(path)?;
                if let Some(parent) = full.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&full, content.as_bytes())?;
                Ok(ToolExecutionResult::success(format!(
                    "wrote {} bytes to {}",
                    content.len(),
                    full.display()
                )))
            }
            ToolInvocation::RunCmd { argv, cwd } => {
                if argv.is_empty() {
                    return Err(ExecutorError::Policy(
                        "run_cmd: argv must not be empty".into(),
                    ));
                }
                let work_dir = match cwd.as_deref() {
                    Some(d) => self.join_under_root(d)?,
                    None => self.root.clone(),
                };
                let mut cmd = Command::new(&argv[0]);
                if argv.len() > 1 {
                    cmd.args(&argv[1..]);
                }
                cmd.current_dir(work_dir);
                let out = cmd.output()?;
                let ok = out.status.success();
                let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                Ok(ToolExecutionResult {
                    ok,
                    stdout,
                    stderr,
                    exit_code: out.status.code(),
                })
            }
            ToolInvocation::SearchReplace {
                path,
                old_string,
                new_string,
            } => {
                let full = self.join_under_root(path)?;
                let text = std::fs::read_to_string(&full)?;
                if !text.contains(old_string.as_str()) {
                    return Err(ExecutorError::OldStringNotFound);
                }
                let updated = text.replacen(old_string, new_string, 1);
                std::fs::write(&full, updated.as_bytes())?;
                Ok(ToolExecutionResult::success(format!(
                    "replaced first match in {}",
                    full.display()
                )))
            }
        }
    }
}

fn join_under_root(root: &Path, rel: &str) -> Result<PathBuf, ExecutorError> {
    let root = root.canonicalize().map_err(ExecutorError::Io)?;
    let mut path = root.clone();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(p) => path.push(p),
            Component::CurDir => {}
            Component::ParentDir => return Err(ExecutorError::PathEscape),
            Component::Prefix(_) | Component::RootDir => return Err(ExecutorError::PathEscape),
        }
    }
    if !path.starts_with(&root) {
        return Err(ExecutorError::PathEscape);
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_cmd_empty_argv() {
        let tmp = std::env::temp_dir().join("rusty_ai_agent_empty_argv");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let ex = RealExecutor::new(&tmp).unwrap();
        let policy = AllowlistPolicy::new(vec!["."], vec![""]);
        let inv = ToolInvocation::RunCmd {
            argv: vec![],
            cwd: None,
        };
        let err = ex.execute(&policy, &inv).unwrap_err();
        assert!(err.to_string().contains("empty"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn join_rejects_dotdot() {
        let tmp = std::env::temp_dir();
        let root = tmp.join("rusty_ai_agent_exec_test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let ex = RealExecutor::new(&root).unwrap();
        assert!(ex.join_under_root("..").is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn search_replace_first_occurrence() {
        let tmp = std::env::temp_dir();
        let root = tmp.join("rusty_ai_agent_sr_test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("t")).unwrap();
        let rel = "t/a.txt";
        let path = root.join(rel);
        std::fs::write(&path, b"alpha beta alpha").unwrap();

        let ex = RealExecutor::new(&root).unwrap();
        let policy = AllowlistPolicy::new(["t"], Vec::<&str>::new());
        let inv = ToolInvocation::SearchReplace {
            path: rel.into(),
            old_string: "alpha".into(),
            new_string: "gamma".into(),
        };
        ex.execute(&policy, &inv).unwrap();
        let got = std::fs::read_to_string(&path).unwrap();
        assert_eq!(got, "gamma beta alpha");
        let _ = std::fs::remove_dir_all(&root);
    }
}
