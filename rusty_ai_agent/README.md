# `rusty_ai_agent`

**Phase 0** des IDE-Roadmaps (siehe [`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`](../docs/ARCHITEKTUR_IDE_ROADMAP_B.md)): austauschbares [`LlmBackend`](src/llm_backend.rs), Chat-/Completion-Typen und ein **JSON-kompatibles Tool-Protokoll** (`read_file`, `write_file`, `run_cmd`) inkl. [JSON Schema](schemas/tool_invocation.json).

Dieses Crate führt **keine** HTTP- oder Dateisystem-Calls aus — nur Typen, Trait und Parsing-Hilfen. Ausführung und Policies liegen im Orchestrierungs- oder Produkt-Code.

## Sicherheit (für den Executor)

- **`read_file` / `write_file`:** Pfade auf den Workspace (oder eine explizite Allowlist) begrenzen; keine absoluten Pfade nach außen ohne Policy.
- **`run_cmd`:** Nur erlaubte Binaries (z. B. `cargo`, `rustc`) und feste Argument-Muster; kein freies Shell-Metazeichen; Timeout und Max-Ausgabelänge setzen.
- **Phase-0-Hinweis:** Das JSON-Schema beschreibt nur die **Form** der Aufrufe — nicht, ob sie ausgeführt werden dürfen.
