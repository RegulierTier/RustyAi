# `rusty_ai_agent`

**Agent-/IDE-Protokoll** für den RustyAi-Workspace: austauschbares [`LlmBackend`](src/llm_backend.rs), Chat-/Completion-Typen und ein **JSON-kompatibles Tool-Protokoll** (`read_file`, `write_file`, `run_cmd`, `search_replace`) inkl. [JSON Schema](schemas/tool_invocation.json).

| Dokument | Inhalt |
| -------- | ------ |
| **[`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`](../docs/ARCHITEKTUR_IDE_ROADMAP_B.md)** | Pfad B: Zielarchitektur, Phasen-Roadmap |
| **[`docs/HANDBUCH.md`](../docs/HANDBUCH.md)** | Abschnitt **2.8** (`rusty_ai_agent`), **3.4** (Ablauf Orchestrierung) |
| **[`SECURITY.md`](SECURITY.md)** | Bedrohungsmodell, Allowlist, Checkliste |
| **[`prompts/v1/`](prompts/v1/)** | Versionierte System-Prompts (Analyse / Migration / Fix), `manifest.json` |
| **[`schemas/`](schemas/)** | JSON-Schema: [`tool_invocation.json`](schemas/tool_invocation.json), [`lsp_diagnostics_export.json`](schemas/lsp_diagnostics_export.json) |

Ohne Features führt dieses Crate **kein** HTTP und kein Dateisystem aus — nur Typen, Trait und Hilfsfunktionen. Ausführung und Policies liegen im Orchestrierungs- oder Produkt-Code (oder hinter den optionalen Features unten).

---

## Features (`Cargo.toml`)

| Feature | Wirkung |
| ------- | ------- |
| *(default)* | Nur Typen, Parsing, Policy, Orchestrierung-Hilfen |
| **`real-exec`** | [`RealExecutor`](src/executor.rs): echtes `std::fs` / `std::process` nach Policy-Check |
| **`http`** | [`OpenAiCompatBackend`](src/openai_compat.rs): `POST …/chat/completions` (blocking `reqwest`) |

Zusätzliches Crate **[`rusty_ai_workspace`](../rusty_ai_workspace/README.md)** (Workspace-Member): Index + Suche; Feature **`embeddings`** dort für HTTP-Embeddings.

```bash
cargo test -p rusty_ai_agent
cargo test -p rusty_ai_agent --all-features   # inkl. http + real-exec
```

---

## Kern-API (Auswahl)

| Bereich | Wichtige Symbole |
| ------- | ---------------- |
| LLM | [`LlmBackend`](src/llm_backend.rs), `CompletionRequest`, `CompletionResponse`, `ModelToolCall`, `ToolDefinition` |
| Tools | [`ToolInvocation`](src/tools.rs), [`names`](src/tools.rs) (`read_file`, …) |
| Parsing | [`tool_invocations_from_model_calls`](src/tool_parse.rs), [`parse_json_arguments_loose`](src/tool_parse.rs), [`tool_invocations_try_each`](src/tool_parse.rs) |
| Policy | [`AllowlistPolicy`](src/policy.rs) |
| Orchestrierung | [`complete_with_tool_parse_retries`](src/orchestrator.rs) |
| Fallback | [`FallbackBackend`](src/fallback_backend.rs) |
| Telemetrie (lokal) | [`LocalTelemetry`](src/telemetry.rs), [`TimedBackend`](src/telemetry.rs) |
| Vorschau | [`format_replace_preview`](src/diff_preview.rs) |
| Diagnosen (Phase 2) | [`parse_cargo_json_stream`](src/diagnostics.rs), [`parse_lsp_diagnostic_json`](src/diagnostics.rs), [`merge_diagnostics`](src/diagnostics.rs), [`format_for_prompt`](src/diagnostics.rs) |
| Prompts (Phase 2) | [`render_embedded`](src/prompts.rs), [`PromptKind`](src/prompts.rs) — Vorlagen [`prompts/v1/`](prompts/v1/) |
| Tests (Phase 2) | [`CargoTestInvocation`](src/cargo_test.rs) |
| HTTP (Feature) | [`OpenAiCompatBackend`](src/openai_compat.rs), [`OpenAiChatConfig`](src/openai_compat.rs), `complete_stream` / `complete_stream_text` |

---

## Beispiele

| Beispiel | Befehl | Features |
| -------- | ------ | -------- |
| **agent_demo** | `cargo run -p rusty_ai_agent --example agent_demo` | — |
|  | `cargo run -p rusty_ai_agent --example agent_demo --features real-exec -- --real` | `real-exec` |
| **agent_retry_demo** | `cargo run -p rusty_ai_agent --example agent_retry_demo` | — |
| **dual_backend_demo** | `cargo run -p rusty_ai_agent --example dual_backend_demo` | — |
| **telemetry_demo** | `cargo run -p rusty_ai_agent --example telemetry_demo` | — |
| **openai_smoke** | `cargo run -p rusty_ai_agent --example openai_smoke --features http` | `http` |
| **openai_stream** | `cargo run -p rusty_ai_agent --example openai_stream --features http` | `http` |
| **cargo_test_demo** | `cargo run -p rusty_ai_agent --example cargo_test_demo` | — |

**HTTP:** Cloud: `OPENAI_API_KEY` setzen; optional `OPENAI_BASE_URL`. **Ollama:** `cargo run … openai_smoke --features http -- --ollama` (ohne Key). Siehe [`examples/openai_smoke.rs`](examples/openai_smoke.rs).

**Workspace-Index:** `cargo run -p rusty_ai_workspace --example workspace_index_demo` (siehe [`rusty_ai_workspace`](../rusty_ai_workspace/README.md)).

---

## Sicherheit (Kurz)

- **`read_file` / `write_file` / `search_replace`:** Pfade über Policy auf den Workspace begrenzen; keine unkontrollierten `..`-Pfade.
- **`run_cmd`:** Nur allowlistete Binaries; siehe **[SECURITY.md](SECURITY.md)**.

---

## `rustdoc`

```bash
cargo doc -p rusty_ai_agent --all-features --no-deps --open
```
