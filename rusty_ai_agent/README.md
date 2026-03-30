# `rusty_ai_agent`

**Agent-/IDE-Protokoll** für den RustyAi-Workspace: austauschbares [`LlmBackend`](src/core/llm_backend.rs), Chat-/Completion-Typen und ein **JSON-kompatibles Tool-Protokoll** (`read_file`, `write_file`, `run_cmd`, `search_replace`) inkl. [JSON Schema](schemas/tool_invocation.json).

| Dokument | Inhalt |
| -------- | ------ |
| **[`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`](../docs/ARCHITEKTUR_IDE_ROADMAP_B.md)** | Pfad B: Zielarchitektur, Phasen-Roadmap |
| **[`docs/HANDBUCH.md`](../docs/HANDBUCH.md)** | Abschnitt **2.8** (`rusty_ai_agent`), **3.4** (Ablauf Orchestrierung) |
| **[`SECURITY.md`](SECURITY.md)** | Bedrohungsmodell, Allowlist, Checkliste |
| **[`prompts/v1/`](prompts/v1/)** | Versionierte System-Prompts (Analyse / Migration / Fix), `manifest.json` |
| **[`schemas/`](schemas/)** | JSON-Schema: [`tool_invocation.json`](schemas/tool_invocation.json), [`lsp_diagnostics_export.json`](schemas/lsp_diagnostics_export.json), [`policy_catalog.example.json`](schemas/policy_catalog.example.json) |

Ohne Features führt dieses Crate **kein** HTTP und kein Dateisystem aus — nur Typen, Trait und Hilfsfunktionen. Ausführung und Policies liegen im Orchestrierungs- oder Produkt-Code (oder hinter den optionalen Features unten).

**Hinweis:** Unter `src/` liegt nur **Rust**-Quelltext dieses Crates. Fremde Skripte (z. B. Python) gehören nicht hierher — bei Bedarf z. B. `contrib/` im Repo oder ein separates Tool-Repository.

---

## Features (`Cargo.toml`)

| Feature | Wirkung |
| ------- | ------- |
| *(default)* | Nur Typen, Parsing, Policy, Orchestrierung-Hilfen |
| **`real-exec`** | [`RealExecutor`](src/execution/executor.rs): echtes `std::fs` / `std::process` nach Policy-Check |
| **`http`** | [`OpenAiCompatBackend`](src/http/openai_compat.rs): `POST …/chat/completions` (blocking `reqwest`) |
| **`minigpt`** | [`MiniGptTinyBackend`](src/minigpt_backend.rs): [`LlmBackend`](src/core/llm_backend.rs) über [`rusty_ai_llm`](../rusty_ai_llm/README.md) ([`MiniGpt`](../rusty_ai_llm/src/model/minigpt.rs), Byte-Tokenizer). **Einschränkungen:** keine Tool-Calls — `CompletionRequest.tools` muss leer sein. **`top_p` / `temperature`:** optional in [`CompletionRequest`](src/core/llm_backend.rs), sonst Backend-Default. **`stop_sequences`:** Substring-Abbruch auf dekodiertem Assist-Text (kein Token-Grenzen-Stopp). **`usage`:** `prompt_tokens` = Byte-Tokenizer-Länge des Prompts; `completion_tokens` = Anzahl gesampelter neuer Token-IDs; `total_tokens` = Summe (bei Stop kann `completion_tokens` größer sein als die Länge des gekürzten `content`). Checkpoint: `MiniGptTinyBackend::from_checkpoint_dir` oder `with_defaults` mit geladenem Modell. |

Zusätzliches Crate **[`rusty_ai_workspace`](../rusty_ai_workspace/README.md)** (Workspace-Member): Index + Suche; Feature **`embeddings`** dort für HTTP-Embeddings.

```bash
cargo test -p rusty_ai_agent
cargo test -p rusty_ai_agent --all-features   # inkl. http + real-exec + minigpt
```

---

## Kern-API (Auswahl)

| Bereich | Wichtige Symbole |
| ------- | ---------------- |
| LLM | [`LlmBackend`](src/core/llm_backend.rs), `CompletionRequest`, `CompletionResponse`, `ModelToolCall`, `ToolDefinition` |
| Tools | [`ToolInvocation`](src/tools/invocation.rs), [`names`](src/tools/invocation.rs) (`read_file`, …) |
| Parsing | [`tool_invocations_from_model_calls`](src/tools/parse.rs), [`parse_json_arguments_loose`](src/tools/parse.rs), [`tool_invocations_try_each`](src/tools/parse.rs) |
| Policy | [`AllowlistPolicy`](src/policy/allowlist.rs) |
| Orchestrierung | [`complete_with_tool_parse_retries`](src/execution/orchestrator.rs) |
| Fallback | [`FallbackBackend`](src/execution/fallback_backend.rs) |
| Lokales Tiny-LLM (Feature **`minigpt`**) | [`MiniGptTinyBackend`](src/minigpt_backend.rs) |
| Telemetrie (lokal) | [`LocalTelemetry`](src/telemetry/mod.rs), [`TimedBackend`](src/telemetry/mod.rs) |
| Vorschau / Text kürzen | [`format_replace_preview`](src/tools/diff_preview.rs), [`truncate_middle`](src/tools/diff_preview.rs), [`truncate_utf8_prefix`](src/tools/diff_preview.rs) (UTF-8-sicheres Präfix; u. a. für Log-Auszüge) |
| Diagnosen (Phase 2) | [`parse_cargo_json_stream`](src/feedback/diagnostics.rs), [`parse_lsp_diagnostic_json`](src/feedback/diagnostics.rs), [`merge_diagnostics`](src/feedback/diagnostics.rs), [`format_for_prompt`](src/feedback/diagnostics.rs) |
| Prompts (Phase 2) | [`render_embedded`](src/feedback/prompts.rs), [`PromptKind`](src/feedback/prompts.rs) — Vorlagen [`prompts/v1/`](prompts/v1/) |
| Tests (Phase 2) | [`CargoTestInvocation`](src/feedback/cargo_test.rs) |
| Policies (Phase 3) | [`PolicyCatalog`](src/policy/catalog.rs), `RUSTY_AI_AGENT_POLICY`, `AllowlistPolicy::preset_dev` / `preset_ci` |
| Batch-Report (Phase 3) | [`BatchReport`](src/batch/batch_report.rs), [`BatchStepRecord`](src/batch/batch_report.rs) |
| Budget (Phase 3) | [`BudgetLlmBackend`](src/batch/budget.rs), [`CompletionUsage`](src/core/llm_backend.rs) |
| HTTP (Feature) | [`OpenAiCompatBackend`](src/http/openai_compat.rs), [`OpenAiChatConfig`](src/http/openai_compat.rs), `complete_stream` / `complete_stream_text` (SSE; Fehlertexte werden UTF-8-sicher gekürzt) |

---

## Beispiele

| Beispiel | Befehl | Features |
| -------- | ------ | -------- |
| **agent_demo** | `cargo run -p rusty_ai_agent --example agent_demo` | — |
|  | `cargo run -p rusty_ai_agent --example agent_demo --features real-exec -- --real` | `real-exec` |
| **agent_retry_demo** | `cargo run -p rusty_ai_agent --example agent_retry_demo` | — |
| **dual_backend_demo** | `cargo run -p rusty_ai_agent --example dual_backend_demo` | — |
| **tiny_llm_fallback_demo** | `cargo run -p rusty_ai_agent --example tiny_llm_fallback_demo --features minigpt` | `minigpt` |
| **telemetry_demo** | `cargo run -p rusty_ai_agent --example telemetry_demo` | — |
| **openai_smoke** | `cargo run -p rusty_ai_agent --example openai_smoke --features http` | `http` |
| **openai_stream** | `cargo run -p rusty_ai_agent --example openai_stream --features http` | `http` |
| **cargo_test_demo** | `cargo run -p rusty_ai_agent --example cargo_test_demo` | — |
| **batch_report_demo** | `cargo run -p rusty_ai_agent --example batch_report_demo` | — |

**HTTP:** Cloud: `OPENAI_API_KEY` setzen; optional `OPENAI_BASE_URL`. **Ollama:** `cargo run … openai_smoke --features http -- --ollama` (ohne Key). Siehe [`examples/openai_smoke.rs`](examples/openai_smoke.rs).

**Workspace-Index:** `cargo run -p rusty_ai_workspace --example workspace_index_demo` (siehe [`rusty_ai_workspace`](../rusty_ai_workspace/README.md)).

---

## Phase 3: Betrieb und CI

| Thema | Kurz |
| ----- | ---- |
| **Policies pro Umgebung** | [`PolicyCatalog`](src/policy/catalog.rs) mappt Namen (`dev`, `ci`, …) auf [`AllowlistPolicy`](src/policy/allowlist.rs). Auswahl: Umgebungsvariable **`RUSTY_AI_AGENT_POLICY`** (Standard `dev`). Voreinstellungen: [`AllowlistPolicy::preset_dev`](src/policy/allowlist.rs) / [`preset_ci`](src/policy/allowlist.rs) — **keine automatische CI-Erkennung**; explizit setzen. Eigene Einträge per JSON: [`schemas/policy_catalog.example.json`](schemas/policy_catalog.example.json), Loader [`PolicyCatalog::from_json_merging_builtin`](src/policy/catalog.rs). |
| **Batch-/Nightly-Reports** | [`BatchReport`](src/batch/batch_report.rs) serialisiert Schritte (LLM, Tool, Check) als JSON; optional Markdown für Lesbarkeit. Kein interaktives Terminal — nur Datenmodell für Orchestrierung/CI. Demo: **`batch_report_demo`**. |
| **Kosten-Limits** | Mit Feature **`http`**: API-Antworten können [`CompletionUsage`](src/core/llm_backend.rs) enthalten; [`BudgetLlmBackend`](src/batch/budget.rs) wrappt ein [`LlmBackend`](src/core/llm_backend.rs) und bricht bei überschrittenen `max_total_tokens` / `max_complete_calls` ab. [`LocalTelemetry`](src/telemetry/mod.rs) kann gemeldete Tokens spiegeln (`total_tokens_reported`). |

**Sicherheit:** Policy-Namen ersetzen **keine** Sandbox — siehe **[SECURITY.md](SECURITY.md)**.

---

## Sicherheit (Kurz)

- **`read_file` / `write_file` / `search_replace`:** Pfade über Policy auf den Workspace begrenzen; keine unkontrollierten `..`-Pfade.
- **`run_cmd`:** Nur allowlistete Binaries; siehe **[SECURITY.md](SECURITY.md)**.

---

## `rustdoc`

```bash
cargo doc -p rusty_ai_agent --all-features --no-deps --open
```
