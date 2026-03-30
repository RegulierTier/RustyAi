# Dokumentation (RustyAi)

Hier liegen **menschenlesbare** Beschreibungen des Projekts. Die Rust-API bleibt maßgeblich in den Quelltext-Kommentaren und in `cargo doc`.

| Datei | Zweck |
| ----- | ----- |
| **[ARCHITEKTUR_IDE_ROADMAP_B.md](ARCHITEKTUR_IDE_ROADMAP_B.md)** | **Pfad B (IDE-nah):** Zielarchitektur (Orchestrierung, LLM-Backends, Tools, Feedback-Schleifen), Einordnung von RustyAi, **phasenweise Roadmap** (MVP → Kontext → Betrieb). |
| **[../rusty_ai_agent/README.md](../rusty_ai_agent/README.md)** | Crate **`rusty_ai_agent`:** Features (`http`, `real-exec`), Kern-API, Phase 2 (Diagnosen, Prompts, Tests), Phase 3 (`PolicyCatalog`, `BatchReport`, `BudgetLlmBackend`), **Beispiel-Tabelle**. |
| **[../rusty_ai_agent/SECURITY.md](../rusty_ai_agent/SECURITY.md)** | Sicherheit: Allowlist, `run_cmd`, `cargo test`, Checkliste (Pfad B). |
| **[../rusty_ai_workspace/README.md](../rusty_ai_workspace/README.md)** | Crate **`rusty_ai_workspace`:** Zeilen-Chunk-Index, Suche, `build_cached`, optionales Feature **`embeddings`** (inkl. `CachingEmbeddingClient`). |
| **[../rusty_ai_agent/prompts/v1/](../rusty_ai_agent/prompts/v1/)** | Versionierte System-Prompt-Vorlagen (Analyse / Migration / Fix) mit `manifest.json`. |
| **[../rusty_ai_agent/schemas/](../rusty_ai_agent/schemas/)** | JSON-Schemas: Tool-Invocations, LSP-Diagnose-Export, Beispiel [`policy_catalog.example.json`](../rusty_ai_agent/schemas/policy_catalog.example.json) (Policy-Katalog). |
| **[HANDBUCH.md](HANDBUCH.md)** | Zentrale Referenz: Architektur, alle Workspace-Crates (**2.8 `rusty_ai_agent`**, **2.9 `rusty_ai_workspace`**, **3.4 Agent-Orchestrierung**), typische Abläufe (MLP, Mini-GPT, **FIM**, **Sliding-Window** / `attention_window`, LLM KV-Cache, Agent), Checkpoints/GPT-2/Candle (Tier 1/2), Grenzen, Glossar. |
| **[plans/](plans/)** | Entwurfs- und Arbeitspläne (Markdown), z. B. FIM/Candle, Phase 4 — **nicht** normativ; maßgeblich bleiben Handbuch und `rustdoc`. |
| **[`../rusty_ai/README.md`](../rusty_ai/README.md)** | Meta-Crate **`rusty_ai`**: Re-Exports (`load_minigpt_checkpoint_bytes`, …), Features (`candle`, `hf-hub`, `gpt2-bpe`), **Beispieltabelle** (`train_mlp`, `train_mini_gpt`, `train_micro_checkpoint`, `mini_local_inference`), Verweise auf Handbuch und Root-README. |
| **[`../rusty_ai/assets/mini_local/README.md`](../rusty_ai/assets/mini_local/README.md)** | Kurzinfo zu den eingecheckten **Mini-Gewichten** (`config.json`, `model.safetensors`) und Aktualisierungsbefehlen. |
| **[BERICHT_PRÜFUNG.md](BERICHT_PRÜFUNG.md)** | Prüfbericht zur Scope-Erweiterung (Korrekturen, Tests, Grenzen). |
| **[../rusty_ai_backend_candle/README.md](../rusty_ai_backend_candle/README.md)** | Kurzbeschreibung des optionalen Candle-Crates (CPU/CUDA, FP8, All-Reduce-Referenz). |

**Einstieg im Repo:** das [README im Projektroot](../README.md) mit Schnellstart, Workspace-Übersicht, Beispielen (`train_mlp`, `train_mini_gpt`) und Kurzbeschreibung der LLM-Pipeline. **Crate-Hinweis LLM:** [`../rusty_ai_llm/README.md`](../rusty_ai_llm/README.md) (Features `hf-hub`, `gpt2-bpe`). **Phase 3 (Policies, CI-Reports, Budgets, Index-Cache):** [Agent-README — Phase 3: Betrieb und CI](../rusty_ai_agent/README.md#phase-3-betrieb-und-ci), [Handbuch](HANDBUCH.md) (Abschnitte **2.9** `rusty_ai_workspace`, **3.4** Agent-Orchestrierung). **Phase 4 (optional):** [Roadmap Phase 4](ARCHITEKTUR_IDE_ROADMAP_B.md), [Handbuch §2.5](HANDBUCH.md) (DPO/Fine-Tuning extern, `generate_from_ids_with_callback`, **FIM**, Kontextlimits). **Candle:** [Handbuch §2.6](HANDBUCH.md), [Crate-README](../rusty_ai_backend_candle/README.md).

**Checkpoints (Kurz):** Eigenes Format: Verzeichnis mit `config.json` (`model_type`: `rusty_ai_minigpt`) und `model.safetensors`. Optional **`attention_window`** in JSON (Sliding-Window; weglassen = unverändertes Verhalten). API: `rusty_ai_llm::save_minigpt_checkpoint` / `load_minigpt_checkpoint` / **`load_minigpt_checkpoint_bytes`** (für eingebettete Assets). GPT-2-HF: `load_minigpt_from_gpt2_safetensors`. **BPE wie HF-GPT-2:** Feature `gpt2-bpe`, `Gpt2Tokenizer` + `tokenizer.json` im Modellordner. Vom Hub nur mit `--features hf-hub` und `load_minigpt_from_hf` — erwartet **RustyAi**-Checkpoints, keine beliebigen GPT-2-Dateien.

**Lokales Mini-LM:** [`rusty_ai/assets/mini_local/`](../rusty_ai/assets/mini_local/); ausführlich im [Handbuch](HANDBUCH.md) (Abschnitte **2.5** und **3.3.1**) und im [Root-README](../README.md) unter „Beispiele“.

**LayerNorm (Kurz):** Reine letzte-Achsen-Normierung: `rusty_ai_nn::layer_norm` / `Variable::layer_norm`. Affine Parameter wie in PyTorch (`γ`, `β`): `layer_norm_affine` bzw. `Variable::layer_norm_affine`; `MiniGpt` bindet eigene γ/β pro Pre-Norm und vor dem LM-Head ein. Ausführlicher: [HANDBUCH.md](HANDBUCH.md) (Abschnitte 2.2–2.3 und `MiniGpt` unter 2.5).
