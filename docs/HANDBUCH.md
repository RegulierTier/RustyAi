# RustyAi — Handbuch

Dieses Handbuch beschreibt die **Architektur**, die **Module** des Workspaces und **typische Arbeitsabläufe**. Es richtet sich an Entwicklerinnen und Entwickler, die das Projekt erweitern oder einbinden möchten.

---

## 1. Überblick

### 1.1 Zielsetzung

RustyAi verbindet **klassisches überwachtes Lernen** (kleine MLPs, Optimierer) mit **LLM-Bausteinen** (Causal Attention, Decoder-Stack, Sampling) in einem einheitlichen Rust-Workspace. Die Implementierung legt Wert auf **Nachvollziehbarkeit**; der Kernpfad bleibt **CPU-Autograd**. Optional stehen **safetensors**-Checkpoints, **GPT-2-Import** und ein **Candle**-Backend (CPU/CUDA, FP8-Hilfen, Referenz-All-Reduce) zur Verfügung.

### 1.2 Architektur auf einen Blick

```text
rusty_ai_core        Tensor, Broadcasting, matmul, softmax, …
       │
       ├── rusty_ai_autograd    Variable, Graph, backward
       │            │
       │            └── rusty_ai_nn    Linear, GELU, LayerNorm
       │                        │
       │                        ├── rusty_ai_ml    Sgd, Adam, Batches
       │                        │
       │                        └── rusty_ai_llm   MiniGpt, TrainableMiniGpt, generate, checkpoints
       │
       ├── rusty_ai_backend_candle   (optional) Matmul, FP8, all_reduce_mean
       │
       └── rusty_ai (Meta-Crate, Re-Exports)

rusty_ai_agent       LlmBackend, Tool-Protokoll, Policy (Pfad B / IDE) — Abschnitt 2.8
rusty_ai_workspace   Workspace-Index (Chunks, Suche, optional Embeddings) — Abschnitt 2.9
```

Datenfluss beim **Training**: Eingabe → `Tensor` / `Variable` → Schichten → Loss → `backward` → Optimizer-Schritt auf Parameter-Tensoren.

Datenfluss bei **LLM-Inferenz**: Token-IDs → Einbettungen → Decoder-Blöcke → Logits → `sample_token` / `generate` (kein Autograd nötig). **LLM-Training** nutzt `TrainableMiniGpt` mit demselben Forward wie `MiniGpt::forward`, Loss z. B. `Variable::cross_entropy_next_token`.

### 1.3 IDE-nähe und Produkt um RustyAi herum

Dieses Handbuch beschreibt den **RustyAi-Workspace**. Für eine **produktionsnahe** Einbindung (Orchestrierung, austauschbare LLM-Backends, Tool-Loops, Compiler-Feedback) siehe **[ARCHITEKTUR_IDE_ROADMAP_B.md](ARCHITEKTUR_IDE_ROADMAP_B.md)** (Pfad B: Architektur und Roadmap). **Phase-0-Typen** (`LlmBackend`, Tool-JSON) liegen im Crate **`rusty_ai_agent`** ([README](../rusty_ai_agent/README.md)); optionales **HTTP** (OpenAI-kompatibel): Feature **`http`**, [`OpenAiCompatBackend`](../rusty_ai_agent/src/http/openai_compat.rs) mit **`complete_stream`** (SSE, inkl. Tool-Deltas); Hilfen: [`tool_invocations_try_each` / `tool_parse_retry_instruction`](../rusty_ai_agent/src/tools/parse.rs), [`format_replace_preview`](../rusty_ai_agent/src/tools/diff_preview.rs), [`complete_with_tool_parse_retries`](../rusty_ai_agent/src/execution/orchestrator.rs), [`FallbackBackend`](../rusty_ai_agent/src/execution/fallback_backend.rs), [`LocalTelemetry` / `TimedBackend`](../rusty_ai_agent/src/telemetry/mod.rs).

---

## 2. Crate-Referenz

### 2.1 `rusty_ai_core`

**Verantwortung:** Speicherlayout, Operationen, Fehlertypen.

- **`Tensor`:** Kontiguierter `f32`-Puffer, `Shape` als `Vec<usize>`, row-major (C-Ordnung).
- **`DType`:** `F32` für `Tensor`-Speicher. FP8 (E4M3) wird nicht im Kern-`Tensor` gehalten; siehe `rusty_ai_backend_candle::fp8`.
- **Broadcasting:** Wie üblich rechtsbündig ausgerichtet (`broadcast_shapes`).
- **Wichtige Operationen:**
  - Elementweise: `add`, `sub`, `mul`, `div` (mit Broadcast)
  - `matmul`: 2D oder Batch `(B,M,K) × (B,K,N)`
  - `transpose_2d`, `transpose_batched_last2`
  - `sum_axis_0` für Bias-Gradienten `(batch, n) → (1, n)`
  - `relu`, `softmax`, `log_softmax` (letzte Achse, numerisch stabil wo nötig)
  - `mse` (mittlerer quadratischer Fehler als Skalar-Tensor)
  - `sqrt` (elementweise)

**Fehler:** `ShapeError`, `TensorError` (u. a. `EmptyTensor`).

---

### 2.2 `rusty_ai_autograd`

**Verantwortung:** Dynamischer Rechengraph und Gradienten.

- **`Variable`:** Hält `Tensor`-Daten in `RefCell` (Optimierer können Gewichte **in-place** setzen), optional `grad`, sowie die Operation (`Op`).
- **Operationen mit Gradient (Auswahl):** `Add` (inkl. Broadcast; Gradient wird wie bei `Mul` auf die jeweilige Parameterform summiert), `BiasAdd` (Batch-Matrix + Zeilen-Bias `(1,n)`), `MatMul` (2D und Batch-3D), `Mul` (Broadcast), `Relu`, `Gelu`, `LayerNorm` (letzte Achse), `SoftmaxLastDim`, `Reshape`, `TransposeBatchedLast2`, `CausalMask` (Scores), **`SlidingCausalMask`** (Sliding-Window auf Scores), **`FimMask`** (FIM-Sichtbarkeit auf Attention-Scores, siehe `rusty_ai_llm::fim`), `EmbeddingGather`, `SplitHeads` / `MergeHeads`, `CrossEntropyNextToken` (Next-Token-LM über alle Positionen `t < seq-1`), **`CrossEntropyNextTokenSubset`** (Next-Token-CE nur an ausgewählten Zeitindizes `t` — für **FIM** mit `fim_middle_prediction_positions`), `Mse` (Ziel ist konstanter `Tensor`, kein Grad aufs Ziel). Affine LayerNorm ist **kein** eigenes `Op`, sondern die Komposition `layer_norm` → `mul` (γ) → `add` (β) über **`Variable::layer_norm_affine`**.
- **`backward(loss)`:** Setzt den eingehenden Gradienten auf den Skalar-Loss auf `1` und verteilt rückwärts.
- **Kontext:** `grad_enabled()`, `set_grad_enabled`, `no_grad(|| { ... })` — im Inferenzpfad keine Graphen erzeugen.

**Typischer Fehler:** Gleiche `Variable`-Knoten müssen zwischen Epochen mit `zero_grad()` geleert werden, bevor ein neuer Forward/Backward-Lauf startet.

---

### 2.3 `rusty_ai_nn`

**Verantwortung:** Baukasten für kleine Netze.

- **`Linear`:** Gewichte und Bias als `Rc<Variable>`; `forward` → `matmul` + `bias_add`.
- **Initialisierung:** `glorot_uniform`, `uniform`, `zeros_bias`, `ones_scale` (γ-Default für LayerNorm, siehe `init.rs`).
- **`gelu`:** Tensor-in-Tensor (tanh-Approximation).
- **`layer_norm`:** Nur Normalisierung über der letzten Dimension (Baustein ohne Affine).
- **`layer_norm_affine`:** `γ · layer_norm(x) + β` mit Broadcast (entspricht `torch.nn.LayerNorm(..., elementwise_affine=True)`); `MiniGpt` nutzt pro Block und vor `lm_head` jeweils eigene γ/β.

---

### 2.4 `rusty_ai_ml`

**Verantwortung:** Training und Daten.

- **`Sgd`:** `step(&[Rc<Variable>])` — liest `grad()` und zieht `lr * grad` vom Parameter ab.
- **`Adam`:** Erster und zweiter Moment je Parameter; Bias-Korrektur wie in der Literatur.
- **`BatchIter`:** Iteriert Mini-Batches aus zwei Tensoren gleicher Zeilenzahl `(n, …)`.
- **`normalize_columns`:** z-Score pro Spalte (Trainingsmatrix).

---

### 2.5 `rusty_ai_llm`

**Verantwortung:** Decoder-only-Modell und Textpipeline.

- **`ByteTokenizer`:** 256 Zeichen; `encode` / `decode`.
- **Feature `gpt2-bpe`:** `Gpt2Tokenizer` lädt Hugging-Face-`tokenizer.json` (Rust-Crate `tokenizers`); `encode` / `decode` mit Token-IDs passend zu HF-GPT-2 (`vocab_size` typisch 50257). Hilfsfunktionen: `generate_from_ids` (Tokenizer-unabhängig), `generate_gpt2_text` (Prompt-String → BPE → Sampling → Text).
- **`MiniGptConfig`:** `vocab_size`, `d_model`, `n_heads`, `n_layers`, `ffn_dim`, `max_seq` (maximale Länge für **Positions-Einbettungen**; längere absolute Positionen werden auf `max_seq - 1` begrenzt); optional **`attention_window: Option<usize>`** — `Some(w)` aktiviert Sliding-Window-Attention und begrenzt den KV-Cache auf die letzten **`w`** Zeitschritte pro Layer.
- **`MiniGpt` / `DecoderBlock`:** Token- und Positions-Einbettung, Pre-LayerNorm (mit γ/β) + Residual + Attention + FFN (GELU), Ausgabe-Linear `lm_head` nach finaler LayerNorm (ebenfalls γ/β).
- **`causal_attention` / `causal_attention_windowed` / `attention_with_additive_mask` / `attention_single_query`:** Skaliertes Dot-Product; `causal_attention` bzw. `causal_attention_windowed` für kausales bzw. Sliding-Window-Verhalten; `attention_with_additive_mask` für benutzerdefinierte additive Score-Masken (z. B. **FIM** über `fim_additive_mask`). `attention_single_query`: ein Query pro Head-Zeile gegen die Key-/Value-Historie (KV-Cache-Decode; mit **`attention_window`** nur die letzten `w` Keys). **FIM-Forward** (`forward_fim`) nutzt weiterhin die volle additive Maske und **kein** `attention_window`.
- **FIM (Fill-in-the-Middle):** Sequenz als `[prefix][middle][suffix]` mit expliziten Längen; **`MiniGpt::forward_fim`** / **`TrainableMiniGpt::forward_fim`**. Inferenz-Hilfen: **`fim_next_logit_timestep`**, **`generate_fim_middle_from_ids`** (iterativ, ohne KV-Cache). Training: **`Variable::cross_entropy_next_token_subset`** mit **`fim_middle_prediction_positions`** — nicht autoregressives „generate“, kein Produktanspruch für IDE-Code-Completion; **`max_seq`** begrenzt Positions-Einbettungen wie beim normalen Forward.
- **`generate` / `generate_from_ids` / `generate_from_ids_with_callback` / `sample_token`:** Autoregressiv; Temperatur, top-p (Nucleus); leeres Logit-Vektor führt zu `TensorError::EmptyTensor`. `generate` nutzt den Byte-Tokenizer; `generate_from_ids` arbeitet auf beliebigen Token-IDs (z. B. nach `Gpt2Tokenizer::encode`). **`generate_from_ids_with_callback`** ruft nach jedem neuen Token einen Callback auf (Rückgabe `false` bricht ab) — für inkrementelle Anzeige oder Offline-Streaming **ohne** HTTP. **Prefill** + **KV-Cache** wie oben. Bei `max_tokens == 0`: `generate` decodiert nur den Prompt; `generate_from_ids` gibt die Prompt-IDs unverändert zurück (ohne Forward).
- **Kontextlänge und Speicher:** `max_seq` begrenzt die Positions-Einbettungen; längere absolute Indizes werden intern gekappt — für sehr lange Kontexte lieber kleineres Modell / kürzere Fenster planen. Kausaler Attention-Forward (ohne FIM-Maske) ist **zeilenweise** umgesetzt (keine volle materialisierte `L×L`-Score-Matrix); FIM / freie Masken nutzen weiterhin die klassische `matmul`-Form ([`attention.rs`](../rusty_ai_llm/src/attention.rs)). Mit KV-Cache wächst der Speicher bei Generierung **linear**, sofern **`attention_window`** nicht gesetzt ist; **`Some(w)`** begrenzt die gespeicherte Historie (**`truncate_last_along_seq`**, **`slice_along_seq`**). CUDA-Flash-Attention ist **nicht** eingebunden; **`rusty_ai_backend_candle`** (Tier-1-Tests optional) dient als separater Matmul-Pfad.
- **`KvCache` / `LayerKv`:** Pro Layer gespeicherte Keys und Values mit Shape `(batch * heads, past_len, d_head)`; `MiniGpt::forward_prefill` und `forward_decode_step` füllen bzw. erweitern den Cache; `forward` / `forward_last` rechnen **ohne** Cache (jedes Mal die volle Sequenz — für Tests und einfache Vergleiche).
- **Weitere API:** `MiniGpt::embed_token_at(id, pos)` — kombinierte Token- und Positionszeile für einen Zeitschritt, Shape `(1, 1, d_model)`; wird intern für Decode-Schritte genutzt.

**Fehlerfälle (Auswahl):** Leere Token-ID-Liste bei `forward`, `forward_prefill` → `TensorError::EmptyTensor`. Inkonsistenter KV-Zustand (nur eines von `k`/`v` gesetzt) in `DecoderBlock::forward_step` → `ShapeError::IncompatibleBroadcast`.

**Checkpoints (RustyAi-Format):** `save_minigpt_checkpoint(dir, &model)` schreibt `config.json` (`model_type`: `rusty_ai_minigpt`, Felder wie `n_embd`/`n_layer`/…) und `model.safetensors` mit Tensor-Namen `tok_embed`, `blocks.{i}.*`, `lm_head_w`, …; `load_minigpt_checkpoint(dir)` rekonstruiert ein [`MiniGpt`]. **`load_minigpt_checkpoint_bytes(config_json, model_safetensors)`** lädt dasselbe Format aus Speicher (z. B. `include_str!` / `include_bytes!` in Binaries, kein Netz).

**Lokales Mini-Bundle (&lt;5 MB):** `MiniGptConfig::micro_local()` definiert ein kleines Byte-Level-Profil (256 Vokabeln, kompatibel mit `ByteTokenizer`); `MiniGptConfig::approx_weight_bytes()` schätzt die FP32-Größe. Im Repo liegen unter [`rusty_ai/assets/mini_local/`](../rusty_ai/assets/mini_local/) Beispiel-`config.json` und `model.safetensors` (Aktualisierung: `cargo test -p rusty_ai_llm bootstrap_rusty_ai_mini_local_assets -- --ignored` für deterministische Zufallsinitialisierung, oder `cargo run -p rusty_ai --example train_micro_checkpoint` zum kurzen Training und Überschreiben). Demo ohne externe Dateien: `cargo run -p rusty_ai --example mini_local_inference`.

**GPT-2-`safetensors`:** `load_minigpt_from_gpt2_safetensors(path, cfg)` mappt HF-Schlüssel (`transformer.h.{i}.attn.c_attn.weight` usw.) auf die interne Struktur. `c_attn` wird in Q/K/V-Matrizen **zerlegt**; `lm_head.weight` akzeptiert sowohl HF-Shape `[vocab, d_model]` als auch RustyAi `[d_model, vocab]`. **Tokenizer:** Feature **`gpt2-bpe`**: `Gpt2Tokenizer::from_file` / `from_model_dir` mit `tokenizer.json` aus dem HF-Modellordner; `ByteTokenizer` bleibt das einfache Byte-Modell für Demos und Training ohne GPT-2-Vokabular.

#### Phase 4 (optional): Fine-Tuning, DPO/Preference — außerhalb des Rust-Workspaces

**Abgrenzung:** **Phase 4** ist die **Ausnahme** zur sonstigen Rust-Ausrichtung des Workspaces: Für SFT, **DPO**, RLHF und ähnliches Alignment ist **Python** ausdrücklich **erlaubt und üblich** — typisch **Hugging Face** (`transformers`, **TRL** für DPO/Preference) auf eurer Maschine, in einer VM oder einem **separaten Repo**; der Trainings-/Export-Code gehört **nicht** zum RustyAi-Pflichtumfang und wird hier **nicht** mitgeliefert. In diesem Repository gibt es **keinen** eingecheckten Python-Trainer und kein verpflichtendes Alignment-Training — nur die folgende **Dokumentations-Checkliste** und Rust-**Inferenz**-APIs sowie das **RustyAi-Checkpoint-Format** (siehe **Checkpoints** und **GPT-2-`safetensors`** oben).

**Empfohlener Ablauf (Checkliste — reproduzierbar als Dokumentationspfad, kein `cargo test`-Pflicht):**

- [ ] **Daten** vorbereiten (z. B. Prompt/Antwort-Paare für SFT oder Preference-Paare je nach Methode).
- [ ] **Training/Alignment** ausführen (eigenes Repo, VM oder Ordner): **Phase 4 — hier ist Python erlaubt**; typisch **Python/HF** mit `transformers` und bei DPO/Preference **TRL**. Kein mitgelieferter Trainer im RustyAi-Clone — ihr führt die Skripte **lokal/extern** aus.
- [ ] **Export:** Gewichte als **`safetensors`** (oder anderes zu den Lade-APIs passendes Format) sichern; bei Anschluss an HF-GPT-2/BPE typisch `model.safetensors` im Modellordner, **dort auch** `tokenizer.json` für [`Gpt2Tokenizer`](../rusty_ai_llm/src/gpt2_tokenizer.rs) (Feature **`gpt2-bpe`**).
- [ ] **In RustyAi laden:** Architektur **HF-GPT-2-kompatibel** → [`load_minigpt_from_gpt2_safetensors`](../rusty_ai_llm/src/gpt2_hf.rs) + passende [`MiniGptConfig`](../rusty_ai_llm/src/model.rs). **RustyAi-Schema** (`config.json` + `model.safetensors`) → [`load_minigpt_checkpoint`](../rusty_ai_llm/src/checkpoint.rs) / [`load_minigpt_checkpoint_bytes`](../rusty_ai_llm/src/checkpoint.rs); ein **eigenes** Exportformat nur, wenn es zur Architektur und zu diesen APIs passt (Export-Code **außerhalb** dieses Repos).
- [ ] **Kein** neues Rust-Crate für DPO/RLHF im Workspace planen — es sei denn, ihr bewusst wollt **Rust-only**-Training (hoher Aufwand).

Siehe [ARCHITEKTUR_IDE_ROADMAP_B.md](ARCHITEKTUR_IDE_ROADMAP_B.md), Phase 4.

---

### 2.6 `rusty_ai_backend_candle`

**Verantwortung:** Optionale Beschleunigung und Verteilungs-Hilfen über [Candle](https://github.com/huggingface/candle).

- **`matmul_f32`:** Matrixmultiplikation auf gewähltem Candle-`Device` (CPU oder CUDA mit Feature **`cuda`**).
- **`f32_tensor_to_f8e4m3` / `f8e4m3_tensor_to_f32`:** FP8 E4M3 (sinnvoll mit CUDA; Candle unterstützt den Datentyp auch CPU-seitig eingeschränkt).
- **`all_reduce_mean_cpu`:** Referenzimplementierung des Mittelwerts über mehrere Gradienten-Kopien (gleiche Länge) — entspricht dem Erwartungswert nach einem All-Reduce-Summe und Division durch `world_size`. Multi-GPU-NCCL: Candle-Feature **`nccl`**, siehe Upstream-Dokumentation.

**Abgrenzung zum LM-Training:** [`TrainableMiniGpt`](../rusty_ai_llm/src/trainable.rs) nutzt **`rusty_ai_core::Tensor`** und **`rusty_ai_autograd::Variable`** auf der CPU. Candle arbeitet mit **eigenen** Tensoren und einem **anderen** Graphen — es gibt **keinen** Drop-in-Ersatz für `TrainableMiniGpt` und kein end-to-end gemeinsames Training im Workspace. Sinnvoll sind: Matmul/FP8/All-Reduce als Bausteine; optional ein **Forward-only**-Vergleich (Gewichte aus `MiniGpt` einmal nach `f32`-Puffern kopieren, z. B. für Benchmarks). Vollständiges Training in Candle wäre ein separates Projekt.

**Stufen (Doku-Konvention):**

| Stufe | Inhalt |
| ----- | ------ |
| **Tier 1** | Gewichte / Teil-Forward (z. B. eine Matrix oder LM-Head) gegen Candle-CPU-`matmul`; **kein** gemeinsamer Gradient mit `Variable`. |
| **Tier 2** | Eigenes Modell in Candle inkl. Training/Optimizer — **nicht** Bestandteil des RustyAi-Kernpfads. |

**Optionale Brückentests:** `cargo test -p rusty_ai_backend_candle -- --ignored` — u. a. `w_q`-Matmul und LM-Head-Linear gegen `matmul_f32` auf CPU-Candle (siehe [`rusty_ai_backend_candle/README.md`](../rusty_ai_backend_candle/README.md) und `lib.rs` des Crates).

---

### 2.7 `rusty_ai`

Re-Exportiert die Untercrates unter den Namen `core`, `autograd`, `nn`, `ml`, `llm` und eine Auswahl häufig genutzter Typen (`Tensor`, `Variable`, `Linear`, `Sgd`, `Adam`, `MiniGpt`, `TrainableMiniGpt`, `KvCache`, `generate_from_ids`, …). Mit Feature **`candle`** zusätzlich `rusty_ai_backend_candle` als Modul `candle`. Mit **`gpt2-bpe`**: `Gpt2Tokenizer`, `generate_gpt2_text`, `Gpt2PipelineError`.

Kurzüberblick und Beispielbefehle: **[`rusty_ai/README.md`](../rusty_ai/README.md)**; Feature-Matrix: Abschnitt **8** unten.

---

### 2.8 `rusty_ai_agent`

**Verantwortung:** **Agent-/IDE-Protokoll** (Pfad B): austauschbares [`LlmBackend`](../rusty_ai_agent/src/core/llm_backend.rs), Chat-Typen (`CompletionRequest` / `CompletionResponse`), OpenAI-artige **Tool-Aufrufe** (`ModelToolCall`) und die ausführbare Repräsentation [`ToolInvocation`](../rusty_ai_agent/src/tools/invocation.rs) (`read_file`, `write_file`, `run_cmd`, `search_replace`). JSON-Schema: [`schemas/tool_invocation.json`](../rusty_ai_agent/schemas/tool_invocation.json).

Das Crate **führt standardmäßig kein Netzwerk** aus; optional **Feature `http`**: [`OpenAiCompatBackend`](../rusty_ai_agent/src/http/openai_compat.rs) (`POST …/chat/completions`, blocking `reqwest`). **Streaming (SSE):** [`OpenAiCompatBackend::complete_stream`](../rusty_ai_agent/src/http/openai_compat.rs) inkl. Aggregation von Text- und Tool-Deltas. Optional **Feature `real-exec`:** [`RealExecutor`](../rusty_ai_agent/src/execution/executor.rs) für echtes Lesen/Schreiben und Subprocess unter [`AllowlistPolicy`](../rusty_ai_agent/src/policy/allowlist.rs).

**Weitere Bausteine (Auswahl):**

| Modul / API | Zweck |
| ----------- | ----- |
| [`tool_parse`](../rusty_ai_agent/src/tools/parse.rs) | `parse_json_arguments_loose`, `tool_invocations_from_model_calls`, `tool_invocations_try_each`, `tool_parse_retry_instruction` |
| [`orchestrator`](../rusty_ai_agent/src/execution/orchestrator.rs) | `complete_with_tool_parse_retries` (mehrere `complete`-Runden; optional `LocalTelemetry`) |
| [`fallback_backend`](../rusty_ai_agent/src/execution/fallback_backend.rs) | `FallbackBackend`: primärer `LlmBackend`, bei Fehler Fallback |
| [`telemetry`](../rusty_ai_agent/src/telemetry/mod.rs) | `LocalTelemetry`, `TimedBackend` (Latenz-/Aufrufzähler), `record_cargo_check` |
| [`diff_preview`](../rusty_ai_agent/src/tools/diff_preview.rs) | `format_replace_preview`, `truncate_middle` (Mitte auslassen), `truncate_utf8_prefix` (Präfix auf Byte-Länge **ohne** UTF-8-Zeichensplit — für Logs und HTTP-Fehlertexte) |
| [`diagnostics`](../rusty_ai_agent/src/feedback/diagnostics.rs) | `parse_cargo_json_stream`, `parse_lsp_diagnostic_json`, `merge_diagnostics`, `format_for_prompt` |
| [`prompts`](../rusty_ai_agent/src/feedback/prompts.rs) | `PromptKind`, `render_embedded`, `load_from_dir` — Vorlagen unter [`prompts/v1/`](../rusty_ai_agent/prompts/v1/) |
| [`cargo_test`](../rusty_ai_agent/src/cargo_test.rs) | `CargoTestInvocation` — sicheres `argv` für `cargo test -p … -- filter` |
| [`policy_catalog`](../rusty_ai_agent/src/policy/catalog.rs) | `PolicyCatalog`, `RUSTY_AI_AGENT_POLICY`, `AllowlistPolicy::preset_dev` / `preset_ci` |
| [`batch_report`](../rusty_ai_agent/src/batch/batch_report.rs) | `BatchReport`, `BatchStepRecord` — JSON/Markdown für CI-Batches |
| [`budget`](../rusty_ai_agent/src/batch/budget.rs) | `BudgetLlmBackend` — Limits für Aufrufe und Token (`CompletionUsage`) |

**Sicherheit und Policy:** Siehe **[`rusty_ai_agent/SECURITY.md`](../rusty_ai_agent/SECURITY.md)**. **Architektur / Roadmap (Pfad B):** **[`ARCHITEKTUR_IDE_ROADMAP_B.md`](ARCHITEKTUR_IDE_ROADMAP_B.md)**.

**Beispiele** (vom Workspace-Root; Details im [Crate-README](../rusty_ai_agent/README.md)):

| Befehl | Inhalt |
| ------ | ------ |
| `cargo run -p rusty_ai_agent --example agent_demo` | Fake-LLM, Dry-Run / Policy |
| `cargo run -p rusty_ai_agent --example agent_demo --features real-exec -- --real` | Echtes FS / `cargo check` |
| `cargo run -p rusty_ai_agent --example agent_retry_demo` | Tool-Parse-Retry-Schleife |
| `cargo run -p rusty_ai_agent --example dual_backend_demo` | Primär + Fallback-Backend |
| `cargo run -p rusty_ai_agent --example telemetry_demo` | `TimedBackend` + `LocalTelemetry` |
| `cargo run -p rusty_ai_agent --example openai_smoke --features http` | Eine Chat-Completion (Cloud; Ollama: `-- --ollama`) |
| `cargo run -p rusty_ai_agent --example openai_stream --features http` | SSE-Streaming |
| `cargo run -p rusty_ai_agent --example cargo_test_demo` | Beispiel-`argv` für gezielte Tests |
| `cargo run -p rusty_ai_agent --example batch_report_demo` | Beispiel-`BatchReport` (JSON/Markdown) |

---

### 2.9 `rusty_ai_workspace`

**Verantwortung:** **Workspace-Index** für Retrieval vor LLM-Aufrufen (Pfad B, Phase 2): [`WorkspaceIndex::build`](../rusty_ai_workspace/src/lib.rs) lädt Textdateien unter einem Root (überspringt u. a. `target/`, `.git/`); Zeilen-Chunks mit Überlappung; [`search_substring`](../rusty_ai_workspace/src/lib.rs).

**Phase 3 — Index-Cache:** [`WorkspaceIndex::build_cached`](../rusty_ai_workspace/src/lib.rs) schreibt unter einem wählbaren `cache_dir` ein Manifest (`index_manifest.json`) und `index_chunks.json`. Gültigkeit: gleicher Root-String, gleicher Fingerprint der [`IndexConfig`](../rusty_ai_workspace/src/lib.rs), unveränderte maximale Datei-Änderungszeit (`mtime`) über alle indexierten Dateien — sonst Neuaufbau; erzwungen auch mit `force_rebuild`. Kein separates Cargo-Feature nötig (nur zusätzliche IO-Pfade).

**Embeddings (optional):** Feature **`embeddings`**: [`embeddings::HttpEmbeddingClient`](../rusty_ai_workspace/src/lib.rs), [`EmbeddingIndex::from_workspace`](../rusty_ai_workspace/src/lib.rs) und Cosinus-Top-k — **Netzwerk nur mit aktiviertem Feature** und konfiguriertem API-Endpunkt. **Phase 3:** [`CachingEmbeddingClient`](../rusty_ai_workspace/src/lib.rs) puffert Embeddings nach Text-Hash (In-Memory).

**Beispiel:** `cargo run -p rusty_ai_workspace --example workspace_index_demo` — ausführlicher: [Crate-README](../rusty_ai_workspace/README.md).

---

## 3. Typische Abläufe

### 3.1 MLP trainieren (Regression)

1. `Linear`-Schichten mit `Linear::new(in, out, &mut seed)` anlegen.
2. Alle trainierbaren `Rc<Variable>` in einem Vektor sammeln (Gewichte + Biases).
3. Pro Epoche: Eingabe als `Variable::leaf(tensor)`, Ziel als `Tensor`, Forward (`relu`, `mse`), `zero_grad` auf allen Parametern, `backward`, Optimizer `step`.

Siehe `rusty_ai/examples/train_mlp.rs`.

### 3.2 LLM: nur Vorwärtsrechnung / Sampling

1. `MiniGpt::random(MiniGptConfig::default(), &mut seed)` oder eigene Konfiguration (bei importiertem GPT-2: `MiniGptConfig` mit passenden `vocab_size`, `d_model`, …).
2. Prompt → Token-IDs: `ByteTokenizer::encode` **oder** mit Feature `gpt2-bpe`: `Gpt2Tokenizer::encode`.
3. Je nach Ziel:
   - Volle Logits `(1, seq, vocab)`: `MiniGpt::forward(&token_ids)`.
   - Nur letztes Zeitschritt (z. B. nächstes Token): `forward_last` — intern ein voller Forward ohne KV-Cache.
   - Text generieren: `generate` (Byte-Tokenizer) **oder** `generate_from_ids` + `decode` **oder** `generate_gpt2_text` (BPE) — jeweils Prefill + KV-Cache pro generiertem Token.

**Manuelles Sampling mit KV-Cache** (gleiche Logik wie `generate`, aber eigenes Stepping):

```text
KvCache::new(cfg.n_layers)
logits_letzter_prompt = forward_prefill(&prompt_ids, &mut cache)   // liefert (1, vocab)
Schleife: nächstes Token wählen (z. B. sample_token)
          ids.push(token)
          logits = forward_decode_step(token, position, &mut cache)
```

Dabei ist **`position`** die **absolute** Indexposition des gerade erzeugten Tokens in der laufenden Sequenz (0-basiert), also nach dem ersten neuen Token typisch `prompt_len`, dann `prompt_len + 1`, … — entspricht der Zeile in `embed_positions` / `embed_token_at`.

#### 3.2.1 GPT-2 mit HF-Gewichten und BPE (Rust-only)

Für **dieselbe Tokenisierung** wie bei HF GPT-2 (Byte-Level-BPE, `tokenizer.json`):

1. Workspace / Crate mit **`--features gpt2-bpe`** bauen (`rusty_ai` oder `rusty_ai_llm`).
2. **`MiniGptConfig`** exakt auf den Checkpoint abstimmen (u. a. `vocab_size`, `d_model`, `n_heads`, `n_layers`, `ffn_dim`, `max_seq`).
3. Modell: **`load_minigpt_from_gpt2_safetensors("…/model.safetensors", cfg)`**.
4. Tokenizer: **`Gpt2Tokenizer::from_model_dir("…/")`**, wobei der Ordner **`tokenizer.json`** enthält (typisch derselbe wie die Modell-Datei).
5. Abgleich: **`tok.vocab_size()`** und **`cfg.vocab_size`** sollten übereinstimmen.
6. Text: **`generate_gpt2_text(&model, &tok, prompt, max_new, temperature, top_p, &mut seed)`** oder **`tok.encode`** → **`generate_from_ids`** → **`tok.decode`**.

Es ist **keine Python-Toolchain** erforderlich; Tests und CI bleiben bei **Cargo**/`cargo test`. Optionaler Integrations-Test: Umgebungsvariable **`RUSTY_AI_TEST_TOKENIZER`** (Pfad zu `tokenizer.json`), dann `cargo test -p rusty_ai_llm --features gpt2-bpe -- --ignored`.

### 3.3 Mini-GPT trainieren (Next-Token, Autograd)

1. `MiniGpt::random(...)` erzeugen, mit `TrainableMiniGpt::from_mini_gpt` übernehmen.
2. `parameters()` liefert alle `Rc<Variable>` für den Optimierer.
3. Pro Schritt: `zero_grad` auf allen Parametern, `forward(&token_ids)` → Logits `(1, seq, vocab)`, Loss `Variable::cross_entropy_next_token(&logits, &token_ids)` (Zielsequenz = Eingabe; pro Position `t` wird das nächste Token vorhergesagt), `backward`, `Adam::step` / `Sgd::step`.

Siehe `rusty_ai/examples/train_mini_gpt.rs`. KV-Cache wird für das Training nicht benötigt (volle Sequenz pro Schritt).

#### 3.3.1 Lokales Mini-Modell (eingebettete Gewichte, ohne Netz)

Für ein **sehr kleines** Byte-Level-LM (typisch **unter 5 MiB** FP32-Gewichte) ohne Hugging Face und ohne Laufzeit-Zugriff auf Dateien neben der Binary:

1. **Konfiguration:** `MiniGptConfig::micro_local()` — passt zu `ByteTokenizer` (`vocab_size` 256). Größenabschätzung: `MiniGptConfig::approx_weight_bytes()`.
2. **Gewichte im Repo:** Verzeichnis [`rusty_ai/assets/mini_local/`](../rusty_ai/assets/mini_local/) mit `config.json` und `model.safetensors` (RustyAi-Format, siehe §2.5).
3. **Aktualisierung der Assets:** deterministische Zufallsinitialisierung: `cargo test -p rusty_ai_llm bootstrap_rusty_ai_mini_local_assets -- --ignored`. Alternativ kurzes Training und Überschreiben: `cargo run -p rusty_ai --example train_micro_checkpoint`.
4. **Einbinden in eine Binary:** `load_minigpt_checkpoint_bytes` mit `include_str!("…/config.json")` und `include_bytes!("…/model.safetensors")` (Pfade relativ zu `CARGO_MANIFEST_DIR` des Crates, das die Includes nutzt).
5. **Referenz-Demo:** `cargo run -p rusty_ai --example mini_local_inference`.

Details: Abschnitt **2.5** in diesem Handbuch und [`rusty_ai_llm/README.md`](../rusty_ai_llm/README.md) (Abschnitt „Lokales Mini-Bundle“).

### 3.4 Agent-Orchestrierung (`rusty_ai_agent`, Pfad B)

1. **Kontrakt wählen:** [`LlmBackend::complete`](../rusty_ai_agent/src/core/llm_backend.rs) synchron; für HTTP ein [`OpenAiCompatBackend`](../rusty_ai_agent/src/http/openai_compat.rs) mit passender [`OpenAiChatConfig`](../rusty_ai_agent/src/http/openai_compat.rs) (Cloud oder Ollama).
2. **Tools definieren:** `CompletionRequest::tools` mit JSON-Schema-artigen [`ToolDefinition`](../rusty_ai_agent/src/core/llm_backend.rs)-Einträgen; Modell liefert `ModelToolCall`s.
3. **Parsen und ausführen:** `tool_invocations_try_each` oder `tool_invocations_from_model_calls` → [`ToolInvocation`](../rusty_ai_agent/src/tools/invocation.rs); vor Ausführung [`AllowlistPolicy::validate`](../rusty_ai_agent/src/policy/allowlist.rs).
4. **Fehlerhafte Tool-JSON:** [`complete_with_tool_parse_retries`](../rusty_ai_agent/src/execution/orchestrator.rs) mit `max_complete_calls` und optional `Some(&telemetry)` für Zähler.
5. **Robustheit:** [`FallbackBackend`](../rusty_ai_agent/src/execution/fallback_backend.rs) (z. B. API ausgefallen → lokal); [`TimedBackend`](../rusty_ai_agent/src/telemetry/mod.rs) + manuell `record_cargo_check` nach `run_cmd`.
6. **Echte Ausführung:** nur mit Feature **`real-exec`** und [`RealExecutor::new`](../rusty_ai_agent/src/execution/executor.rs) (Workspace-Root kanonisieren).
7. **Kontext (Phase 2):** [`WorkspaceIndex`](../rusty_ai_workspace/src/lib.rs) für relevante Chunks; Compiler- und LSP-Ausgaben mit [`merge_diagnostics`](../rusty_ai_agent/src/feedback/diagnostics.rs) und [`format_for_prompt`](../rusty_ai_agent/src/feedback/diagnostics.rs) in die nächste Nachricht; System-Prompt aus [`render_embedded`](../rusty_ai_agent/src/feedback/prompts.rs); schnelle Tests mit [`CargoTestInvocation`](../rusty_ai_agent/src/feedback/cargo_test.rs) + `run_cmd`.
8. **Betrieb (Phase 3):** [`PolicyCatalog`](../rusty_ai_agent/src/policy/catalog.rs) mit eingebauten Presets und Auswahl per **`RUSTY_AI_AGENT_POLICY`** (oder eigenes JSON + [`from_json_merging_builtin`](../rusty_ai_agent/src/policy/catalog.rs)) — die aktive [`AllowlistPolicy`](../rusty_ai_agent/src/policy/allowlist.rs) vor jeder Tool-Ausführung verwenden. Für **CI/Nightly** ohne UI: Schritte in [`BatchReport`](../rusty_ai_agent/src/batch/batch_report.rs) sammeln und als JSON/Markdown ablegen (Beispiel `batch_report_demo`). **Kosten-Schutz:** HTTP-Backend mit [`BudgetLlmBackend`](../rusty_ai_agent/src/batch/budget.rs) umhüllen; [`CompletionUsage`](../rusty_ai_agent/src/core/llm_backend.rs) aus API-Antworten wird gezählt (siehe [`LocalTelemetry`](../rusty_ai_agent/src/telemetry/mod.rs)). **Index:** bei wiederholten Läufen [`WorkspaceIndex::build_cached`](../rusty_ai_workspace/src/lib.rs) nutzen; Embeddings optional mit [`CachingEmbeddingClient`](../rusty_ai_workspace/src/lib.rs).

Ausführliche Beispielbefehle: Abschnitt **2.8**, Phase-3-Kurzüberblick im [Agent-README](../rusty_ai_agent/README.md#phase-3-betrieb-und-ci), Sicherheit in [`SECURITY.md`](../rusty_ai_agent/SECURITY.md).

---

## 4. Grenzen und bekannte Einschränkungen

- **Training:** Der Kern (`TrainableMiniGpt`, Optimierer) läuft auf der **CPU**. GPU-Training über Candle ist nicht 1:1 angebunden; Candle dient als **zusätzlicher** Pfad für Matmul/Quantization/Verteilungs-Experimente (Tier-1-Paritätstests optional in `rusty_ai_backend_candle`, kein gemeinsamer Autograd-Graph mit `Variable`).
- **Autograd** deckt die implementierten Ops ab; erweiterte Ops erfordern eigene `Op`-Varianten und Ableitungen.
- **Broadcasting im Autograd** ist nicht vollständig für alle Kombinationen in jedem Op abgebildet; `BiasAdd` ist für den üblichen Linear-Bias-Pfad `(batch, n) + (1, n)` gedacht; `Add`/`Mul` reduzieren Gradienten bei Broadcast auf die Eltern-Shapes.
- **LayerNorm:** `MiniGpt` nutzt affine LayerNorm (`layer_norm_affine`); die Basisfunktion `layer_norm` bleibt für Normierung ohne γ/β nutzbar. Extern gespeicherte Gewichte ohne die neuen γ/β-Tensoren sind nicht abwärtskompatibel.
- **KV-Cache:** Optional begrenzt durch **`MiniGptConfig::attention_window`** (Sliding-Window); ohne diese Option wächst der Cache bei langen Generierungen pro Schritt (Konkatenation). Batch-Größe größer als 1 für LLM-Pfade ist nicht der Fokus der API.

---

## 5. Qualitätssicherung

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

CI (falls eingerichtet): siehe `.github/workflows/ci.yml`.

---

## 6. Glossar

| Begriff | Bedeutung in RustyAi |
| ------- | -------------------- |
| **Broadcasting** | Automatische Anpassung kleinerer Formen an gemeinsame Ausgabeform bei elementweisen Ops. |
| **Causal / masked attention** | Attention nur auf vergangene und aktuelle Tokens (untere Dreiecksmatrix in den Scores). |
| **Decoder-only** | Transformer, der nur „nach links“ sieht (wie GPT), ohne Encoder. |
| **KV-Cache** | Zwischengespeicherte Key- und Value-Tensoren pro Schicht, um bei autoregressiver Generierung nicht die komplette bisherige Sequenz erneut zu verarbeiten. Mit **`attention_window`** optional auf die letzten **`w`** Zeitschritte begrenzt (Sliding-Window). |
| **LayerNorm (affine)** | Normierung über die letzte Dimension plus elementweise **γ** (Skalierung, Initialisierung typisch Einsen) und **β** (Verschiebung, typisch Nullen); in RustyAi: `layer_norm_affine` / `Variable::layer_norm_affine`. Ohne γ/β: nur `layer_norm`. |
| **Prefill** | Erster Inferenzschritt über den ganzen Prompt; füllt den KV-Cache und liefert Logits für die letzte Prompt-Position. |
| **top-p (Nucleus)** | Sampling aus der kleinsten Menge höchster Wahrscheinlichkeiten, deren Summe ≥ p. |
| **TrainableMiniGpt** | Decoder-only-Modell mit `Variable`-Gewichten; Forward wie `MiniGpt`, für Training mit `backward` + Optimierer. |
| **Next-Token-Cross-Entropy** | Mittlerer negativer Log-Likelihood über Positionen `t`, die `token[t+1]` aus `logits[0,t,:]` vorhersagen (`Variable::cross_entropy_next_token`). |
| **FIM (Fill-in-the-Middle)** | Trainingslayout `[prefix][middle][suffix]` ohne neue Vokabel-IDs; Attention über `fim_additive_mask`; Forward [`MiniGpt::forward_fim`](../rusty_ai_llm/src/model.rs); Loss nur in der Mitte über `Variable::cross_entropy_next_token_subset` + `fim_middle_prediction_positions`. Inferenz-Hilfen: `generate_fim_middle_from_ids`, `fim_next_logit_timestep`. Kein KV-Cache für FIM in der Referenz-API. |
| **Sliding-Window-Attention** | Optional über `MiniGptConfig::attention_window = Some(w)`; kausale Attention nur über ein Fenster der Länge `w` (und entsprechend begrenzter KV-Cache). FIM-Forward (`forward_fim`) nutzt weiterhin die volle FIM-Maske, nicht dieses Fenster. |
| **Candle Tier 1 / Tier 2** | **Tier 1:** Forward- oder Matmul-Parität (z. B. LM-Head) gegen Candle-CPU — siehe §2.6. **Tier 2:** eigenes Candle-Training — außerhalb des Kernpfads. |
| **Variable** | Knoten mit Daten und optional Gradient im Autograd-Graphen. |
| **LlmBackend** | Trait für eine synchrone Chat-/Completion-Anfrage (`complete`); Implementierungen: z. B. HTTP-Client oder Fake für Tests. |
| **ToolInvocation** | Konkretes, ausführbares Tool (`read_file`, `write_file`, …) nach Parsing aus `ModelToolCall`. |
| **AllowlistPolicy** | Erlaubte Pfad-Präfixe und Binaries für `run_cmd` vor Ausführung durch einen Executor. |
| **Pfad B** | IDE-nähe: Orchestrierung, externe LLMs, Tool-Loops, Compiler-Feedback — siehe `ARCHITEKTUR_IDE_ROADMAP_B.md`. |
| **WorkspaceIndex** | Zeilen-Chunks aus Dateien unter einem konfigurierbaren Root; Substring-Suche; optional HTTP-Embeddings (`rusty_ai_workspace`, Feature `embeddings`). |
| **UnifiedDiagnostic** | Gemeinsames Format für rustc-/Cargo-JSON und LSP-Subset; `merge_diagnostics`, `format_for_prompt` (`rusty_ai_agent::diagnostics`). |
| **CargoTestInvocation** | Validiertes `argv` für `cargo test -p … -- filter` ohne Shell (`rusty_ai_agent`). |
| **PolicyCatalog** | Namen → [`AllowlistPolicy`](../rusty_ai_agent/src/policy/allowlist.rs); Auswahl z. B. über `RUSTY_AI_AGENT_POLICY` (`rusty_ai_agent`). |
| **BatchReport** | Serieller Bericht über LLM-/Tool-/Check-Schritte für CI (JSON/Markdown), ohne Terminal-UI (`rusty_ai_agent`). |
| **BudgetLlmBackend** | Wrapper um [`LlmBackend`](../rusty_ai_agent/src/core/llm_backend.rs) mit harten Grenzen für Token- und Aufrufanzahl (`rusty_ai_agent`). |
| **micro_local** | Festes kleines `MiniGptConfig`-Profil für Byte-Tokenizer und lokale Demos; siehe §2.5 und §3.3.1. |
| **Mini-Bundle** | `config.json` + `model.safetensors` im RustyAi-Format; oft unter `rusty_ai/assets/mini_local/` für eingebettete Inferenz. |

---

## 7. Versionshinweise

Dieses Handbuch bezieht sich auf den Stand des Repositories zum Zeitpunkt der letzten Bearbeitung. Für API-Details sind die `rustdoc`-Kommentare in den Quellen und `cargo doc` maßgeblich. Ein **Einstiegsindex** für die Dokumentation liegt in [`README.md`](README.md) im gleichen Ordner; das **Projekt-README** liegt im Repository-Root. **Phase-2/3-Bausteine** (Index, Diagnosen, Prompts, Policies, Batch-Reports, Budgets, Cache): Abschnitte **2.8–2.9** und Roadmap in [`ARCHITEKTUR_IDE_ROADMAP_B.md`](ARCHITEKTUR_IDE_ROADMAP_B.md). **Phase 4 (optional):** Fine-Tuning/DPO (Python/HF/TRL **ausnahmsweise** erlaubt, siehe Checkliste), `generate_from_ids_with_callback`, **FIM** — Abschnitt **2.5** (`rusty_ai_llm`); Candle-Backend Abschnitt **2.6**. Roadmap: [`ARCHITEKTUR_IDE_ROADMAP_B.md`](ARCHITEKTUR_IDE_ROADMAP_B.md). Eine **Prüfzusammenfassung** zur Erweiterung (Checkpoints, GPT-2, Candle) steht in [`BERICHT_PRÜFUNG.md`](BERICHT_PRÜFUNG.md).

---

## 8. Feature-Matrix (Meta-Crate `rusty_ai`)

| Feature | Wirkung |
| ------- | ------- |
| `candle` | Bindet `rusty_ai_backend_candle` ein; Modul `rusty_ai::candle`. |
| `candle-cuda` | Wie `candle`, Candle mit CUDA. |
| `hf-hub` | `rusty_ai_llm` mit Hub-Download; Re-Export `load_minigpt_from_hf`. |
| `gpt2-bpe` | `tokenizers`-Dependency; `Gpt2Tokenizer`, `generate_gpt2_text`, `Gpt2PipelineError` (Re-Exports in `rusty_ai`). |

Direkt auf `rusty_ai_llm` kann ebenfalls `hf-hub` aktiviert werden, ohne die Meta-Crate.
