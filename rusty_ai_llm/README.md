# `rusty_ai_llm`

Decoder-only Transformer (**`MiniGpt`**), Byte-Tokenizer, Autoregressions-**`generate`**, safetensors-Checkpoints und optional **GPT-2-Gewichtsimport** aus Hugging-Face-`safetensors`.

## Features (Cargo)

| Feature | Bedeutung |
| ------- | --------- |
| *(default)* | Kern-API ohne Hub, ohne schwere Tokenizer-Abhängigkeit. |
| **`hf-hub`** | Download von **RustyAi**-Checkpoints vom Hugging Face Hub (`load_minigpt_from_hf`). |
| **`gpt2-bpe`** | [`tokenizers`](https://crates.io/crates/tokenizers) (Rust): **`Gpt2Tokenizer`** lädt `tokenizer.json`; **`generate_gpt2_text`**, **`generate_from_ids`**. Keine Python-Runtime nötig. |

```bash
cargo test -p rusty_ai_llm
cargo test -p rusty_ai_llm --features gpt2-bpe
```

## Lokales Mini-Bundle (ohne Hub, &lt;5 MB)

- **`MiniGptConfig::micro_local()`** — kleines Profil für Byte-Tokenizer (`vocab_size` 256); **`approx_weight_bytes()`** schätzt die Speichergröße der Gewichte (FP32).
- **`load_minigpt_checkpoint_bytes`** — Checkpoint aus JSON-String + `model.safetensors`-Bytes (z. B. eingebettet mit `include_str!` / `include_bytes!`).
- **Assets:** [`rusty_ai/assets/mini_local/`](../rusty_ai/assets/mini_local/) (`config.json`, `model.safetensors`). Aktualisierung: `cargo test -p rusty_ai_llm bootstrap_rusty_ai_mini_local_assets -- --ignored` (Zufallsinit, Seed 42) oder `cargo run -p rusty_ai --example train_micro_checkpoint` (kurzes Training).
- **Demo:** `cargo run -p rusty_ai --example mini_local_inference` — lädt eingebettete Bytes, **kein Netzwerk**.

## GPT-2-Parität (Kurz)

1. **`MiniGptConfig`** an den Checkpoint anpassen (z. B. `vocab_size: 50257`, `d_model: 768`, …).
2. **`load_minigpt_from_gpt2_safetensors("model.safetensors", cfg)`** — mappt HF-Tensor-Namen auf `MiniGpt`.
3. Mit **`gpt2-bpe`:** **`Gpt2Tokenizer::from_model_dir(dir)`** — `dir` enthält `tokenizer.json` (und typischerweise dieselbe `model.safetensors`-Quelle).
4. **`tok.vocab_size()`** sollte mit **`cfg.vocab_size`** übereinstimmen.
5. **`generate_gpt2_text(&model, &tok, prompt, max_new, temperature, top_p, &mut seed)`** oder manuell `encode` → **`generate_from_ids`** → `decode`.

### Fill-in-the-Middle (FIM)

- **Layout:** flache Token-ID-Liste als `[prefix][middle][suffix]`; Längen `prefix_len` und `middle_len` werden explizit übergeben (Rest = Suffix). Optionale Spezialmarker (z. B. `<|fim_*|>`) sind nur eine Vokabel-/Tokenizer-Frage und nicht Teil des Kern-API.
- **`fim_additive_mask` / `fim_allowed`:** Sichtbarkeitsmaske für Attention (Präfix kausal; Mitte sieht Präfix, eigene Region und Suffix; Suffix sieht Präfix, ganze Mitte und kausal den Rest).
- **`MiniGpt::forward_fim` / `TrainableMiniGpt::forward_fim`:** Forward mit dieser Maske (kein KV-Cache — eigene Inferenz-Pfade später).
- **Training:** Next-Token-Loss nur auf Positionen in der Mitte: `rusty_ai_autograd::Variable::cross_entropy_next_token_subset` mit Ziel-Indizes aus **`fim_middle_prediction_positions`** (Mittelwert nur über diese `t`).

### Inkrementelle Generierung (Phase 4)

- **`generate_from_ids_with_callback`** — wie `generate_from_ids`, aber nach jedem neuen Token-ID wird `on_token(id) -> bool` aufgerufen; bei `false` wird gestoppt. Kein async nötig; geeignet für UI-Updates oder Logging. **Nicht** zu verwechseln mit HTTP-SSE (`rusty_ai_agent`, Feature `http`).

### Kontextlänge und Speicher

- **`MiniGptConfig::max_seq`** — obere Grenze für Positions-Einbettungen; sehr lange Prompts sind nicht das Ziel dieser kleinen Referenz-API.
- **Attention** im vollen Forward: grob **O(L²)** ([`src/attention.rs`](src/attention.rs)); KV-Cache linear im Kontext, aber ohne Flash-/Window-Attention.

## Tests (Auswahl)

```bash
cargo test -p rusty_ai_llm
cargo test -p rusty_ai_backend_candle -- --ignored   # optional: MiniGpt-Gewicht → Candle-Matmul-Brücke
```

## Dokumentation

| Ressource | Inhalt |
| --------- | ------ |
| **[`docs/HANDBUCH.md`](../docs/HANDBUCH.md)** | Abschnitt **2.5** (`rusty_ai_llm`, inkl. FIM), **2.6** (Candle-Abgrenzung), **3.3.1** (lokales Mini-Modell), Abläufe LLM/KV-Cache, Phase 4 (Fine-Tuning extern, Callback-Generierung) |
| **[`docs/README.md`](../docs/README.md)** | Index aller Dokumentationsdateien |
| **[`README.md`](../README.md)** (Repo-Root) | Workspace-Schnellstart, Beispiele, Dokumentations-Index |
| **[`rusty_ai/README.md`](../rusty_ai/README.md)** | Meta-Crate: Beispiele `train_micro_checkpoint`, `mini_local_inference` |

```bash
cargo doc -p rusty_ai_llm --no-deps --open
```
