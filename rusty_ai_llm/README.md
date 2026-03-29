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

## GPT-2-Parität (Kurz)

1. **`MiniGptConfig`** an den Checkpoint anpassen (z. B. `vocab_size: 50257`, `d_model: 768`, …).
2. **`load_minigpt_from_gpt2_safetensors("model.safetensors", cfg)`** — mappt HF-Tensor-Namen auf `MiniGpt`.
3. Mit **`gpt2-bpe`:** **`Gpt2Tokenizer::from_model_dir(dir)`** — `dir` enthält `tokenizer.json` (und typischerweise dieselbe `model.safetensors`-Quelle).
4. **`tok.vocab_size()`** sollte mit **`cfg.vocab_size`** übereinstimmen.
5. **`generate_gpt2_text(&model, &tok, prompt, max_new, temperature, top_p, &mut seed)`** oder manuell `encode` → **`generate_from_ids`** → `decode`.

Siehe auch das **[Projekt-Handbuch](../docs/HANDBUCH.md)** (Abschnitt `rusty_ai_llm`, Ablauf LLM).
