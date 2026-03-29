# Prüfbericht: Scope-Erweiterung (Checkpoints, GPT-2, Candle, verteilte Referenz)

**Stand:** Nach Code-Review und Tests im Workspace.

## 1. Geprüfte Bereiche

| Bereich | Ergebnis |
| ------- | -------- |
| `rusty_ai_llm::checkpoint` | Roundtrip `safetensors` + `config.json`; `MiniGptConfigFile`-Validierung um positive Dimensionen ergänzt. |
| `rusty_ai_llm::gpt2_hf` | Mapping HF → `MiniGpt`; **`c_attn.weight`** unterstützt jetzt **`[d, 3·d]`** (HF-Standard) und **`[3·d, d]`** (transponiert). |
| `load_minigpt_from_hf` | Dokumentation: lädt **RustyAi**-Checkpoints vom Hub, **nicht** rohe GPT-2-Archive (dafür `load_minigpt_from_gpt2_safetensors`). |
| `rusty_ai_backend_candle` | Matmul-/FP8-/All-Reduce-Referenz; sinnvoll nur mit passenden Features gebaut. |
| `rusty_ai` (Meta-Crate) | Features `candle`, `candle-cuda`, **`hf-hub`** dokumentiert und `load_minigpt_from_hf` re-exportiert. |

## 2. Behobene / ergänzte Punkte

1. **GPT-2 QKV-Layout:** Zusätzlicher Pfad für transponiertes `c_attn.weight`, um Export-Varianten abzudecken; neuer Test `gpt2_mapping_accepts_transposed_c_attn`.
2. **Konfiguration:** `TryFrom<MiniGptConfigFile>` lehnt `vocab_size == 0`, `n_layer == 0`, `n_inner == 0`, `n_positions == 0` ab.
3. **Hub-Klarstellung:** Rustdoc zu `load_minigpt_from_hf` präzisiert erwartetes Checkpoint-Format.
4. **Dokumentation:** Dieses Dokument, `docs/README.md`, `rusty_ai_backend_candle/README.md`.

## 3. Bekannte Grenzen (absichtlich)

- **Training** des Mini-GPT bleibt auf dem CPU-Autograd; Candle ersetzt das nicht vollständig.
- **GPT-2-Tokenizer (BPE)** ist nicht nachgebaut; Gewichte können geladen werden, die Textpipeline mit Byte-Tokenizer weicht von OpenAI ab.
- **NCCL/Multi-GPU:** Nur über Candle-Umfeld und eigene Orchestrierung; `all_reduce_mean_cpu` ist eine mathematische Referenz ohne Netzwerk.

## 4. Empfohlene Tests

```bash
cargo test --workspace
cargo test -p rusty_ai_llm
cargo test -p rusty_ai_llm --features hf-hub   # optional Hub-API
cargo check -p rusty_ai --features candle,hf-hub
```
