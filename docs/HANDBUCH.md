# RustyAi — Handbuch

Dieses Handbuch beschreibt die **Architektur**, die **Module** des Workspaces und **typische Arbeitsabläufe**. Es richtet sich an Entwicklerinnen und Entwickler, die das Projekt erweitern oder einbinden möchten.

---

## 1. Überblick

### 1.1 Zielsetzung

RustyAi verbindet **klassisches überwachtes Lernen** (kleine MLPs, Optimierer) mit **LLM-Bausteinen** (Causal Attention, Decoder-Stack, Sampling) in einem einheitlichen Rust-Workspace. Die Implementierung legt Wert auf **Nachvollziehbarkeit**; Performance ist zweitrangig gegenüber Klarheit (CPU, keine eigenen CUDA-Kernels).

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
       │                        └── rusty_ai_llm   MiniGpt, TrainableMiniGpt, generate
       │
       └── rusty_ai (Meta-Crate, Re-Exports)
```

Datenfluss beim **Training**: Eingabe → `Tensor` / `Variable` → Schichten → Loss → `backward` → Optimizer-Schritt auf Parameter-Tensoren.

Datenfluss bei **LLM-Inferenz**: Token-IDs → Einbettungen → Decoder-Blöcke → Logits → `sample_token` / `generate` (kein Autograd nötig). **LLM-Training** nutzt `TrainableMiniGpt` mit demselben Forward wie `MiniGpt::forward`, Loss z. B. `Variable::cross_entropy_next_token`.

---

## 2. Crate-Referenz

### 2.1 `rusty_ai_core`

**Verantwortung:** Speicherlayout, Operationen, Fehlertypen.

- **`Tensor`:** Kontiguierter `f32`-Puffer, `Shape` als `Vec<usize>`, row-major (C-Ordnung).
- **`DType`:** Derzeit im Wesentlichen `F32`.
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
- **Operationen mit Gradient (Auswahl):** `Add`, `BiasAdd` (Batch-Matrix + Zeilen-Bias `(1,n)`), `MatMul` (2D und Batch-3D), `Mul` (Broadcast), `Relu`, `Gelu`, `LayerNorm` (letzte Achse), `SoftmaxLastDim`, `Reshape`, `TransposeBatchedLast2`, `CausalMask` (Scores), `EmbeddingGather`, `SplitHeads` / `MergeHeads`, `CrossEntropyNextToken` (Next-Token-LM, Ziele als Indizes), `Mse` (Ziel ist konstanter `Tensor`, kein Grad aufs Ziel).
- **`backward(loss)`:** Setzt den eingehenden Gradienten auf den Skalar-Loss auf `1` und verteilt rückwärts.
- **Kontext:** `grad_enabled()`, `set_grad_enabled`, `no_grad(|| { ... })` — im Inferenzpfad keine Graphen erzeugen.

**Typischer Fehler:** Gleiche `Variable`-Knoten müssen zwischen Epochen mit `zero_grad()` geleert werden, bevor ein neuer Forward/Backward-Lauf startet.

---

### 2.3 `rusty_ai_nn`

**Verantwortung:** Baukasten für kleine Netze.

- **`Linear`:** Gewichte und Bias als `Rc<Variable>`; `forward` → `matmul` + `bias_add`.
- **Initialisierung:** `glorot_uniform`, `uniform`, `zeros_bias` (siehe `init.rs`).
- **`gelu`:** Tensor-in-Tensor (tanh-Approximation).
- **`layer_norm`:** Nur Normalisierung über der letzten Dimension (keine lernbaren γ/β).

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
- **`MiniGptConfig`:** `vocab_size`, `d_model`, `n_heads`, `n_layers`, `ffn_dim`, `max_seq` (maximale Länge für **Positions-Einbettungen**; längere absolute Positionen werden auf `max_seq - 1` begrenzt).
- **`MiniGpt` / `DecoderBlock`:** Token- und Positions-Einbettung, vor-layernorm + Residual + Attention + FFN (GELU), Ausgabe-Linear `lm_head`.
- **`causal_attention` / `attention_single_query`:** Skaliertes Dot-Product; ersteres mit kausaler Maske für volle Sequenz, letzteres für einen Query mit Sequenzlänge 1 gegen die vollständige Key-/Value-Historie (nach KV-Konkatenation). Keine zusätzliche Maske nötig, da alle Keys „Vergangenheit“ sind.
- **`generate` / `sample_token`:** Autoregressiv; Temperatur, top-p (Nucleus); leeres Logit-Vektor führt zu `TensorError::EmptyTensor`. `generate` nutzt **Prefill** (einmalige volle Vorwärtsrechnung über den Prompt) und danach **KV-Cache** pro Layer. Bei `max_tokens == 0` wird nur encodiert/decodiert, ohne Modellaufruf.
- **`KvCache` / `LayerKv`:** Pro Layer gespeicherte Keys und Values mit Shape `(batch * heads, past_len, d_head)`; `MiniGpt::forward_prefill` und `forward_decode_step` füllen bzw. erweitern den Cache; `forward` / `forward_last` rechnen **ohne** Cache (jedes Mal die volle Sequenz — für Tests und einfache Vergleiche).
- **Weitere API:** `MiniGpt::embed_token_at(id, pos)` — kombinierte Token- und Positionszeile für einen Zeitschritt, Shape `(1, 1, d_model)`; wird intern für Decode-Schritte genutzt.

**Fehlerfälle (Auswahl):** Leere Token-ID-Liste bei `forward`, `forward_prefill` → `TensorError::EmptyTensor`. Inkonsistenter KV-Zustand (nur eines von `k`/`v` gesetzt) in `DecoderBlock::forward_step` → `ShapeError::IncompatibleBroadcast`.

---

### 2.6 `rusty_ai`

Re-Exportiert die Untercrates unter den Namen `core`, `autograd`, `nn`, `ml`, `llm` und eine Auswahl häufig genutzter Typen (`Tensor`, `Variable`, `Linear`, `Sgd`, `Adam`, `MiniGpt`, `TrainableMiniGpt`, `KvCache`, …).

---

## 3. Typische Abläufe

### 3.1 MLP trainieren (Regression)

1. `Linear`-Schichten mit `Linear::new(in, out, &mut seed)` anlegen.
2. Alle trainierbaren `Rc<Variable>` in einem Vektor sammeln (Gewichte + Biases).
3. Pro Epoche: Eingabe als `Variable::leaf(tensor)`, Ziel als `Tensor`, Forward (`relu`, `mse`), `zero_grad` auf allen Parametern, `backward`, Optimizer `step`.

Siehe `rusty_ai/examples/train_mlp.rs`.

### 3.2 LLM: nur Vorwärtsrechnung / Sampling

1. `MiniGpt::random(MiniGptConfig::default(), &mut seed)` oder eigene Konfiguration.
2. Prompt mit `ByteTokenizer::encode` → Token-IDs.
3. Je nach Ziel:
   - Volle Logits `(1, seq, vocab)`: `MiniGpt::forward(&token_ids)`.
   - Nur letztes Zeitschritt (z. B. nächstes Token): `forward_last` — intern ein voller Forward ohne KV-Cache.
   - Text generieren: `generate(&model, prompt, max_tokens, temperature, top_p, &mut seed)` — Prefill + KV-Cache pro generiertem Token.

**Manuelles Sampling mit KV-Cache** (gleiche Logik wie `generate`, aber eigenes Stepping):

```text
KvCache::new(cfg.n_layers)
logits_letzter_prompt = forward_prefill(&prompt_ids, &mut cache)   // liefert (1, vocab)
Schleife: nächstes Token wählen (z. B. sample_token)
          ids.push(token)
          logits = forward_decode_step(token, position, &mut cache)
```

Dabei ist **`position`** die **absolute** Indexposition des gerade erzeugten Tokens in der laufenden Sequenz (0-basiert), also nach dem ersten neuen Token typisch `prompt_len`, dann `prompt_len + 1`, … — entspricht der Zeile in `embed_positions` / `embed_token_at`.

### 3.3 Mini-GPT trainieren (Next-Token, Autograd)

1. `MiniGpt::random(...)` erzeugen, mit `TrainableMiniGpt::from_mini_gpt` übernehmen.
2. `parameters()` liefert alle `Rc<Variable>` für den Optimierer.
3. Pro Schritt: `zero_grad` auf allen Parametern, `forward(&token_ids)` → Logits `(1, seq, vocab)`, Loss `Variable::cross_entropy_next_token(&logits, &token_ids)` (Zielsequenz = Eingabe; pro Position `t` wird das nächste Token vorhergesagt), `backward`, `Adam::step` / `Sgd::step`.

Siehe `rusty_ai/examples/train_mini_gpt.rs`. KV-Cache wird für das Training nicht benötigt (volle Sequenz pro Schritt).

---

## 4. Grenzen und bekannte Einschränkungen

- **Nur CPU** in der Standardimplementierung.
- **Autograd** deckt die implementierten Ops ab; erweiterte Ops erfordern eigene `Op`-Varianten und Ableitungen.
- **Broadcasting im Autograd** ist nicht vollständig für alle Kombinationen in jedem Op abgebildet; `BiasAdd` ist für den üblichen Linear-Bias-Pfad gedacht.
- **LayerNorm** ohne affine Parameter — für Forschungs- oder Produktionsmodelle ggf. erweitern.
- **KV-Cache:** Kein Paging oder Ringpuffer; bei langen Generierungen wächst der Speicher pro Schritt (Konkatenation). Batch-Größe größer als 1 für LLM-Pfade ist nicht der Fokus der API.

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
| **KV-Cache** | Zwischengespeicherte Key- und Value-Tensoren pro Schicht, um bei autoregressiver Generierung nicht die komplette bisherige Sequenz erneut zu verarbeiten. |
| **Prefill** | Erster Inferenzschritt über den ganzen Prompt; füllt den KV-Cache und liefert Logits für die letzte Prompt-Position. |
| **top-p (Nucleus)** | Sampling aus der kleinsten Menge höchster Wahrscheinlichkeiten, deren Summe ≥ p. |
| **TrainableMiniGpt** | Decoder-only-Modell mit `Variable`-Gewichten; Forward wie `MiniGpt`, für Training mit `backward` + Optimierer. |
| **Next-Token-Cross-Entropy** | Mittlerer negativer Log-Likelihood über Positionen `t`, die `token[t+1]` aus `logits[0,t,:]` vorhersagen (`Variable::cross_entropy_next_token`). |
| **Variable** | Knoten mit Daten und optional Gradient im Autograd-Graphen. |

---

## 7. Versionshinweise

Dieses Handbuch bezieht sich auf den Stand des Repositories zum Zeitpunkt der letzten Bearbeitung. Für API-Details sind die `rustdoc`-Kommentare in den Quellen und `cargo doc` maßgeblich. Ein **Einstiegsindex** für die Dokumentation liegt in [`README.md`](README.md) im gleichen Ordner; das **Projekt-README** liegt im Repository-Root.
