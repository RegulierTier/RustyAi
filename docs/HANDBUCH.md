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
       │                        └── rusty_ai_llm   MiniGpt, generate
       │
       └── rusty_ai (Meta-Crate, Re-Exports)
```

Datenfluss beim **Training**: Eingabe → `Tensor` / `Variable` → Schichten → Loss → `backward` → Optimizer-Schritt auf Parameter-Tensoren.

Datenfluss bei **LLM-Inferenz**: Token-IDs → Einbettungen → Decoder-Blöcke → Logits → `sample_token` / `generate` (kein Autograd nötig).

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
- **Operationen mit Gradient:** `Add`, `BiasAdd` (Batch-Matrix + Zeilen-Bias `(1,n)`), `MatMul`, `Relu`, `Mse` (Ziel ist konstanter `Tensor`, kein Grad aufs Ziel).
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
- **`MiniGpt` / `DecoderBlock`:** Token- und Positions-Einbettung, vor-layernorm + Residual + Attention + FFN (GELU), Ausgabe-Linear `lm_head`.
- **`causal_attention`:** Skaliertes Dot-Product mit Maskierung oberhalb der Diagonale.
- **`generate` / `sample_token`:** Autoregressiv; Temperatur, top-p (Nucleus); leeres Vokabular führt zu `TensorError::EmptyTensor`.
- **`LayerKv`:** Platzhalter-Struktur für zukünftigen KV-Cache.

**Hinweis:** Die Standard-Generierung **ohne** KV-Cache ist einfacher, aber bei langer Sequenz langsamer (volle Vorwärtsrechnung pro Schritt).

---

### 2.6 `rusty_ai`

Re-Exportiert die Untercrates unter den Namen `core`, `autograd`, `nn`, `ml`, `llm` und eine Auswahl häufig genutzter Typen (`Tensor`, `Variable`, `Linear`, `Sgd`, `Adam`, `MiniGpt`, …).

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
3. `forward` oder `forward_last` für Logits; `generate` für fortlaufende Ausgabe.

Training des LLM ist im Framework **nicht** als fertiger Trainingsloop vorgefunden; Gewichte werden typischerweise für Demozwecke zufällig initialisiert.

---

## 4. Grenzen und bekannte Einschränkungen

- **Nur CPU** in der Standardimplementierung.
- **Autograd** deckt die implementierten Ops ab; erweiterte Ops erfordern eigene `Op`-Varianten und Ableitungen.
- **Broadcasting im Autograd** ist nicht vollständig für alle Kombinationen in jedem Op abgebildet; `BiasAdd` ist für den üblichen Linear-Bias-Pfad gedacht.
- **LayerNorm** ohne affine Parameter — für Forschungs- oder Produktionsmodelle ggf. erweitern.

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
| **top-p (Nucleus)** | Sampling aus der kleinsten Menge höchster Wahrscheinlichkeiten, deren Summe ≥ p. |
| **Variable** | Knoten mit Daten und optional Gradient im Autograd-Graphen. |

---

## 7. Versionshinweise

Dieses Handbuch bezieht sich auf den Stand des Repositories zum Zeitpunkt der letzten Bearbeitung. Für API-Details sind die `rustdoc`-Kommentare in den Quellen und `cargo doc` maßgeblich.
