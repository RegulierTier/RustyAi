# Sicherheit — `rusty_ai_agent`

Dieses Dokument ergänzt die Kurzregeln im [README](README.md) und ist für **Produkt-Teams** gedacht, die `AllowlistPolicy`, `RealExecutor` oder eigene Executors einbinden.

## Bedrohungsmodell

- **Vertrauensgrenze:** Alles, was ein LLM vorschlägt (Tool-JSON, Pfade, Argumente), gilt als **unvertrauenswürdig**, bis eure Policy es erlaubt.
- **Kein Sandbox-Ersatz:** `RealExecutor` führt Prozesse und Dateizugriffe aus — er ersetzt **keine** OS-Sandbox, Container oder AppArmor.

## Pfad- und Workspace-Regeln

1. **Arbeitsverzeichnis:** Alle relativen Pfade werden gegen ein explizites **Workspace-Root** aufgelöst (`RealExecutor::new`).
2. **`..` verbieten:** Wie in [`AllowlistPolicy`](src/policy.rs) — verhindert naive Pfad-Eskalation.
3. **Präfix-Allowlist:** Nur Pfade unter erlaubten Präfixen (z. B. `rusty_ai_agent/`, `examples/`).
4. **Symlinks:** Für Hochrisiko-Umgebungen: nach Auflösung prüfen, ob das kanonische Ziel noch unter dem Workspace liegt (Policy-Erweiterung im Produkt).

## `run_cmd`

1. **Binary-Allowlist** (z. B. nur `cargo`, `rustc`) — keine freie Shell (`sh -c`, `powershell -Command`, …).
2. **Argumente:** Keine vom Nutzer ungeprüften Strings in `argv` einfügen; feste Muster pro Aufgabe.
3. **Timeout und Ausgabelimits:** Lange `cargo`-Läufe begrenzen; stdout/stderr truncaten, um Speicher zu schützen.
4. **Umgebungsvariablen:** Keine Secrets in der Child-Umgebung ohne Review.

## `cargo test` (gezieltes Feedback)

Für schnelle Schleifen ist `cargo test -p <crate> -- <filter>` üblich. Nutze [`CargoTestInvocation`](src/cargo_test.rs), um `argv` **ohne Shell** zusammenzusetzen und gefährliche Zeichen in Filtern zu vermeiden. Die Binary-Allowlist muss weiterhin **`cargo`** enthalten; das Arbeitsverzeichnis (`cwd`) sollte auf das Workspace-Root zeigen. Keine zusammengesetzten Shell-Befehle (`sh -c "cargo test …"`).

## Netzwerk

Mit Feature **`http`** ([`OpenAiCompatBackend`](src/openai_compat.rs)): TLS über das System-`reqwest`-Backend; **API-Keys** nur über Umgebungsvariablen oder sicheren Konfigurationskanal — nie in Repos oder Logs. Für Ollama lokal oft **ohne** Bearer-Token (`api_key` leer).

## Checkliste vor Produktion

- [ ] Policy pro Umgebung (Dev / CI) mit unterschiedlichen Allowlists
- [ ] Maximale Tool-Runden / Token-Budgets
- [ ] Menschliche Freigabe für große `write_file`-Diffs (oder CI-only-Modus)

## Weiterführend

- **[`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`](../docs/ARCHITEKTUR_IDE_ROADMAP_B.md)** — Pfad B, Roadmap
- **[`docs/HANDBUCH.md`](../docs/HANDBUCH.md)** — Abschnitt **2.8** (`rusty_ai_agent`), **3.4** (Ablauf), Glossar
- **Beispiele** mit Policy/Executor: [`examples/agent_demo.rs`](examples/agent_demo.rs), Übersicht aller Beispiele im [Crate-README](README.md)
