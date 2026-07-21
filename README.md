# Multimodal Image Gate for Evidence KG

An English Wikipedia-based research prototype. It selects images that represent a document's core claims or add extractable knowledge, then connects selected images to a provenance-preserving evidence knowledge graph.

## MVP Scope

- Corpus: 10 English Wikipedia articles
- Enrichment: image download, OCR, BLIP caption, and keywords
- Gate: Ollama VLM returns validated structured `keep/drop` decisions
- Evaluation: 20-30 single-annotator pilot labels
- Output: evidence KG JSON and a Markdown report

Agent orchestration, Neo4j, full GraphRAG, fine-tuning, and a full gold set are outside the MVP.

## Setup

Python 3.12 is installed locally. Run PowerShell from repository root:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[vision,dev]"
```

`vision` installs BLIP dependencies including PyTorch. Use a GPU-compatible PyTorch build when available.

## Configuration

1. Copy `config/articles.example.yaml` to `config/articles.yaml`.
2. Set the Ollama vision model name in `config/articles.yaml`.
3. Start Ollama locally and ensure the configured vision model is available:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
   ```

   The current local setup includes vision-capable `gemma4:e4b-it-q4_K_M`.
4. Keep downloaded data and generated reports local; `data/raw/`, `data/images/`, `data/processed/`, and `output/` are ignored by Git.

## Pipeline

```text
collect -> download -> enrich -> gate -> annotate -> build KG -> report
```

```powershell
python scripts/run_pipeline.py --config config/articles.yaml
```

The command is currently a scaffold. Each stage will be implemented in `src/mm_gate/`.

## Project Layout

```text
config/       Article list and annotation guidelines
data/         Local raw, image, and processed data
docs/         Research direction and implementation plan
scripts/      Command-line entry points
src/mm_gate/  Collection, enrichment, gate, KG, and report modules
tests/        Automated tests
```

## Project Docs

- `PROJECT_MEMORY.md`: durable project context, decisions, progress, and open items.
- `AGENTS.md`: contributor and AI-agent working rules.
- `docs/decisions/`: short records for significant technical decisions.
- `docs/research-direction.md`: multimodal research direction and execution plan.
- `docs/implementation-plan.md`: MVP scope, pipeline, and evidence KG schema.
