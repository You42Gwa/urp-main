# Multimodal Image Gate for Evidence KG

An English Wikipedia-based research prototype. It selects images that represent a document's core claims or add extractable knowledge, then connects selected images to a provenance-preserving evidence knowledge graph.

## Research Question

> Before multimodal RAG or an evidence KG admits a document image, can a vision MLLM decide whether that image represents a core document claim or adds useful visual evidence?

The gate receives the image plus document title, lead, section, Wikipedia caption, BLIP caption, and OCR text. It returns validated structured `keep/drop` evidence with a reason and confidence.

## Current MVP Status

Completed on a local English Wikipedia pilot corpus:

- 10 articles and 94 downloaded images with source and license provenance
- Tesseract OCR and BLIP captions completed for 90 raster images; 4 SVG images intentionally skipped
- Balanced 10-image gate pilot completed with 10 valid structured decisions
- Pilot mean gate confidence: 0.93; evidence KG: 40 nodes and 40 edges
- Automated checks: 10 tests pass and Ruff passes

The pilot images were selected from document-leading images, so all 10 were `keep`. This verifies the pipeline, not filtering accuracy. Human labels and mixed keep/drop evaluation are next.

## MVP Scope

- Corpus: 10 English Wikipedia articles
- Enrichment: image download, OCR, BLIP caption, and keywords
- Gate: Ollama VLM returns validated structured `keep/drop` decisions
- Next evaluation: 20-30 mixed single-annotator pilot labels
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
collect -> download -> enrich -> caption -> gate -> build KG -> report
```

```powershell
python scripts/run_pipeline.py collect --config config/articles.example.yaml
python scripts/run_pipeline.py download
python scripts/run_pipeline.py enrich
python scripts/run_pipeline.py caption
python scripts/run_pipeline.py gate --max-per-article 3
python scripts/run_pipeline.py kg
python scripts/run_pipeline.py report
```

Collection, download, OCR/keyword enrichment, BLIP captioning, Ollama gating, evidence KG construction, and reporting are implemented.

## Current Gate Stack

- Gate runtime/model: Ollama 0.20.2 with `gemma4:e4b-it-q4_K_M`
- Gate output: `representative`, `knowledge_contribution`, `image_type`, `keep/drop`, `confidence`, and `reason`
- OCR: Tesseract 5.4 with English language data
- Caption model: `Salesforce/blip-image-captioning-base`
- CLIP is not in the current pipeline; it is a planned similarity baseline for later evaluation.

## Local-Only Artifacts

The following are intentionally Git-ignored and must not be committed: runtime corpus data, generated KG/report files, presentation files/scripts under `output/`, local virtual environment/cache, `PROJECT_MEMORY.md`, and local agent skill files.

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
