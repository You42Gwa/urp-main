# MVP Implementation Plan

## Fixed Scope

Input: 10 English Wikipedia articles.

Output: image gate decisions, 20-30 single-annotator pilot labels, evidence KG JSON, and a Markdown report.

Excluded: agent orchestration, Neo4j, training/fine-tuning, full GraphRAG, and large-scale gold-set construction.

## Pipeline

```text
collect -> download -> enrich -> gate -> annotate -> build KG -> report
```

1. `collect`: retrieve article lead, sections, image metadata, caption, URL, license, and revision.
2. `download`: fetch allowed image files and validate MIME type/size.
3. `enrich`: generate BLIP caption, OCR text, and keywords.
4. `gate`: call Ollama VLM with image plus document context; validate structured JSON.
5. `annotate`: save manual pilot labels without overwriting model predictions.
6. `build KG`: create only provenance-preserving evidence triples.
7. `report`: summarize counts, keep/drop ratio, latency, failures, and representative cases.

## Evidence KG Schema

```text
Document -HAS_SECTION-> Section
Section  -HAS_CLAIM-> Claim
Section  -HAS_IMAGE-> Image
Image    -SUPPORTS-> Claim
Image    -DEPICTS-> Entity
Claim    -MENTIONS-> Entity
```

Every edge stores `source_document`, `section`, `evidence`, `model`, and `confidence`.

## Completion Criteria

- One command runs all stages after configuration.
- All image decisions retain prompt context, raw model response, parsed result, timing, and errors.
- Graph JSON includes source evidence for every generated edge.
- Report contains at least three keep, three drop, and one failure/uncertain case.
