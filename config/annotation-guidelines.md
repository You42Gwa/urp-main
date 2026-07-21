# Pilot Annotation Guidelines

Use document title, lead summary, section context, image, and caption together.

## Labels

- `representative = 1`: image or caption directly shows lead topic or one central claim.
- `knowledge_contribution = 1`: image adds a number, relation, spatial structure, temporal change, or visual distinction not recoverable from lead and caption alone.
- `keep_for_rag = 1`: either label is 1.
- `keep_for_kg = 1`: image provides extractable entity, relation, number, or structure with explicit evidence.

Record one short reason and supported section/claim. If uncertain, set `uncertain = true`; do not force a positive label.

This is a single-annotator pilot, not final inter-annotator evaluation.
