# Phase 2 implementation — recommend_hybrid foundation

## Scope delivered

Phase 2 implements `HybridPredictionAdapter`, immutable contracts, `ObservedStateBuilder`, the controlled action catalog, `HybridCandidateGenerator`, and blinded expert export/import. The validated flow is pre-cutoff data → frozen Hybrid CNN-BiLSTM outputs/embeddings → observed state → eligible candidates → expert review package.

No predictor weight, checkpoint byte, feature protocol, fold, seed, stage, split, prediction result, legacy artefact, database or production API is changed. No action ranker, pseudo-label, automatic relevance score, Top-K selection, constraint solver, learning-plan builder or training is present.

## Frozen adapter

The adapter loads only manifest-authorized checkpoint payloads, validates file SHA-256, architecture hash and 160,492 parameters, calls `eval()` under `torch.inference_mode()`, and reads existing forward outputs. Phase 2 validates all five shared-stage fold-0 seeds at `MIDDLE_50`; Phase 1 authority validation continues to validate all 30 canonical files and 75 stage/fold/seed mappings.

Five-seed probability is the arithmetic mean of seed probabilities. The predicted class uses the fold/stage threshold frozen in the canonical training authority, never an invented default threshold. Seed disagreement is population standard deviation (`ddof=0`), matching the repository's validated diagnostic convention. Uncertainty is binary predictive entropy of mean risk probability. Embeddings are seed means of the existing 64-D and 32-D outputs. Direct and adapted execution on the same CPU/float32 path are bit-exact.

## Observed state and catalog

Typed activity/assessment events are rejected if their known evidence time is at or after cutoff. Scores without a verified release timestamp remain unavailable. Missing activity or assessment evidence stays `None` with explicit missing masks and lineage. Course progress comes only from the canonical stage mapping.

Ten catalog actions are metadata-validated. Candidate generation evaluates stage, evidence, prerequisites, contraindications and human-review policy only. `FINAL_EVALUATION` yields zero eligible intervention candidates.

## Expert package

The pilot contains 60 representative canonical `MIDDLE_50` cases across three outer folds and probability bands. It provides 533 eligible candidate rows per blank reviewer template. The importer rejects invalid score and duplicate-rating smoke cases; those temporary validation rows are deleted and are not expert data.

## Reproduction

```powershell
.venv-oulad-v2/Scripts/python.exe scripts/recommend_hybrid/export_expert_cases.py --count 60
.venv-oulad-v2/Scripts/python.exe -m pytest tests/recommend_hybrid -q
.venv-oulad-v2/Scripts/python.exe scripts/recommend_hybrid/validate_phase2.py
```

Phase 3 remains blocked until real expert labels are supplied and explicitly approved.
