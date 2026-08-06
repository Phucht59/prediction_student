# LLM Annotation Protocol & Batch Guidelines

## Overview
This protocol defines how blinded student cases are exported and formatted for multi-review LLM annotations.

## Panel Structure
- **Panel A (Development & Calibration)**: Used for LF tuning, Snorkel preliminary models, and score calibration.
- **Panel B (Held-Out Benchmark)**: Independent pseudo-expert benchmark panel. Zero student overlap with Panel A.

## Multi-Reviewer Setup
Each case can be evaluated by multiple LLM model/prompt variants:
- `LLM_A_PROMPT_1`
- `LLM_A_PROMPT_2`
- `LLM_B_PROMPT_1`

## Instructions for External LLM Execution
1. Send request batches from `artifacts/recommend_hybrid/explainable_v2/annotations/prompts/panel_a_request_batches/`.
2. Format responses according to `artifacts/recommend_hybrid/explainable_v2/annotations/prompts/response_schema.json`.
3. Place returned JSON/JSONL responses into `artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw/`.
4. Run the importer:
   ```bash
   python scripts/recommend_hybrid/explainable_v2/import_llm_annotations.py
   ```
