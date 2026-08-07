# External LLM Annotation Dispatch & Import Guide

## Overview
This guide provides step-by-step instructions for dispatching blinded student intervention evaluation cases to external LLM provider APIs (OpenAI, Anthropic, Google Gemini, etc.) and importing verified raw responses.

---

## 1. Request Package & Batch Locations

| Resource | Path |
|---|---|
| System Prompt | artifacts/recommend_hybrid/explainable_v2/annotations/prompts/system_prompt.txt |
| Annotation Instructions | artifacts/recommend_hybrid/explainable_v2/annotations/prompts/annotation_instructions.md |
| Response Schema | artifacts/recommend_hybrid/explainable_v2/annotations/prompts/response_schema.json |
| Panel A Request Batches | artifacts/recommend_hybrid/explainable_v2/annotations/prompts/panel_a_request_batches/batch_*.jsonl |
| Panel B Request Batches | artifacts/recommend_hybrid/explainable_v2/annotations/prompts/panel_b_request_batches/batch_*.jsonl |

---

## 2. Dispatching to External LLM Providers

### Target Reviewer Configurations (Minimum 2 Independent Sources Required)
1. REVIEWER_A: Behavioral Evidence Specialist (e.g. OpenAI GPT-4o)
2. REVIEWER_B: Stage Appropriateness Specialist (e.g. Anthropic Claude 3.5 Sonnet)
3. REVIEWER_C: Pedagogical Safety Specialist (e.g. Google Gemini 1.5 Pro)

---

## 3. Storage Structure for Returned Raw Responses

Place returned raw JSONL response files in:
artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw/

Name files according to the format:
- panel_a_<reviewer_id>_<batch_id>.jsonl
- panel_b_<reviewer_id>_<batch_id>.jsonl

---

## 4. Response Provenance Fields Required

Each review record inside the response JSONL MUST include authentic provider metadata:
- case_id / query_id
- action_id
- relevance_score: Integer (0, 1, 2, 3) or abstain: true
- reviewer_id: String (e.g. REVIEWER_A)
- reviewer_type: REAL_EXTERNAL_LLM_REVIEW
- provider: String (e.g. OpenAI, Anthropic, Google)
- model_name: String (e.g. gpt-4o, claude-3-5-sonnet, gemini-1-5-pro)
- request_id: Provider request/response ID string (e.g. req_abc123)
- rationale: Non-empty text justification

---

## 5. Execution & Import Commands

python scripts/recommend_hybrid/explainable_v2/import_llm_annotations.py
python scripts/recommend_hybrid/explainable_v2/audit_annotation_independence.py
python scripts/recommend_hybrid/explainable_v2/supervisor.py

---

## 6. Current Scientific Status

- EXTERNAL_PROVIDER_STATUS: UNAVAILABLE
- SCIENTIFIC_STATUS: BLOCKED_PENDING_EXTERNAL_LLM_ACCESS
- FINAL_VERIFIER_EXIT_CODE: 2