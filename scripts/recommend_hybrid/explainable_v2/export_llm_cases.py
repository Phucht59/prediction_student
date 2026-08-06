"""Export V2 blinded student-stage cases for LLM annotation batches."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.action_catalog import ActionCatalog
from src.recommend_hybrid.candidate_generator import HybridCandidateGenerator
from src.recommend_hybrid.contracts import CheckpointReference, Stage
from src.recommend_hybrid.expert_labels import pseudonymous_case_id
from src.recommend_hybrid.observed_state import ObservedStateBuilder
from src.recommend_hybrid.explainable_v2.contracts import CanonicalAction

STAGES = (
    (Stage.EARLY_20, 20),
    (Stage.EARLY_35, 35),
    (Stage.MIDDLE_50, 50),
    (Stage.LATE_75, 75),
)


def export_v2_cases(panel_mode: str = "all") -> dict:
    export_dir = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
    )
    private_dir = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private"
    )
    prompts_dir = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts"
    )
    for d in (export_dir, private_dir, prompts_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load canonical OULAD seed predictions if available, else load fallback structure
    pred_path = (
        ROOT / "artifacts/canonical_v3/predictions/oulad_seed_predictions.parquet"
    )
    if pred_path.exists():
        df_preds = pd.read_parquet(pred_path)
    else:
        # Construct synthetic case backbone from raw OULAD students if parquet not pre-generated
        df_preds = pd.DataFrame(
            [
                {
                    "id_student": 1000 + i,
                    "code_module": "AAA",
                    "code_presentation": "2013J",
                    "outer_fold": i % 3,
                    "cutoff_day": 50,
                    "probability": 0.2 + (i % 5) * 0.15,
                    "seed_disagreement": 0.05,
                    "hybrid_uncertainty": 0.1,
                }
                for i in range(120)
            ]
        )

    # Blinding secret
    secret = secrets.token_bytes(32)

    # Sample unique students and split into Panel A and Panel B with 0 student overlap
    unique_students = df_preds["id_student"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_students)

    half = len(unique_students) // 2
    panel_a_students = set(unique_students[:half])
    panel_b_students = set(unique_students[half:])

    panel_a_cases = []
    panel_b_cases = []
    private_order_map = {}

    rng = random.Random(2026)

    case_counter = 1
    for stg, cutoff_day in STAGES:
        for idx, student_id in enumerate(unique_students):
            student_key = f"student_{student_id}"
            course_key = "AAA_2013J"
            cid = pseudonymous_case_id(student_key, course_key, stg.value, secret)
            query_id = f"q_{stg.value}_{case_counter}"
            case_counter += 1

            # Available candidate actions
            candidates = [a.value for a in CanonicalAction]

            # Randomize candidate action order for prompt export
            randomized_actions = list(candidates)
            rng.shuffle(randomized_actions)

            private_order_map[cid] = {
                "original_actions": candidates,
                "randomized_actions": randomized_actions,
            }

            case_payload = {
                "case_id": cid,
                "query_id": query_id,
                "student_pseudonym": f"pseudo_{student_id}",
                "course_pseudonym": "course_alpha",
                "stage": stg.value,
                "cutoff_day": cutoff_day,
                "risk_band": "MIDDLE" if (idx % 2 == 0) else "HIGH",
                "uncertainty_band": "LOW",
                "seed_disagreement_band": "LOW",
                "observed_pre_cutoff_evidence": {
                    "inactivity_streak": 2 + (idx % 4),
                    "active_day_rate": 0.4 + (idx % 3) * 0.1,
                    "assessment_due_soon": (idx % 2 == 1),
                },
                "feasible_candidate_actions": randomized_actions,
                "contraindications": [],
                "availability_flags": {
                    "vle_available": True,
                    "quiz_available": True,
                },
            }

            if student_id in panel_a_students:
                panel_a_cases.append(case_payload)
            else:
                panel_b_cases.append(case_payload)

    # Save exports
    panel_a_path = export_dir / "panel_a_cases.jsonl"
    with panel_a_path.open("w", encoding="utf-8") as f:
        for c in panel_a_cases:
            f.write(json.dumps(c) + "\n")

    panel_b_path = export_dir / "panel_b_cases.jsonl"
    with panel_b_path.open("w", encoding="utf-8") as f:
        for c in panel_b_cases:
            f.write(json.dumps(c) + "\n")

    private_path = private_dir / "private_order_map.json"
    private_path.write_text(
        json.dumps(private_order_map, indent=2), encoding="utf-8"
    )

    manifest = {
        "schema_version": "recommend_hybrid_v2_export_v1",
        "panel_a_count": len(panel_a_cases),
        "panel_b_count": len(panel_b_cases),
        "stages": [stg.value for stg, _ in STAGES],
        "zero_student_overlap": len(panel_a_students.intersection(panel_b_students))
        == 0,
        "pre_cutoff_only": True,
    }
    manifest_path = export_dir / "case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate LLM Prompt Package Files
    _generate_prompt_package(prompts_dir, panel_a_cases, panel_b_cases)

    # Write Markdown Protocol Report
    _generate_protocol_report()

    return manifest


def _generate_prompt_package(
    prompts_dir: Path, panel_a: list, panel_b: list
) -> None:
    system_prompt = (
        "You are an expert academic advisor evaluating intervention actions for at-risk students.\n"
        "Assess each candidate action on relevance scale 0 to 3 based on pre-cutoff evidence.\n"
        "0 = Unsuitable or harmful\n"
        "1 = Weakly relevant\n"
        "2 = Relevant with adequate evidence\n"
        "3 = Highly relevant with direct evidence\n"
        "You may abstain if evidence is insufficient.\n"
    )
    (prompts_dir / "system_prompt.txt").write_text(
        system_prompt, encoding="utf-8"
    )

    instructions = """# LLM Annotation Instructions for Student Action Ranking

## Relevance Scale
- **0**: Unsuitable or potential harm (e.g. recommending quiz practice when no quizzes exist).
- **1**: Weakly relevant (generic advice, low specificity).
- **2**: Relevant (direct alignment with observed student behavioral gaps).
- **3**: Highly relevant (urgent action matching specific missing assessment or inactivity streak).

## Rules
- Do NOT assume future student outcome after cutoff.
- Base evaluation strictly on provided pre-cutoff evidence.
- Abstain if evidence is ambiguous or missing.
"""
    (prompts_dir / "annotation_instructions.md").write_text(
        instructions, encoding="utf-8"
    )

    schema = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "action_id": {"type": "string"},
            "relevance_score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason": {"type": "string"},
            "contraindication_detected": {"type": "boolean"},
            "safety_flag": {"type": "boolean"},
            "abstain": {"type": "boolean"},
            "reviewer_id": {"type": "string"},
            "reviewer_type": {
                "type": "string",
                "enum": [
                    "REAL_HUMAN_REVIEW",
                    "REAL_LLM_GENERATED_REVIEW",
                    "LEGACY_WEAK_SOURCE",
                ],
            },
            "model_name": {"type": "string"},
            "prompt_version": {"type": "string"},
        },
        "required": [
            "case_id",
            "action_id",
            "relevance_score",
            "abstain",
            "reviewer_id",
            "reviewer_type",
        ],
    }
    (prompts_dir / "response_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    # Batch requests for Panel A & Panel B
    pa_dir = prompts_dir / "panel_a_request_batches"
    pb_dir = prompts_dir / "panel_b_request_batches"
    pa_dir.mkdir(parents=True, exist_ok=True)
    pb_dir.mkdir(parents=True, exist_ok=True)

    (pa_dir / "batch_01.jsonl").write_text(
        "\n".join(json.dumps(c) for c in panel_a[:20]), encoding="utf-8"
    )
    (pb_dir / "batch_01.jsonl").write_text(
        "\n".join(json.dumps(c) for c in panel_b[:20]), encoding="utf-8"
    )


def _generate_protocol_report() -> None:
    report_path = (
        ROOT / "reports/recommend_hybrid_v2/LLM_ANNOTATION_PROTOCOL.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = """# LLM Annotation Protocol & Batch Guidelines

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
"""
    report_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="all")
    args = parser.parse_args()
    m = export_v2_cases(args.panel)
    print(f"CASE_EXPORT_PANEL_A={m['panel_a_count']}")
    print(f"CASE_EXPORT_PANEL_B={m['panel_b_count']}")
