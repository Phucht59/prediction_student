"""Export blinded query-level student-stage cases for LLM annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.action_eligibility import (
    evaluate_action_eligibility,
)
from src.recommend_hybrid.explainable_v2.query_evidence import (
    AVAILABILITY_FIELDS,
    QUERY_EVIDENCE_FIELDS,
)
from src.recommend_hybrid.explainable_v2.sampling import (
    perform_grouped_stratified_sampling,
)

QUERY_EVIDENCE_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/features"
    / "query_level_evidence.parquet"
)
CANDIDATES_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/features"
    / "action_candidates.parquet"
)
FEATURE_MANIFEST_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/features"
    / "QUERY_EVIDENCE_MANIFEST.json"
)
EXPORT_DIR = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
)
PRIVATE_DIR = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/private"
)
PROMPTS_DIR = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts"
)

PANEL_A_TARGET = 300
PANEL_B_TARGET = 150
ALL_ACTIONS = [
    "QUIZ_RETRIEVAL_PRACTICE",
    "ASSESSMENT_COMPLETION",
    "RECOVER_ENGAGEMENT",
    "STUDY_REGULARITY",
    "TARGETED_CONTENT_REVIEW",
]


def _blinded_case_id(
    raw_query_id: str,
    salt: str | None = None,
) -> str:
    if salt is None:
        if "CASE_EXPORT_SALT" not in os.environ:
            raise KeyError(
                "CASE_EXPORT_SALT environment variable is required"
            )
        salt = os.environ["CASE_EXPORT_SALT"]
    return (
        "case_"
        + hashlib.sha256(
            salt.encode("utf-8")
            + b"_"
            + raw_query_id.encode("utf-8")
        ).hexdigest()[:24]
    )


def _row_sha256(row_dict: dict) -> str:
    row_bytes = json.dumps(
        row_dict,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(row_bytes).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _python_value(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def export_v2_cases() -> dict:
    if "CASE_EXPORT_SALT" not in os.environ:
        raise RuntimeError(
            "CASE_EXPORT_SALT environment variable is required"
        )
    salt = os.environ["CASE_EXPORT_SALT"]

    for directory in (EXPORT_DIR, PRIVATE_DIR, PROMPTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for path in (
        QUERY_EVIDENCE_PATH,
        CANDIDATES_PATH,
        FEATURE_MANIFEST_PATH,
    ):
        if not path.exists():
            raise RuntimeError(
                f"MISSING_QUERY_EVIDENCE_ARTIFACT={path}"
            )

    query_df = pd.read_parquet(QUERY_EVIDENCE_PATH)
    candidate_df = pd.read_parquet(CANDIDATES_PATH)

    if query_df["query_id"].duplicated().any():
        raise RuntimeError(
            "QUERY_EVIDENCE_DUPLICATE_QUERY_ID"
        )
    if candidate_df.duplicated(
        ["query_id", "action_id"]
    ).any():
        raise RuntimeError(
            "CANDIDATE_DUPLICATE_QUERY_ACTION"
        )
    if len(candidate_df) != len(query_df) * 5:
        raise RuntimeError(
            "CANDIDATE_ROW_COUNT_NOT_EXACTLY_5X"
        )

    shared = [
        *QUERY_EVIDENCE_FIELDS,
        *AVAILABILITY_FIELDS,
    ]
    varying = (
        candidate_df.groupby("query_id")[shared]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if bool(varying.any()):
        raise RuntimeError(
            "ACTION_CONDITIONED_EVIDENCE_DETECTED"
        )

    query_df = (
        query_df.sort_values("query_id")
        .reset_index(drop=True)
    )
    query_groups = query_df.groupby(
        "query_id",
        sort=True,
    )

    public_cases: dict[str, dict] = {}
    private_mappings: dict[str, dict] = {}
    student_to_queries: dict[str, list[str]] = {}
    query_strata: dict[str, str] = {}

    query_evidence_sha = _file_sha256(
        QUERY_EVIDENCE_PATH
    )
    candidate_sha = _file_sha256(CANDIDATES_PATH)
    manifest_sha = _file_sha256(
        FEATURE_MANIFEST_PATH
    )

    for first in query_df.itertuples(index=False):
        query_id = str(first.query_id)
        stage_value = str(first.stage)
        student_group_id = str(first.student_group_id)
        outer_fold = int(first.outer_fold)
        risk_band = str(first.risk_band)
        uncertainty = float(first.hybrid_uncertainty)
        uncertainty_band = (
            "HIGH"
            if uncertainty > 0.3
            else (
                "MEDIUM"
                if uncertainty > 0.15
                else "LOW"
            )
        )

        pre_cutoff = {
            field: _python_value(
                getattr(first, field)
            )
            for field in QUERY_EVIDENCE_FIELDS
        }
        availability = {
            "vle_available": bool(
                first.vle_available
            ),
            "study_material_available": bool(
                first.study_material_available
            ),
            "quiz_available": bool(
                first.quiz_available
            ),
        }
        case_features = {
            **pre_cutoff,
            **availability,
            "stage": stage_value,
        }

        candidate_actions = []
        contraindications = []
        for action in ALL_ACTIONS:
            eligible, code = (
                evaluate_action_eligibility(
                    case_features,
                    action,
                )
            )
            if eligible:
                candidate_actions.append(action)
            elif code.startswith("CONTRAINDICATED"):
                contraindications.append(action)

        candidate_actions = [
            action
            for action in candidate_actions
            if action not in contraindications
        ]
        routing_status = (
            "FEASIBLE"
            if candidate_actions
            else "NO_FEASIBLE_ACTION"
        )

        case_id = _blinded_case_id(
            query_id,
            salt=salt,
        )
        public_cases[query_id] = {
            "case_id": case_id,
            "panel_id": "PENDING_PANEL_ASSIGNMENT",
            "stage": stage_value,
            "cutoff_day": int(first.cutoff_day),
            "risk_band": risk_band,
            "uncertainty_band": uncertainty_band,
            "routing_status": routing_status,
            "observed_pre_cutoff_evidence": (
                pre_cutoff
            ),
            "candidate_actions": candidate_actions,
            "availability_flags": availability,
            "contraindications": contraindications,
        }

        feature_row = {
            column: _python_value(
                getattr(first, column)
            )
            for column in query_df.columns
        }
        private_mappings[case_id] = {
            "case_id": case_id,
            "source_query_id": query_id,
            "source_student_group_id": (
                student_group_id
            ),
            "outer_fold": outer_fold,
            "source_query_evidence_row_sha256": (
                _row_sha256(feature_row)
            ),
            "source_query_evidence_sha256": (
                query_evidence_sha
            ),
            "source_candidate_table_sha256": (
                candidate_sha
            ),
            "source_query_evidence_manifest_sha256": (
                manifest_sha
            ),
        }

        student_to_queries.setdefault(
            student_group_id,
            [],
        ).append(query_id)
        query_strata[query_id] = (
            f"fold{outer_fold}_"
            f"{stage_value}_{risk_band}"
        )

    panel_a_qids, panel_b_qids, sampling_audit = (
        perform_grouped_stratified_sampling(
            df=query_df,
            query_groups=query_groups,
            student_to_queries=student_to_queries,
            query_strata=query_strata,
            panel_a_target=PANEL_A_TARGET,
            panel_b_target=PANEL_B_TARGET,
            seed=2026,
        )
    )

    panel_a_cases = []
    for query_id in panel_a_qids:
        case = dict(public_cases[query_id])
        case["panel_id"] = "PANEL_A"
        panel_a_cases.append(case)

    panel_b_cases = []
    for query_id in panel_b_qids:
        case = dict(public_cases[query_id])
        case["panel_id"] = "PANEL_B"
        panel_b_cases.append(case)

    forbidden_keys = {
        "query_id",
        "source_query_id",
        "id_student",
        "student_group_id",
        "module",
        "presentation",
        "outer_fold",
    }
    for case in panel_a_cases + panel_b_cases:
        leaked = forbidden_keys & set(case)
        if leaked:
            raise RuntimeError(
                "PRIVACY_LEAK_DETECTED="
                + str(sorted(leaked))
            )

    with (
        EXPORT_DIR / "panel_a_cases.jsonl"
    ).open("w", encoding="utf-8") as handle:
        for case in panel_a_cases:
            handle.write(json.dumps(case) + "\n")

    with (
        EXPORT_DIR / "panel_b_cases.jsonl"
    ).open("w", encoding="utf-8") as handle:
        for case in panel_b_cases:
            handle.write(json.dumps(case) + "\n")

    (
        PRIVATE_DIR / "private_case_mapping.json"
    ).write_text(
        json.dumps(private_mappings, indent=2) + "\n",
        encoding="utf-8",
    )
    (
        EXPORT_DIR / "SAMPLING_AUDIT.json"
    ).write_text(
        json.dumps(sampling_audit, indent=2) + "\n",
        encoding="utf-8",
    )

    action_counts_a = {
        action: sum(
            action in case["candidate_actions"]
            for case in panel_a_cases
        )
        for action in ALL_ACTIONS
    }
    action_counts_b = {
        action: sum(
            action in case["candidate_actions"]
            for case in panel_b_cases
        )
        for action in ALL_ACTIONS
    }
    manifest = {
        "source_query_evidence_sha256": (
            query_evidence_sha
        ),
        "source_candidates_sha256": candidate_sha,
        "source_manifest_sha256": manifest_sha,
        "total_eligible_queries": len(public_cases),
        "total_eligible_students": len(
            student_to_queries
        ),
        "panel_a_case_count": len(panel_a_cases),
        "panel_b_case_count": len(panel_b_cases),
        "panel_student_overlap_count": (
            sampling_audit["student_overlap"]
        ),
        "panel_query_overlap_count": (
            sampling_audit["query_overlap"]
        ),
        "zero_student_overlap": (
            sampling_audit["student_overlap"] == 0
        ),
        "zero_query_overlap": (
            sampling_audit["query_overlap"] == 0
        ),
        "public_privacy_verified": True,
        "synthetic_fixture_used": False,
        "lineage_source": (
            "query_level_evidence.parquet"
        ),
        "case_export_classification": (
            "VERIFIED_OULAD_QUERY_LEVEL_LINEAGE_V4"
        ),
        "query_level_evidence_invariant_across_actions": (
            True
        ),
        "panel_a_action_candidate_counts": (
            action_counts_a
        ),
        "panel_b_action_candidate_counts": (
            action_counts_b
        ),
        "runtime_authorized": False,
    }
    (
        EXPORT_DIR / "case_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    _generate_prompt_package(
        PROMPTS_DIR,
        panel_a_cases,
        panel_b_cases,
    )
    return manifest


def _generate_prompt_package(
    prompts_dir: Path,
    panel_a: list,
    panel_b: list,
) -> None:
    system_prompt = """You are an expert academic advisor evaluating intervention actions for at-risk students.
Assess each candidate action on relevance scale 0 to 3 based on pre-cutoff evidence.
0 = Unsuitable or harmful
1 = Weakly relevant
2 = Relevant with adequate evidence
3 = Highly relevant with direct evidence
You may abstain if evidence is insufficient.
"""
    (
        prompts_dir / "system_prompt.txt"
    ).write_text(
        system_prompt,
        encoding="utf-8",
    )

    instructions = """# LLM Annotation Instructions for Student Action Ranking

All public evidence is query-level and constructed before action expansion.

## Relevance Scale
- **0**: Unsuitable or potential harm.
- **1**: Weakly relevant.
- **2**: Relevant with adequate evidence.
- **3**: Highly relevant with direct evidence.

External provenance is validated separately by the fail-closed importer.
"""
    (
        prompts_dir / "annotation_instructions.md"
    ).write_text(
        instructions,
        encoding="utf-8",
    )

    panel_a_dir = (
        prompts_dir / "panel_a_request_batches"
    )
    panel_b_dir = (
        prompts_dir / "panel_b_request_batches"
    )
    panel_a_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    panel_b_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old in panel_a_dir.glob("batch_*.jsonl"):
        old.unlink()
    for old in panel_b_dir.glob("batch_*.jsonl"):
        old.unlink()

    for index in range(0, len(panel_a), 50):
        number = index // 50 + 1
        batch = panel_a[index : index + 50]
        (
            panel_a_dir
            / f"batch_{number:02d}.jsonl"
        ).write_text(
            "\n".join(
                json.dumps(case)
                for case in batch
            ),
            encoding="utf-8",
        )

    for index in range(0, len(panel_b), 50):
        number = index // 50 + 1
        batch = panel_b[index : index + 50]
        (
            panel_b_dir
            / f"batch_{number:02d}.jsonl"
        ).write_text(
            "\n".join(
                json.dumps(case)
                for case in batch
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        default="all",
    )
    parser.parse_args()
    result = export_v2_cases()
    print(
        "CASE_EXPORT_STATUS="
        "VERIFIED_OULAD_QUERY_LEVEL_LINEAGE_V4"
    )
    print("PUBLIC_PRIVACY_VERIFIED=TRUE")
    print(
        "CASE_EXPORT_PANEL_A="
        + str(result["panel_a_case_count"])
    )
    print(
        "CASE_EXPORT_PANEL_B="
        + str(result["panel_b_case_count"])
    )
