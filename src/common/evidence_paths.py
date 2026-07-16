"""Stable resolver from frozen protocol paths to thesis-friendly namespaces.

The protocol JSON files are immutable scientific inputs whose hashes are part of
the evidence chain.  Their historical path strings therefore remain unchanged.
This resolver lets current code use the cleaner repository layout without
rewriting those frozen protocol bytes.
"""

from __future__ import annotations

from pathlib import Path


PATH_ALIASES = {
    "artifacts/protocol_v2": "artifacts/student_mat/development_splits",
    "artifacts/legacy_v1": "artifacts/archive/student_mat/legacy_dataset",
    "artifacts/strategy_b_phase_c/strategy-b-phase-c-20260714-5d34a66": "artifacts/student_mat/model_comparison",
    "artifacts/strategy_b_phase_e_prediction/strategy-b-phase-e-prediction-20260714-9007144": "artifacts/student_mat/prediction",
    "artifacts/final_repository_closure/final-repository-closure-corrected-20260715-6ab785d": "artifacts/student_mat/final",
    "artifacts/strategy_b_phase_d_recommendation/strategy-b-phase-d-recommendation-20260715-407ac0f": "artifacts/student_mat/recommendation",
    "artifacts/study_b_student_por/study-b-student-por-20260715-v1": "artifacts/student_por/final",
    "reports/study_b_student_por/study-b-student-por-20260715-v1": "reports/student_por/final",
    "artifacts/study_c_oulad/study-c-oulad-20260715-v1": "artifacts/oulad/baseline",
    "reports/study_c_oulad/study-c-oulad-20260715-v1": "reports/oulad/baseline",
    "artifacts/study_c_oulad_v2/oulad-deep-v2-f2-20260716-v1": "artifacts/oulad/tuning",
    "reports/study_c_oulad_v2/oulad-deep-v2-f2-20260716-v1": "reports/oulad/tuning",
    "artifacts/study_c_oulad_v3/oulad-deep-v3-f2-20260716-v1": "artifacts/oulad/temporal",
    "reports/study_c_oulad_v3/oulad-deep-v3-f2-20260716-v1": "reports/oulad/temporal",
    "artifacts/study_c_oulad_v3_closure/oulad-v3-fair-db-closure-20260716-v1": "artifacts/oulad/final",
    "reports/study_c_oulad_v3_closure/oulad-v3-fair-db-closure-20260716-v1": "reports/oulad/final",
}


OFFICIAL_EVIDENCE = {
    "student_mat": "artifacts/student_mat/final",
    "student_por": "artifacts/student_por/final",
    "oulad": "artifacts/oulad/final",
    "recommendation": "artifacts/student_mat/recommendation",
}


def resolve_evidence_path(root: Path, path: str | Path) -> Path:
    """Resolve a frozen relative path into the current repository layout."""

    relative = Path(path).as_posix().lstrip("./")
    for historical, current in sorted(PATH_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if relative == historical:
            return root / current
        prefix = historical + "/"
        if relative.startswith(prefix):
            return root / current / relative[len(prefix) :]
    return root / relative


def official_evidence_paths(root: Path) -> dict[str, Path]:
    return {name: root / path for name, path in OFFICIAL_EVIDENCE.items()}
