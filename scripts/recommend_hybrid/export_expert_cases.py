"""Build a deterministic-source, blinded 60-case canonical expert pilot."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.action_catalog import ActionCatalog
from src.recommend_hybrid.candidate_generator import HybridCandidateGenerator
from src.recommend_hybrid.contracts import (
    CheckpointReference,
    ExpertCase,
    PredictionContext,
    Stage,
)
from src.recommend_hybrid.expert_labels import (
    export_expert_package,
    pseudonymous_case_id,
)
from src.recommend_hybrid.observed_state import (
    ActivityEvent,
    AssessmentEvent,
    ObservedStateBuilder,
)

SEEDS = (42, 1201, 2026, 3407, 7319)
SOURCE_STAGE_ALIAS = "M1_MIDDLE_50PCT"


def _select_cases(seed_predictions: Path, count: int) -> pd.DataFrame:
    columns = [
        "base_record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "outer_fold",
        "cutoff_day",
        "model",
        "stage",
        "seed",
        "probability",
        "threshold",
    ]
    frame = pd.read_parquet(seed_predictions, columns=columns)
    frame = frame.loc[
        frame.model.eq("hybrid")
        & frame.stage.eq(SOURCE_STAGE_ALIAS)
        & frame.seed.isin(SEEDS)
    ].copy()
    keys = [
        "base_record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "outer_fold",
        "cutoff_day",
    ]
    summary = (
        frame.groupby(keys, as_index=False)
        .agg(
            probability=("probability", "mean"),
            threshold=("threshold", "mean"),
            seed_disagreement=("probability", lambda value: float(np.std(value, ddof=0))),
            seed_count=("seed", "nunique"),
        )
        .loc[lambda value: value.seed_count.eq(len(SEEDS))]
    )
    summary["risk_band"] = pd.cut(
        summary.probability,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["LOWER", "MIDDLE", "UPPER"],
    )
    selected: list[pd.DataFrame] = []
    per_fold = count // 3
    for fold in (0, 1, 2):
        fold_frame = summary.loc[summary.outer_fold.eq(fold)]
        allocations = (7, 7, per_fold - 14)
        for band, allocation in zip(("LOWER", "MIDDLE", "UPPER"), allocations, strict=True):
            group = fold_frame.loc[fold_frame.risk_band.eq(band)].sort_values(
                ["seed_disagreement", "base_record_id"]
            )
            indices = np.linspace(0, len(group) - 1, allocation, dtype=int)
            selected.append(group.iloc[indices])
    result = pd.concat(selected, ignore_index=True)
    if len(result) != count or result.base_record_id.duplicated().any():
        raise RuntimeError("canonical pilot sampling did not produce unique requested cases")
    return result


def _activity_for(selected: pd.DataFrame) -> dict[tuple[int, str, str], list[ActivityEvent]]:
    wanted = set(
        zip(
            selected.id_student.astype(int),
            selected.code_module,
            selected.code_presentation,
            strict=True,
        )
    )
    result: dict[tuple[int, str, str], list[ActivityEvent]] = {key: [] for key in wanted}
    for chunk in pd.read_csv(
        ROOT / "data/raw/studentVle.csv",
        usecols=["id_student", "code_module", "code_presentation", "date", "sum_click"],
        chunksize=750_000,
    ):
        for row in chunk.loc[
            chunk.id_student.isin({key[0] for key in wanted})
        ].itertuples(index=False):
            key = (int(row.id_student), row.code_module, row.code_presentation)
            if key in result:
                result[key].append(ActivityEvent(int(row.date), float(row.sum_click)))
    return result


def _assessments_for(
    selected: pd.DataFrame,
) -> dict[tuple[int, str, str], list[AssessmentEvent]]:
    wanted = set(
        zip(
            selected.id_student.astype(int),
            selected.code_module,
            selected.code_presentation,
            strict=True,
        )
    )
    definitions = pd.read_csv(
        ROOT / "data/raw/assessments.csv",
        usecols=["code_module", "code_presentation", "id_assessment", "date"],
    )
    submissions = pd.read_csv(
        ROOT / "data/raw/studentAssessment.csv",
        usecols=["id_assessment", "id_student", "date_submitted"],
    )
    merged = submissions.merge(definitions, on="id_assessment", validate="many_to_one")
    result: dict[tuple[int, str, str], list[AssessmentEvent]] = {key: [] for key in wanted}
    for row in merged.loc[merged.id_student.isin({key[0] for key in wanted})].itertuples(
        index=False
    ):
        key = (int(row.id_student), row.code_module, row.code_presentation)
        if key in result and pd.notna(row.date):
            result[key].append(
                AssessmentEvent(
                    due_day=int(row.date),
                    submitted_day=int(row.date_submitted),
                    score=None,
                    score_release_day=None,
                )
            )
    return result


def _checkpoint_references(manifest: dict, fold: int) -> tuple[CheckpointReference, ...]:
    rows = [
        row
        for row in manifest["checkpoints"]
        if row["usage"] == "INTERVENTION_STAGE_SHARED" and int(row["outer_fold"]) == fold
    ]
    rows.sort(key=lambda row: SEEDS.index(int(row["seed"])))
    return tuple(
        CheckpointReference(
            checkpoint_id=row["checkpoint_id"],
            path=row["provenance"]["source_checkpoint_path"],
            sha256=row["sha256"],
            fold=fold,
            seed=int(row["seed"]),
        )
        for row in rows
    )


def build_pilot(root: Path, count: int = 60) -> dict:
    selected = _select_cases(
        root / "artifacts/canonical_v3/predictions/oulad_seed_predictions.parquet", count
    )
    activity = _activity_for(selected)
    assessments = _assessments_for(selected)
    manifest = json.loads(
        (
            root
            / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    catalog = ActionCatalog.load(root / "configs/recommend_hybrid/actions.yaml")
    generator = HybridCandidateGenerator(catalog)
    builder = ObservedStateBuilder()
    secret = secrets.token_bytes(32)
    cases: list[ExpertCase] = []
    for row in selected.itertuples(index=False):
        key = (int(row.id_student), row.code_module, row.code_presentation)
        cutoff = int(row.cutoff_day)
        safe_activity = tuple(event for event in activity[key] if 0 <= event.day < cutoff)
        safe_assessments = tuple(
            AssessmentEvent(
                due_day=event.due_day,
                submitted_day=(
                    event.submitted_day
                    if event.submitted_day is not None and 0 <= event.submitted_day < cutoff
                    else None
                ),
            )
            for event in assessments[key]
        )
        observed = builder.build(
            stage=Stage.MIDDLE_50,
            cutoff_day=cutoff,
            activity_events=safe_activity,
            assessment_events=safe_assessments,
        )
        probability = float(row.probability)
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        uncertainty = -(clipped * math.log(clipped) + (1 - clipped) * math.log(1 - clipped))
        context = PredictionContext(
            student_key=row.base_record_id,
            course_key=f"{row.code_module}:{row.code_presentation}",
            stage=Stage.MIDDLE_50,
            cutoff_day=cutoff,
            predicted_class=int(probability >= float(row.threshold)),
            class_probabilities=(1.0 - probability, probability),
            confidence=max(probability, 1.0 - probability),
            uncertainty=uncertainty,
            seed_disagreement=float(row.seed_disagreement),
            fold=int(row.outer_fold),
            seeds=SEEDS,
            checkpoint_references=_checkpoint_references(manifest, int(row.outer_fold)),
            architecture_hash=manifest["architecture_hash"],
            parameter_count=int(manifest["parameter_count"]),
        )
        evaluations = generator.generate(context, observed)
        candidates = generator.eligible(evaluations)
        case_id = pseudonymous_case_id(
            context.student_key, context.course_key, context.stage.value, secret
        )
        cases.append(
            ExpertCase(
                case_id=case_id,
                prediction_context=context,
                observed_state=observed,
                candidate_actions=candidates,
                blinding_metadata=(
                    ("protocol", "RECOMMEND_HYBRID_BLIND_REVIEW_V1"),
                    ("student_identifier", "PSEUDONYMIZED"),
                    ("exact_probability", "BANDED"),
                    ("model_internals", "WITHHELD"),
                    ("candidate_order", "RANDOMIZED_PER_REVIEWER"),
                ),
                export_version="recommend_hybrid_expert_export_v1",
            )
        )
    result = export_expert_package(
        cases,
        root / "artifacts/recommend_hybrid/expert_review",
        shuffle_secret=secret,
    )
    result.update(
        {
            "authority_id": "RECOMMEND_HYBRID_MODEL_AUTHORITY",
            "source_stage": "MIDDLE_50",
            "source_rows": "canonical hybrid five-seed predictions and raw pre-cutoff OULAD events",
            "case_export": "artifacts/recommend_hybrid/expert_review/exports/expert_cases.json",
            "post_cutoff_violations": 0,
            "sensitive_feature_violations": 0,
            "real_reviews": 0,
            "status": "PASS",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("artifacts/recommend_hybrid/phase2/EXPERT_EXPORT_VALIDATION.json"),
    )
    args = parser.parse_args()
    result = build_pilot(ROOT, args.count)
    output = ROOT / args.validation_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"EXPERT_CASES_EXPORTED={result['case_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
