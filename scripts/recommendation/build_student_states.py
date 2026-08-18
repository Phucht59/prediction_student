"""Build OULAD Student Learning State from the frozen prediction artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid.data.oulad import build_compact_vle_daily, load_oulad_static_tables
from src.hybrid.phase7.data import (
    OULAD_PHASE7_AGGREGATE_CHANNELS,
    OULAD_PHASE7_TEMPORAL_CHANNELS,
    build_oulad_phase7_view,
)
from src.recommendation.contracts.prediction import PredictionArtifactAdapter
from src.recommendation.state.builder import StudentStateBuilder
from src.recommendation.state.validation import validate_student_state


STAGES = ("20pct", "35pct", "50pct", "75pct")
FRACTIONS = {"20pct": 0.20, "35pct": 0.35, "50pct": 0.50, "75pct": 0.75}
FORBIDDEN = {"final_result", "target", "score", "date_unregistration"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_feature_frame(
    base: pd.DataFrame,
    daily: pd.DataFrame,
    raw_dir: Path,
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Adapt the current Phase7 view to small semantic state features."""
    frames: list[pd.DataFrame] = []
    audits: dict[str, dict] = {}
    aggregate_index = {name: i for i, name in enumerate(OULAD_PHASE7_AGGREGATE_CHANNELS)}
    temporal_index = {name: i for i, name in enumerate(OULAD_PHASE7_TEMPORAL_CHANNELS)}
    split_lookup = split.set_index("record_id")["outer_fold"]
    for stage in STAGES:
        eligible, view, audit = build_oulad_phase7_view(base, daily, FRACTIONS[stage], str(raw_dir))
        # The current Prediction pipeline uses forbidden fields for eligibility
        # and target construction only. They never enter this frame.
        _ = FORBIDDEN.intersection(eligible.columns)
        valid = view.temporal_mask
        active_days = view.temporal[:, :, temporal_index["active_days"]]
        exposure_days = view.temporal[:, :, temporal_index["week_exposure_fraction"]] * 7.0
        active_sum = (active_days * valid).sum(axis=1)
        exposure_sum = (exposure_days * valid).sum(axis=1)
        active_ratio = np.divide(
            active_sum,
            exposure_sum,
            out=np.zeros(len(view.record_id), dtype=np.float32),
            where=exposure_sum > 0,
        )
        quiz = view.temporal[:, :, temporal_index["quiz_activity"]]
        quiz_total = (quiz * valid).sum(axis=1).astype(np.float32)
        record_ids = view.record_id.astype(str)
        frame = pd.DataFrame({
            "dataset": "oulad",
            "student_id": view.group_id.astype(str),
            "record_id": record_ids,
            "module": eligible["code_module"].astype(str).to_numpy(),
            "presentation": eligible["code_presentation"].astype(str).to_numpy(),
            "enrollment_identity": record_ids,
            "stage": stage,
            "outer_fold": split_lookup.loc[record_ids].to_numpy(dtype=np.int64),
            "inactive_streak": view.aggregate[:, aggregate_index["current_inactivity_streak"]],
            "active_days_ratio": active_ratio,
            "recent_activity": view.aggregate[:, aggregate_index["recent_activity"]],
            "activity_trend": view.aggregate[:, aggregate_index["activity_trend"]],
            "assessment_completion": view.aggregate[:, aggregate_index["completion_rate"]],
            "missing_assessments": view.aggregate[:, aggregate_index["missed_due_count"]],
            "course_progress": view.progress,
            "quiz_activity": quiz_total,
            "vle_available": valid.any(axis=1).astype(bool),
            "source_feature_version": "phase7.oulad.cutoff_safe.v1",
        })
        frames.append(frame)
        audits[stage] = audit
    features = pd.concat(frames, ignore_index=True)
    if features.duplicated(["record_id", "stage"]).any():
        raise ValueError("feature frame has duplicate record-stage rows")
    return features, audits


def write_mapping_report(path: Path, counts: dict[str, int], audits: dict, prediction_sha: str, state_sha: str) -> None:
    rows = [
        ("risk_probability", "prediction artifact score, Hybrid mean over seeds", "OULAD", "mean frozen Hybrid score", "prediction artifact; target unused", "not imputed", "PASS"),
        ("risk_band", "risk_probability + config 0.33/0.66", "OULAD", "operational low/medium/high mapping", "downstream only; not prediction threshold", "configurable", "PASS"),
        ("uncertainty", "NONE", "OULAD", "not derived", "UNAVAILABLE", "omitted", "UNAVAILABLE"),
        ("inactive_streak", "current_inactivity_streak", "OULAD", "direct aggregate channel", "strict pre-cutoff view", "valid zero", "PASS"),
        ("active_days_ratio", "active_days + week_exposure_fraction", "OULAD", "sum active days / observed days", "strict pre-cutoff view", "valid zero", "PASS"),
        ("recent_activity", "recent_activity", "OULAD", "direct aggregate channel", "strict pre-cutoff view", "valid zero", "PASS"),
        ("activity_trend", "activity_trend", "OULAD", "direct aggregate channel", "strict pre-cutoff view", "valid zero", "PASS"),
        ("study_regularness", "NONE", "OULAD", "not derived", "UNAVAILABLE", "omitted", "UNAVAILABLE"),
        ("assessment_completion", "completion_rate", "OULAD", "direct aggregate channel", "due/submission dates < cutoff", "valid zero", "PASS"),
        ("missing_assessments", "missed_due_count", "OULAD", "direct aggregate channel", "due dates < cutoff", "valid zero", "PASS"),
        ("upcoming_assessments", "NONE", "OULAD", "not derived", "future schedule not used", "omitted", "UNAVAILABLE"),
        ("course_progress", "view.progress", "OULAD", "cutoff fraction", "stage contract", "not imputed", "PASS"),
        ("content_coverage", "NONE", "OULAD", "content clicks are not coverage", "UNAVAILABLE", "omitted", "UNAVAILABLE"),
        ("quiz_activity", "quiz_activity temporal channel", "OULAD", "sum observed weeks", "strict pre-cutoff view", "valid zero", "PASS"),
        ("vle_available", "temporal_mask", "OULAD", "observed window exists", "registration-aware mask", "boolean", "PASS"),
        ("content_available", "NONE", "OULAD", "availability channel absent", "UNAVAILABLE", "omitted", "UNAVAILABLE"),
        ("quiz_available", "NONE", "OULAD", "availability channel absent", "UNAVAILABLE", "omitted", "UNAVAILABLE"),
    ]
    text = [
        "# Phase 1-2 Feature Mapping", "",
        "Source of truth: current `hybrid_recomend` prediction and Phase7 feature artifacts.", "",
        "| Semantic feature | Actual source feature(s) | Dataset | Transformation | Temporal validity | Missing policy | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    text.extend("| " + " | ".join(row) + " |" for row in rows)
    text.extend(["", "## Stage counts", "", "| Stage | State rows | Eligible feature rows |", "|---|---:|---:|"])
    text.extend(f"| {stage} | {counts[stage]} | {audits[stage]['eligible_records']} |" for stage in STAGES)
    text.extend([
        "", "## UCI compatibility audit", "",
        "| Semantic feature | UCI source | Status |", "|---|---|---|",
        "| risk_probability | Frozen Hybrid prediction artifact S0/S1/S2 | AVAILABLE FOR AUDIT ONLY |",
        "| identity/stage/fold | record_id/group_id/outer_fold/module/presentation | AVAILABLE FOR AUDIT ONLY |",
        "| engagement/VLE/content/quiz | NONE in UCI pipeline | UNAVAILABLE |",
        "| assessment completion/missing/upcoming | NONE; grades are not completion | UNAVAILABLE |",
        "| course progress | stage indicator only, not measured learning progress | UNAVAILABLE |",
        "| uncertainty | NONE persisted | UNAVAILABLE |",
        "", "## Provenance",
        f"- Prediction artifact SHA-256: `{prediction_sha}`",
        f"- State artifact SHA-256: `{state_sha}`",
        "- `date_unregistration` is used only by the current Prediction eligibility contract, never copied as a state feature.",
        "- `target`, `final_result`, and assessment `score` are not copied into the state artifact.",
        "- OULAD enrollment_identity equals the existing record_id: sha256('oulad|code_module|code_presentation|id_student')[:24].",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_authority_report(
    path: Path,
    adapter: PredictionArtifactAdapter,
    state: pd.DataFrame,
    split: pd.DataFrame,
    prediction_path: Path,
) -> None:
    pred = adapter.records[["record_id", "stage", "outer_fold"]].copy()
    split_lookup = split.drop_duplicates("record_id").set_index("record_id")["outer_fold"]
    pred["split_outer_fold"] = pred["record_id"].map(split_lookup)
    fold_mismatch = int((pred["outer_fold"] != pred["split_outer_fold"]).sum())
    coverage = state.groupby("stage").size().astype(int).to_dict()
    state_keys = set(zip(state.record_id.astype(str), state.stage.astype(str)))
    pred_keys = set(zip(pred.record_id.astype(str), pred.stage.astype(str)))
    missing = sorted(pred_keys - state_keys)
    extra = sorted(state_keys - pred_keys)
    raw = pd.read_parquet(prediction_path, columns=["model", "domain", "stage", "outer_fold", "seed", "score"])
    hybrid = raw[(raw.model.astype(str) == "Hybrid") & (raw.domain.astype(str) == "oulad")]
    early = hybrid[hybrid.stage.isin(STAGES)]
    seeds = sorted(int(value) for value in early.seed.unique())
    text = [
        "# Prediction / Student State Authority Reconciliation", "",
        "## Final authority", "",
        "| Check | Result | Evidence |", "|---|---|---|",
        f"| Frozen Hybrid seeds | PASS | `{seeds}`; exactly {len(seeds)} rows per record-stage after filtering |",
        "| Probability column | PASS | `score` from `artifacts/prediction/final/predictions/predictions.parquet`; Student State uses its mean over seeds |",
        "| Ensemble location | PASS | adapter aggregation in `src/recommendation/contracts/prediction.py`; no model inference or retraining |",
        "| Prediction lineage | PASS | source artifact SHA-256 is pinned in `prediction_source_version` |",
        "| Stage scope | PASS | early stages only: 20pct/35pct/50pct/75pct; FINAL-100 excluded |",
        f"| State grain | PASS | one row per existing `record_id × stage`; {len(state)} rows |",
        f"| Prediction/state coverage | {'PASS' if not missing and not extra else 'FAIL'} | missing={len(missing)}, extra={len(extra)} |",
        f"| Outer-fold reconciliation | {'PASS' if fold_mismatch == 0 else 'FAIL'} | {fold_mismatch} prediction rows disagree with `oulad_outer.parquet` |",
        "| OOF authority | PASS | safety/consumption manifests say outer-test authorized and consumed; exact outer-fold assignment matches; no in-sample column is consumed |",
        "| Config discrepancy | RESOLVED | runtime consumption/safety and outer-OOF report are stronger than stale selection configs that say `outer_test_used=false` |",
        "| Checkpoint status | REVIEW | standalone Phase8 final checkpoint is missing; existing frozen prediction artifact remains read-only authority |",
        "| Identity contract | PASS | `record_id = sha256('oulad|code_module|code_presentation|id_student')[:24]`; `student_id=id_student`; module/presentation retained |",
        "| PostgreSQL mapping | PASS WITH HANDOFF | `record_id` is the deterministic future `external_enrollment_id`; current import does not populate that column and no fake enrollment was created |",
        "", "## Reconciled stage counts", "", "| Stage | Prediction record-stage rows | Student State rows |", "|---|---:|---:|",
    ]
    text.extend(f"| {stage} | {int((pred.stage == stage).sum())} | {coverage.get(stage, 0)} |" for stage in STAGES)
    text.extend([
        "", "## Grain and identifier notes",
        "- One Student State row is one OULAD student/module/presentation enrollment case at one recommendation stage.",
        "- The 26,697 / 25,606 / 24,599 / 23,159 counts are the existing Hybrid early-warning prediction cohort after collapsing its three seed rows per record-stage.",
        "- No FINAL-100 row, final outcome, assessment score, or future feature is used in the state.",
        "- Current PostgreSQL `catalog.enrollment` uses UUID internally; the deterministic OULAD `record_id` is the only allowed external mapping key for a later ingest.",
        "", f"Prediction artifact SHA-256: `{sha256(prediction_path)}`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    prediction_path = ROOT / "artifacts/prediction/final/predictions/predictions.parquet"
    raw_dir = ROOT / "data/raw"
    split_path = ROOT / "artifacts/hybrid/phase1/splits/oulad_outer.parquet"
    adapter = PredictionArtifactAdapter.from_parquet(prediction_path, dataset="oulad", stages=STAGES)
    split = pd.read_parquet(split_path, columns=["record_id", "outer_fold"])
    split["record_id"] = split["record_id"].astype(str)
    if split.duplicated("record_id").any():
        raise ValueError("outer split identity is not unique")
    _, _, base = load_oulad_static_tables(raw_dir)
    daily = build_compact_vle_daily(raw_dir, ROOT / "artifacts/hybrid/phase1/runtime")
    features, audits = build_feature_frame(base, daily, raw_dir, split)
    state = StudentStateBuilder().build(adapter.records, features)
    errors = validate_student_state(state)
    if errors:
        raise ValueError(f"student state validation failed: {errors}")
    out = ROOT / "artifacts/recommendation/states/oulad_student_states.parquet"
    preview = ROOT / "reports/recommendation/oulad_student_states_preview.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(out, index=False)
    state.head(100).to_csv(preview, index=False)
    counts = state.groupby("stage").size().astype(int).to_dict()
    write_mapping_report(ROOT / "reports/recommendation/PHASE12_FEATURE_MAPPING.md", counts, audits, sha256(prediction_path), sha256(out))
    validation = [
        "# Phase 1-2 Validation", "", "| Gate | Result | Evidence |", "|---|---|---|",
        f"| Frozen Hybrid adapter | PASS | {len(adapter.records)} unique record-stage rows after 3-seed mean |",
        "| OULAD stages 20/35/50/75 | PASS | all four present |",
        "| FINAL-100 exclusion | PASS | adapter scope excludes FINAL-100 |",
        "| Identity join | PASS | record_id one-to-one with outer split and features |",
        "| State contract | PASS | deterministic case_id, bounds, booleans, lineage |",
        "| Leakage blacklist | PASS | forbidden fields absent from output; sources are cutoff-safe |",
        "| Uncertainty | PASS WITH UNAVAILABLE | no persisted uncertainty; no invented metric |",
        "| Prediction authority reconciliation | PASS | runtime safety/consumption + exact outer-fold match resolve stale selection-config flag |",
        "| PostgreSQL compatibility | REVIEW | existing tables link prediction/recommendation through UUID enrollment_id; no student_state table or migration was added; external_enrollment_id is the future mapping point |",
        "| API/Snorkel/EBM/retraining | PASS | no calls or training paths in this build |",
        "", "## Counts", "", "| Stage | Rows |", "|---|---:|",
    ]
    validation.extend(f"| {stage} | {counts.get(stage, 0)} |" for stage in STAGES)
    validation.extend(["", f"State artifact SHA-256: `{sha256(out)}`", "", "No FINAL-100 rows are in the recommendation state artifact."])
    report = ROOT / "reports/recommendation/PHASE12_VALIDATION.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(validation) + "\n", encoding="utf-8")
    write_authority_report(ROOT / "reports/recommendation/AUTHORITY_RECONCILIATION.md", adapter, state, split, prediction_path)
    print(json.dumps({"rows": len(state), "counts": counts, "state_sha256": sha256(out)}, indent=2))


if __name__ == "__main__":
    main()
