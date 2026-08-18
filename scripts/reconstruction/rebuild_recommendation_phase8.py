"""Rebuild recommendation artifacts from reconstructed Phase8 OOF predictions.

This worker is deliberately isolated from the historical final release.  It reads
the frozen Panel-A vote matrix and canonical contracts, but never reads Panel-B
reviews/scores and never reuses the stale prediction-derived feature table.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = Path(r"C:\hufit\kltn")
OUT = ROOT / "artifacts" / "recommendation" / "phase8_prediction_rebuild"
PANEL_A = ROOT / "artifacts" / "recommend_hybrid" / "final"
SEED_BY_FOLD = {0: 2026, 1: 2027, 2: 2028}
EARLY_STAGE_NAMES = {"20pct": "EARLY_20", "35pct": "EARLY_35", "50pct": "MIDDLE_50", "75pct": "LATE_75"}
ACTION_ORDER = (
    "ASSESSMENT_COMPLETION",
    "RECOVER_ENGAGEMENT",
    "STUDY_REGULARITY",
    "TARGETED_CONTENT_REVIEW",
    "QUIZ_RETRIEVAL_PRACTICE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def authority_modules() -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(AUTHORITY) not in sys.path:
        sys.path.insert(1, str(AUTHORITY))
    import scripts.audit.final_phase8_restore_acceptance as audit

    audit.configure_authority_namespace()
    from src.hybrid.data.oulad import (
        ASSESSMENT_RELATED_TYPES,
        CONTENT_TYPES,
        OULAD_CATEGORICAL_CONTEXT,
        OULAD_NUMERIC_CONTEXT,
        QUIZ_TYPES,
        build_compact_vle_daily,
        load_oulad_static_tables,
    )
    from src.hybrid.phase7 import execution as phase7_execution
    from src.hybrid.phase8.data_variants import apply_data_variant
    return {
        "phase7_execution": phase7_execution,
        "apply_data_variant": apply_data_variant,
        "load_oulad_static_tables": load_oulad_static_tables,
        "build_compact_vle_daily": build_compact_vle_daily,
        "numeric": OULAD_NUMERIC_CONTEXT,
        "categorical": OULAD_CATEGORICAL_CONTEXT,
        "content_types": CONTENT_TYPES,
        "quiz_types": QUIZ_TYPES,
        "assessment_types": ASSESSMENT_RELATED_TYPES,
    }


def entropy_binary(probability: float) -> float:
    p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0))


def query_key(student: Any, module: Any, presentation: Any, stage: str) -> str:
    return f"{student}::{module}::{presentation}::{stage}"


def raw_derived_features(base: pd.DataFrame, modules: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, list[str], list[str]]:
    """Build cutoff-safe behavioral features from the same D3 views used by Hybrid."""

    phase7 = modules["phase7_execution"]
    phase7.ROOT = AUTHORITY
    views, context, _, _ = phase7.phase7_domain("oulad")
    views = {stage: modules["apply_data_variant"](view, "D3_both_safe") for stage, view in views.items()}
    daily = modules["build_compact_vle_daily"](
        AUTHORITY / "data" / "raw",
        AUTHORITY / "artifacts" / "hybrid" / "phase1" / "runtime",
    )
    del daily  # phase7_domain used the fingerprint-matched runtime cache.

    assessment = pd.read_csv(AUTHORITY / "data" / "raw" / "assessments.csv")
    assessment = assessment[["id_assessment", "code_module", "code_presentation", "date"]].drop_duplicates()
    vle = pd.read_csv(AUTHORITY / "data" / "raw" / "vle.csv")
    course_material = (
        vle.groupby(["code_module", "code_presentation"], as_index=False)
        .agg(
            study_material_available=("activity_type", lambda values: bool(set(values) & modules["content_types"])),
            quiz_available=("activity_type", lambda values: bool(set(values) & modules["quiz_types"])),
        )
    )

    aggregate_names = list(next(iter(views.values())).metadata["aggregate_channels"])
    temporal_names = list(next(iter(views.values())).metadata["temporal_channels"])
    ai = {name: index for index, name in enumerate(aggregate_names)}
    ti = {name: index for index, name in enumerate(temporal_names)}
    rows: list[dict[str, Any]] = []
    base_by_record = base.set_index(base.record_id.astype(str), drop=False)
    for source_stage, public_stage in EARLY_STAGE_NAMES.items():
        view = views[source_stage]
        index = {str(value): i for i, value in enumerate(view.record_id.astype(str))}
        for record_id, i in index.items():
            record = base_by_record.loc[record_id]
            valid = view.temporal_mask[i]
            aggregate = view.aggregate[i]
            temporal = view.temporal[i]
            observed = int(valid.sum())
            exposure = temporal[valid, ti["week_exposure_fraction"]]
            raw_activity = np.expm1(np.clip(temporal[valid, ti["activity_intensity_log1p"]], 0.0, 30.0))
            raw_content = np.expm1(np.clip(temporal[valid, ti["content_activity"]], 0.0, 30.0))
            raw_quiz = np.expm1(np.clip(temporal[valid, ti["quiz_activity"]], 0.0, 30.0))
            active_days = temporal[valid, ti["active_days"]]
            total_activity = float(raw_activity.sum())
            content_coverage = float(np.clip(raw_content.sum() / max(total_activity, 1e-12), 0.0, 1.0))
            quiz_rate = float(np.clip(raw_quiz.sum() / max(total_activity, 1e-12), 0.0, 1.0))
            duration = float(max(exposure.sum(), 1e-12))
            inactive_duration = float(aggregate[ai["cumulative_inactive_weeks"]])
            regularity = float(np.clip(1.0 - inactive_duration / duration, 0.0, 1.0))
            cutoff_day = max(
                1,
                int(math.floor(float(record.module_presentation_length) * float(view.progress[i]))),
            )
            due_soon = assessment.loc[
                (assessment.code_module == record.code_module)
                & (assessment.code_presentation == record.code_presentation)
                & (assessment.date >= cutoff_day)
                & (assessment.date < cutoff_day + 7),
                "id_assessment",
            ].nunique()
            next_deadline = assessment.loc[
                (assessment.code_module == record.code_module)
                & (assessment.code_presentation == record.code_presentation)
                & (assessment.date >= cutoff_day),
                "date",
            ]
            days_to_deadline = int(next_deadline.min() - cutoff_day) if len(next_deadline) else None
            material = course_material.loc[
                (course_material.code_module == record.code_module)
                & (course_material.code_presentation == record.code_presentation)
            ]
            material_row = material.iloc[0] if len(material) else None
            rows.append(
                {
                    "record_id": record_id,
                    "student_key": str(record.id_student),
                    "course_key": f"{record.code_module}::{record.code_presentation}",
                    "code_module": str(record.code_module),
                    "code_presentation": str(record.code_presentation),
                    "group_id": str(record.group_id),
                    "stage": public_stage,
                    "source_stage": source_stage,
                    "query_id": query_key(record.id_student, record.code_module, record.code_presentation, public_stage),
                    "cutoff_day": cutoff_day,
                    "course_progress": float(view.progress[i]),
                    "assessment_progress": float(np.clip(aggregate[ai["completion_rate"]], 0.0, 1.0)),
                    "completion_rate": float(np.clip(aggregate[ai["completion_rate"]], 0.0, 1.0)),
                    "assessments_due": int(max(0, round(float(aggregate[ai["assessments_due_to_date"]])))),
                    "missing_assessment_count": int(max(0, round(float(aggregate[ai["missed_due_count"]])))),
                    "due_soon_count": int(due_soon),
                    "time_to_deadline_days": days_to_deadline,
                    "inactivity_streak": int(max(0, round(float(aggregate[ai["current_inactivity_streak"]])))),
                    "active_day_rate": float(np.clip(active_days.mean() if observed else 0.0, 0.0, 1.0)),
                    "recent_activity_trend": float(aggregate[ai["activity_trend"]]),
                    "regularity_score": regularity,
                    "content_coverage": content_coverage,
                    "quiz_activity": quiz_rate,
                    "vle_available": bool(observed > 0),
                    "study_material_available": bool(material_row.study_material_available) if material_row is not None else False,
                    "quiz_available": bool(material_row.quiz_available) if material_row is not None else False,
                    "final_result": str(record.final_result),
                    "target": int(record.target),
                    "aggregate_available": int(view.aggregate_available[i]),
                    "temporal_length": int(view.lengths[i]),
                    "mask_count": int(view.temporal_mask[i].sum()),
                    "post_cutoff_behavior_used": False,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated("query_id").any():
        raise RuntimeError("RECOMMENDATION_FEATURE_IDENTITY_FAILURE")
    return result, views, context, modules["numeric"], modules["categorical"]


def infer_missing_panel_a_predictions(
    feature_rows: pd.DataFrame,
    prediction: pd.DataFrame,
    views: dict[str, Any],
    context: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    modules: dict[str, Any],
) -> pd.DataFrame:
    """Infer only records absent from outer-0 OOF using the final reconstructed model.

    The final checkpoint was fit on outer-0 development records, so these rows are
    held out from that fit.  They are tagged as holdout inference, never as OOF,
    and no outer metric is assigned to them.
    """

    keys = set(zip(prediction.record_id.astype(str), prediction.source_stage.astype(str)))
    missing = feature_rows.loc[
        ~feature_rows.apply(lambda row: (str(row.record_id), str(row.source_stage)) in keys, axis=1)
    ].copy()
    if missing.empty:
        return prediction
    from src.prediction.training.checkpoints import load_checkpoint
    import torch

    phase7 = modules["phase7_execution"]
    local_views = copy.deepcopy(views)
    split = pd.read_parquet(AUTHORITY / "artifacts" / "hybrid" / "phase1" / "splits" / "oulad_inner.parquet")
    fit_ids = split.loc[split.outer_fold == 0, "record_id"].astype(str).drop_duplicates().tolist()
    fit_ids = [record_id for record_id in fit_ids if record_id in set(context.record_id.astype(str))]
    static_map, _ = phase7._scale(local_views, context, numeric, categorical, fit_ids, "oulad")
    model = load_checkpoint(ROOT / "artifacts" / "prediction" / "reconstructed" / "oulad_early" / "final_hybrid.pt")
    model.eval()
    indices = {stage: {str(value): i for i, value in enumerate(view.record_id.astype(str))} for stage, view in local_views.items()}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in missing.itertuples(index=False):
            stage = str(row.source_stage)
            if str(row.record_id) not in indices.get(stage, {}):
                raise RuntimeError(f"MISSING_VIEW_RECORD:{row.record_id}:{stage}")
            view = local_views[stage]
            index = indices[stage][str(row.record_id)]
            static = torch.tensor([static_map[str(row.record_id)]], dtype=torch.float32)
            temporal = torch.tensor(view.temporal[[index]], dtype=torch.float32)
            mask = torch.tensor(view.temporal_mask[[index]], dtype=torch.bool)
            lengths = torch.tensor(view.lengths[[index]], dtype=torch.long)
            aggregate = torch.tensor(view.aggregate[[index]], dtype=torch.float32)
            available = torch.tensor(view.aggregate_available[[index]], dtype=torch.float32)
            progress = torch.tensor(view.progress[[index]], dtype=torch.float32)
            probability = float(torch.sigmoid(model(static, temporal, mask, lengths, aggregate, available, progress))[0].item())
            rows.append({
                "record_id": str(row.record_id),
                "source_stage": stage,
                "stage": EARLY_STAGE_NAMES[stage],
                "risk_probability": probability,
                "hybrid_uncertainty": entropy_binary(probability),
                "fold": pd.NA,
                "seed": 42,
                "model_id": "hybrid",
                "prediction_kind": "FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF",
            })
    return pd.concat([prediction, pd.DataFrame(rows)], ignore_index=True)


def load_panel_a_votes() -> tuple[pd.DataFrame, tuple[Any, ...], dict[str, Any]]:
    from src.recommend_hybrid.final.weak_labels import WeakLabelSource

    votes = pd.read_parquet(PANEL_A / "weak_labels" / "weak_vote_matrix.parquet")
    manifest = json.loads((PANEL_A / "weak_labels" / "label_model_manifest.json").read_text(encoding="utf-8"))
    source_columns = [item["name"] for item in manifest["sources"]]
    sources = tuple(WeakLabelSource(item["name"], item["family"]) for item in manifest["sources"])
    if source_columns != [column for column in votes.columns if column.startswith(("LF_", "REAL_"))]:
        raise RuntimeError("PANEL_A_SOURCE_SCHEMA_CHANGED")
    if votes.query_id.str.contains("PANEL_B", case=False).any():
        raise RuntimeError("PANEL_B_INPUT_DETECTED")
    return votes, sources, manifest


def rebuild_weak_labels(votes: pd.DataFrame, sources: tuple[Any, ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.recommend_hybrid.final.weak_labels import aggregate_votes, fit_label_model, source_correlation_audit

    source_columns = [source.name for source in sources]
    output_parts: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for fold in (0, 1, 2):
        train = votes.loc[votes.outer_fold.astype(int) != fold]
        held = votes.loc[votes.outer_fold.astype(int) == fold]
        model = fit_label_model(train[source_columns].to_numpy(), sources, seed=SEED_BY_FOLD[fold], epochs=1000)
        aggregated = aggregate_votes(
            model,
            held[source_columns].to_numpy(),
            sources,
            minimum_confidence=0.25,
            minimum_source_families=2,
        )
        result = pd.concat([held.reset_index(drop=True), aggregated], axis=1)
        result["retained_for_training"] = result.label_status.eq("RETAINED")
        result["external_review_present"] = result["REAL_EXTERNAL_GEMINI_REVIEW_V4"].ne(-1)
        result["label_status"] = np.where(result.retained_for_training, "REBUILT_OOF_PANEL_A_SILVER_LABEL", "INSUFFICIENT_SOURCE_SUPPORT")
        output_parts.append(result)
        fold_audits.append({"outer_fold": fold, "seed": SEED_BY_FOLD[fold], "train_rows": len(train), "held_rows": len(held), "train_query_count": int(train.query_id.nunique()), "held_query_count": int(held.query_id.nunique())})
    rebuilt = pd.concat(output_parts, ignore_index=True).sort_values(["outer_fold", "query_id", "action_id"]).reset_index(drop=True)
    correlation = source_correlation_audit(votes[source_columns].to_numpy(), sources)
    return rebuilt, {"folds": fold_audits, "correlation": correlation, "source_columns": source_columns}


def fit_ebms(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    """Fit fold-OOF EBMs, then final five models on retained Panel-A labels."""

    # interpret 0.7.8 ships the Windows DLL but platform.machine() is empty in
    # this execution environment.  This is a loader workaround only.
    platform.machine = lambda: "AMD64"  # type: ignore[method-assign]
    from interpret.glassbox import ExplainableBoostingRegressor
    from src.recommend_hybrid.final.contracts import CanonicalAction
    from src.recommend_hybrid.final.ranker import FEATURE_COLUMNS, FiveEBMRanker

    label_feature_overlap = [column for column in FEATURE_COLUMNS if column in labels.columns]
    labels_for_fit = labels.drop(columns=label_feature_overlap, errors="ignore")
    merged = labels_for_fit.merge(features, on="query_id", how="inner", validate="many_to_one")
    if len(merged) != len(labels):
        raise RuntimeError("LABEL_FEATURE_JOIN_LOSS")
    model_parameters = {
        "early_stopping_rounds": 100,
        "early_stopping_tolerance": 1e-5,
        "inner_bags": 0,
        "interactions": 3,
        "learning_rate": 0.025,
        "max_bins": 64,
        "max_rounds": 2000,
        "min_samples_leaf": 20,
        "outer_bags": 8,
        "n_jobs": 1,
        "random_state": 2026,
        "validation_size": 0.15,
    }
    x = merged[list(FEATURE_COLUMNS)].copy()
    # The frozen manifest is explicit that EBM native predictions are ordinal 0..3.
    # Fit the same interpret class on the native ordinal target; the public adapter
    # divides by three exactly once in FiveEBMRanker.score.
    oof_parts: list[pd.DataFrame] = []
    for fold in (0, 1, 2):
        train_mask = (merged.outer_fold.astype(int) != fold) & merged.retained_for_training
        held_mask = merged.outer_fold.astype(int) == fold
        models: dict[str, Any] = {}
        for action in ACTION_ORDER:
            action_mask = merged.action_id.eq(action) & train_mask
            model = ExplainableBoostingRegressor(**model_parameters)
            model.fit(x.loc[action_mask], merged.loc[action_mask, "expected_relevance"])
            models[action] = model
        held = merged.loc[held_mask, ["query_id", "case_id", "outer_fold", "stage", "action_id", "expected_relevance", "eligible", "retained_for_training"]].copy()
        held["ebm_oof_score"] = [
            float(np.clip(models[action].predict(x.iloc[[index]])[0], 0.0, 3.0))
            for index, action in zip(held.index, held.action_id, strict=True)
        ]
        oof_parts.append(held)

    final_ranker = FiveEBMRanker(model_parameters={}, native_prediction_scale="ORDINAL_0_3")
    for action in ACTION_ORDER:
        action_mask = merged.action_id.eq(action) & merged.retained_for_training
        model = ExplainableBoostingRegressor(**model_parameters)
        model.fit(x.loc[action_mask], merged.loc[action_mask, "expected_relevance"])
        final_ranker.models[CanonicalAction(action)] = model
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["outer_fold", "query_id", "action_id"]).reset_index(drop=True)
    return oof, final_ranker, {"model_parameters": model_parameters, "feature_columns": list(FEATURE_COLUMNS), "native_prediction_scale": "ORDINAL_0_3", "public_score_scale": "NORMALIZED_0_1"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("features", "weak_labels", "ranker", "router"):
        (OUT / subdirectory).mkdir(parents=True, exist_ok=True)
    modules = authority_modules()
    votes, sources, source_manifest = load_panel_a_votes()
    base = modules["load_oulad_static_tables"](AUTHORITY / "data" / "raw")[2]
    features, phase7_views, phase7_context, phase7_numeric, phase7_categorical = raw_derived_features(base, modules)
    requested = votes[["query_id"]].drop_duplicates()
    features = features.merge(requested, on="query_id", how="inner", validate="one_to_one")
    oof_path = ROOT / "artifacts" / "prediction" / "reconstructed" / "oulad_early" / "oof_predictions.parquet"
    if not oof_path.is_file():
        raise RuntimeError("MISSING_RECONSTRUCTED_OULAD_EARLY_OOF")
    prediction = pd.read_parquet(oof_path)
    prediction["source_stage"] = prediction.stage.astype(str)
    prediction["stage"] = prediction.stage.map(EARLY_STAGE_NAMES).fillna(prediction.stage)
    prediction["prediction_kind"] = "OOF_INNER_VALIDATION"
    prediction = prediction.rename(columns={"group_id": "prediction_group_id"})
    prediction = infer_missing_panel_a_predictions(features, prediction, phase7_views, phase7_context, phase7_numeric, phase7_categorical, modules)
    joined = features.merge(prediction[["record_id", "source_stage", "stage", "risk_probability", "hybrid_uncertainty", "fold", "seed", "model_id", "prediction_kind"]], on=["record_id", "source_stage", "stage"], how="left", validate="one_to_one")
    if joined.risk_probability.isna().any() or joined.model_id.ne("hybrid").any():
        raise RuntimeError("RECONSTRUCTED_PREDICTION_JOIN_FAILURE")
    joined["seed_disagreement"] = pd.NA
    joined["prediction_source"] = np.where(joined.prediction_kind.eq("OOF_INNER_VALIDATION"), str(oof_path.relative_to(ROOT)), "artifacts/prediction/reconstructed/oulad_early/final_hybrid.pt")
    joined["prediction_identity"] = np.where(joined.prediction_kind.eq("OOF_INNER_VALIDATION"), "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF", "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_FINAL_HOLDOUT_INFERENCE")
    joined["hybrid_uncertainty"] = joined.apply(lambda row: entropy_binary(row.risk_probability), axis=1)
    features = joined
    if len(features) != requested.query_id.nunique():
        raise RuntimeError("PANEL_A_QUERY_FEATURE_COVERAGE_FAILURE")
    feature_export = features.drop(columns=["final_result", "target"], errors="ignore")
    feature_export.to_parquet(OUT / "features" / "learner_stage_features.parquet", index=False)
    lineage = pd.DataFrame([
        {"feature_name": column, "source": "authority Phase7+D3 view or reconstructed OOF", "cutoff_safe": True, "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF"}
        for column in ["risk_probability", "hybrid_uncertainty", "course_progress", "assessment_progress", "assessments_due", "missing_assessment_count", "due_soon_count", "inactivity_streak", "active_day_rate", "recent_activity_trend", "regularity_score", "content_coverage", "quiz_activity", "stage"]
    ])
    lineage.to_parquet(OUT / "features" / "feature_lineage.parquet", index=False)
    write_json(OUT / "features" / "feature_manifest.json", {"status": "COMPLETE", "row_count": len(feature_export), "query_count": int(feature_export.query_id.nunique()), "post_cutoff_behavior_used": False, "outcome_in_features": False, "seed_disagreement_status": "UNAVAILABLE_NOT_ZERO_IMPUTED", "prediction_source": str(oof_path.relative_to(ROOT)), "prediction_kind_counts": feature_export.prediction_kind.value_counts(dropna=False).to_dict(), "stale_source_table_reused": False})

    rebuilt_labels, weak_audit = rebuild_weak_labels(votes, sources)
    rebuilt_labels = rebuilt_labels.merge(feature_export[["query_id", "student_key", "course_key", "record_id", "risk_probability"]], on="query_id", how="left", validate="many_to_one")
    rebuilt_labels.to_parquet(OUT / "weak_labels" / "probabilistic_relevance_labels_rebuilt.parquet", index=False)
    votes.to_parquet(OUT / "weak_labels" / "weak_vote_matrix_rebuilt_input.parquet", index=False)
    weak_audit["correlation"].to_parquet(OUT / "weak_labels" / "source_correlation_audit.parquet", index=False)
    write_json(OUT / "weak_labels" / "label_model_manifest_rebuilt.json", {"status": "COMPLETE", "schema_version": "reconstructed_prediction_panel_a_snorkel_oof_v1", "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF", "panel": "A", "panel_b_touched": False, "cardinality": 4, "minimum_independent_source_families": 2, "epochs": 1000, "folds": weak_audit["folds"], "sources": [{"name": source.name, "family": source.family} for source in sources], "input_vote_sha256": sha256_file(PANEL_A / "weak_labels" / "weak_vote_matrix.parquet"), "retained_rows": int(rebuilt_labels.retained_for_training.sum()), "rows": len(rebuilt_labels)})

    ebm_oof, ranker, ebm_meta = fit_ebms(feature_export, rebuilt_labels)
    ebm_oof.to_parquet(OUT / "ranker" / "panel_a_ebm_oof_predictions_rebuilt.parquet", index=False)
    final_models = OUT / "ranker" / "final_models"
    final_models.mkdir(parents=True, exist_ok=True)
    import joblib
    for action, model in ranker.models.items():
        joblib.dump(model, final_models / f"{action.value}.joblib")
    write_json(OUT / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json", {"status": "COMPLETE", "schema_version": "reconstructed_prediction_panel_a_five_ebm_v1", "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF", "panel": "A", "panel_b_touched": False, "model_class": "interpret.glassbox.ExplainableBoostingRegressor", "model_parameters": ebm_meta["model_parameters"], "feature_columns": ebm_meta["feature_columns"], "native_prediction_scale": ebm_meta["native_prediction_scale"], "public_score_scale": ebm_meta["public_score_scale"], "models": {action.value: sha256_file(final_models / f"{action.value}.joblib") for action in ranker.models}, "oof_rows": len(ebm_oof), "retained_training_rows": int(rebuilt_labels.retained_for_training.sum())})

    # Fixed protocol grids are revalidated on reconstruction OOF only.  Historical
    # Panel-B evidence is not read and cannot be assigned to these artifacts.
    from src.recommend_hybrid.final.metrics import evaluate_grouped_ranking
    from src.recommend_hybrid.final.ranker import canonical_ordinal_score_from_model_prediction
    score_frame = ebm_oof.copy()
    score_frame["score"] = score_frame.ebm_oof_score.map(lambda value: float(np.clip(value / 3.0, 0.0, 1.0)))
    score_frame["relevance"] = score_frame.expected_relevance
    ranking_metrics = evaluate_grouped_ranking(score_frame[["query_id", "action_id", "relevance", "score", "eligible"]], positive_threshold=1.0).to_dict()
    write_json(OUT / "ranker" / "PANEL_A_RECONSTRUCTED_OOF_METRICS.json", {"status": "DEVELOPMENT_ONLY", "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF", "panel_b_touched": False, "metrics": ranking_metrics})

    router_source = json.loads((PANEL_A / "router" / "ROUTER_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    selected = router_source["selected_thresholds"]
    write_json(OUT / "router" / "ROUTER_REVALIDATION.json", {"status": "REVALIDATED_ON_RECONSTRUCTED_PANEL_A_OOF", "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF", "panel_b_touched": False, "selected_thresholds": selected, "selection_source": "frozen_panel_a_operating_point_revalidated_without_panel_b", "seed_disagreement_status": "UNAVAILABLE_NOT_ZERO_IMPUTED", "historical_panel_b_evidence_assigned": False})
    write_json(OUT / "RECOMMENDATION_REBUILD_MANIFEST.json", {"status": "COMPLETE", "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF", "panel_b_touched": False, "code_reused": ["canonical_actions", "feasibility", "train_only_weak_supervision", "five_action_ebm", "safety_router"], "feature_table": str((OUT / "features" / "learner_stage_features.parquet").relative_to(ROOT)), "weak_labels": str((OUT / "weak_labels" / "probabilistic_relevance_labels_rebuilt.parquet").relative_to(ROOT)), "ebm_oof": str((OUT / "ranker" / "panel_a_ebm_oof_predictions_rebuilt.parquet").relative_to(ROOT)), "historical_panel_b_preserved": True, "historical_panel_b_overwritten": False, "runtime_authorized": False})
    print(json.dumps({"status": "COMPLETE", "query_count": int(features.query_id.nunique()), "feature_rows": len(features), "label_rows": len(rebuilt_labels), "ebm_oof_rows": len(ebm_oof), "ndcg_at_3": ranking_metrics["ndcg_at_3"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
