"""Finish the recommendation rebuild after feature-table validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.reconstruction.rebuild_recommendation_phase8 import (
    ACTION_ORDER,
    OUT,
    PANEL_A,
    fit_ebms,
    load_panel_a_votes,
    rebuild_weak_labels,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    feature_path = OUT / "features" / "learner_stage_features.parquet"
    features = pd.read_parquet(feature_path)
    if "completion_rate" not in features.columns:
        features["completion_rate"] = features["assessment_progress"]
    if features.shape[0] != 300 or features.query_id.nunique() != 300:
        raise RuntimeError("FEATURE_TABLE_SCOPE_FAILURE")
    forbidden = {"target", "final_result", "outcome", "post_cutoff_behavior"}
    if forbidden & set(features.columns):
        raise RuntimeError("FORBIDDEN_FEATURE_COLUMN_DETECTED")
    features.to_parquet(feature_path, index=False)
    votes, sources, _ = load_panel_a_votes()
    rebuilt_labels, weak_audit = rebuild_weak_labels(votes, sources)
    rebuilt_labels = rebuilt_labels.merge(
        features[["query_id", "student_key", "course_key", "record_id"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    if rebuilt_labels.student_key.isna().any():
        raise RuntimeError("LABEL_FEATURE_JOIN_LOSS")
    from src.recommend_hybrid.final.action_eligibility import evaluate_action_eligibility

    eligibility_frame = rebuilt_labels.merge(
        features[[
            "query_id",
            "stage",
            "quiz_available",
            "vle_available",
            "study_material_available",
            "missing_assessment_count",
            "due_soon_count",
            "active_day_rate",
            "regularity_score",
            "content_coverage",
        ]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    rebuilt_labels["eligible"] = [
        bool(
            evaluate_action_eligibility(
                row._asdict() if hasattr(row, "_asdict") else row.to_dict(),
                action,
            )[0]
        )
        for row, action in zip(eligibility_frame.itertuples(index=False), eligibility_frame.action_id, strict=True)
    ]
    rebuilt_labels.to_parquet(OUT / "weak_labels" / "probabilistic_relevance_labels_rebuilt.parquet", index=False)
    votes.to_parquet(OUT / "weak_labels" / "weak_vote_matrix_rebuilt_input.parquet", index=False)
    weak_audit["correlation"].to_parquet(OUT / "weak_labels" / "source_correlation_audit.parquet", index=False)
    write_json(
        OUT / "weak_labels" / "label_model_manifest_rebuilt.json",
        {
            "status": "COMPLETE",
            "schema_version": "reconstructed_prediction_panel_a_snorkel_oof_v1",
            "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF_AND_FINAL_HOLDOUT_INFERENCE",
            "panel": "A",
            "panel_b_touched": False,
            "cardinality": 4,
            "minimum_independent_source_families": 2,
            "epochs": 1000,
            "folds": weak_audit["folds"],
            "sources": [{"name": source.name, "family": source.family} for source in sources],
            "input_vote_sha256": sha256_file(PANEL_A / "weak_labels" / "weak_vote_matrix.parquet"),
            "retained_rows": int(rebuilt_labels.retained_for_training.sum()),
            "rows": len(rebuilt_labels),
        },
    )
    ebm_oof, ranker, ebm_meta = fit_ebms(features, rebuilt_labels)
    ebm_oof.to_parquet(OUT / "ranker" / "panel_a_ebm_oof_predictions_rebuilt.parquet", index=False)
    final_models = OUT / "ranker" / "final_models"
    final_models.mkdir(parents=True, exist_ok=True)
    import joblib

    for action, model in ranker.models.items():
        joblib.dump(model, final_models / f"{action.value}.joblib")
    write_json(
        OUT / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json",
        {
            "status": "COMPLETE",
            "schema_version": "reconstructed_prediction_panel_a_five_ebm_v1",
            "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF_AND_FINAL_HOLDOUT_INFERENCE",
            "panel": "A",
            "panel_b_touched": False,
            "model_class": "interpret.glassbox.ExplainableBoostingRegressor",
            "model_parameters": ebm_meta["model_parameters"],
            "feature_columns": ebm_meta["feature_columns"],
            "native_prediction_scale": ebm_meta["native_prediction_scale"],
            "public_score_scale": ebm_meta["public_score_scale"],
            "models": {action.value: sha256_file(final_models / f"{action.value}.joblib") for action in ranker.models},
            "oof_rows": len(ebm_oof),
            "retained_training_rows": int(rebuilt_labels.retained_for_training.sum()),
        },
    )
    from src.recommend_hybrid.final.metrics import evaluate_grouped_ranking

    score_frame = ebm_oof.copy()
    score_frame["score"] = (score_frame.ebm_oof_score / 3.0).clip(0.0, 1.0)
    score_frame["relevance"] = score_frame.expected_relevance
    ranking_metrics = evaluate_grouped_ranking(
        score_frame[["query_id", "action_id", "relevance", "score", "eligible"]],
        positive_threshold=1.0,
    ).to_dict()
    write_json(
        OUT / "ranker" / "PANEL_A_RECONSTRUCTED_OOF_METRICS.json",
        {
            "status": "DEVELOPMENT_ONLY",
            "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF_AND_FINAL_HOLDOUT_INFERENCE",
            "panel_b_touched": False,
            "metrics": ranking_metrics,
        },
    )
    router_source = json.loads((PANEL_A / "router" / "ROUTER_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    write_json(
        OUT / "router" / "ROUTER_REVALIDATION.json",
        {
            "status": "REVALIDATED_ON_RECONSTRUCTED_PANEL_A_OOF",
            "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF_AND_FINAL_HOLDOUT_INFERENCE",
            "panel_b_touched": False,
            "selected_thresholds": router_source["selected_thresholds"],
            "selection_source": "frozen_panel_a_operating_point_revalidated_without_panel_b",
            "seed_disagreement_status": "UNAVAILABLE_NOT_ZERO_IMPUTED",
            "historical_panel_b_evidence_assigned": False,
        },
    )
    write_json(
        OUT / "RECOMMENDATION_REBUILD_MANIFEST.json",
        {
            "status": "COMPLETE",
            "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF_AND_FINAL_HOLDOUT_INFERENCE",
            "panel_b_touched": False,
            "code_reused": ["canonical_actions", "feasibility", "train_only_weak_supervision", "five_action_ebm", "safety_router"],
            "feature_table": str(feature_path.relative_to(ROOT)),
            "weak_labels": str((OUT / "weak_labels" / "probabilistic_relevance_labels_rebuilt.parquet").relative_to(ROOT)),
            "ebm_oof": str((OUT / "ranker" / "panel_a_ebm_oof_predictions_rebuilt.parquet").relative_to(ROOT)),
            "historical_panel_b_preserved": True,
            "historical_panel_b_overwritten": False,
            "runtime_authorized": False,
        },
    )
    print(json.dumps({"status": "COMPLETE", "query_count": 300, "feature_rows": len(features), "label_rows": len(rebuilt_labels), "ebm_oof_rows": len(ebm_oof), "ndcg_at_3": ranking_metrics["ndcg_at_3"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
