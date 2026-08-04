"""Executable nested-OOF ablations for outcome-grounded V2.1."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scientific_core import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeaturePreprocessor,
    RelevanceTransformer,
    aggregate_metrics,
    fit_ranker,
    predict_ranker,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/outcome_grounded_v2_1"
DATA = OUT / "dataset"
FINAL = OUT / "final_oof"
ABLATION_OUT = OUT / "ablations_executed"
SEED = 20260804

ABLATIONS: dict[str, dict[str, Any]] = {
    "FULL": {"remove": [], "interactions": True},
    "NO_RISK_PROFILE": {
        "remove": ["risk_probability", "risk_uncertainty"],
        "interactions": True,
    },
    "NO_BEHAVIOR_STATE": {
        "remove": [
            "active_days",
            "inactive_streak",
            "activity_trend",
            "assessment_progress",
            "vle_intensity",
        ],
        "interactions": True,
    },
    "NO_OPPORTUNITY": {"remove": ["opportunity_count"], "interactions": True},
    "NO_DEFICIT": {"remove": ["deficit_score"], "interactions": True},
    "NO_COUNTERFACTUAL_DELTA": {
        "remove": ["counterfactual_v1_delta"],
        "interactions": True,
    },
    "NO_ACTION_INTERACTIONS": {"remove": [], "interactions": False},
    "NO_WORKLOAD": {"remove": ["workload_minutes"], "interactions": True},
    "ACTION_PRIOR_ONLY": {"prior_only": True},
    "NO_CONSTRAINTS_OFFLINE_ONLY": {
        "remove": [],
        "interactions": True,
        "all_candidates": True,
    },
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["group_id", "action_family"], kind="stable").reset_index(drop=True)


def selected_model(outer_fold: int) -> tuple[str, dict[str, Any]]:
    payload = json.loads(
        (OUT / "model_selection" / f"fold_{outer_fold}_selected.json").read_text(
            encoding="utf-8"
        )
    )
    return str(payload["model"]), dict(payload["parameters"])


def unavailable_top_rate(frame: pd.DataFrame, score_column: str) -> float:
    if "action_available" not in frame.columns:
        return 0.0
    top = frame.loc[frame.groupby("group_id")[score_column].idxmax()]
    return float((pd.to_numeric(top["action_available"], errors="coerce").fillna(0) <= 0).mean())


def evaluate_ablation(
    eligible_raw: pd.DataFrame,
    all_raw: pd.DataFrame,
    name: str,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = []
    fold_records = []
    source = all_raw if spec.get("all_candidates") else eligible_raw
    for outer_fold in [0, 1, 2]:
        raw_train = ordered(source[source["outer_fold"] != outer_fold].copy())
        raw_test = ordered(source[source["outer_fold"] == outer_fold].copy())
        relevance = RelevanceTransformer(seed=SEED + outer_fold)
        train = relevance.fit_transform(raw_train)
        test = relevance.transform(raw_test)

        if spec.get("prior_only"):
            prior = train.groupby("action_family", observed=True)["continuous_relevance"].mean()
            test["ablation_score"] = test["action_family"].map(prior).fillna(0.0)
        else:
            numeric = [column for column in NUMERIC_FEATURES if column not in spec.get("remove", [])]
            preprocessor = FeaturePreprocessor(
                numeric_features=numeric,
                categorical_features=CATEGORICAL_FEATURES,
                include_interactions=bool(spec.get("interactions", True)),
            )
            train_matrix = preprocessor.fit_transform(train)
            test_matrix = preprocessor.transform(test)
            family, parameters = selected_model(outer_fold)
            ranker = fit_ranker(
                family,
                train_matrix,
                train,
                parameters,
                SEED + outer_fold,
            )
            test["ablation_score"] = predict_ranker(ranker, test_matrix)

        metrics = aggregate_metrics(test, "ablation_score")
        metrics["unavailable_top_action_rate"] = unavailable_top_rate(
            test, "ablation_score"
        )
        fold_records.append({"outer_fold": outer_fold, **metrics})
        test["ablation"] = name
        predictions.append(test)

    oof = pd.concat(predictions, ignore_index=True)
    metrics = aggregate_metrics(oof, "ablation_score")
    metrics["unavailable_top_action_rate"] = unavailable_top_rate(
        oof, "ablation_score"
    )
    metrics["folds"] = fold_records
    return oof, metrics


def main() -> None:
    ABLATION_OUT.mkdir(parents=True, exist_ok=True)
    eligible = pd.read_parquet(DATA / "candidate_rows.parquet")
    all_path = DATA / "all_candidate_rows_before_rankability.parquet"
    all_candidates = pd.read_parquet(all_path) if all_path.exists() else eligible.copy()
    rows = []
    registry = []
    for name, spec in ABLATIONS.items():
        predictions_path = ABLATION_OUT / f"{name}.parquet"
        metrics_path = ABLATION_OUT / f"{name}.json"
        if predictions_path.exists() and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            predictions, metrics = evaluate_ablation(
                eligible,
                all_candidates,
                name,
                spec,
            )
            temporary = predictions_path.with_suffix(".tmp.parquet")
            predictions.to_parquet(temporary, index=False)
            os.replace(temporary, predictions_path)
            atomic_json(metrics_path, metrics)
        rows.append({"ablation": name, **{k: v for k, v in metrics.items() if k != "folds"}})
        registry.append(
            {
                "ablation": name,
                "status": "COMPLETE",
                "specification": spec,
                "prediction_file": predictions_path.name,
                "metrics_file": metrics_path.name,
            }
        )

    pd.DataFrame(rows).to_csv(ABLATION_OUT / "SUMMARY.csv", index=False)
    atomic_json(ABLATION_OUT / "registry.json", {"status": "COMPLETE", "ablations": registry})
    progress_path = OUT / "PROGRESS.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.setdefault("stages", {})["ABLATIONS_EXECUTED"] = {
        "status": "COMPLETE",
        "count": len(ABLATIONS),
    }
    atomic_json(progress_path, progress)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
