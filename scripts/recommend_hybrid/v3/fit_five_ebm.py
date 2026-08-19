"""Fit Five-EBM-C0 and baselines on group-safe inner folds. Development only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommend_hybrid.v3.contracts import CanonicalAction
from src.recommend_hybrid.v3.metrics import evaluate_grouped_ranking
from src.recommend_hybrid.v3.ranker import FEATURE_COLUMNS, FiveEBMC0Ranker

ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "labels" / "v3_action_rows.parquet"
OUT = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "ranker"

PARAMS = {
    "interactions": 3,
    "learning_rate": 0.025,
    "max_bins": 64,
    "max_rounds": 500,
    "min_samples_leaf": 20,
    "outer_bags": 4,
    "random_state": 2026,
    "validation_size": 0.15,
}


def _frame(rows: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({column: rows.get(column) for column in FEATURE_COLUMNS})
    frame["vle_available"] = rows["vle_access_available"]
    frame["stage"] = rows["stage"].astype(str)
    frame["risk_margin"] = rows["risk_probability"] - rows["prediction_threshold"]
    frame["uncertainty"] = rows["uncertainty"]
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(LABELS)
    rows = rows.loc[rows.label_status.eq("RETAINED")].copy()
    portable = rows.loc[rows.portability_status.eq("CONDITIONALLY_PORTABLE")]
    extra = (
        rows.loc[rows.portability_status.ne("CONDITIONALLY_PORTABLE")]
        .groupby("action_id", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), 8000), random_state=2026))
    )
    rows = pd.concat([portable, extra], ignore_index=True)
    oof_parts = []
    for fold in sorted(rows.inner_fold.dropna().unique()):
        train = rows.loc[rows.inner_fold != fold]
        valid = rows.loc[rows.inner_fold == fold]
        if train.empty or valid.empty:
            continue
        ranker = FiveEBMC0Ranker(PARAMS)
        targets = {action: train.loc[train.action_id.eq(action.value), "expected_relevance"] for action in CanonicalAction}
        aligned = train.copy()
        # one row per query for features, targets per action series aligned by index of action subset
        feature_train = _frame(aligned)
        # Fit using only rows of each action
        models = FiveEBMC0Ranker(PARAMS)
        from interpret.glassbox import ExplainableBoostingRegressor

        for action in CanonicalAction:
            subset = train.loc[train.action_id.eq(action.value)]
            x = _frame(subset)
            y = pd.to_numeric(subset["expected_relevance"], errors="coerce")
            keep = y.notna()
            model = ExplainableBoostingRegressor(**PARAMS)
            model.fit(x.loc[keep, list(FEATURE_COLUMNS)], y.loc[keep] / 3.0)
            models.models[action] = model
        for action in CanonicalAction:
            subset = valid.loc[valid.action_id.eq(action.value)].copy()
            if subset.empty:
                continue
            pred = models.models[action].predict(_frame(subset))
            subset = subset.copy()
            subset["score"] = np.clip(pred, 0.0, 1.0)
            oof_parts.append(subset)
        joblib_dir = OUT / "folds" / f"fold{int(fold)}"
        joblib_dir.mkdir(parents=True, exist_ok=True)
        import joblib

        for action, model in models.models.items():
            joblib.dump(model, joblib_dir / f"{action.value}.joblib")
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["relevance"] = oof["expected_relevance"]
    metrics = evaluate_grouped_ranking(oof, relevance_column="relevance", eligible_column="eligible")
    # B0 action-stage prior
    prior = rows.groupby(["stage", "action_id"])["expected_relevance"].mean() / 3.0
    b0 = oof.copy()
    b0["score"] = [float(prior.get((row.stage, row.action_id), 0.0)) for row in b0.itertuples(index=False)]
    b0_metrics = evaluate_grouped_ranking(b0, relevance_column="relevance", eligible_column="eligible")
    # final fit on all retained
    final = FiveEBMC0Ranker(PARAMS)
    from interpret.glassbox import ExplainableBoostingRegressor
    import joblib

    final_dir = OUT / "final_models"
    final_dir.mkdir(parents=True, exist_ok=True)
    for action in CanonicalAction:
        subset = rows.loc[rows.action_id.eq(action.value)]
        model = ExplainableBoostingRegressor(**PARAMS)
        y = pd.to_numeric(subset["expected_relevance"], errors="coerce")
        keep = y.notna()
        model.fit(_frame(subset).loc[keep, list(FEATURE_COLUMNS)], y.loc[keep] / 3.0)
        joblib.dump(model, final_dir / f"{action.value}.joblib")
    results = pd.DataFrame(
        [
            {"model": "B0_action_stage", **b0_metrics.to_dict()},
            {"model": "B2_five_ebm_c0", **metrics.to_dict()},
        ]
    )
    results.to_csv(OUT / "BASELINE_RESULTS.csv", index=False)
    (OUT / "FIVE_EBM_MANIFEST.json").write_text(
        json.dumps(
            {
                "model": "Five-EBM-C0",
                "features": list(FEATURE_COLUMNS),
                "parameters": PARAMS,
                "development_metrics": metrics.to_dict(),
                "baseline_b0": b0_metrics.to_dict(),
                "panel_b_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
