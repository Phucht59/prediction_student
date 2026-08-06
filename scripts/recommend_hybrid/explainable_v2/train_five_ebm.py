"""Train Five-EBM Action Ranker with strict interpret dependency checking and official feature schema."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.contracts import CanonicalAction
from src.recommend_hybrid.explainable_v2.ranker import FEATURE_COLUMNS, FiveEBMRanker


def check_interpret_dependency():
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor
    except ModuleNotFoundError:
        return None


def prepare_ebm_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = pd.DataFrame(index=df.index)
    feature_types = []

    for col in FEATURE_COLUMNS:
        if col == "stage":
            X[col] = df["stage"].astype(str)
            feature_types.append("nominal")
        elif col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            feature_types.append("continuous")
        else:
            X[col] = 0.0
            feature_types.append("continuous")

    return X, feature_types


def train_five_ebm() -> dict:
    ebm_class = check_interpret_dependency()
    models_dir = ROOT / "artifacts/recommend_hybrid/explainable_v2/models/five_ebm"
    calib_dir = ROOT / "artifacts/recommend_hybrid/explainable_v2/calibration"
    models_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    if ebm_class is None:
        blocked_manifest = {
            "status": "BLOCKED",
            "reason": "BLOCKED_MISSING_INTERPRET_DEPENDENCY",
            "model_type": "Five-EBM Explainable Action Ranker",
            "runtime_authorized": False,
        }
        (models_dir / "five_ebm_manifest.json").write_text(
            json.dumps(blocked_manifest, indent=2), encoding="utf-8"
        )
        print("FIVE_EBM_TRAINING_STATUS=BLOCKED_MISSING_INTERPRET_DEPENDENCY")
        return blocked_manifest

    candidates_path = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
    )
    labels_path = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/labels/probabilistic_relevance_labels.parquet"
    )

    if not candidates_path.exists() or not labels_path.exists():
        raise RuntimeError("missing candidate features or probabilistic relevance labels")

    df_cand = pd.read_parquet(candidates_path)
    df_labels = pd.read_parquet(labels_path)

    df = df_cand.merge(
        df_labels[["query_id", "action_id", "expected_relevance", "label_confidence"]],
        on=["query_id", "action_id"],
        how="inner",
    )

    fitted_models = {}
    calibrators = {}

    for action in CanonicalAction:
        act_val = action.value
        sub_df = df[df["action_id"] == act_val].copy()
        y = sub_df["expected_relevance"].to_numpy()
        weights = sub_df["label_confidence"].to_numpy()

        X, feature_types = prepare_ebm_dataframe(sub_df)

        model = ebm_class(
            feature_types=feature_types,
            max_bins=128,
            random_state=42,
        )
        model.fit(X, y / 3.0, sample_weight=weights)

        fitted_models[action.value] = model

        # Calibration model on predictions vs target
        preds = model.predict(X) * 3.0
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(preds, y)
        calibrators[action.value] = iso

        # Save model checkpoint
        with (models_dir / f"ebm_{act_val}.pkl").open("wb") as f:
            pickle.dump(model, f)
        with (calib_dir / f"calib_{act_val}.pkl").open("wb") as f:
            pickle.dump(iso, f)

    manifest = {
        "status": "PASS",
        "model_type": "Five-EBM Explainable Action Ranker",
        "action_count": len(fitted_models),
        "trained_actions": list(fitted_models.keys()),
        "feature_count": len(FEATURE_COLUMNS),
        "calibration": "IsotonicRegression",
    }
    (models_dir / "five_ebm_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _write_five_ebm_report(manifest)
    print("FIVE_EBM_TRAINING_STATUS=PASS")
    return manifest


def _write_five_ebm_report(manifest: dict) -> None:
    report_path = (
        ROOT / "reports/recommend_hybrid_v2/FIVE_EBM_MODEL_REPORT.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Five-EBM Explainable Action Ranker Report

## Architecture Summary
- **Model Type**: Five Independent Explainable Boosting Regressors (`FiveEBMRanker`)
- **Trained Actions**: `{', '.join(manifest.get('trained_actions', []))}`
- **Features Used**: `{manifest.get('feature_count', 0)}` learner state features
- **Action ID Feature**: **Excluded** (Zero Action-Stage Identity Shortcut)
- **Score Calibration**: `{manifest.get('calibration', 'IsotonicRegression')}` mapping predictions to shared relevance scale [0, 3]

## Action Models
1. `EBM_ASSESSMENT_COMPLETION`: Predicts assessment urgency relevance.
2. `EBM_RECOVER_ENGAGEMENT`: Predicts engagement recovery relevance.
3. `EBM_STUDY_REGULARITY`: Predicts study pattern regularity relevance.
4. `EBM_TARGETED_CONTENT_REVIEW`: Predicts topic review relevance.
5. `EBM_QUIZ_RETRIEVAL_PRACTICE`: Predicts quiz practice relevance.
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> int:
    manifest = train_five_ebm()
    if manifest.get("status") == "BLOCKED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
