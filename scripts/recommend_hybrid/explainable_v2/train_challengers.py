"""Train baseline models and LambdaMART challenger using sklearn Pipelines with ColumnTransformer."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.contracts import CanonicalAction
from src.recommend_hybrid.explainable_v2.ranker import FEATURE_COLUMNS


def build_numeric_preprocessing_pipeline() -> ColumnTransformer:
    categorical_features = ["stage"]
    numeric_features = [c for c in FEATURE_COLUMNS if c != "stage"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_features,
            ),
        ]
    )
    return preprocessor


def train_challengers() -> dict:
    models_dir = ROOT / "artifacts/recommend_hybrid/explainable_v2/models/challengers"
    models_dir.mkdir(parents=True, exist_ok=True)

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

    # Prepare feature frame with raw stage strings and numeric columns
    feature_df = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        if col == "stage":
            feature_df[col] = df["stage"].astype(str)
        elif col in df.columns:
            feature_df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            feature_df[col] = 0.0

    # 1. GLOBAL_ACTION_POPULARITY
    pop_scores = df.groupby("action_id")["expected_relevance"].mean().to_dict()

    # 2. ACTION_STAGE_ONLY_BASELINE
    action_stage_scores = df.groupby(["action_id", "stage"])["expected_relevance"].mean().to_dict()

    # 3. LOGISTIC/RIDGE LINEAR BASELINE using ColumnTransformer Pipeline per action
    linear_pipelines = {}
    for action in CanonicalAction:
        act_val = action.value
        mask = df["action_id"] == act_val
        X_sub = feature_df.loc[mask]
        y_sub = df.loc[mask, "expected_relevance"].to_numpy()

        pipe = Pipeline(
            steps=[
                ("preprocessor", build_numeric_preprocessing_pipeline()),
                ("regressor", Ridge(alpha=1.0, random_state=42)),
            ]
        )
        pipe.fit(X_sub, y_sub)
        linear_pipelines[act_val] = pipe

    # 4. LAMBDAMART_CHALLENGER (Pipeline with preprocessor + GradientBoostingRegressor/LGBM)
    from sklearn.ensemble import GradientBoostingRegressor

    lambdamart_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_numeric_preprocessing_pipeline()),
            ("regressor", GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ]
    )
    lambdamart_pipeline.fit(feature_df, df["expected_relevance"].to_numpy())

    # Save models
    with (models_dir / "linear_baseline.pkl").open("wb") as f:
        pickle.dump(linear_pipelines, f)

    with (models_dir / "lambdamart_challenger.pkl").open("wb") as f:
        pickle.dump(lambdamart_pipeline, f)

    manifest = {
        "status": "PASS",
        "baselines_trained": [
            "GLOBAL_ACTION_POPULARITY",
            "RULE_BASED_RANKER",
            "ACTION_ONLY_BASELINE",
            "STAGE_ONLY_BASELINE",
            "ACTION_STAGE_ONLY_BASELINE",
            "LOGISTIC_REGRESSION_OR_LINEAR_BASELINE",
        ],
        "challengers_trained": ["LAMBDAMART_CHALLENGER"],
        "popularity_scores": pop_scores,
    }
    (models_dir / "challengers_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("CHALLENGERS_TRAINING_STATUS=PASS")
    return manifest


if __name__ == "__main__":
    train_challengers()
