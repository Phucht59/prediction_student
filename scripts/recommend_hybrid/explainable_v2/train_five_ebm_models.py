"""Train five action-specific Explainable Boosting Models on frozen Panel A labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTIONS = (
    "ASSESSMENT_COMPLETION",
    "RECOVER_ENGAGEMENT",
    "STUDY_REGULARITY",
    "TARGETED_CONTENT_REVIEW",
    "QUIZ_RETRIEVAL_PRACTICE",
)
EXPECTED_PANEL_A_CASES = 300
EXPECTED_ROWS = 1500
EXPECTED_SNORKEL_SHA = "4a4871426880bdcd1257dc15c29a36c23de34481f07be68d8e5095dc20efefb9"

LABEL_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
LABEL_PATH = LABEL_DIR / "probabilistic_relevance_labels.parquet"
LABEL_MANIFEST_PATH = LABEL_DIR / "label_model_manifest.json"
CANDIDATES_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"

OUTPUT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"

NUMERIC_FEATURES = (
    "risk_probability",
    "hybrid_uncertainty",
    "course_progress",
    "inactivity_streak",
    "active_day_rate",
    "assessments_due",
    "regularity_score",
    "content_coverage",
    "quiz_activity",
    "missing_assessment_count",
    "due_soon_count",
    "completion_rate",
)
BINARY_FEATURES = (
    "vle_available",
    "study_material_available",
    "quiz_available",
)
CATEGORICAL_FEATURES = (
    "stage",
)
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

EBM_PARAMS = {
    "interactions": 3,
    "max_bins": 64,
    "max_rounds": 2000,
    "learning_rate": 0.025,
    "min_samples_leaf": 20,
    "outer_bags": 8,
    "inner_bags": 0,
    "validation_size": 0.15,
    "early_stopping_rounds": 100,
    "early_stopping_tolerance": 1e-05,
    "random_state": 2026,
}

LOCKED_GRID_SELECTED_CONFIG_ID = "a70599afad40"
LOCKED_GRID_SELECTION_RELATIVE_PATH = (
    "artifacts/recommend_hybrid/explainable_v2/ranker_development/"
    "ebm_locked_grid_v1/EBM_GRID_SELECTION.json"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for p in (LABEL_PATH, LABEL_MANIFEST_PATH, CANDIDATES_PATH):
        if not p.exists():
            raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT={p}")

    manifest = json.loads(LABEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("SNORKEL_LABEL_MANIFEST_NOT_PASS")
    if manifest.get("panel") != "A":
        raise RuntimeError("SNORKEL_LABEL_MANIFEST_NOT_PANEL_A")
    if manifest.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_CONTAMINATION_DETECTED")
    if manifest.get("runtime_authorized") is not False:
        raise RuntimeError("RUNTIME_AUTHORIZED_MUST_REMAIN_FALSE")
    if manifest.get("frozen_panel_a_sha256") != EXPECTED_SNORKEL_SHA:
        raise RuntimeError("FROZEN_PANEL_A_LINEAGE_SHA_MISMATCH")
    if int(manifest.get("cardinality", -1)) != 4:
        raise RuntimeError("SNORKEL_CARDINALITY_MUST_BE_4")

    labels = pd.read_parquet(LABEL_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)

    required_label = {
        "query_id", "case_id", "outer_fold", "stage", "action_id",
        "expected_relevance", "label_confidence", "label_entropy", "eligible",
    }
    missing = required_label - set(labels.columns)
    if missing:
        raise RuntimeError("LABEL_FIELDS_MISSING=" + ",".join(sorted(missing)))
    missing_features = set(FEATURES) - set(candidates.columns)
    if missing_features:
        raise RuntimeError("CANDIDATE_FEATURES_MISSING=" + ",".join(sorted(missing_features)))

    if len(labels) != EXPECTED_ROWS:
        raise RuntimeError(f"LABEL_ROW_COUNT={len(labels)} expected={EXPECTED_ROWS}")
    if labels["case_id"].nunique() != EXPECTED_PANEL_A_CASES:
        raise RuntimeError("PANEL_A_CASE_COUNT_MISMATCH")
    if labels.duplicated(["query_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_LABEL_QUERY_ACTION")
    if candidates.duplicated(["query_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_CANDIDATE_QUERY_ACTION")

    cols = ["query_id", "action_id", *FEATURES]
    features = candidates.loc[:, cols].copy()
    merged = labels.merge(
        features,
        on=["query_id", "action_id", "stage"],
        how="left",
        validate="one_to_one",
    )
    if merged[list(FEATURES)].isna().all(axis=1).any():
        raise RuntimeError("FEATURE_JOIN_FAILED_FOR_SOME_ROWS")

    action_counts = merged.groupby("action_id").size().to_dict()
    expected_counts = {action: EXPECTED_PANEL_A_CASES for action in ACTIONS}
    if action_counts != expected_counts:
        raise RuntimeError(f"ACTION_ROW_COUNTS={action_counts} expected={expected_counts}")

    return merged, candidates, manifest


def _prepare_X(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.loc[:, FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in BINARY_FEATURES:
        X[col] = X[col].astype("boolean").astype("Int64").astype(float)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return X


def _new_model(seed: int) -> ExplainableBoostingRegressor:
    params = dict(EBM_PARAMS)
    params["random_state"] = seed
    return ExplainableBoostingRegressor(**params)


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3) -> float:
    order = np.argsort(-y_score)[:k]
    ideal = np.argsort(-y_true)[:k]
    gains = np.power(2.0, y_true) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts[: len(order)]))
    idcg = float(np.sum(gains[ideal] * discounts[: len(ideal)]))
    return 0.0 if idcg <= 0.0 else dcg / idcg


def _ranking_metrics(oof: pd.DataFrame) -> dict:
    ndcgs = []
    top1_agree = []
    for _, group in oof.groupby("query_id", sort=False):
        y = group["expected_relevance"].to_numpy(dtype=float)
        s = group["ebm_oof_score"].to_numpy(dtype=float)
        ndcgs.append(_ndcg_at_k(y, s, 3))
        top1_agree.append(int(np.argmax(y) == np.argmax(s)))
    return {
        "development_ndcg_at_3": float(np.mean(ndcgs)),
        "development_top1_agreement": float(np.mean(top1_agree)),
        "query_count": int(len(ndcgs)),
        "scope": "PANEL_A_DEVELOPMENT_ONLY_NOT_FINAL_EVALUATION",
    }


def run() -> int:
    data, _, snorkel_manifest = _load_inputs()
    folds = sorted(int(x) for x in pd.unique(data["outer_fold"]))
    if folds != [0, 1, 2]:
        raise RuntimeError(f"EXPECTED_OUTER_FOLDS_0_1_2_GOT={folds}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT_DIR / "final_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    oof_parts = []
    action_reports = []

    for action_index, action in enumerate(ACTIONS):
        action_df = data[data["action_id"] == action].copy().reset_index(drop=True)
        X_all = _prepare_X(action_df)
        y_all = action_df["expected_relevance"].to_numpy(dtype=float)

        oof_pred = np.full(len(action_df), np.nan, dtype=float)
        fold_rows = []

        for fold in folds:
            train_mask = action_df["outer_fold"].astype(int).to_numpy() != fold
            hold_mask = ~train_mask

            model = _new_model(2026 + action_index * 100 + fold)
            model.fit(X_all.loc[train_mask], y_all[train_mask])
            pred = np.asarray(model.predict(X_all.loc[hold_mask]), dtype=float)
            pred = np.clip(pred, 0.0, 3.0)
            oof_pred[hold_mask] = pred

            fold_rows.append({
                "outer_fold": fold,
                "train_rows": int(train_mask.sum()),
                "holdout_rows": int(hold_mask.sum()),
                "rmse": float(math.sqrt(mean_squared_error(y_all[hold_mask], pred))),
                "mae": float(mean_absolute_error(y_all[hold_mask], pred)),
            })

        if np.isnan(oof_pred).any():
            raise RuntimeError(f"OOF_PREDICTIONS_INCOMPLETE action={action}")

        final_model = _new_model(2026 + action_index * 1000 + 99)
        final_model.fit(X_all, y_all)

        model_path = models_dir / f"{action}.joblib"
        joblib.dump(final_model, model_path)

        part = action_df[
            ["query_id", "case_id", "outer_fold", "stage", "action_id",
             "expected_relevance", "label_confidence", "label_entropy", "eligible"]
        ].copy()
        part["ebm_oof_score"] = oof_pred
        oof_parts.append(part)

        action_reports.append({
            "action_id": action,
            "rows": int(len(action_df)),
            "oof_rmse": float(math.sqrt(mean_squared_error(y_all, oof_pred))),
            "oof_mae": float(mean_absolute_error(y_all, oof_pred)),
            "fold_metrics": fold_rows,
            "final_model_sha256": _sha256(model_path),
            "final_model_path": str(model_path.relative_to(ROOT)),
        })

    oof = pd.concat(oof_parts, ignore_index=True)
    if len(oof) != EXPECTED_ROWS:
        raise RuntimeError("OOF_EBM_ROW_COUNT_MISMATCH")
    if oof.duplicated(["query_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_OOF_QUERY_ACTION")

    oof_path = OUTPUT_DIR / "panel_a_ebm_oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)

    ranking = _ranking_metrics(oof)
    manifest = {
        "schema_version": "panel_a_five_ebm_v1",
        "status": "PASS",
        "panel": "A",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "model_class": "interpret.glassbox.ExplainableBoostingRegressor",
        "action_model_count": 5,
        "actions": list(ACTIONS),
        "features": list(FEATURES),
        "locked_grid_selected_config_id": LOCKED_GRID_SELECTED_CONFIG_ID,
        "locked_grid_selection_path": LOCKED_GRID_SELECTION_RELATIVE_PATH,
        "ranker_calibration": "NONE_RAW_EBM_SELECTED",
        "excluded_all_nan_features": ["seed_disagreement"],
        "missing_value_policy": {
            "inactivity_streak": "retain_native_EBM_missing_bin",
            "completion_rate": "retain_native_EBM_missing_bin",
        },
        "hyperparameters": EBM_PARAMS,
        "training_target": "Snorkel OOF expected_relevance",
        "training_protocol": "ACTION_SPECIFIC_OUTER_FOLD_OOF_THEN_FINAL_FIT_ON_ALL_PANEL_A",
        "snorkel_manifest_sha256": _sha256(LABEL_MANIFEST_PATH),
        "snorkel_labels_sha256": _sha256(LABEL_PATH),
        "frozen_panel_a_sha256": snorkel_manifest["frozen_panel_a_sha256"],
        "development_metrics": ranking,
        "action_reports": action_reports,
        "oof_predictions_sha256": _sha256(oof_path),
        "scientific_constraints": [
            "Panel B is not read or used.",
            "Development metrics are not final held-out performance.",
            "Each action has a distinct ExplainableBoostingRegressor.",
            "Outer-fold OOF predictions are generated without fitting that action model on the held-out fold.",
            "Final action models are fit only after OOF development predictions are complete.",
            "RUNTIME_AUTHORIZED remains FALSE.",
        ],
    }
    manifest_path = OUTPUT_DIR / "FIVE_EBM_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== FIVE ACTION-SPECIFIC EBM MODELS ===")
    print("MODEL_CLASS=ExplainableBoostingRegressor")
    print("ACTION_MODELS=5")
    print("PANEL_A_CASES=300")
    print("ACTION_ROWS=1500")
    print("OUTER_FOLDS=0,1,2")
    for report in action_reports:
        print(
            f"ACTION={report['action_id']} "
            f"OOF_RMSE={report['oof_rmse']:.6f} "
            f"OOF_MAE={report['oof_mae']:.6f}"
        )
    print(f"DEVELOPMENT_NDCG_AT_3={ranking['development_ndcg_at_3']:.6f}")
    print(f"DEVELOPMENT_TOP1_AGREEMENT={ranking['development_top1_agreement']:.6f}")
    print("FINAL_METRICS_CLAIMED=FALSE")
    print("PANEL_B_TOUCHED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    print("FIVE_EBM_TRAINING=PASS")
    print("NEXT_ACTION=BUILD_AND_FREEZE_RECOMMENDATION_RANKER")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
