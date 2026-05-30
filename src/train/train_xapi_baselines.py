from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import classification_metrics, save_confusion_matrix_plot  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "xapi" / "xapi_behavior"
RESULTS_PATH = PROJECT_ROOT / "reports" / "results" / "xapi_baseline_results.csv"
SKIPPED_PATH = PROJECT_ROOT / "reports" / "results" / "xapi_baseline_skipped_records.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "saved" / "xapi_baselines"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "xapi_baselines"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xAPI traditional baseline models.")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_processed() -> dict:
    required = (
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train_class.npy",
        "y_val_class.npy",
        "y_test_class.npy",
        "metadata.json",
    )
    missing = [filename for filename in required if not (PROCESSED_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing xAPI processed files: {missing}")
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("leakage_checks", {}).get("passed") is not True:
        raise ValueError(f"xAPI leakage check failed in metadata: {metadata.get('leakage_checks')}")
    data = {
        "X_train": np.load(PROCESSED_DIR / "X_train.npy", allow_pickle=False),
        "X_val": np.load(PROCESSED_DIR / "X_val.npy", allow_pickle=False),
        "X_test": np.load(PROCESSED_DIR / "X_test.npy", allow_pickle=False),
        "y_train": np.load(PROCESSED_DIR / "y_train_class.npy", allow_pickle=False),
        "y_val": np.load(PROCESSED_DIR / "y_val_class.npy", allow_pickle=False),
        "y_test": np.load(PROCESSED_DIR / "y_test_class.npy", allow_pickle=False),
        "metadata": metadata,
    }
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"xAPI {split}: X/y row mismatch.")
        if np.isnan(X).any():
            raise ValueError(f"xAPI {split}: X contains NaN.")
        if not set(np.unique(y).tolist()).issubset({0, 1, 2}):
            raise ValueError(f"xAPI {split}: invalid labels.")
    return data


def get_models(seed: int) -> dict:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "svm_rbf": SVC(kernel="rbf", random_state=seed),
    }


def class_counts(y: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(y, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def resample_train(X: np.ndarray, y: np.ndarray, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "none":
        return X, y
    try:
        if strategy == "random_over":
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(random_state=seed)
        elif strategy in {"smote", "borderline_smote", "adasyn"}:
            counts = class_counts(y)
            min_count = min(counts.values())
            if min_count < 2:
                raise ValueError(f"Not enough minority samples for {strategy}: min_count={min_count}")
            k_neighbors = min(5, min_count - 1)
            if strategy == "smote":
                from imblearn.over_sampling import SMOTE

                sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors)
            elif strategy == "borderline_smote":
                from imblearn.over_sampling import BorderlineSMOTE

                sampler = BorderlineSMOTE(random_state=seed, k_neighbors=k_neighbors)
            else:
                from imblearn.over_sampling import ADASYN

                sampler = ADASYN(random_state=seed, n_neighbors=k_neighbors)
        else:
            raise ValueError(f"Unknown resampling strategy: {strategy}")
        return sampler.fit_resample(X, y)
    except Exception as exc:
        raise RuntimeError(f"Resampling strategy {strategy} failed: {exc}") from exc


def supports_class_weight(model) -> bool:
    return "class_weight" in model.get_params()


def save_model(model, model_name: str, strategy: str) -> None:
    output_dir = MODEL_DIR / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"{model_name}.joblib")


def main() -> int:
    args = parse_args()
    data = load_processed()
    models = get_models(args.seed)
    strategies = ("none", "random_over", "smote", "borderline_smote", "adasyn", "class_weight_balanced")
    rows: list[dict] = []
    skipped: list[dict] = []

    for model_name, base_model in models.items():
        for strategy in strategies:
            try:
                model = clone(base_model)
                if strategy == "class_weight_balanced":
                    if not supports_class_weight(model):
                        raise ValueError("model does not support class_weight")
                    model.set_params(class_weight="balanced")
                    X_fit, y_fit = data["X_train"], data["y_train"]
                else:
                    X_fit, y_fit = resample_train(data["X_train"], data["y_train"], strategy, args.seed)

                print(f"Training xAPI baseline: {model_name}/{strategy}, rows={len(y_fit)}")
                model.fit(X_fit, y_fit)
                save_model(model, model_name, strategy)

                for split in ("train", "val", "test"):
                    X = data[f"X_{split}"]
                    y = data[f"y_{split}"]
                    y_pred = model.predict(X)
                    metrics = classification_metrics(y, y_pred)
                    rows.append(
                        {
                            "dataset": "xapi",
                            "scenario": "xapi_behavior",
                            "model_name": model_name,
                            "imbalance_strategy": strategy,
                            "split": split,
                            **metrics,
                        }
                    )
                    if split == "test":
                        save_confusion_matrix_plot(
                            y,
                            y_pred,
                            FIGURE_DIR / strategy / f"{model_name}_test_confusion_matrix.png",
                            f"xAPI {model_name} {strategy} test confusion matrix",
                        )
                test_row = rows[-1]
                print(
                    "  test_f1_macro={:.4f} test_acc={:.4f} test_recall_weak={:.4f}".format(
                        test_row["f1_macro"],
                        test_row["accuracy"],
                        test_row["recall_weak"],
                    )
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"SKIP xAPI baseline {model_name}/{strategy}: {reason}")
                skipped.append(
                    {
                        "dataset": "xapi",
                        "scenario": "xapi_behavior",
                        "model_name": model_name,
                        "imbalance_strategy": strategy,
                        "reason": reason,
                        "traceback": traceback.format_exc(),
                    }
                )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    pd.DataFrame(skipped).to_csv(SKIPPED_PATH, index=False)
    print(f"xAPI baseline result rows: {len(rows)}")
    print(f"xAPI baseline skipped rows: {len(skipped)}")
    print(f"Saved to: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
