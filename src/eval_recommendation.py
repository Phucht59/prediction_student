"""Held-out evaluation for the MLP recommendation risk ranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, RECOMMENDATIONS_DIR
from src.recommendation import MLPLearningPathEngine, reference_risk_targets
from src.utils import setup_logger


logger = setup_logger("eval_recommendation")
DATASETS = ("student-mat", "student-por", "xapi")


def _ranking_metrics(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    precisions = []
    recalls = []
    ndcgs = []
    for truth, row_scores in zip(y_true, scores):
        order = np.argsort(-row_scores)[:k]
        hits = float(truth[order].sum())
        precisions.append(hits / k)
        relevant = float(truth.sum())
        if relevant > 0:
            recalls.append(hits / relevant)
            gains = truth[order] / np.log2(np.arange(2, len(order) + 2))
            ideal_count = min(int(relevant), k)
            ideal = np.ones(ideal_count) / np.log2(np.arange(2, ideal_count + 2))
            ndcgs.append(float(gains.sum() / ideal.sum()))
    return {
        f"precision_at_{k}": float(np.mean(precisions)),
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def _structural_quality(engine: MLPLearningPathEngine, test_frame: pd.DataFrame) -> dict[str, float]:
    nonempty = 0
    complete_steps = 0
    staged = 0
    for record in test_frame.to_dict("records"):
        result = engine.generate(record, predicted_class=1, confidence=0.5)
        path = result["learning_path"]
        nonempty += bool(path)
        complete_steps += bool(path) and all({"phase", "goal", "actions"}.issubset(step) for step in path)
        staged += len({step["phase"] for step in path}) >= 1
    count = max(len(test_frame), 1)
    return {
        "nonempty_path_rate": nonempty / count,
        "complete_step_schema_rate": complete_steps / count,
        "staged_path_rate": staged / count,
    }


def evaluate_dataset(dataset_name: str, force_retrain: bool = False) -> dict:
    train_path = PROCESSED_DIR / f"{dataset_name}_3class_train_pool.csv"
    test_path = PROCESSED_DIR / f"{dataset_name}_3class_locked_test.csv"
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)

    if force_retrain:
        from src.recommendation import load_or_train_recommendation_model

        load_or_train_recommendation_model(dataset_name, train_frame, force_retrain=True)
    engine = MLPLearningPathEngine(dataset_name, train_frame=train_frame)
    scores = engine.predict_scores(test_frame)
    y_true = reference_risk_targets(test_frame, dataset_name)
    y_pred = (scores >= 0.5).astype(int)

    metrics = {
        "dataset": dataset_name,
        "evaluation_set": test_path.name,
        "training_set": train_path.name,
        "training_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "supervision": "domain_criteria_weak_supervision",
        "interpretation": (
            "Ranking and multilabel metrics quantify fidelity to the explicit domain reference criteria; "
            "they do not establish causal improvement in student outcomes."
        ),
        "multilabel": {
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
        },
        "ranking": {},
        "structural_quality": _structural_quality(engine, test_frame),
        "llm_judge": {
            "status": "not_run",
            "score": None,
            "reason": "No external LLM annotations or validated human rating set was supplied.",
        },
        "model": {
            "architecture": "8/7-64-32-6 PyTorch MLP",
            "checkpoint_schema_version": engine.checkpoint["schema_version"],
            "epochs_completed": engine.checkpoint["epochs_completed"],
            "best_validation_loss": engine.checkpoint["best_validation_loss"],
            "seed": engine.checkpoint["seed"],
        },
    }
    for k in (1, 3, 5):
        metrics["ranking"].update(_ranking_metrics(y_true, scores, k))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    selected = args.dataset or list(DATASETS)
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in selected:
        result = evaluate_dataset(dataset_name, force_retrain=args.force_retrain)
        out_path = RECOMMENDATIONS_DIR / f"{dataset_name.replace('-', '_')}_evaluation.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved recommendation evaluation to %s", out_path)


if __name__ == "__main__":
    main()
