from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.metrics import f1_score

from src.studies.v5.common.metrics import multiclass_metrics
from src.studies.v5.common.uci_runner import _baseline, _sample_baseline

from ..common.artifacts import atomic_write_json, safe_v5_1_root
from ..common.protocol import ROOT
from ..common.uci_data import UCIDataV51
from ..common.uci_training import UCIInputsV51, prepare_partition
from .runner import FINAL_SEEDS, _inner_splits, _outer_indices


BASELINES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
    "hist_gradient_boosting",
)
IMBALANCE = ("none", "class_weight", "random_oversampling", "smote", "adasyn")


def _matrix(inputs: UCIInputsV51) -> np.ndarray:
    return np.concatenate([inputs.temporal.reshape(len(inputs.target), -1), inputs.context], axis=1)


def _resample(x: np.ndarray, y: np.ndarray, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if strategy in {"none", "class_weight"}:
        return x, y
    minimum = int(np.bincount(y, minlength=3).min())
    samplers = {
        "random_oversampling": RandomOverSampler(random_state=seed),
        "smote": SMOTE(random_state=seed, k_neighbors=min(5, max(1, minimum - 1))),
        "adasyn": ADASYN(random_state=seed, n_neighbors=min(5, max(1, minimum - 1))),
    }
    sampled_x, sampled_y = samplers[strategy].fit_resample(x, y)
    if len(sampled_y) <= len(y):
        raise RuntimeError(f"{strategy} did not add training rows")
    return sampled_x, sampled_y


def _score(
    data: UCIDataV51,
    splits,
    name: str,
    parameters: dict[str, Any],
    imbalance: str,
    seed: int,
) -> float:
    scores = []
    for train_index, validation_index in splits:
        train, transformer = prepare_partition(data, train_index, train_index)
        validation, _ = prepare_partition(data, train_index, validation_index, fitted=transformer)
        train_x, train_y = _resample(_matrix(train), train.target, imbalance, seed)
        model = _baseline(name, parameters, seed, class_weight=imbalance == "class_weight")
        model.fit(train_x, train_y)
        probability = model.predict_proba(_matrix(validation))
        scores.append(
            f1_score(validation.target, probability.argmax(axis=1), average="macro", zero_division=0)
        )
    return float(np.mean(scores))


def tune_and_evaluate_baselines(
    dataset: str,
    data: UCIDataV51,
    protocol: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    artifact = safe_v5_1_root(ROOT / "artifacts/v5_1" / dataset.replace("-", "_"))
    output = artifact / "ml_final_metrics.json"
    if output.is_file() and not force:
        return {"status": "COMPLETE", "metrics": pd.read_json(output).to_dict(orient="records")}
    storage = f"sqlite:///{(artifact / 'optuna.db').as_posix()}"
    selected_rows = []
    prediction_rows = []
    model_root = artifact / "ml_models"
    model_root.mkdir(parents=True, exist_ok=True)
    for outer_fold, (outer_train, outer_validation) in enumerate(_outer_indices(protocol)):
        splits = _inner_splits(
            data,
            outer_train,
            split_seed=int(protocol["splits"]["split_seed"]),
            outer_fold=outer_fold,
        )
        train, transformer = prepare_partition(data, outer_train, outer_train)
        validation, _ = prepare_partition(data, outer_train, outer_validation, fitted=transformer)
        for name in BASELINES:
            study = optuna.create_study(
                study_name=f"{dataset}-v5.1-ml-{name}-outer-{outer_fold}",
                direction="maximize",
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=int(protocol["search"]["sampler_seed"])),
            )

            def objective(trial: optuna.Trial) -> float:
                parameters = _sample_baseline(trial, name)
                imbalance = trial.suggest_categorical("imbalance", list(IMBALANCE))
                trial.set_user_attr("parameters", parameters)
                trial.set_user_attr("imbalance", imbalance)
                return _score(data, splits, name, parameters, imbalance, 3407)

            budget = 12
            completed = len(
                [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            )
            if completed < budget:
                study.optimize(objective, n_trials=budget - completed, catch=(RuntimeError, ValueError))
            best = study.best_trial
            parameters = dict(best.user_attrs["parameters"])
            imbalance = str(best.user_attrs["imbalance"])
            selected_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate": name,
                    "inner_macro_f1": float(best.value),
                    "parameters": parameters,
                    "imbalance": imbalance,
                    "completed_trials": len(
                        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                    ),
                }
            )
            for seed in FINAL_SEEDS:
                train_x, train_y = _resample(_matrix(train), train.target, imbalance, seed)
                model = _baseline(name, parameters, seed, class_weight=imbalance == "class_weight")
                model.fit(train_x, train_y)
                probability = model.predict_proba(_matrix(validation))
                model_path = model_root / f"{name}_outer_{outer_fold}_seed_{seed}.joblib"
                joblib.dump({"model": model, "preprocessor": transformer}, model_path)
                for index, record_index in enumerate(outer_validation):
                    prediction_rows.append(
                        {
                            "record_id": data.record_ids[record_index],
                            "source_row": int(record_index),
                            "outer_fold": outer_fold,
                            "candidate": name,
                            "seed": seed,
                            "target": int(validation.target[index]),
                            "p_low": float(probability[index, 0]),
                            "p_medium": float(probability[index, 1]),
                            "p_high": float(probability[index, 2]),
                            "model_path": model_path.relative_to(ROOT).as_posix(),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_parquet(artifact / "ml_oof_predictions.parquet", index=False)
    metric_rows = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"]):
        metrics = multiclass_metrics(
            frame.target.to_numpy(), frame[["p_low", "p_medium", "p_high"]].to_numpy()
        )
        metric_rows.append({"candidate": candidate, "seed": int(seed), **metrics})
    for candidate, frame in predictions.groupby("candidate"):
        ensemble = (
            frame.groupby(["record_id", "source_row", "outer_fold", "target"], as_index=False)
            .agg(p_low=("p_low", "mean"), p_medium=("p_medium", "mean"), p_high=("p_high", "mean"))
            .sort_values("source_row")
        )
        metrics = multiclass_metrics(
            ensemble.target.to_numpy(), ensemble[["p_low", "p_medium", "p_high"]].to_numpy()
        )
        metric_rows.append({"candidate": f"{candidate}_ensemble", "seed": -1, **metrics})
    metrics = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    metrics.to_json(output, orient="records", indent=2)
    atomic_write_json(artifact / "ml_selected_configs.json", selected_rows)
    return {"status": "COMPLETE", "metrics": metrics.to_dict(orient="records")}


__all__ = ["BASELINES", "IMBALANCE", "tune_and_evaluate_baselines"]
