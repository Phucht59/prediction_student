from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.studies.v5.common.metrics import binary_metrics_per_record_threshold

from ..common.artifacts import atomic_write_json, safe_v5_1_root
from ..common.protocol import ROOT
from .data import prepare_oulad_inputs
from .training import choose_threshold


BASELINES = ("logistic_regression", "hist_gradient_boosting", "xgboost_compact", "aggregate_only_mlp")
IMBALANCE = ("none", "class_weight", "random_oversampling", "smote", "adasyn")
FINAL_SEEDS = (42, 1201, 2026, 3407, 7319)


def _matrix(inputs) -> np.ndarray:
    return np.concatenate([inputs.aggregate, inputs.static], axis=1)


def _resample(x: np.ndarray, y: np.ndarray, strategy: str, seed: int):
    if strategy in {"none", "class_weight"}:
        return x, y
    minimum = int(np.bincount(y.astype(int), minlength=2).min())
    samplers = {
        "random_oversampling": RandomOverSampler(random_state=seed),
        "smote": SMOTE(random_state=seed, k_neighbors=min(5, max(1, minimum - 1))),
        "adasyn": ADASYN(random_state=seed, n_neighbors=min(5, max(1, minimum - 1))),
    }
    sampled_x, sampled_y = samplers[strategy].fit_resample(x, y)
    if len(sampled_y) <= len(y):
        raise RuntimeError(f"{strategy} did not add training records")
    return sampled_x, sampled_y


def _parameters(trial: optuna.Trial, candidate: str) -> dict[str, Any]:
    if candidate == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if candidate == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-5, 10, log=True),
            "max_iter": trial.suggest_categorical("max_iter", [100, 200, 300]),
        }
    if candidate == "xgboost_compact":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 350, 500]),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if candidate == "aggregate_only_mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", ["64", "96", "96_48"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 3e-3, log=True),
        }
    raise ValueError(candidate)


def _model(candidate: str, parameters: dict[str, Any], seed: int):
    if candidate == "logistic_regression":
        return LogisticRegression(C=float(parameters["C"]), max_iter=3000, random_state=seed)
    if candidate == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **parameters)
    if candidate == "xgboost_compact":
        return XGBClassifier(
            random_state=seed,
            n_jobs=6,
            eval_metric="logloss",
            tree_method="hist",
            **parameters,
        )
    if candidate == "aggregate_only_mlp":
        hidden = tuple(int(value) for value in str(parameters["hidden_layer_sizes"]).split("_"))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=float(parameters["alpha"]),
            learning_rate_init=float(parameters["learning_rate_init"]),
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        )
    raise ValueError(candidate)


def _fit(model, x: np.ndarray, y: np.ndarray, imbalance: str):
    if imbalance == "class_weight":
        model.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(x, y)
    return model


def evaluate_oulad_baselines(data, v4_protocol: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    artifact = safe_v5_1_root(ROOT / "artifacts/v5_1/oulad")
    output = artifact / "ml_final_metrics.json"
    if output.is_file() and not force:
        return {"status": "COMPLETE", "metrics": pd.read_json(output).to_dict(orient="records")}
    from src.studies.oulad_v4.data import build_v4_inner_manifest, manifest_indices

    storage = f"sqlite:///{(artifact / 'optuna.db').as_posix()}"
    selected_rows = []
    prediction_rows = []
    model_root = artifact / "ml_models"
    model_root.mkdir(parents=True, exist_ok=True)
    for outer_fold in range(3):
        outer_train, outer_validation = data.v2.outer_indices(outer_fold)
        inner_manifest = build_v4_inner_manifest(data, outer_fold, v4_protocol)
        inner_splits = [
            manifest_indices(data.v2, inner_manifest, int(fold))
            for fold in sorted(inner_manifest.inner_fold.unique())
        ]
        for candidate in BASELINES:
            study = optuna.create_study(
                study_name=f"oulad-v5.1-ml-{candidate}-outer-{outer_fold}",
                direction="maximize",
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=3407),
            )

            def objective(trial: optuna.Trial) -> float:
                parameters = _parameters(trial, candidate)
                imbalance = trial.suggest_categorical("imbalance", list(IMBALANCE))
                probabilities = []
                targets = []
                for train_index, validation_index in inner_splits:
                    train = prepare_oulad_inputs(data, train_index, train_index)
                    validation = prepare_oulad_inputs(
                        data, train_index, validation_index, fitted=train.preprocessors
                    )
                    x, y = _resample(_matrix(train), train.target.astype(int), imbalance, 3407)
                    model = _fit(_model(candidate, parameters, 3407), x, y, imbalance)
                    probabilities.append(model.predict_proba(_matrix(validation))[:, 1])
                    targets.append(validation.target.astype(int))
                probability = np.concatenate(probabilities)
                target = np.concatenate(targets)
                threshold = choose_threshold(target, probability)
                trial.set_user_attr("parameters", parameters)
                trial.set_user_attr("imbalance", imbalance)
                trial.set_user_attr("threshold", threshold)
                return float(threshold["inner_macro_f1"])

            budget = 12
            completed = len(
                [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            )
            if completed < budget:
                study.optimize(
                    objective,
                    n_trials=budget - completed,
                    catch=(RuntimeError, ValueError),
                )
            best = study.best_trial
            parameters = dict(best.user_attrs["parameters"])
            imbalance = str(best.user_attrs["imbalance"])
            threshold = dict(best.user_attrs["threshold"])
            selected_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate": candidate,
                    "parameters": parameters,
                    "imbalance": imbalance,
                    "threshold": threshold,
                    "inner_macro_f1": float(best.value),
                }
            )
            train = prepare_oulad_inputs(data, outer_train, outer_train)
            validation = prepare_oulad_inputs(
                data, outer_train, outer_validation, fitted=train.preprocessors
            )
            for seed in FINAL_SEEDS:
                x, y = _resample(_matrix(train), train.target.astype(int), imbalance, seed)
                model = _fit(_model(candidate, parameters, seed), x, y, imbalance)
                probability = model.predict_proba(_matrix(validation))[:, 1]
                model_path = model_root / f"{candidate}_outer_{outer_fold}_seed_{seed}.joblib"
                joblib.dump({"model": model, "preprocessors": train.preprocessors}, model_path)
                for index, probability_value in zip(outer_validation, probability):
                    prediction_rows.append(
                        {
                            "record_id": str(data.base.record_ids[index]),
                            "id_student": int(data.groups[index]),
                            "outer_fold": outer_fold,
                            "candidate": candidate,
                            "seed": seed,
                            "target": int(data.y[index]),
                            "probability": float(probability_value),
                            "threshold": float(threshold["threshold"]),
                            "model_path": model_path.relative_to(ROOT).as_posix(),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_parquet(artifact / "ml_oof_predictions.parquet", index=False)
    metric_rows = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"]):
        metrics = binary_metrics_per_record_threshold(
            frame.target.to_numpy(), frame.probability.to_numpy(), frame.threshold.to_numpy()
        )
        metric_rows.append({"candidate": candidate, "seed": int(seed), **metrics})
    for candidate, frame in predictions.groupby("candidate"):
        ensemble = (
            frame.groupby(["record_id", "id_student", "outer_fold", "target", "threshold"], as_index=False)
            .probability.mean()
        )
        metrics = binary_metrics_per_record_threshold(
            ensemble.target.to_numpy(), ensemble.probability.to_numpy(), ensemble.threshold.to_numpy()
        )
        metric_rows.append({"candidate": f"{candidate}_ensemble", "seed": -1, **metrics})
    metrics = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    metrics.to_json(output, orient="records", indent=2)
    atomic_write_json(artifact / "ml_selected_configs.json", selected_rows)
    return {"status": "COMPLETE", "metrics": metrics.to_dict(orient="records")}


__all__ = ["BASELINES", "IMBALANCE", "evaluate_oulad_baselines"]
