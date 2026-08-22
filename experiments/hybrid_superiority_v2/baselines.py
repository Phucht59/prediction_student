"""Fair CPU baselines including XGBoost and CatBoost. Tune AP on inner splits only."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import os

from .metrics import binary_metrics, select_stop_threshold
from .protocol import BASELINE_ROSTER  # noqa: F401

N_JOBS = max(1, os.cpu_count() or 8)
USE_GPU_TREES = os.environ.get("HS_V2_GPU_TREES", "1") != "0"


def _cuda_ok() -> bool:
    if not USE_GPU_TREES:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


def preprocessor(frame: pd.DataFrame, columns: list[str], categorical: list[str]) -> ColumnTransformer:
    cats = [c for c in categorical if c in columns]
    nums = [c for c in columns if c not in cats]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cats,
            ),
        ],
        remainder="drop",
    )


def default_params(name: str, seed: int) -> dict[str, Any]:
    if name == "LR":
        return {"C": 1.0, "class_weight": "balanced", "max_iter": 2000}
    if name == "DT":
        return {"max_depth": 8, "min_samples_leaf": 20, "class_weight": "balanced", "random_state": seed}
    if name == "RF":
        return {"n_estimators": 200, "min_samples_leaf": 2, "max_depth": None, "class_weight": "balanced", "random_state": seed, "n_jobs": N_JOBS}
    if name == "SVM":
        return {"kernel": "linear", "C": 1.0, "class_weight": "balanced"}
    if name == "MLP":
        return {"hidden": 128, "depth": 2, "alpha": 1e-4, "lr": 1e-3, "dropout_like_alpha": 1e-4}
    if name == "XGB":
        return {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2.0, "reg_lambda": 1.0, "reg_alpha": 0.0}
    if name == "CatBoost":
        return {"depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 3.0, "random_strength": 0.5, "bagging_temperature": 0.2, "iterations": 200}
    raise ValueError(name)


def make_estimator(name: str, params: dict[str, Any], seed: int, y_train: np.ndarray):
    if name == "LR":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=int(params.get("max_iter", 2000)),
            solver="lbfgs",
            random_state=seed,
        )
    if name == "DT":
        return DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", 8)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            class_weight="balanced",
            random_state=seed,
        )
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_depth=None if not params.get("max_depth") else int(params["max_depth"]),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=seed,
            n_jobs=N_JOBS,
        )
    if name == "SVM":
        kernel = params.get("kernel", "linear")
        C = float(params.get("C", 1.0))
        cw = params.get("class_weight", "balanced")
        if kernel == "linear":
            base = LinearSVC(C=C, class_weight=cw, max_iter=int(params.get("max_iter", 2000)), dual="auto", random_state=seed)
            if params.get("calibrate", True):
                return CalibratedClassifierCV(base, method="sigmoid", cv=int(params.get("cv", 2)))
            return base
        return SVC(kernel="rbf", C=C, gamma=params.get("gamma", "scale"), class_weight=cw, probability=True, random_state=seed, cache_size=500)
    if name == "MLP":
        hidden = int(params.get("hidden", 128))
        depth = int(params.get("depth", 2))
        layers = tuple([hidden] * depth)
        return MLPClassifier(
            hidden_layer_sizes=layers,
            alpha=float(params.get("alpha", 1e-4)),
            learning_rate_init=float(params.get("lr", 1e-3)),
            max_iter=int(params.get("max_iter", 120)),
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
    if name == "XGB":
        if XGBClassifier is None:
            raise RuntimeError("XGBOOST_MISSING")
        n_pos = max(1, int(y_train.sum()))
        n_neg = max(1, int(len(y_train) - n_pos))
        xgb_kw: dict[str, Any] = {
            "n_estimators": int(params.get("n_estimators", 400)),
            "max_depth": int(params.get("max_depth", 5)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "min_child_weight": float(params.get("min_child_weight", 2.0)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "reg_alpha": float(params.get("reg_alpha", 0.0)),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1 if (_cuda_ok() and params.get("task_type", "GPU") != "CPU") else N_JOBS,
            "random_state": seed,
            "scale_pos_weight": n_neg / n_pos,
            "verbosity": 0,
        }
        if params.get("early_stopping", True):
            xgb_kw["early_stopping_rounds"] = 40
        if _cuda_ok() and params.get("task_type", "GPU") != "CPU":
            xgb_kw["device"] = "cuda"
        return XGBClassifier(**xgb_kw)
    if name == "CatBoost":
        if CatBoostClassifier is None:
            raise RuntimeError("CATBOOST_MISSING")
        cb_kw: dict[str, Any] = {
            "depth": int(params.get("depth", 6)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "l2_leaf_reg": float(params.get("l2_leaf_reg", 3.0)),
            "random_strength": float(params.get("random_strength", 0.5)),
            "iterations": int(params.get("iterations", 400)),
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "auto_class_weights": "Balanced",
            "allow_writing_files": False,
        }
        gpu = _cuda_ok() and params.get("task_type", "GPU") != "CPU"
        if gpu:
            cb_kw["task_type"] = "GPU"
            cb_kw["devices"] = "0"
            cb_kw["thread_count"] = 1
        else:
            cb_kw["task_type"] = "CPU"
            cb_kw["thread_count"] = N_JOBS
            cb_kw["bagging_temperature"] = float(params.get("bagging_temperature", 0.2))
        return CatBoostClassifier(**cb_kw)
    raise ValueError(name)


def sample_space(name: str, trial, *, domain: str = "uci") -> dict[str, Any]:
    if name == "LR":
        return {"C": trial.suggest_float("C", 1e-3, 30.0, log=True), "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])}
    if name == "DT":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40),
        }
    if name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if name == "SVM":
        kernel = "linear" if domain == "oulad" else trial.suggest_categorical("kernel", ["linear", "rbf"])
        params = {"kernel": kernel, "C": trial.suggest_float("C", 1e-3, 20.0, log=True), "class_weight": "balanced"}
        if kernel == "rbf":
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        return params
    if name == "MLP":
        return {
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "depth": trial.suggest_int("depth", 1, 3),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        }
    if name == "XGB":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        }
    if name == "CatBoost":
        return {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "iterations": trial.suggest_int("iterations", 100, 300, step=50),
        }
    raise ValueError(name)


def _predict_proba(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    scores = np.asarray(scores, dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def fit_eval(
    name: str,
    frame: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
    fit_ids: list[str],
    stop_ids: list[str],
    valid_ids: list[str],
    seed: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = frame.record_id.astype(str)
    train = frame[ids.isin(fit_ids)]
    stop = frame[ids.isin(stop_ids)]
    valid = frame[ids.isin(valid_ids)]
    params = params or default_params(name, seed)
    if name == "CatBoost":
        cats = [c for c in categorical if c in columns]
        nums = [c for c in columns if c not in cats]
        x_train = train[nums + cats].copy()
        x_stop = stop[nums + cats].copy()
        x_valid = valid[nums + cats].copy()
        for c in cats:
            x_train[c] = x_train[c].astype(str).fillna("Unknown")
            x_stop[c] = x_stop[c].astype(str).fillna("Unknown")
            x_valid[c] = x_valid[c].astype(str).fillna("Unknown")
        for c in nums:
            med = x_train[c].median()
            x_train[c] = x_train[c].fillna(med)
            x_stop[c] = x_stop[c].fillna(med)
            x_valid[c] = x_valid[c].fillna(med)
        model = make_estimator(name, params, seed, train.target.to_numpy())
        cat_idx = list(range(len(nums), len(nums) + len(cats)))
        try:
            model.fit(x_train, train.target.to_numpy(), cat_features=cat_idx, eval_set=(x_stop, stop.target.to_numpy()), use_best_model=True)
        except Exception:
            cpu_params = dict(params)
            cpu_params["task_type"] = "CPU"
            model = make_estimator(name, cpu_params, seed, train.target.to_numpy())
            model.fit(x_train, train.target.to_numpy(), cat_features=cat_idx, eval_set=(x_stop, stop.target.to_numpy()), use_best_model=True)
        stop_p = model.predict_proba(x_stop)[:, 1]
        valid_p = model.predict_proba(x_valid)[:, 1]
        n_features = x_train.shape[1]
    else:
        prep = preprocessor(train, columns, categorical)
        x_train = prep.fit_transform(train)
        x_stop = prep.transform(stop)
        x_valid = prep.transform(valid)
        y_train = train.target.to_numpy()
        model = make_estimator(name, params, seed, y_train)
        if name == "XGB":
            try:
                try:
                    model.fit(x_train, y_train, eval_set=[(x_stop, stop.target.to_numpy())], verbose=False)
                except TypeError:
                    model.fit(x_train, y_train, eval_set=[(x_stop, stop.target.to_numpy())])
            except Exception:
                cpu_params = dict(params)
                cpu_params["task_type"] = "CPU"
                cpu_params["early_stopping"] = False
                model = make_estimator(name, cpu_params, seed, y_train)
                model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train)
        stop_p = _predict_proba(model, x_stop)
        valid_p = _predict_proba(model, x_valid)
        n_features = int(x_train.shape[1])
    threshold = select_stop_threshold(stop.target.to_numpy(), stop_p)
    metrics = binary_metrics(valid.target.to_numpy(), valid_p, threshold=threshold)
    metrics.update(
        {
            "stop_ap": binary_metrics(stop.target.to_numpy(), stop_p)["ap"],
            "family": name,
            "n_features": n_features,
            "params": params,
            "outer_test_used": False,
            "valid_record_id": valid.record_id.astype(str).to_numpy(),
            "valid_p": np.asarray(valid_p, dtype=np.float32),
            "valid_y": valid.target.to_numpy(),
            "valid_group": valid.group_id.astype(str).to_numpy(),
        }
    )
    return metrics


def predictor_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    skip = {"record_id", "group_id", "target", "final_result", "id_student", "G1", "G2", "stage"}
    columns = [c for c in frame.columns if c not in skip]
    categorical = [c for c in columns if frame[c].dtype == object or str(frame[c].dtype) in {"string", "category"}]
    return columns, categorical


def _subset(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return frame[frame.record_id.astype(str).isin(set(map(str, ids)))]


def _fit_one(
    name: str,
    train: pd.DataFrame,
    stop: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
    seed: int,
    params: dict[str, Any],
):
    """Fit a single estimator on stacked FIT rows. STOP is only for tree early-stop."""
    params = dict(params)
    if name == "CatBoost":
        cats = [c for c in categorical if c in columns]
        nums = [c for c in columns if c not in cats]
        x_train = train[nums + cats].copy()
        x_stop = stop[nums + cats].copy()
        for c in cats:
            x_train[c] = x_train[c].astype(str).fillna("Unknown")
            x_stop[c] = x_stop[c].astype(str).fillna("Unknown")
        for c in nums:
            med = x_train[c].median()
            x_train[c] = x_train[c].fillna(med)
            x_stop[c] = x_stop[c].fillna(med)
        model = make_estimator(name, params, seed, train.target.to_numpy())
        cat_idx = list(range(len(nums), len(nums) + len(cats)))
        try:
            model.fit(x_train, train.target.to_numpy(), cat_features=cat_idx, eval_set=(x_stop, stop.target.to_numpy()), use_best_model=True)
        except Exception:
            cpu_params = dict(params)
            cpu_params["task_type"] = "CPU"
            model = make_estimator(name, cpu_params, seed, train.target.to_numpy())
            model.fit(x_train, train.target.to_numpy(), cat_features=cat_idx, eval_set=(x_stop, stop.target.to_numpy()), use_best_model=True)

        def predict(part: pd.DataFrame) -> np.ndarray:
            x = part[nums + cats].copy()
            for c in cats:
                x[c] = x[c].astype(str).fillna("Unknown")
            for c in nums:
                x[c] = x[c].fillna(med)
            return model.predict_proba(x)[:, 1]

        return model, predict, int(x_train.shape[1])

    prep = preprocessor(train, columns, categorical)
    x_train = prep.fit_transform(train)
    x_stop = prep.transform(stop)
    y_train = train.target.to_numpy()
    model = make_estimator(name, params, seed, y_train)
    if name == "XGB":
        try:
            try:
                model.fit(x_train, y_train, eval_set=[(x_stop, stop.target.to_numpy())], verbose=False)
            except TypeError:
                model.fit(x_train, y_train, eval_set=[(x_stop, stop.target.to_numpy())])
        except Exception:
            cpu_params = dict(params)
            cpu_params["task_type"] = "CPU"
            cpu_params["early_stopping"] = False
            model = make_estimator(name, cpu_params, seed, y_train)
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)

    def predict(part: pd.DataFrame) -> np.ndarray:
        return _predict_proba(model, prep.transform(part))

    return model, predict, int(x_train.shape[1])


def fit_eval_stacked(
    name: str,
    frame: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
    fit_ids: list[str],
    stop_ids: list[str],
    valid_ids: list[str],
    seed: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One fitted estimator, all stages. Per-stage threshold from STOP only."""
    if "stage" not in frame.columns:
        raise RuntimeError("STACKED_FRAME_MISSING_STAGE")
    train = _subset(frame, fit_ids)
    stop = _subset(frame, stop_ids)
    valid = _subset(frame, valid_ids)
    if len(train) < 20 or len(valid) < 10:
        raise RuntimeError("STACKED_TOO_SMALL")
    params = params or default_params(name, seed)
    model, predict, n_features = _fit_one(name, train, stop, columns, categorical, seed, params)
    stop_p = predict(stop)
    valid_p = predict(valid)
    stop = stop.copy()
    valid = valid.copy()
    stop["_p"] = np.asarray(stop_p, dtype=np.float64)
    valid["_p"] = np.asarray(valid_p, dtype=np.float64)
    stages: dict[str, Any] = {}
    oof_rows = []
    for stage, valid_s in valid.groupby("stage", sort=False):
        stop_s = stop[stop.stage == stage]
        if len(valid_s) < 8 or len(np.unique(valid_s.target.to_numpy())) < 2:
            continue
        if len(stop_s) < 8 or len(np.unique(stop_s.target.to_numpy())) < 2:
            threshold = 0.5
        else:
            threshold = select_stop_threshold(stop_s.target.to_numpy(), stop_s["_p"].to_numpy())
        metrics = binary_metrics(valid_s.target.to_numpy(), valid_s["_p"].to_numpy(), threshold=threshold)
        metrics["stop_ap"] = binary_metrics(stop_s.target.to_numpy(), stop_s["_p"].to_numpy())["ap"] if len(stop_s) >= 8 and len(np.unique(stop_s.target.to_numpy())) >= 2 else None
        metrics["stage"] = str(stage)
        metrics["threshold"] = float(threshold)
        stages[str(stage)] = metrics
        for record_id, group_id, y, p in zip(
            valid_s.record_id.astype(str).to_numpy(),
            valid_s.group_id.astype(str).to_numpy(),
            valid_s.target.to_numpy(),
            valid_s["_p"].to_numpy(),
        ):
            oof_rows.append({"stage": str(stage), "record_id": str(record_id), "group_id": str(group_id), "y": int(y), "p": float(p)})
    if not stages:
        raise RuntimeError("STACKED_NO_STAGE_METRICS")
    return {
        "n_models": 1,
        "one_weight_all_stages": True,
        "family": name,
        "n_features": n_features,
        "n_train_rows": int(len(train)),
        "n_stop_rows": int(len(stop)),
        "n_valid_rows": int(len(valid)),
        "params": params,
        "outer_test_used": False,
        "estimator_id": id(model),
        "stages": stages,
        "oof": oof_rows,
    }
