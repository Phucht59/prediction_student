"""Shared scientific evaluation utilities for outcome-grounded recommender V2.1.

The module enforces three invariants:
1. labels and preprocessing are fitted on training partitions only;
2. every ranking model is trained and evaluated within learner-grouped splits;
3. all reported metrics are computed per learner-stage ranking group.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence
import math
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRanker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMRanker = None

try:
    from xgboost import XGBRanker, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRanker = None
    XGBRegressor = None


NUMERIC_FEATURES = [
    "risk_probability",
    "risk_uncertainty",
    "active_days",
    "inactive_streak",
    "activity_trend",
    "assessment_progress",
    "vle_intensity",
    "opportunity_count",
    "deficit_score",
    "evidence_strength",
    "workload_minutes",
    "counterfactual_v1_delta",
    "action_needed",
]

STATE_FEATURES = [
    "risk_probability",
    "risk_uncertainty",
    "active_days",
    "inactive_streak",
    "activity_trend",
    "assessment_progress",
    "vle_intensity",
    "opportunity_count",
    "deficit_score",
]

CATEGORICAL_FEATURES = ["stage", "course", "presentation", "action_family"]

ACTION_SPECIFIC_FEATURES = [
    "action_family",
    "opportunity_count",
    "deficit_score",
    "evidence_strength",
    "workload_minutes",
    "counterfactual_v1_delta",
    "action_needed",
    "action_available",
]

POLICY_ORDER = {
    "ASSESSMENT_COMPLETION": 5.0,
    "VLE_ENGAGEMENT": 4.0,
    "STUDY_REGULARITY": 3.0,
    "QUIZ_OR_RETRIEVAL_PRACTICE": 2.0,
    "CONTENT_REVIEW": 1.0,
}


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def _safe_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.reindex(columns=columns).copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


@dataclass
class _ExpectedSignalSpec:
    behavior_model: Any
    behavior_constant: float
    behavior_lo: float
    behavior_hi: float
    behavior_mean: float
    behavior_std: float
    proximal_model: Any | None
    proximal_constant: float
    proximal_lo: float
    proximal_hi: float
    proximal_mean: float
    proximal_std: float
    has_proximal: bool
    grade_thresholds: tuple[float, float, float]


class RelevanceTransformer:
    """Fit future-signal residuals and grade thresholds using training data only."""

    def __init__(
        self,
        state_features: Sequence[str] = STATE_FEATURES,
        behavior_weight: float = 0.60,
        proximal_weight: float = 0.40,
        min_model_rows: int = 30,
        seed: int = 20260804,
    ) -> None:
        self.state_features = list(state_features)
        self.behavior_weight = float(behavior_weight)
        self.proximal_weight = float(proximal_weight)
        self.min_model_rows = int(min_model_rows)
        self.seed = int(seed)
        self.specs: dict[tuple[str, str], _ExpectedSignalSpec] = {}
        self.global_spec: _ExpectedSignalSpec | None = None
        self.global_medians: pd.Series | None = None
        self.is_fitted = False

    def _fit_regressor(self, x: np.ndarray, y: np.ndarray) -> tuple[Any | None, float]:
        finite = np.isfinite(y)
        x = x[finite]
        y = y[finite]
        constant = float(np.nanmean(y)) if len(y) else 0.0
        if len(y) < self.min_model_rows or np.unique(y).size < 2:
            return None, constant
        model = HistGradientBoostingRegressor(
            max_iter=120,
            max_depth=3,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=self.seed,
        )
        model.fit(x, y)
        return model, constant

    @staticmethod
    def _residual_parameters(residual: np.ndarray) -> tuple[float, float, float, float]:
        residual = residual[np.isfinite(residual)]
        if residual.size == 0:
            return -1.0, 1.0, 0.0, 1.0
        lo, hi = np.quantile(residual, [0.01, 0.99])
        clipped = np.clip(residual, lo, hi)
        mean = float(np.mean(clipped))
        std = float(np.std(clipped))
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        return float(lo), float(hi), mean, std

    def _fit_spec(self, frame: pd.DataFrame, medians: pd.Series) -> _ExpectedSignalSpec:
        numeric = _safe_numeric_frame(frame, self.state_features)
        x = numeric.fillna(medians).to_numpy(dtype=np.float32)

        behavior_y = pd.to_numeric(frame["future_behavior_signal"], errors="coerce").to_numpy(dtype=float)
        behavior_model, behavior_constant = self._fit_regressor(x, behavior_y)
        behavior_pred = np.full(len(frame), behavior_constant, dtype=float)
        if behavior_model is not None:
            behavior_pred = np.asarray(behavior_model.predict(x), dtype=float)
        behavior_residual = behavior_y - behavior_pred
        behavior_lo, behavior_hi, behavior_mean, behavior_std = self._residual_parameters(
            behavior_residual
        )
        behavior_z = (
            np.clip(behavior_residual, behavior_lo, behavior_hi) - behavior_mean
        ) / behavior_std

        proximal_available = (
            frame.get("proximal_outcome_available", pd.Series(False, index=frame.index))
            .fillna(0)
            .astype(bool)
            .to_numpy()
        )
        proximal_y = pd.to_numeric(
            frame.get("future_proximal_signal", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        proximal_valid = proximal_available & np.isfinite(proximal_y)
        proximal_model = None
        proximal_constant = 0.0
        proximal_lo, proximal_hi, proximal_mean, proximal_std = -1.0, 1.0, 0.0, 1.0
        proximal_z = np.full(len(frame), np.nan, dtype=float)
        has_proximal = bool(np.count_nonzero(proximal_valid) >= self.min_model_rows)

        if has_proximal:
            proximal_model, proximal_constant = self._fit_regressor(
                x[proximal_valid], proximal_y[proximal_valid]
            )
            proximal_pred = np.full(len(frame), proximal_constant, dtype=float)
            if proximal_model is not None:
                proximal_pred = np.asarray(proximal_model.predict(x), dtype=float)
            proximal_residual = proximal_y - proximal_pred
            proximal_lo, proximal_hi, proximal_mean, proximal_std = self._residual_parameters(
                proximal_residual[proximal_valid]
            )
            proximal_z[proximal_valid] = (
                np.clip(proximal_residual[proximal_valid], proximal_lo, proximal_hi)
                - proximal_mean
            ) / proximal_std

        relevance = behavior_z.copy()
        combine = proximal_valid & np.isfinite(proximal_z)
        relevance[combine] = (
            self.behavior_weight * behavior_z[combine]
            + self.proximal_weight * proximal_z[combine]
        )
        finite_relevance = relevance[np.isfinite(relevance)]
        if finite_relevance.size:
            q50, q75, q90 = np.quantile(finite_relevance, [0.50, 0.75, 0.90])
        else:
            q50 = q75 = q90 = 0.0

        return _ExpectedSignalSpec(
            behavior_model=behavior_model,
            behavior_constant=behavior_constant,
            behavior_lo=behavior_lo,
            behavior_hi=behavior_hi,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            proximal_model=proximal_model,
            proximal_constant=proximal_constant,
            proximal_lo=proximal_lo,
            proximal_hi=proximal_hi,
            proximal_mean=proximal_mean,
            proximal_std=proximal_std,
            has_proximal=has_proximal,
            grade_thresholds=(float(q50), float(q75), float(q90)),
        )

    def fit(self, frame: pd.DataFrame) -> "RelevanceTransformer":
        required = {
            "action_family",
            "stage",
            "future_behavior_signal",
            *self.state_features,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing relevance columns: {missing}")
        if frame.empty:
            raise ValueError("Cannot fit relevance transformer on an empty frame")

        self.global_medians = _safe_numeric_frame(frame, self.state_features).median().fillna(0.0)
        self.specs = {}
        for key, group in frame.groupby(["action_family", "stage"], sort=True):
            self.specs[(str(key[0]), str(key[1]))] = self._fit_spec(group, self.global_medians)
        self.global_spec = self._fit_spec(frame, self.global_medians)
        self.is_fitted = True
        return self

    @staticmethod
    def _predict_expected(model: Any | None, constant: float, x: np.ndarray) -> np.ndarray:
        if model is None:
            return np.full(len(x), constant, dtype=float)
        return np.asarray(model.predict(x), dtype=float)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self.global_spec is None or self.global_medians is None:
            raise RuntimeError("RelevanceTransformer must be fitted before transform")
        result = frame.copy()
        result["continuous_relevance"] = np.nan
        result["graded_relevance"] = 0
        result["label_transform_available"] = False

        numeric = _safe_numeric_frame(result, self.state_features)
        for key, indices in result.groupby(["action_family", "stage"], sort=False).groups.items():
            str_key = (str(key[0]), str(key[1]))
            spec = self.specs.get(str_key, self.global_spec)
            idx = pd.Index(indices)
            x = numeric.loc[idx].fillna(self.global_medians).to_numpy(dtype=np.float32)

            behavior_y = pd.to_numeric(
                result.loc[idx, "future_behavior_signal"], errors="coerce"
            ).to_numpy(dtype=float)
            behavior_pred = self._predict_expected(
                spec.behavior_model, spec.behavior_constant, x
            )
            behavior_residual = behavior_y - behavior_pred
            behavior_z = (
                np.clip(behavior_residual, spec.behavior_lo, spec.behavior_hi)
                - spec.behavior_mean
            ) / spec.behavior_std

            relevance = behavior_z.copy()
            if spec.has_proximal:
                available = (
                    result.loc[idx]
                    .get("proximal_outcome_available", pd.Series(False, index=idx))
                    .fillna(0)
                    .astype(bool)
                    .to_numpy()
                )
                proximal_y = pd.to_numeric(
                    result.loc[idx].get(
                        "future_proximal_signal", pd.Series(np.nan, index=idx)
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float)
                valid = available & np.isfinite(proximal_y)
                if np.any(valid):
                    proximal_pred = self._predict_expected(
                        spec.proximal_model, spec.proximal_constant, x
                    )
                    proximal_residual = proximal_y - proximal_pred
                    proximal_z = (
                        np.clip(proximal_residual, spec.proximal_lo, spec.proximal_hi)
                        - spec.proximal_mean
                    ) / spec.proximal_std
                    relevance[valid] = (
                        self.behavior_weight * behavior_z[valid]
                        + self.proximal_weight * proximal_z[valid]
                    )

            q50, q75, q90 = spec.grade_thresholds
            grades = np.select(
                [relevance <= q50, relevance <= q75, relevance <= q90],
                [0, 1, 2],
                default=3,
            ).astype(np.int8)
            result.loc[idx, "continuous_relevance"] = relevance
            result.loc[idx, "graded_relevance"] = grades
            result.loc[idx, "label_transform_available"] = np.isfinite(relevance)

        result["continuous_relevance"] = pd.to_numeric(
            result["continuous_relevance"], errors="coerce"
        ).fillna(0.0)
        result["graded_relevance"] = pd.to_numeric(
            result["graded_relevance"], errors="coerce"
        ).fillna(0).astype(np.int8)
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


class FeaturePreprocessor:
    """Train-only sparse feature transformer with explicit learner-action interactions."""

    def __init__(
        self,
        numeric_features: Sequence[str] = NUMERIC_FEATURES,
        categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
        state_features: Sequence[str] = STATE_FEATURES,
        include_interactions: bool = True,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.state_features = [c for c in state_features if c in self.numeric_features]
        self.include_interactions = bool(include_interactions)

        numeric_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        self.base = ColumnTransformer(
            [
                ("numeric", numeric_pipe, self.numeric_features),
                ("categorical", categorical_pipe, self.categorical_features),
            ],
            sparse_threshold=1.0,
        )
        self.state = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        self.action = _make_one_hot_encoder()
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        missing = sorted(
            set(self.numeric_features + self.categorical_features).difference(frame.columns)
        )
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        self.base.fit(frame)
        if self.include_interactions:
            self.state.fit(_safe_numeric_frame(frame, self.state_features))
            self.action.fit(frame[["action_family"]].astype(str))
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform")
        base_csr = sparse.csr_matrix(self.base.transform(frame), dtype=np.float32)
        if not self.include_interactions:
            return base_csr

        state = np.asarray(
            self.state.transform(_safe_numeric_frame(frame, self.state_features)),
            dtype=np.float32,
        )
        action = sparse.csr_matrix(
            self.action.transform(frame[["action_family"]].astype(str)),
            dtype=np.float32,
        )
        interaction_blocks = []
        for column_index in range(action.shape[1]):
            membership = action[:, column_index].toarray().reshape(-1, 1)
            interaction_blocks.append(sparse.csr_matrix(state * membership))
        interactions = (
            sparse.hstack(interaction_blocks, format="csr")
            if interaction_blocks
            else sparse.csr_matrix((len(frame), 0), dtype=np.float32)
        )
        return sparse.hstack([base_csr, interactions], format="csr", dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        return self.fit(frame).transform(frame)


@dataclass
class RankerBundle:
    family: str
    model: Any
    config: dict[str, Any]
    score_mode: str


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    sizes = frame.groupby("group_id", sort=False).size().astype(int).tolist()
    if sum(sizes) != len(frame):
        raise AssertionError("Ranking group sizes do not sum to row count")
    return sizes


def _pairwise_training_data(
    x: sparse.csr_matrix,
    relevance: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    pair_rows = []
    labels = []
    for _, group in frame.groupby("group_id", sort=False):
        positions = group["_row_position"].to_numpy(dtype=int)
        for left_index in range(len(positions)):
            for right_index in range(left_index + 1, len(positions)):
                left = positions[left_index]
                right = positions[right_index]
                if relevance[left] == relevance[right]:
                    continue
                preferred, other = (
                    (left, right) if relevance[left] > relevance[right] else (right, left)
                )
                difference = x[preferred] - x[other]
                pair_rows.append(difference)
                labels.append(1)
                pair_rows.append(-difference)
                labels.append(0)
    if not pair_rows:
        raise ValueError("Pairwise ranker has no non-tied action pairs")
    pair_x = sparse.vstack(pair_rows, format="csr")
    pair_y = np.asarray(labels, dtype=np.int8)
    if np.unique(pair_y).size != 2:
        raise AssertionError("Pairwise training data must contain both classes")
    return pair_x, pair_y


def fit_ranker(
    family: str,
    x: sparse.csr_matrix,
    train_frame: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
) -> RankerBundle:
    ordered = train_frame.reset_index(drop=True).copy()
    ordered["_row_position"] = np.arange(len(ordered), dtype=int)
    grades = ordered["graded_relevance"].to_numpy(dtype=int)
    relevance = ordered["continuous_relevance"].to_numpy(dtype=float)

    if family == "interaction_logistic":
        model = LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 2000)),
            solver="lbfgs",
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            model.fit(x, grades)
        return RankerBundle(family, model, dict(config), "expected_grade")

    if family == "pairwise_logistic":
        pair_x, pair_y = _pairwise_training_data(x, relevance, ordered)
        model = LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 2000)),
            solver="liblinear",
            random_state=seed,
        )
        model.fit(pair_x, pair_y)
        return RankerBundle(family, model, dict(config), "linear_utility")

    if family == "lambdamart":
        group_sizes = _group_sizes(ordered)
        if LGBMRanker is not None:
            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=int(config.get("n_estimators", 250)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                num_leaves=int(config.get("num_leaves", 15)),
                max_depth=int(config.get("max_depth", -1)),
                min_child_samples=int(config.get("min_child_samples", 20)),
                subsample=float(config.get("subsample", 1.0)),
                colsample_bytree=float(config.get("colsample_bytree", 1.0)),
                reg_lambda=float(config.get("reg_lambda", 0.0)),
                random_state=seed,
                n_jobs=int(config.get("n_jobs", 4)),
                verbosity=-1,
            )
            model.fit(x, grades, group=group_sizes)
            return RankerBundle(family, model, dict(config), "direct")
        if XGBRanker is not None:
            num_leaves = max(2, int(config.get("num_leaves", 15)))
            max_depth = int(config.get("max_depth", max(2, math.ceil(math.log2(num_leaves)))))
            model = XGBRanker(
                objective="rank:ndcg",
                eval_metric="ndcg@3",
                n_estimators=int(config.get("n_estimators", 250)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                max_depth=max_depth,
                min_child_weight=float(config.get("min_child_weight", 1.0)),
                subsample=float(config.get("subsample", 1.0)),
                colsample_bytree=float(config.get("colsample_bytree", 1.0)),
                reg_lambda=float(config.get("reg_lambda", 1.0)),
                tree_method="hist",
                random_state=seed,
                n_jobs=int(config.get("n_jobs", 4)),
            )
            model.fit(x, grades, group=group_sizes)
            return RankerBundle(family, model, dict(config), "direct")
        raise RuntimeError(
            "LambdaMART requires lightgbm or xgboost. Install one before scientific execution."
        )

    if family == "boosted_tree":
        if XGBRegressor is None:
            raise RuntimeError("boosted_tree requires xgboost for sparse feature support")
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(config.get("n_estimators", 250)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_depth=int(config.get("max_depth", 3)),
            min_child_weight=float(config.get("min_child_weight", 1.0)),
            subsample=float(config.get("subsample", 1.0)),
            colsample_bytree=float(config.get("colsample_bytree", 1.0)),
            reg_lambda=float(config.get("reg_lambda", 1.0)),
            tree_method="hist",
            random_state=seed,
            n_jobs=int(config.get("n_jobs", 4)),
        )
        model.fit(x, relevance)
        return RankerBundle(family, model, dict(config), "direct")

    raise ValueError(f"Unknown ranker family: {family}")


def predict_ranker(bundle: RankerBundle, x: sparse.csr_matrix) -> np.ndarray:
    if bundle.score_mode == "expected_grade":
        probabilities = np.asarray(bundle.model.predict_proba(x), dtype=float)
        classes = np.asarray(bundle.model.classes_, dtype=float)
        return probabilities @ classes
    if bundle.score_mode == "linear_utility":
        coefficient = np.asarray(bundle.model.coef_, dtype=float).reshape(-1)
        return np.asarray(x @ coefficient, dtype=float).reshape(-1)
    return np.asarray(bundle.model.predict(x), dtype=float).reshape(-1)


def add_baseline_scores(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    result = test_frame.copy()
    action_prior = (
        train_frame.groupby("action_family", observed=True)["continuous_relevance"]
        .mean()
        .to_dict()
    )
    result["random_debug_score"] = np.random.default_rng(seed).random(len(result))
    result["popular_score"] = result["action_family"].map(action_prior).fillna(0.0)
    result["workload_score"] = -pd.to_numeric(
        result["workload_minutes"], errors="coerce"
    ).fillna(np.inf)
    result["policy_score"] = result["action_family"].map(POLICY_ORDER).fillna(0.0)
    result["counterfactual_score"] = pd.to_numeric(
        result["counterfactual_v1_delta"], errors="coerce"
    ).fillna(-1e9)
    return result


def group_metric_rows(frame: pd.DataFrame, score_column: str, k: int = 3) -> pd.DataFrame:
    rows = []
    for group_id, group in frame.groupby("group_id", sort=False):
        scores = pd.to_numeric(group[score_column], errors="coerce").fillna(-np.inf).to_numpy()
        grades = pd.to_numeric(group["graded_relevance"], errors="coerce").fillna(0).to_numpy(dtype=float)
        continuous = pd.to_numeric(
            group["continuous_relevance"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        order = np.argsort(-scores, kind="stable")
        top_k = min(k, len(group))
        binary = grades > 0
        top_hits = binary[order[:top_k]]

        gains = grades - float(np.min(grades))
        ndcg_1 = float(ndcg_score([gains], [scores], k=1)) if np.any(gains > 0) else 0.0
        ndcg_k = float(ndcg_score([gains], [scores], k=top_k)) if np.any(gains > 0) else 0.0
        ndcg_all = float(ndcg_score([gains], [scores], k=len(group))) if np.any(gains > 0) else 0.0

        relevant_count = int(np.count_nonzero(binary))
        precision_at_k = float(np.mean(top_hits)) if top_k else 0.0
        recall_at_k = float(np.count_nonzero(top_hits) / max(relevant_count, 1))
        ap_numerator = 0.0
        hits_so_far = 0
        for rank, hit in enumerate(top_hits, start=1):
            if hit:
                hits_so_far += 1
                ap_numerator += hits_so_far / rank
        ap_denominator = max(min(relevant_count, top_k), 1)
        average_precision = float(ap_numerator / ap_denominator)
        first_hit_positions = np.flatnonzero(top_hits)
        reciprocal_rank = float(1.0 / (first_hit_positions[0] + 1)) if first_hit_positions.size else 0.0

        rows.append(
            {
                "group_id": group_id,
                "base_record_id": str(group["base_record_id"].iloc[0]),
                "stage": str(group["stage"].iloc[0]),
                "outer_fold": int(group["outer_fold"].iloc[0]),
                "ndcg_at_1": ndcg_1,
                "ndcg_at_3": ndcg_k,
                "ndcg_all": ndcg_all,
                "precision_at_1": float(binary[order[0]]) if len(order) else 0.0,
                "precision_at_3": precision_at_k,
                "recall_at_3": recall_at_k,
                "map_at_3": average_precision,
                "mrr": reciprocal_rank,
                "top1_relevance": float(continuous[order[0]]) if len(order) else 0.0,
                "top_action": str(group["action_family"].iloc[order[0]]) if len(order) else "",
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    group_rows = group_metric_rows(frame, score_column)
    metric_columns = [
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_all",
        "precision_at_1",
        "precision_at_3",
        "recall_at_3",
        "map_at_3",
        "mrr",
        "top1_relevance",
    ]
    output = {
        column: float(group_rows[column].mean()) if len(group_rows) else 0.0
        for column in metric_columns
    }
    output["groups"] = int(len(group_rows))
    output["learners"] = int(group_rows["base_record_id"].nunique()) if len(group_rows) else 0
    output["action_diversity"] = int(group_rows["top_action"].nunique()) if len(group_rows) else 0
    output["top_action_concentration"] = (
        float(group_rows["top_action"].value_counts(normalize=True).max()) if len(group_rows) else 0.0
    )
    return output


def random_null_distribution(frame: pd.DataFrame, repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for replicate in range(repetitions):
        replicate_frame = frame.copy()
        replicate_frame["_random_null_score"] = rng.random(len(replicate_frame))
        values[replicate] = aggregate_metrics(replicate_frame, "_random_null_score")["ndcg_at_3"]
    return values


def hyperparameter_configs(model_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not model_config:
        return [{}]
    keys = sorted(model_config)
    value_lists = []
    for key in keys:
        value = model_config[key]
        value_lists.append(list(value) if isinstance(value, list) else [value])
    return [dict(zip(keys, values)) for values in product(*value_lists)]


def model_selection_key(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result.get("mean_ndcg_at_3", -np.inf)),
        float(result.get("mean_precision_at_1", -np.inf)),
        float(result.get("worst_ndcg_at_3", -np.inf)),
        float(result.get("mean_action_diversity", -np.inf)),
    )
