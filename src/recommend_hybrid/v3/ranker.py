"""Five action-specific EBMs on C0 risk + student evidence. No action_id feature."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .contracts import ActionScore, CanonicalAction, RecommendationFeatures

FEATURE_COLUMNS = (
    "risk_probability",
    "uncertainty",
    "risk_margin",
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
    "vle_available",
    "study_material_available",
    "quiz_available",
    "stage",
)

FORBIDDEN_FEATURES = frozenset(
    {
        "action_id",
        "final_result",
        "target",
        "label_conflict",
        "label_confidence",
        "expected_relevance",
        "seed_disagreement",
    }
)
ORDINAL_MAX = 3.0


def feature_frame(features: RecommendationFeatures) -> pd.DataFrame:
    raw = asdict(features)
    row = {column: raw.get(column) for column in FEATURE_COLUMNS}
    row["uncertainty"] = features.uncertainty
    row["risk_margin"] = features.risk_margin
    row["vle_available"] = features.vle_access_available
    row["stage"] = features.stage.value
    return pd.DataFrame([row], columns=list(FEATURE_COLUMNS))


class FiveEBMC0Ranker:
    def __init__(self, model_parameters: dict | None = None) -> None:
        self.model_parameters = dict(model_parameters or {})
        self.models: dict[CanonicalAction, object] = {}

    @classmethod
    def from_artifacts(cls, model_dir: Path) -> "FiveEBMC0Ranker":
        ranker = cls()
        for action in CanonicalAction:
            path = Path(model_dir) / f"{action.value}.joblib"
            if not path.is_file():
                raise RuntimeError(f"missing V3 action model: {path}")
            ranker.models[action] = joblib.load(path)
        return ranker

    @staticmethod
    def _regressor_class():
        module = import_module("interpret.glassbox")
        return module.ExplainableBoostingRegressor

    def fit(
        self,
        frame: pd.DataFrame,
        targets: dict[CanonicalAction, pd.Series],
        sample_weights: dict[CanonicalAction, pd.Series] | None = None,
    ) -> "FiveEBMC0Ranker":
        leaked = set(frame.columns) & FORBIDDEN_FEATURES
        if leaked:
            raise ValueError(f"forbidden ranker features: {sorted(leaked)}")
        missing = set(FEATURE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"missing ranker features: {sorted(missing)}")
        regressor_class = self._regressor_class()
        for action in CanonicalAction:
            target = pd.to_numeric(targets[action], errors="coerce")
            retained = target.notna() & target.ge(0) & target.le(3)
            if int(retained.sum()) < 30:
                raise ValueError(f"insufficient labels for {action.value}")
            model = regressor_class(**self.model_parameters)
            kwargs: dict = {}
            if sample_weights is not None:
                weights = pd.to_numeric(sample_weights[action], errors="coerce")
                kwargs["sample_weight"] = weights.loc[retained].to_numpy(dtype=float)
            model.fit(frame.loc[retained, list(FEATURE_COLUMNS)], target.loc[retained] / ORDINAL_MAX, **kwargs)
            self.models[action] = model
        return self

    def score(
        self,
        features: RecommendationFeatures,
        eligible_actions: tuple[CanonicalAction, ...],
    ) -> tuple[ActionScore, ...]:
        frame = feature_frame(features)
        results: list[ActionScore] = []
        for action in eligible_actions:
            model = self.models.get(action)
            if model is None:
                raise RuntimeError(f"no V3 model for {action.value}")
            raw = float(np.asarray(model.predict(frame))[0])
            ordinal = float(np.clip(raw, 0.0, ORDINAL_MAX)) if raw > 1.5 else float(np.clip(raw * ORDINAL_MAX, 0.0, ORDINAL_MAX))
            # Frozen V3 artifacts train on target/3 so predict is already ~[0,1].
            public = float(np.clip(raw if raw <= 1.5 else ordinal / ORDINAL_MAX, 0.0, 1.0))
            results.append(ActionScore(action, public))
        results.sort(key=lambda item: (-item.score, item.action.value))
        return tuple(results)


class ActionStagePriorRanker:
    """B0: action+stage frequency prior from development labels."""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = dict(scores)

    def score(self, features: RecommendationFeatures, eligible_actions: tuple[CanonicalAction, ...]):
        results = []
        for action in eligible_actions:
            value = self.scores.get((features.stage.value, action.value), 0.0)
            results.append(ActionScore(action, float(np.clip(value, 0.0, 1.0))))
        results.sort(key=lambda item: (-item.score, item.action.value))
        return tuple(results)


def rule_score_for_action(action: CanonicalAction, features: RecommendationFeatures) -> float:
    if action is CanonicalAction.ASSESSMENT_COMPLETION:
        missing = float(features.missing_assessment_count or 0)
        due = float(features.due_soon_count or 0)
        return min(1.0, 0.35 * missing + 0.25 * due + (0.4 if (features.completion_rate or 1) < 0.8 else 0.0))
    if action is CanonicalAction.RECOVER_ENGAGEMENT:
        rate = 1.0 - float(features.active_day_rate or 1.0)
        streak = min(1.0, float(features.inactivity_streak or 0) / 14.0)
        return min(1.0, 0.6 * rate + 0.4 * streak)
    if action is CanonicalAction.STUDY_REGULARITY:
        return min(
            1.0,
            (1.0 - float(features.regularity_score or 1.0)) * 0.7
            + (1.0 - float(features.active_day_rate or 1.0)) * 0.3,
        )
    if action is CanonicalAction.TARGETED_CONTENT_REVIEW:
        return min(1.0, 1.0 - float(features.content_coverage or 1.0))
    if action is CanonicalAction.QUIZ_RETRIEVAL_PRACTICE:
        return min(1.0, 0.5 + 0.5 * (1.0 - float(features.quiz_activity or 0.0)))
    return 0.0


class RuleScoreRanker:
    """B1: deterministic evidence severity in [0, 1]."""

    def score(self, features: RecommendationFeatures, eligible_actions: tuple[CanonicalAction, ...]):
        results = [
            ActionScore(action, float(rule_score_for_action(action, features)))
            for action in eligible_actions
        ]
        results.sort(key=lambda item: (-item.score, item.action.value))
        return tuple(results)
