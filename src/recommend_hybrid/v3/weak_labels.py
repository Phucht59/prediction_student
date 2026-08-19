"""Train-only ordinal weak labels. label_conflict is audit-only, never a runtime feature."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

ABSTAIN = -1


def behavioral_vote(row: pd.Series, action: str) -> int:
    if action == "ASSESSMENT_COMPLETION":
        missing = int(row.get("missing_assessment_count") or 0)
        due = int(row.get("due_soon_count") or 0)
        if missing <= 0 and due <= 0:
            return 0
        return 3 if missing >= 2 or due >= 2 else 2
    if action == "RECOVER_ENGAGEMENT":
        rate = row.get("active_day_rate")
        streak = row.get("inactivity_streak")
        if rate is None or pd.isna(rate):
            return ABSTAIN
        if float(rate) < 0.35 or (pd.notna(streak) and int(streak) >= 7):
            return 3
        if float(rate) < 0.5:
            return 2
        return 0
    if action == "STUDY_REGULARITY":
        regularity = row.get("regularity_score")
        if regularity is None or pd.isna(regularity):
            return ABSTAIN
        if float(regularity) < 0.4:
            return 3
        if float(regularity) < 0.8:
            return 2
        return 1
    if action == "TARGETED_CONTENT_REVIEW":
        if str(row.get("stage")) == "EARLY_20":
            return 0
        coverage = row.get("content_coverage")
        if coverage is None or pd.isna(coverage):
            return ABSTAIN
        if float(coverage) < 0.5:
            return 3
        if float(coverage) < 0.8:
            return 2
        return 0
    if action == "QUIZ_RETRIEVAL_PRACTICE":
        if not bool(row.get("quiz_available", False)):
            return 0
        quiz = row.get("quiz_activity")
        if quiz is None or pd.isna(quiz):
            return 2
        return 1 if float(quiz) >= 0.6 else 3
    return ABSTAIN


def feasibility_vote(eligible: bool) -> int:
    return 1 if eligible else 0


def gemini_vote(score) -> int:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return ABSTAIN
    value = int(score)
    return value if value in {0, 1, 2, 3} else ABSTAIN


def fit_label_model(votes: np.ndarray, *, seed: int, epochs: int = 500):
    label_model_class = import_module("snorkel.labeling.model").LabelModel
    model = label_model_class(cardinality=4, verbose=False)
    model.fit(L_train=np.asarray(votes, dtype=int), n_epochs=epochs, seed=seed, log_freq=max(1, epochs // 10))
    return model


def aggregate(model, votes: np.ndarray, *, min_families: int, source_families: list[str]) -> pd.DataFrame:
    matrix = np.asarray(votes, dtype=int)
    probabilities = np.asarray(model.predict_proba(L=matrix), dtype=float)
    expected = probabilities @ np.arange(4, dtype=float)
    families = []
    for row in matrix:
        active = {source_families[i] for i, vote in enumerate(row) if vote != ABSTAIN}
        families.append(len(active))
    family_count = np.asarray(families)
    confidence = probabilities.max(axis=1)
    retained = family_count >= min_families
    return pd.DataFrame(
        {
            "expected_relevance": expected,
            "hard_relevance": probabilities.argmax(axis=1),
            "label_confidence": confidence,
            "independent_source_families": family_count,
            "label_conflict": 1.0 - confidence,
            "label_status": np.where(retained, "RETAINED", "ABSTAINED"),
        }
    )
