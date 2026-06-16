import numpy as np
import pandas as pd
from typing import Any

R1_LOW_PRIOR_PERFORMANCE = "R1_LOW_PRIOR_PERFORMANCE"
R2_DECLINING_TREND = "R2_DECLINING_TREND"
R3_ATTENDANCE_RISK = "R3_ATTENDANCE_RISK"
R4_LOW_ENGAGEMENT = "R4_LOW_ENGAGEMENT"
R5_INSUFFICIENT_STUDY_TIME = "R5_INSUFFICIENT_STUDY_TIME"
R6_HIGH_FAILURE_PROBABILITY = "R6_HIGH_FAILURE_PROBABILITY"


def _number(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, default)
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def generate_weak_labels(df: pd.DataFrame, dataset_kind: str) -> np.ndarray:
    """
    Generate auditable weak labels for academic risks.

    These labels train/evaluate the risk-diagnosis part of the recommender.
    They rely on observable profile/engagement signals instead of final targets.
    """
    kind = "student" if "student" in dataset_kind.lower() else "xapi"
    targets: list[list[float]] = []

    for record in df.to_dict("records"):
        if kind == "student":
            failures = _number(record, "failures")
            g1 = _number(record, "G1")
            g2 = _number(record, "G2")
            absences = _number(record, "absences")
            freetime = _number(record, "freetime", 1.0)
            goout = _number(record, "goout", 1.0)
            activities = str(record.get("activities", "")).strip().lower()
            study_time = _number(record, "studytime", 1.0)

            r1 = float(failures > 0 or g1 < 10)
            r2 = float(g1 > 0 and g2 < g1)
            r3 = float(absences >= 10)
            r4 = float(goout >= 4 or freetime >= 4 or activities == "no")
            r5 = float(study_time <= 1)
            # Do not use G3/true final grade in operational weak labels.
            r6 = float(failures > 0 or g1 <= 8 or g2 <= 9 or (g1 > 0 and g2 < g1 - 2))

            targets.append([r1, r2, r3, r4, r5, r6])
        else:
            absences_str = str(record.get("StudentAbsenceDays", "")).strip().lower()
            visited = _number(record, "VisITedResources")
            raised = _number(record, "raisedhands")
            discussion = _number(record, "Discussion")
            announcements = _number(record, "AnnouncementsView")
            parent_answer = str(record.get("ParentAnsweringSurvey", "")).strip().lower()
            school_satisfaction = str(record.get("ParentschoolSatisfaction", "")).strip().lower()

            engagement_values = np.asarray([visited, raised, discussion, announcements], dtype=float)
            engagement_avg = float(np.mean(engagement_values))
            low_signal_count = int(np.sum(engagement_values < 30))

            r3 = float(absences_str == "above-7")
            r4 = float(engagement_avg < 35 or low_signal_count >= 2 or visited < 25 or raised < 25)

            # Do not use the true xAPI Class label in operational weak labels.
            weak_support = parent_answer == "no" or school_satisfaction == "bad"
            severe_engagement_gap = engagement_avg < 25 or (visited < 20 and raised < 20)
            compound_risk = (r3 == 1.0 and engagement_avg < 45) or (weak_support and engagement_avg < 40)
            r6 = float(severe_engagement_gap or compound_risk)

            targets.append([r3, r4, r6])

    return np.asarray(targets, dtype=np.float32)
