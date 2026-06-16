import numpy as np
import pandas as pd
from typing import Any

# Define the 6 academic risks
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
    Generate weak labels for academic risks.
    - For student data (mat/por), all 6 risks are returned: shape (N, 6)
    - For xapi data, only R3, R4, R6 have features, so 3 risks are returned: shape (N, 3)
    """
    kind = "student" if "student" in dataset_kind.lower() else "xapi"
    targets: list[list[float]] = []
    
    for record in df.to_dict("records"):
        if kind == "student":
            failures = _number(record, "failures")
            g1 = _number(record, "G1")
            g2 = _number(record, "G2")
            g3 = _number(record, "G3", np.nan)
            absences = _number(record, "absences")
            freetime = _number(record, "freetime", 1.0)
            goout = _number(record, "goout", 1.0)
            activities = str(record.get("activities", "")).strip().lower()
            study_time = _number(record, "studytime", 1.0)
            
            # R1_LOW_PRIOR_PERFORMANCE (failures > 0 or G1 < 10)
            r1 = float(failures > 0 or g1 < 10)
            # R2_DECLINING_TREND (G2 < G1)
            r2 = float(g1 > 0 and g2 < g1)
            # R3_ATTENDANCE_RISK (absences >= 10)
            r3 = float(absences >= 10)
            # R4_LOW_ENGAGEMENT (goout >= 4 or freetime >= 4 or activities == "no")
            r4 = float(goout >= 4 or freetime >= 4 or activities == "no")
            # R5_INSUFFICIENT_STUDY_TIME (studytime <= 1)
            r5 = float(study_time <= 1)
            # R6_HIGH_FAILURE_PROBABILITY (failure history, weak prior grade, or low final grade when available)
            r6 = float(failures > 0 or g1 <= 8 or (not pd.isna(g3) and g3 <= 9))
            
            targets.append([r1, r2, r3, r4, r5, r6])
        else:
            absences_str = str(record.get("StudentAbsenceDays", "")).strip().lower()
            visited = _number(record, "VisITedResources")
            raised = _number(record, "raisedhands")
            discussion = _number(record, "Discussion")
            announcements = _number(record, "AnnouncementsView")
            cls = str(record.get("Class", "")).strip().upper()
            
            # R3_ATTENDANCE_RISK (StudentAbsenceDays == 'Above-7')
            r3 = float(absences_str == "above-7")
            
            # R4_LOW_ENGAGEMENT (VisITedResources/raisedhands/Discussion/AnnouncementsView)
            r4 = float(visited < 30 or raised < 30 or discussion < 30 or announcements < 30)
            
            # R6_HIGH_FAILURE_PROBABILITY (observed low class or very low engagement with no parent support)
            parent_answer = str(record.get("ParentAnsweringSurvey", "")).strip().lower()
            r6 = float(cls == "L" or (visited < 20 and raised < 20 and parent_answer == "no"))
            
            targets.append([r3, r4, r6])
            
    return np.asarray(targets, dtype=np.float32)
