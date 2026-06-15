import numpy as np
import pandas as pd
from typing import Any

def _number(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, default)
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default

def generate_weak_labels(df: pd.DataFrame, dataset_kind: str) -> np.ndarray:
    """
    Generate weak labels for academic risks R1 to R6.
    
    dataset_kind: should be "student" or "xapi" (or a specific dataset name like "student-mat", "student-por")
    """
    kind = "student" if "student" in dataset_kind.lower() else "xapi"
    targets: list[list[float]] = []
    
    for record in df.to_dict("records"):
        if kind == "student":
            absences = _number(record, "absences")
            study_time = _number(record, "studytime", 1.0)
            failures = _number(record, "failures")
            g1 = _number(record, "G1")
            g2 = _number(record, "G2")
            alcohol = _number(record, "Dalc", 1.0) + _number(record, "Walc", 1.0)
            goout = _number(record, "goout", 1.0)
            ratio = absences / max(study_time, 0.5)
            
            targets.append(
                [
                    float(absences >= 10 or ratio >= 5),          # R1: attendance
                    float(failures > 0),                          # R2: failure_history
                    float(g2 < 10 or (g1 > 0 and g2 < g1)),       # R3: grade_gap
                    float(study_time <= 1),                       # R4: study_time
                    float(alcohol >= 6),                          # R5: wellbeing
                    float(goout >= 4),                            # R6: time_management
                ]
            )
        else:
            targets.append(
                [
                    float(str(record.get("StudentAbsenceDays", "")).strip().lower() == "above-7"), # R1: attendance
                    float(_number(record, "VisITedResources") < 40),                                # R2: resource_usage
                    float(_number(record, "raisedhands") < 30 or _number(record, "Discussion") < 30), # R3: class_engagement
                    float(_number(record, "AnnouncementsView") < 30),                               # R4: course_updates
                    float(str(record.get("ParentAnsweringSurvey", "")).strip().lower() == "no"),     # R5: parent_support
                    float(str(record.get("ParentschoolSatisfaction", "")).strip().lower() == "bad"),  # R6: school_support
                ]
            )
            
    return np.asarray(targets, dtype=np.float32)
