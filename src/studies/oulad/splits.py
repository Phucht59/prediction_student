from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.studies.oulad.cohort import presentation_sort_key


def _support(frame: pd.DataFrame) -> dict[str, int]:
    return {"total": len(frame), "positive": int(frame["target_at_risk"].sum()), "negative": int((1 - frame["target_at_risk"]).sum())}


def _meets(values: dict[str, int], *, total: int, positive: int, negative: int) -> bool:
    return values["total"] >= total and values["positive"] >= positive and values["negative"] >= negative


def build_forecast_roles(cohort: pd.DataFrame, targets: pd.DataFrame, support_rule: dict[str, int]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frame = cohort.merge(targets[["record_id", "target_at_risk"]], on="record_id", validate="one_to_one")
    frame["role"] = "historical_development"
    audits = []
    for module, module_frame in frame.groupby("code_module"):
        presentations = sorted(module_frame["code_presentation"].unique(), key=presentation_sort_key)
        latest = presentations[-1]
        future = module_frame[module_frame["code_presentation"] == latest]
        historical = module_frame[module_frame["code_presentation"] != latest]
        future_support, historical_support = _support(future), _support(historical)
        eligible = _meets(historical_support, total=support_rule["historical_total_min"], positive=support_rule["historical_positive_min"], negative=support_rule["historical_negative_min"]) and _meets(future_support, total=support_rule["future_total_min"], positive=support_rule["future_positive_min"], negative=support_rule["future_negative_min"])
        frame.loc[future.index, "role"] = "future_candidate" if eligible else "descriptive_only"
        audits.append({"code_module": module, "latest_presentation": latest, "pre_overlap_eligible": eligible, **{f"historical_{key}": value for key, value in historical_support.items()}, **{f"future_{key}": value for key, value in future_support.items()}})
    return frame, audits


def build_common_split_manifests(frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]], support_rule: dict[str, int], seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    role_frames = {}
    audit_rows = []
    all_future_students: set[int] = set()
    for forecast_id, (cohort, targets) in frames.items():
        roles, audits = build_forecast_roles(cohort, targets, support_rule)
        role_frames[forecast_id] = roles
        all_future_students.update(roles.loc[roles["role"] == "future_candidate", "id_student"].astype(int))
        audit_rows.extend({"forecast_id": forecast_id, **row} for row in audits)

    manifest_rows = []
    future_rows = []
    historical_pool = []
    for forecast_id, roles in role_frames.items():
        overlap = (roles["role"] == "historical_development") & roles["id_student"].astype(int).isin(all_future_students)
        roles.loc[overlap, "role"] = "excluded_future_student_overlap"
        # Recheck the future support after the global student exclusion. Future rows are unchanged;
        # historical support is deliberately conservative after removing every future student.
        for module, group in roles.groupby("code_module"):
            future = group[group["role"] == "future_candidate"]
            historical = group[group["role"] == "historical_development"]
            if future.empty:
                continue
            historical_support, future_support = _support(historical), _support(future)
            if not (_meets(historical_support, total=support_rule["historical_total_min"], positive=support_rule["historical_positive_min"], negative=support_rule["historical_negative_min"]) and _meets(future_support, total=support_rule["future_total_min"], positive=support_rule["future_positive_min"], negative=support_rule["future_negative_min"])):
                roles.loc[future.index, "role"] = "descriptive_only"
        historical_pool.append(roles.loc[roles["role"] == "historical_development", ["record_id", "id_student", "target_at_risk"]].assign(forecast_id=forecast_id))
        role_frames[forecast_id] = roles

    combined = pd.concat(historical_pool, ignore_index=True)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    student_to_fold: dict[int, int] = {}
    for fold, (_, validation) in enumerate(splitter.split(combined, combined["target_at_risk"], groups=combined["id_student"])):
        for student in combined.iloc[validation]["id_student"].astype(int).unique():
            if student in student_to_fold and student_to_fold[student] != fold:
                raise RuntimeError("Student assigned to multiple outer folds")
            student_to_fold[student] = fold

    for forecast_id, roles in role_frames.items():
        for row in roles.itertuples(index=False):
            fold = student_to_fold.get(int(row.id_student)) if row.role == "historical_development" else None
            manifest_rows.append({"forecast_id": forecast_id, "record_id": row.record_id, "code_module": row.code_module, "code_presentation": row.code_presentation, "id_student": int(row.id_student), "target_at_risk": int(row.target_at_risk), "role": row.role, "outer_fold": fold})
            if row.role in {"future_candidate", "descriptive_only"}:
                future_rows.append({"forecast_id": forecast_id, "record_id": row.record_id, "code_module": row.code_module, "code_presentation": row.code_presentation, "id_student": int(row.id_student), "target_at_risk": int(row.target_at_risk), "role": row.role})
    manifest = pd.DataFrame(manifest_rows)
    historical = manifest[manifest["role"] == "historical_development"]
    future = manifest[manifest["role"] == "future_candidate"]
    if set(historical["id_student"]) & set(future["id_student"]):
        raise RuntimeError("Future student leaked into historical development")
    if historical["outer_fold"].isna().any():
        raise RuntimeError("Historical record lacks outer fold")
    return manifest, pd.DataFrame(future_rows), pd.DataFrame(audit_rows)
