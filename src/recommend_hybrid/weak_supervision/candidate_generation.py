"""Build valid student-state × action candidates from frozen OOF predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical_v3.oulad_data import build_canonical_bundle

from .registry import load_action_mappings
from .split import split_for_student

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/scientific_labeling/candidates.parquet"
STAGE_MAP = {"S0_EARLY_NO_GRADE": "S0", "S1_MID_G1_ONLY": "S1", "S2_LATE_G1_G2": "S2", "E1_EARLY_20PCT": "EARLY_20", "E2_EARLY_35PCT": "EARLY_35", "M1_MIDDLE_FROZEN": "MIDDLE_50", "M1_MIDDLE_50PCT": "MIDDLE_50", "L1_LATE_75PCT": "LATE_75"}


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def _entropy(probabilities: list[float]) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-(values * np.log(values)).sum() / np.log(len(values)))


def _query_id(dataset: str, student_key: str, stage: str, cutoff: float) -> str:
    return hashlib.sha256(f"{dataset}|{student_key}|{stage}|{cutoff}".encode()).hexdigest()[:32]


def _uci_queries(dataset: str) -> list[dict]:
    raw_name = "student-mat.csv" if dataset == "student_mat" else "student-por.csv"
    namespace = "student-mat" if dataset == "student_mat" else "student-por"
    raw = pd.read_csv(ROOT / "data/raw" / raw_name, sep=";")
    raw["record_id"] = [_stable_id(namespace, index) for index in range(len(raw))]
    pred = pd.read_parquet(ROOT / "artifacts/final/unified_stage_aware_uci/predictions.parquet")
    pred = pred[(pred.dataset == dataset) & (pred.model_id == f"cnn_bilstm_{dataset.split('_')[-1]}")].copy()
    pred = pred[pred.prediction_stage.isin(STAGE_MAP)].merge(raw, on="record_id", validate="many_to_one")
    result = []
    for row in pred.itertuples(index=False):
        stage = STAGE_MAP[row.prediction_stage]
        values = {"absences": float(row.absences), "study_time": float(row.studytime), "previous_failures": float(row.failures)}
        if stage in {"S1", "S2"}: values["G1"] = float(row.G1)
        if stage == "S2": values["G2"] = float(row.G2)
        probabilities = [float(row.p_low), float(row.p_medium), float(row.p_high)]
        result.append({"student_key": row.record_id, "dataset": dataset, "stage": stage, "requested_cutoff": float({"S0": 0, "S1": 1, "S2": 2}[stage]), "prediction_model_id": f"cnn_bilstm_{dataset.split('_')[-1]}", "prediction_authority": "FINAL_THESIS_MODEL_AUTHORITY", "predicted_class": int(np.argmax(probabilities)), "class_probabilities": probabilities, "uncertainty": _entropy(probabilities), "checkpoint_lineage": [f"frozen_uci_oof_outer_{int(row.outer_fold)}", "artifacts/final/unified_stage_aware_uci/predictions.parquet"], "evidence_values": values, "evidence_observation_end": float({"S0": -1, "S1": 0, "S2": 1}[stage])})
    return result


def _oulad_queries() -> list[dict]:
    pred = pd.read_parquet(ROOT / "artifacts/final/h1_final/predictions.parquet")
    pred = pred[pred.candidate.eq("H1_TABULAR_RESIDUAL_EXPERT")].copy()
    bundle = build_canonical_bundle()
    by_key = {(stage, str(row.base_record_id)): index for stage, data in bundle.stages.items() if stage != "FINAL" for index, row in data.frame.iterrows()}
    result = []
    for row in pred.itertuples(index=False):
        canonical_stage = "M1_MIDDLE_50PCT" if row.prediction_stage == "M1_MIDDLE_FROZEN" else row.prediction_stage
        stage = STAGE_MAP[canonical_stage]
        data = bundle.stages[canonical_stage]
        index = by_key[(canonical_stage, str(row.base_record_id))]
        length = int(data.lengths[index]); sequence = data.sequence[index, :length]
        values = {"activity_level": float(sequence[:, 0].mean()), "recent_activity_trend": float(sequence[-1, 21]), "inactivity_streak": float(sequence[-1, 15]), "assessment_progress": None, "assessments_due": None, "grade_trend": None, "knowledge_gap": None, "course_progress": float(data.frame.iloc[index].progress_fraction)}
        probability = float(row.probability); probabilities = [1.0 - probability, probability]
        result.append({"student_key": str(row.base_record_id), "dataset": "oulad", "stage": stage, "requested_cutoff": float(row.cutoff_day), "prediction_model_id": "h1_tabular_residual_oulad", "prediction_authority": "RECOMMEND_HYBRID_MODEL_AUTHORITY", "predicted_class": int(probability >= 0.5), "class_probabilities": probabilities, "uncertainty": _entropy(probabilities), "checkpoint_lineage": [f"frozen_h1_oof_outer_{int(row.outer_fold)}", "artifacts/final/h1_final/predictions.parquet"], "evidence_values": values, "evidence_observation_end": float(row.cutoff_day - 1)})
    return result


def build_candidates(output: Path = OUT) -> pd.DataFrame:
    actions = load_action_mappings(ROOT / "artifacts/recommend_hybrid/scientific_labeling/action_evidence_map.yaml")
    queries = _uci_queries("student_mat") + _uci_queries("student_por") + _oulad_queries()
    rows: list[dict] = []
    for query in queries:
        query_id = _query_id(query["dataset"], query["student_key"], query["stage"], query["requested_cutoff"])
        for action in actions:
            required = list(action.required_evidence)
            missing = [key for key in required if query["evidence_values"].get(key) is None]
            rows.append({**query, **query["evidence_values"], "query_id": query_id, "action_id": action.action_id, "action_status": action.status, "action_datasets": [value.value for value in action.supported_datasets], "action_stages": list(action.supported_stages), "required_evidence": required, "human_review_required": action.human_review_required, "available_evidence": sorted(key for key, value in query["evidence_values"].items() if value is not None), "missingness_flags": missing, "prediction_risk": max(query["class_probabilities"]), "split": split_for_student(query["student_key"])})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["query_id", "action_id"]).any(): raise ValueError("duplicate candidate")
    output.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(output, index=False)
    return frame


__all__ = ["OUT", "build_candidates"]
