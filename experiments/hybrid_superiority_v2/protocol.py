"""Preregistered scientific protocol. Frozen before baseline HPO / Hybrid search."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from . import PROTOCOL_VERSION
from .io_utils import sha256_json
from .paths import DATA_ROOT

PROTOCOL_ID = PROTOCOL_VERSION
PRIMARY_METRIC = "ap"
PRIMARY_METRIC_FN = "sklearn.metrics.average_precision_score"

UCI_STAGES = ("S0", "S1", "S2")
OULAD_STAGES = ("20pct", "35pct", "50pct", "75pct", "100pct")
WARM_STAGES = {
    "uci": ("S1", "S2"),
    "oulad": ("35pct", "50pct", "75pct", "100pct"),
}
COLD_STAGES = {
    "uci": ("S0",),
    "oulad": ("20pct",),
}
WARM_SET = ("uci:S1", "uci:S2", "oulad:35pct", "oulad:50pct", "oulad:75pct", "oulad:100pct")
COLD_SET = ("uci:S0", "oulad:20pct")

UCI_TARGET = "G3 < 10"
OULAD_TARGET = "final_result in {Fail, Withdrawn}"
FORBIDDEN_UCI = ("G1", "G2", "G3", "absences")
FORBIDDEN_OULAD = ("final_result", "target", "score", "date_unregistration")
HYBRID_UCI_GRADE_BRANCH = "temporal_only"

CANDIDATES = ("C0-R", "C1-R", "C2-S", "C3-G")
PREFERRED_CANDIDATE = "C3-G"
ABLATIONS = (
    "tabular_only",
    "tabular_cnn",
    "tabular_bilstm",
    "serial_no_tabular",
    "full",
    "full_no_gate",
    "full_no_rank",
    "full_no_kd",
    "full_no_multiprefix",
)

BASELINE_ROSTER = ("LR", "DT", "RF", "SVM", "XGB", "CatBoost", "MLP")
PANEL_A = "same_raw_information_representation_native"
PANEL_B = "strict_feature_parity_diagnostic"

SEEDS_ROBUST = (42, 1201, 2026)
SEEDS_FINAL = (42, 1201, 2026, 3407, 7777)
SCREEN_SEED = 42
SCREEN_FOLD = 0
N_OUTER = 3
N_INNER = 3
SPLIT_SEED = 42
DEVELOPMENT_OUTER_FOLD = 0

STAGE_WEIGHTS = {
    "S0": 0.40,
    "S1": 1.00,
    "S2": 1.00,
    "20pct": 0.40,
    "35pct": 1.00,
    "50pct": 1.00,
    "75pct": 1.00,
    "100pct": 1.00,
}

COLD_GUARDRAIL = {
    "uci:S0": 0.05,
    "oulad:20pct": 0.02,
    "recall_at_20_drop": 0.05,
}
MATERIAL_FLOOR = 0.010
MATERIAL_SCALE = 0.10
ABLATION_MARGIN = 0.005
FULL_VS_SHUFFLE_ORDER_THRESHOLD = 0.003

PARAM_TARGET_MIN = 50_000
PARAM_TARGET_MAX = 200_000

HPO_BUDGET = {
    "smoke_trials": 2,
    "screen_trials_per_candidate": 24,
    "robust_trials_per_survivor": 80,
    "baseline_trials_uci": 40,
    "baseline_trials_oulad": 28,
}

BOOTSTRAP_REPLICATES = 10_000
HOLM_ALPHA = 0.05

RAW_SHA256 = {
    "student-mat.csv": "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80",
    "student-por.csv": "a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f",
    "assessments.csv": "8cc738fb88ad760571d6f2a23059bfee0ffcae3bcd830514c9cbd5c6d5a046f1",
    "courses.csv": "4f16eee7454b15e109b0a21a0e43be820e6846ed6f9301bb7feb5ab5ad737a75",
    "studentAssessment.csv": "fd5320786328d05af841ee7dd4b5871b9dada3b9fe9d6a3642b2f42635510a6e",
    "studentInfo.csv": "7e6f3e474a5eee00639d2a414a6c7e928745823c2d2c2563ca1780145f99b0d6",
    "studentRegistration.csv": "0d32676285372aaf2e7a80304e5b274b4fba24313e2ca4c04317225e1ec90170",
    "studentVle.csv": "52668253d876c5becbcb72185977152700cecab2942aca807fecc3dd54b937f0",
    "vle.csv": "d1b28303dea802ad87b4484e1196e878e06824850b9a4fe8aa34693439fe87e9",
}

DATA_SOURCES = {
    "uci": {
        "name": "UCI Student Performance (Mathematics + Portuguese)",
        "url": "https://archive.ics.uci.edu/dataset/320/student+performance",
        "citation": "Cortez & Silva, 2008",
        "license": "CC BY 4.0 (UCI)",
        "files": ["student-mat.csv", "student-por.csv"],
    },
    "oulad": {
        "name": "Open University Learning Analytics Dataset",
        "url": "https://analyse.kmi.open.ac.uk/open_dataset",
        "citation": "Kuzilek, Hlosta & Zdrahal, Scientific Data 2017, DOI 10.1038/sdata.2017.171",
        "license": "CC BY 4.0",
        "files": [
            "assessments.csv",
            "courses.csv",
            "studentAssessment.csv",
            "studentInfo.csv",
            "studentRegistration.csv",
            "studentVle.csv",
            "vle.csv",
        ],
    },
}

AUTHORITY_REF_COMMIT = "0cb02479154a734240b55bf5525a96e11a72e863"
AUTHORITY_UNTOUCHED = True
OUTER_TEST_USED_FOR_SELECTION = False


def stages_for(domain: str) -> tuple[str, ...]:
    if domain == "uci":
        return UCI_STAGES
    if domain == "oulad":
        return OULAD_STAGES
    raise ValueError(domain)


def warm_for(domain: str) -> tuple[str, ...]:
    return WARM_STAGES[domain]


def cold_for(domain: str) -> tuple[str, ...]:
    return COLD_STAGES[domain]


def material_margin(ap_baseline: float) -> float:
    return max(MATERIAL_FLOOR, MATERIAL_SCALE * (1.0 - float(ap_baseline)))


def normalized_margin(delta_ap: float, ap_baseline: float) -> float:
    denom = material_margin(ap_baseline)
    return float(delta_ap) / denom if denom > 0 else float("nan")


def lexicographic_key(n_warm_nonpositive: int, min_norm: float, mean_clip: float, variance: float, n_params: int, runtime: float):
    return (n_warm_nonpositive, -min_norm, -mean_clip, variance, n_params, runtime)


def scalar_objective(r_s: list[float], delta_std: float, n_params: int, hard_penalty: float = 0.0) -> float:
    arr = np.asarray(r_s, dtype=float)
    return float(
        arr.min()
        + 0.25 * np.clip(arr, -2.0, 2.0).mean()
        - 0.10 * delta_std
        - 0.02 * max(0.0, np.log10(max(n_params, 1) / 100_000))
        - hard_penalty
    )


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
    else:
        torch.backends.cudnn.benchmark = True


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_fn": PRIMARY_METRIC_FN,
        "uci_target": UCI_TARGET,
        "oulad_target": OULAD_TARGET,
        "uci_stages": list(UCI_STAGES),
        "oulad_stages": list(OULAD_STAGES),
        "warm_set": list(WARM_SET),
        "cold_set": list(COLD_SET),
        "forbidden_uci": list(FORBIDDEN_UCI),
        "forbidden_oulad": list(FORBIDDEN_OULAD),
        "hybrid_uci_grade_branch": HYBRID_UCI_GRADE_BRANCH,
        "candidates": list(CANDIDATES),
        "preferred_candidate": PREFERRED_CANDIDATE,
        "baseline_roster": list(BASELINE_ROSTER),
        "panel_a": PANEL_A,
        "panel_b": PANEL_B,
        "seeds_robust": list(SEEDS_ROBUST),
        "seeds_final": list(SEEDS_FINAL),
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "split_seed": SPLIT_SEED,
        "development_outer_fold": DEVELOPMENT_OUTER_FOLD,
        "stage_weights": STAGE_WEIGHTS,
        "cold_guardrail": COLD_GUARDRAIL,
        "material_floor": MATERIAL_FLOOR,
        "material_scale": MATERIAL_SCALE,
        "ablation_margin": ABLATION_MARGIN,
        "param_target": [PARAM_TARGET_MIN, PARAM_TARGET_MAX],
        "hpo_budget": HPO_BUDGET,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "raw_sha256": RAW_SHA256,
        "data_root_default": str(DATA_ROOT),
        "authority_ref_commit": AUTHORITY_REF_COMMIT,
        "authority_untouched": AUTHORITY_UNTOUCHED,
        "outer_test_used_for_selection": OUTER_TEST_USED_FOR_SELECTION,
        "no_dataset_topology_branch": True,
        "one_checkpoint_scores_all_stages": True,
        "separate_oulad_100_model": False,
        "smote_on_hybrid_tensors": False,
        "r2_rmse_used": False,
    }


def protocol_hash() -> str:
    return sha256_json(protocol_payload())
