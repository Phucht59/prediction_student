"""Protocol c4_star_v2_1. Frozen before new HPO. Does not regenerate locked outer splits."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from experiments.hybrid_superiority_v2.protocol import (
    COLD_GUARDRAIL,
    COLD_SET,
    FORBIDDEN_OULAD,
    FORBIDDEN_UCI,
    MATERIAL_FLOOR,
    MATERIAL_SCALE,
    OULAD_STAGES,
    OULAD_TARGET,
    RAW_SHA256,
    SEEDS_ROBUST,
    UCI_STAGES,
    UCI_TARGET,
    WARM_SET,
    WARM_STAGES,
    material_margin,
    protocol_hash as parent_protocol_hash,
    protocol_payload as parent_protocol_payload,
)

PROTOCOL_ID = "c4_star_v2.1"
PARENT_PROTOCOL_ID = "hybrid_superiority_v2.0"
PRIMARY_METRIC = "ap"
PRIMARY_METRIC_FN = "sklearn.metrics.average_precision_score"
HYBRID_UCI_GRADE_BRANCH = "temporal_only"

BACKBONES = ("C0-R", "C1-R", "C2-S", "C3-G")
MECHANISMS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
BASELINE_ROSTER = ("LR", "DT", "RF", "SVM", "XGB", "CatBoost", "MLP", "TemporalSummaryCatBoost")
SEQUENCE_ONLY = ("CNN", "BiLSTM", "CNN_BiLSTM")

SEEDS_SCREEN = (42, 1201, 2026)
SCREEN_FOLD = 0
N_INNER = 3
SPLIT_SEED = 42
DEVELOPMENT_OUTER_FOLD = 0
OUTER_TEST_USED_FOR_SELECTION = False
AUTHORITY_UNTOUCHED = True

PARAM_TARGET_MIN = 100_000
PARAM_TARGET_MAX = 350_000
BOOTSTRAP_REPLICATES = 10_000
HOLM_ALPHA = 0.05
ABLATION_MARGIN = 0.005
FULL_VS_SHUFFLE_ORDER_THRESHOLD = 0.003
TEMP_HARD_C = 80

HPO_BUDGET = {
    "LR": {"min_trials": 40, "plateau": 20},
    "RF": {"min_trials": 80, "plateau": 30},
    "XGB": {"min_trials": 120, "plateau": 40},
    "CatBoost": {"min_trials": 120, "plateau": 40},
    "SVM": {"min_trials": 60, "plateau": 25},
    "MLP": {"min_trials": 120, "plateau": 40},
    "TemporalSummaryCatBoost": {"min_trials": 160, "plateau": 50},
    "DT": {"min_trials": 20, "plateau": 10, "timeout_sec": 90},
    "hybrid_screen_trials": 24,
    "c4_min_trials": 160,
    "c4_plateau": 50,
}

DT_TIMEOUT_SEC = 90
WARMUP_EPOCHS = 4
MAX_EPOCHS = 40
PATIENCE = 10


def stages_for(domain: str) -> tuple[str, ...]:
    if domain == "uci":
        return UCI_STAGES
    if domain == "oulad":
        return OULAD_STAGES
    raise ValueError(domain)


def warm_for(domain: str) -> tuple[str, ...]:
    return WARM_STAGES[domain]


def cold_for(domain: str) -> tuple[str, ...]:
    return {"uci": ("S0",), "oulad": ("20pct",)}[domain]


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_protocol_hash": parent_protocol_hash(),
        "amendment": "joint_domain_candidate_selection",
        "amendment_reason": "v2 screened long-sequence candidates almost entirely on UCI T<=2; v2.1 requires UCI short-prefix + OULAD long-sequence joint screen before C4-STAR.",
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
        "backbones": list(BACKBONES),
        "mechanisms": list(MECHANISMS),
        "baseline_roster": list(BASELINE_ROSTER),
        "sequence_only": list(SEQUENCE_ONLY),
        "seeds_robust": list(SEEDS_ROBUST),
        "seeds_screen": list(SEEDS_SCREEN),
        "n_inner": N_INNER,
        "split_seed": SPLIT_SEED,
        "development_outer_fold": DEVELOPMENT_OUTER_FOLD,
        "outer_splits_regenerated": False,
        "cold_guardrail": dict(COLD_GUARDRAIL),
        "material_floor": MATERIAL_FLOOR,
        "material_scale": MATERIAL_SCALE,
        "ablation_margin": ABLATION_MARGIN,
        "shuffle_gap_threshold": FULL_VS_SHUFFLE_ORDER_THRESHOLD,
        "param_target": [PARAM_TARGET_MIN, PARAM_TARGET_MAX],
        "hpo_budget": HPO_BUDGET,
        "dt_timeout_sec": DT_TIMEOUT_SEC,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "warmup_epochs": WARMUP_EPOCHS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "holm_alpha": HOLM_ALPHA,
        "raw_sha256": dict(RAW_SHA256),
        "authority_untouched": AUTHORITY_UNTOUCHED,
        "outer_test_used_for_selection": OUTER_TEST_USED_FOR_SELECTION,
        "one_checkpoint_scores_all_stages": True,
        "no_dataset_topology_branch": True,
        "smote_on_hybrid_tensors": False,
        "gemini_in_prediction_hpo": False,
        "temp_hard_c": TEMP_HARD_C,
        "speed_finish_not_confirmatory": True,
        "locked_split_source": "artifacts/research/hybrid_superiority_v2/cache/splits",
        "parent_payload_keys": sorted(parent_protocol_payload().keys()),
    }


def protocol_hash() -> str:
    blob = json.dumps(protocol_payload(), sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = [
    "PROTOCOL_ID",
    "PARENT_PROTOCOL_ID",
    "material_margin",
    "protocol_hash",
    "protocol_payload",
    "stages_for",
    "warm_for",
    "cold_for",
    "BACKBONES",
    "MECHANISMS",
    "BASELINE_ROSTER",
    "SEEDS_SCREEN",
    "HPO_BUDGET",
    "DT_TIMEOUT_SEC",
    "MAX_EPOCHS",
    "PATIENCE",
    "TEMP_HARD_C",
]
