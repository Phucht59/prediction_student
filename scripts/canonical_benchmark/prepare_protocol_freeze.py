"""Freeze the canonical V3 protocol before any new benchmark scoring."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines import oulad

OUT = ROOT / "artifacts" / "canonical_v3"
CONFIG = ROOT / "configs" / "canonical_v3"
STAGES = (
    "E1_EARLY_20PCT",
    "E2_EARLY_35PCT",
    "M1_MIDDLE_50PCT",
    "L1_LATE_75PCT",
    "FINAL",
)


def stable(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = frame.loc[:, columns].drop_duplicates().sort_values(columns)
    return hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def build_monotonicity() -> tuple[dict[str, Any], pd.DataFrame]:
    base, old_cutoffs = oulad._base_and_cutoffs()
    length = base.module_presentation_length.astype(int)
    cutoffs = old_cutoffs.rename(columns={"M1_MIDDLE_FROZEN": "M1_MIDDLE_50PCT"}).copy()
    cutoffs["FINAL"] = length.to_numpy() - 14
    ordered = cutoffs.loc[:, list(STAGES)].to_numpy(dtype=int)
    relations = {
        f"{left}_subset_{right}": bool(np.all(ordered[:, index] <= ordered[:, index + 1]))
        for index, (left, right) in enumerate(zip(STAGES, STAGES[1:]))
    }
    if not all(relations.values()):
        raise RuntimeError("canonical cutoff monotonicity failed")

    feature_names = {
        "temporal": list(oulad.CHANNELS),
        "aggregate": [f"aggregate_{index:03d}" for index in range(165)],
        "static": list(oulad.STATIC_COLUMNS),
    }
    flat_features = sorted({name for values in feature_names.values() for name in values})
    sets = {stage: flat_features for stage in STAGES}

    joined = base.merge(cutoffs, on="base_record_id", validate="one_to_one")
    eligibility: list[dict[str, Any]] = []
    for stage in STAGES:
        cutoff = joined[stage]
        registered = joined.date_registration.notna() & (joined.date_registration < cutoff)
        unregistered = joined.date_unregistration.notna() & (joined.date_unregistration < cutoff)
        eligible = registered & ~unregistered
        current = joined.loc[eligible, ["base_record_id", "id_student", "outer_fold"]]
        eligibility.append(
            {
                "stage": stage,
                "eligible_records": int(len(current)),
                "eligible_students": int(current.id_student.nunique()),
                "fold_hash": frame_hash(current, ["base_record_id", "outer_fold"]),
            }
        )

    old_feature_lineage = json.loads(
        (ROOT / "artifacts/final/unified_stage_aware_oulad/feature_lineage.json").read_text(
            encoding="utf-8"
        )
    )
    phase7 = json.loads(
        (ROOT / "artifacts/audit/phase7/endpoint_protocol_audit.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": "canonical_v3_feature_monotonicity_v1",
        "status": "PASS",
        "information_policy": "STRICT_REAL_TIME",
        "score_values_excluded_all_stages": True,
        "feature_name_sets": sets,
        "feature_group_names": feature_names,
        "relations": relations,
        "event_set_proof": "identical event-key semantics and nondecreasing cutoff; allowed iff 0<=event_day<cutoff_day",
        "75_only_features": [],
        "final_only_features": [],
        "shared_features": flat_features,
        "eligibility": eligibility,
        "eligibility_semantics": "dynamic registered risk set; common-cohort diagnostic required for cross-stage attribution",
        "old_comparison": {
            "early_warning_75_stage": "L1_LATE_75PCT",
            "early_warning_score_policy": old_feature_lineage["score_policy"],
            "endpoint_id": phase7["endpoint_id"],
            "endpoint_cutoff_fraction": phase7["cutoff_fraction"],
            "endpoint_score_policy": "score values excluded; identical strict policy",
            "same_score_policy": True,
            "directly_comparable_as_75_vs_final": False,
            "reason": "Phase 7 endpoint is F2 at 50%, not an observation point after 75%; it also uses a separately trained checkpoint.",
            "root_cause_classification": [
                "G_OLD_STAGE_EVIDENCE_NOT_PROTOCOL_COMPATIBLE",
                "C_SEPARATE_TRAINING_MISMATCH",
            ],
        },
    }
    return payload, joined


def uci_fold_hashes() -> dict[str, str]:
    prediction = pd.read_parquet(
        ROOT / "artifacts/final/unified_stage_aware_uci/predictions.parquet"
    )
    result = {}
    for dataset in ("student_mat", "student_por"):
        current = prediction.loc[
            prediction.dataset.eq(dataset)
            & prediction.model_family.eq("logistic_regression")
            & prediction.prediction_stage.eq("S2_LATE_G1_G2")
        ]
        result[dataset] = frame_hash(current, ["record_id", "outer_fold"])
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    monotonicity, _ = build_monotonicity()
    write_json(OUT / "oulad_feature_monotonicity.json", monotonicity)

    policy_path = CONFIG / "oulad_information_policy.yaml"
    protocol_path = CONFIG / "benchmark_protocol.yaml"
    search_path = CONFIG / "model_search_spaces.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    search = yaml.safe_load(search_path.read_text(encoding="utf-8"))
    phase3_configs = json.loads(
        (ROOT / "artifacts/audit/phase3/selected_configs.json").read_text(encoding="utf-8")
    )
    freeze = {
        "schema_version": "canonical_benchmark_freeze_v3",
        "authority_id": "UNIFIED_CANONICAL_BENCHMARK_V3",
        "status": "IMMUTABLE_PRE_BENCHMARK",
        "benchmark_label": "CANONICAL_NESTED_CV_BENCHMARK",
        "parent_authority_commit": git_head(),
        "policy_hash": stable(policy),
        "protocol_hash": stable(protocol),
        "search_space_hash": stable(search),
        "feature_monotonicity_hash": stable(monotonicity),
        "uci_topology_hash": stable(protocol["uci"]["hybrid"]["topology"]),
        "oulad_architecture_hash": protocol["oulad"]["hybrid"]["architecture_hash"],
        "oulad_temporal_backbone_hash": protocol["oulad"]["hybrid"]["temporal_backbone_hash"],
        "oulad_parameter_count": protocol["oulad"]["hybrid"]["parameter_count"],
        "fold_hashes": {
            "uci": uci_fold_hashes(),
            "oulad_by_stage": {
                row["stage"]: row["fold_hash"] for row in monotonicity["eligibility"]
            },
        },
        "seeds": {"uci": protocol["uci"]["seeds"], "oulad": protocol["oulad"]["seeds"]},
        "phase3_training_configs": {
            fold: {"config": value["config"], "config_hash": value["config_hash"]}
            for fold, value in phase3_configs.items()
        },
        "source_hashes": {
            "information_policy": file_hash(policy_path),
            "benchmark_protocol": file_hash(protocol_path),
            "search_spaces": file_hash(search_path),
            "raw_manifest": file_hash(ROOT / "data/manifests/extension_raw_manifest.json"),
            "old_uci_predictions": file_hash(
                ROOT / "artifacts/final/unified_stage_aware_uci/predictions.parquet"
            ),
            "old_oulad_predictions": file_hash(
                ROOT / "artifacts/final/unified_stage_aware_oulad/predictions.parquet"
            ),
            "old_h1_predictions": file_hash(ROOT / "artifacts/final/h1_final/predictions.parquet"),
        },
        "outer_labels_used_to_define_policy": False,
        "new_benchmark_metrics_observed_before_freeze": False,
        "architecture_search": False,
        "post_result_tuning": "PROHIBITED",
    }
    freeze["canonical_benchmark_hash"] = stable(freeze)
    write_json(OUT / "CANONICAL_BENCHMARK_FREEZE.json", freeze)
    write_json(
        OUT / "old_75_vs_endpoint_protocol_audit.json", monotonicity["old_comparison"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
