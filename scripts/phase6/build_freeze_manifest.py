"""Create the immutable H1 scientific freeze before any Phase 6 outer access."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines import oulad  # noqa: E402
from src.training.control import stable_hash  # noqa: E402
from src.training.phase3_optuna import write_json  # noqa: E402
from src.training.phase5_mlp_gap import architecture_registry  # noqa: E402

OUT = ROOT / "artifacts" / "final_candidate_freeze"
REPORT = ROOT / "reports" / "final_candidate" / "FINAL_H1_FREEZE.md"
AUTHORITY = ROOT / "configs" / "registry" / "oulad_unified_stage_aware_v2.yaml"
PROTOCOL = ROOT / "configs" / "final" / "oulad_prediction.yaml"
PHASE3_SELECTED = ROOT / "artifacts" / "audit" / "phase3" / "selected_configs.json"
PHASE5_RUNS = ROOT / "artifacts" / "audit" / "phase5" / "runtime" / "runs"
SEEDS = [42, 1201, 2026, 3407, 7319]
MODELS = ["H1_TABULAR_RESIDUAL_EXPERT", "H0_CURRENT_HYBRID", "M0_MLP"]


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _phase5_screening(candidate: str, outer_fold: int) -> dict[str, Any]:
    paths = sorted(
        PHASE5_RUNS.glob(f"screening_{candidate}_outer{outer_fold}_seed42_*.json")
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one Phase 5 screening authority for {candidate} fold {outer_fold}"
        )
    return json.loads(paths[0].read_text(encoding="utf-8"))


def build_manifest() -> dict[str, Any]:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    selected = json.loads(PHASE3_SELECTED.read_text(encoding="utf-8"))
    registry = {row["architecture_id"]: row for row in architecture_registry()}
    inner_authority: dict[str, Any] = {}
    for candidate in MODELS:
        inner_authority[candidate] = {}
        for fold in range(3):
            result = _phase5_screening(candidate, fold)
            inner_authority[candidate][str(fold)] = {
                "source_run_id": result["run_id"],
                "source_commit": result["commit_sha"] if "commit_sha" in result else "PHASE5_COMMIT",
                "training_seed": 42,
                "inner_selected_epochs": result.get("inner_selected_epochs", []),
                "selected_refit_epoch": result.get("aggregated_epoch"),
                "research_thresholds": result["research_thresholds"],
                "threshold_source": "pooled_inner_oof_only",
                "outer_labels_used": False,
            }
    architecture = {
        **authority["architecture"],
        "model_class": (
            "src.models.oulad_tabular_residual.CNNBiLSTMTabularResidualOULAD"
        ),
        "candidate_id": "H1_TABULAR_RESIDUAL_EXPERT",
        "tabular_residual_expert": {
            "input": ["aggregate_165", "static_runtime_13"],
            "layers": [
                "Linear(178,48)",
                "LayerNorm(48)",
                "GELU",
                "Dropout",
                "Linear(48,32)",
                "GELU",
                "Linear(32,1)",
            ],
            "maximum_hidden_transforms": 2,
            "residual_formula": "z_final=z_hybrid+sigmoid(a)*z_tabular",
            "alpha_initial": 0.05,
            "alpha_policy": "learnable_bounded_sigmoid",
            "sample_dependent": False,
        },
    }
    feature_schema = {
        **authority["dataset_schema"],
        "temporal_channel_order": list(oulad.CHANNELS),
        "temporal_base_channel_order": list(oulad.BASE_CHANNELS),
        "aggregate_contract": "10 statistics x 16 stage-safe channels + inactive_count + 4 context",
        "aggregate_feature_count": 165,
        "static_columns": list(oulad.STATIC_COLUMNS),
        "categorical_columns": list(oulad.CATEGORICAL),
        "stage_context": list(oulad.CONTEXT_COLUMNS),
        "forbidden_predictors": protocol["feature_contract"]["forbidden_predictors"],
    }
    preprocessing = {
        "class": "src.pipelines.oulad._DeepPreprocessor",
        "fit_scope": "outer_train_only",
        "aggregate": "train_mean_std_nan_to_zero",
        "static_numeric": "train_mean_std",
        "categorical": "train_fit_one_hot_unknown_all_zero",
        "future_mask": "event_day<cutoff_day_and_zero_padded",
    }
    training_by_fold = {
        fold: {
            **selected[fold]["config"],
            "selected_phase3_trial": selected[fold]["trial_number"],
        }
        for fold in sorted(selected)
    }
    training_policy = {
        "optimizer": "AdamW",
        "per_outer_fold_config": training_by_fold,
        "scheduler": None,
        "max_epochs": 15,
        "checkpoint_criterion": "minimize_mean_stage_validation_nll",
        "inner_epoch_selection_rule": "round_half_up_median",
        "gradient_clip_norm": 1.0,
        "augmentation": None,
        "class_imbalance": "Phase3 selected loss policy; no resampling",
        "pretraining_requested": False,
        "pretraining_executed": False,
    }
    evaluation_protocol = {
        "protocol_id": "h1_final_outer_v1",
        "base_protocol_id": protocol["pipeline_id"],
        "stage_policy_version": authority["stage_policy_version"],
        "outer_folds": 3,
        "inner_folds": 2,
        "final_seeds": SEEDS,
        "stages": protocol["stages"],
        "one_checkpoint_all_stages": True,
        "candidate_count_h1": 1,
        "comparators": ["H0_CURRENT_HYBRID", "M0_MLP"],
        "comparator_reuse": "RECOMPUTE_PROTOCOL_MATCHED",
        "research_threshold": {
            "source": "Phase5 pooled inner OOF seed42",
            "objective": "maximize_macro_f1",
            "topology": "candidate_outer_fold_stage",
            "outer_labels_used": False,
        },
        "operational_threshold": {
            "objective": "maximize_risk_recall_subject_to_precision>=0.75",
            "role": "deployment_analysis_only",
            "affects_scientific_comparison": False,
        },
        "seed_aggregation": "record_aligned_mean_probability_all_fixed_seeds",
        "bootstrap": {
            "replicates": int(protocol["bootstrap"]["replicates"]),
            "grouped_by": protocol["bootstrap"]["grouped_by"],
            "paired": True,
            "seed": 7319,
        },
        "optuna_trials": 0,
        "post_outer_development": "PROHIBITED",
    }
    hashes = {
        "architecture_hash": registry["H1_TABULAR_RESIDUAL_EXPERT"][
            "architecture_hash"
        ],
        "temporal_backbone_hash": registry["H1_TABULAR_RESIDUAL_EXPERT"][
            "temporal_backbone_hash"
        ],
        "feature_schema_hash": stable_hash(feature_schema),
        "preprocessing_hash": stable_hash(preprocessing),
        "training_policy_hash": stable_hash(training_policy),
        "evaluation_protocol_hash": stable_hash(evaluation_protocol),
    }
    scientific_configuration = {
        "candidate_id": "H1_TABULAR_RESIDUAL_EXPERT",
        "architecture": architecture,
        "feature_schema": feature_schema,
        "preprocessing": preprocessing,
        "training_policy": training_policy,
        "evaluation_protocol": evaluation_protocol,
        "inner_authority": inner_authority,
        "parameter_count": 160492,
        "loss_policy": {
            "risk": "per_fold_Phase3_selected",
            "survival_auxiliary": "per_fold_Phase3_selected",
            "outcome_auxiliary": "per_fold_Phase3_selected",
        },
    }
    hashes["final_candidate_hash"] = stable_hash(scientific_configuration)
    manifest = {
        "schema_version": "final_h1_freeze_manifest_v1",
        "freeze_status": "IMMUTABLE_PRE_OUTER",
        "candidate_id": "H1_TABULAR_RESIDUAL_EXPERT",
        "model_class": architecture["model_class"],
        "parameter_count": 160492,
        **hashes,
        "scientific_configuration": scientific_configuration,
        "dataset_version": {
            "dataset": "OULAD",
            "feature_contract": authority["config_version"],
            "target": protocol["target"],
            "raw_data_hash_deferred": (
                "not_read_pre_freeze_to_preserve_outer-label firewall"
            ),
        },
        "protocol_version": evaluation_protocol["protocol_id"],
        "source_commit": git_head(),
        "phase5_commit": "2c8baa96f563d7a4a5188abfd0c700828a91e301",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "outer_test_accessed_before_freeze": False,
        "architecture_frozen": True,
        "hyperparameters_frozen": True,
        "features_frozen": True,
        "seeds_frozen": True,
        "outer_folds_frozen": True,
        "threshold_policy_frozen": True,
        "h2_rejected": True,
    }
    return manifest


def main() -> int:
    manifest = build_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "FINAL_H1_FREEZE_MANIFEST.json", manifest)
    write_json(OUT / "freeze_manifest.json", manifest)
    REPORT.write_text(
        f"""# Final H1 Freeze

Candidate: `H1_TABULAR_RESIDUAL_EXPERT`

- ARCHITECTURE FROZEN: YES
- HYPERPARAMETERS FROZEN: YES
- FEATURES FROZEN: YES
- SEEDS FROZEN: YES
- OUTER FOLDS FROZEN: YES
- THRESHOLD POLICY FROZEN: YES
- OUTER TEST ACCESSED BEFORE FREEZE: NO
- H2 DISTILLATION INCLUDED: NO
- OPTUNA TRIALS AUTHORIZED: 0

Final candidate hash: `{manifest["final_candidate_hash"]}`

Architecture hash: `{manifest["architecture_hash"]}`

Feature schema hash: `{manifest["feature_schema_hash"]}`

Training policy hash: `{manifest["training_policy_hash"]}`

Evaluation protocol hash: `{manifest["evaluation_protocol_hash"]}`

This file and its machine-readable manifest must be committed before the Phase 6
supervisor may access any outer-test label or metric. After that access, development
is permanently closed.
""",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
