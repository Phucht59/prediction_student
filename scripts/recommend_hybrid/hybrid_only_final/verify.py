"""Verify hybrid-only determinism, feature boundaries, and runtime safety."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
SCRIPT = Path(__file__).resolve().parent / "tune_and_evaluate.py"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml"
FORBIDDEN_TOKENS = (
    "xgboost",
    "lightgbm",
    "lambdamart",
    "logisticregression",
    "randomforest",
    "gradientboosting",
    "sklearn.",
)
PROTECTED_NAMES = {
    "gender",
    "age_band",
    "disability",
    "region",
    "imd_band",
    "final_result",
    "date_unregistration",
    "target",
    "outer_label",
}
FUTURE_NAMES = {
    "silver_positive",
    "future_behavior_signal",
    "future_proximal_signal",
    "group_has_positive",
}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values("group_id", kind="stable")[columns].reset_index(drop=True)
    payload = ordered.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_tuning_module():
    spec = importlib.util.spec_from_file_location("hybrid_only_tuning", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load hybrid-only tuning module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    schema = json.loads((OUT / "dataset/schema.json").read_text(encoding="utf-8"))
    candidates = pd.read_parquet(OUT / "dataset/candidate_rows.parquet")
    official = pd.read_parquet(OUT / "evaluation/OOF_PREDICTIONS.parquet")
    tuning = _load_tuning_module()

    runtime_features = set(schema["runtime_features"])
    future_in_runtime = sorted(runtime_features & FUTURE_NAMES)
    protected_in_runtime = sorted(runtime_features & PROTECTED_NAMES)
    prohibited_columns_present = sorted(
        (set(candidates.columns) & PROTECTED_NAMES) - {"outer_fold"}
    )

    package_root = ROOT / "src/recommend_hybrid/hybrid_only_final"
    source_text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in package_root.glob("*.py")
    )
    forbidden_runtime_tokens = sorted(
        token for token in FORBIDDEN_TOKENS if token in source_text
    )

    replay_rows: list[pd.DataFrame] = []
    for outer_fold in protocol["evaluation"]["outer_folds"]:
        train = candidates[candidates["outer_fold"] != outer_fold].copy()
        test = candidates[candidates["outer_fold"] == outer_fold].copy()
        selected = json.loads(
            (OUT / f"model_selection/fold_{outer_fold}_selected.json").read_text(
                encoding="utf-8"
            )
        )
        replay, _, _ = tuning._evaluate_partition(train, test, selected)
        replay["outer_fold"] = int(outer_fold)
        replay_rows.append(replay)
    replay = pd.concat(replay_rows, ignore_index=True)

    comparison_columns = [
        "group_id",
        "runtime_action_id",
        "hybrid_score",
        "top_margin",
        "issued",
        "correct_top1",
    ]
    official_hash = _sha_frame(official, comparison_columns)
    replay_hash = _sha_frame(replay, comparison_columns)
    deterministic = official_hash == replay_hash

    valid_actions = {
        "ASSESSMENT_COMPLETION",
        "STUDY_SCHEDULE",
        "VLE_ENGAGEMENT",
        "RETRIEVAL_PRACTICE",
        "LEARNING_CONSOLIDATION",
    }
    unknown_actions = sorted(set(candidates["runtime_action_id"]) - valid_actions)
    constraint_violations = int(
        ((candidates["action_available"] != 1) | (candidates["prerequisite_status"] != 1)).sum()
    )

    gates = {
        "deterministic_replay": deterministic,
        "future_features_in_scoring": len(future_in_runtime) == 0,
        "protected_features_in_scoring": len(protected_in_runtime) == 0,
        "protected_columns_in_candidates": len(prohibited_columns_present) == 0,
        "forbidden_runtime_models": len(forbidden_runtime_tokens) == 0,
        "unknown_runtime_actions": len(unknown_actions) == 0,
        "constraint_violations": constraint_violations == 0,
        "protocol_forbids_learned_ranker": not bool(
            protocol["architecture"]["learned_recommendation_model_allowed"]
        ),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "official_replay_sha256": official_hash,
        "recomputed_replay_sha256": replay_hash,
        "future_features_in_runtime": future_in_runtime,
        "protected_features_in_runtime": protected_in_runtime,
        "protected_columns_present": prohibited_columns_present,
        "forbidden_runtime_tokens": forbidden_runtime_tokens,
        "unknown_runtime_actions": unknown_actions,
        "constraint_violation_count": constraint_violations,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write(OUT / "evaluation/VERIFICATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
