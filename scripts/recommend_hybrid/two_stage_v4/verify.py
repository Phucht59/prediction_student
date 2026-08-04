"""Verify V4 frozen authority, feature boundaries, and exact replay."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v4"
V3_CACHE = ROOT / "artifacts/recommend_hybrid/two_stage_v3/cache"
sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.two_stage_v3.data import (  # noqa: E402
    FeatureScaler,
    apply_scaler,
    load_two_stage_arrays,
)
from src.recommend_hybrid.two_stage_v4.metrics import (  # noqa: E402
    ACTION_COUNT,
    ActionAwareThresholds,
    make_decisions,
)
from src.recommend_hybrid.two_stage_v4.model import (  # noqa: E402
    ActionAwareHeadConfig,
    HybridActionAwareRecommendationHeads,
)

PROHIBITED_FEATURES = {
    "final_result",
    "date_unregistration",
    "future_behavior_signal",
    "silver_positive",
    "group_has_positive",
    "target",
    "gender",
    "age_band",
    "disability",
    "region",
    "imd_band",
}
ALLOWED_SKLEARN_MODULES = {"sklearn.metrics", "sklearn.model_selection"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_config(payload: dict) -> ActionAwareHeadConfig:
    return ActionAwareHeadConfig(
        group_feature_dim=int(payload["group_feature_dim"]),
        action_feature_dim=int(payload["action_feature_dim"]),
        group_hidden_dim=int(payload["group_hidden_dim"]),
        action_embedding_dim=int(payload["action_embedding_dim"]),
        dropout=float(payload["dropout"]),
        recommendability_loss_weight=float(
            payload["recommendability_loss_weight"]
        ),
        listwise_loss_weight=float(payload["listwise_loss_weight"]),
        candidate_binary_loss_weight=float(
            payload["candidate_binary_loss_weight"]
        ),
        consistency_loss_weight=float(payload["consistency_loss_weight"]),
        focal_gamma=float(payload["focal_gamma"]),
    )


def _thresholds(payload: dict) -> ActionAwareThresholds:
    return ActionAwareThresholds(
        stage_gate_probability=tuple(
            float(value) for value in payload["stage_gate_probability"]
        ),
        direct_action_blend=float(payload["direct_action_blend"]),
        minimum_action_probability=float(payload["minimum_action_probability"]),
        minimum_action_margin=float(payload["minimum_action_margin"]),
        action_probability_by_id=tuple(
            float(value) for value in payload["action_probability_by_id"]
        ),
    )


def _predict(
    state_dict: dict[str, torch.Tensor],
    arrays,
    indexes: np.ndarray,
    config_payload: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = HybridActionAwareRecommendationHeads(_head_config(config_payload)).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(arrays.group_features[indexes]),
        torch.from_numpy(arrays.action_features[indexes]),
        torch.from_numpy(arrays.action_ids[indexes]),
        torch.from_numpy(arrays.action_mask[indexes]),
    )
    direct_rows = []
    action_rows = []
    with torch.inference_mode():
        for group, action, action_ids, mask in DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
        ):
            output = model(
                group.to(device),
                action.to(device),
                action_ids.to(device),
                mask.to(device),
            )
            direct_rows.append(output.direct_gate_logit.cpu().numpy())
            action_rows.append(output.action_logits.cpu().numpy())
    return np.concatenate(direct_rows), np.concatenate(action_rows)


def _sklearn_import_audit(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("sklearn") and alias.name not in ALLOWED_SKLEARN_MODULES:
                    violations.append(alias.name)
        if module and module.startswith("sklearn") and module not in ALLOWED_SKLEARN_MODULES:
            violations.append(module)
    return sorted(set(violations))


def main() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/recommend_hybrid/two_stage_v4_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (V3_CACHE / "CACHE_REGISTRY.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (OUT / "final_oof/NESTED_OOF_RESULTS.json").read_text(encoding="utf-8")
    )
    official = pd.read_parquet(OUT / "final_oof/OOF_PREDICTIONS.parquet")
    action_candidates = pd.read_parquet(V3_CACHE / "ACTION_CANDIDATES.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_rows = []
    checkpoint_authority = []

    for outer_fold in protocol["evaluation"]["outer_folds"]:
        selected = json.loads(
            (OUT / f"final_oof/fold_{outer_fold}/selected.json").read_text(
                encoding="utf-8"
            )
        )
        group_features = pd.read_parquet(
            V3_CACHE / f"outer_{outer_fold}/GROUP_FEATURES.parquet"
        )
        arrays, schema = load_two_stage_arrays(group_features, action_candidates)
        if schema != selected["feature_schema"]:
            raise RuntimeError(f"fold {outer_fold}: V4 feature schema drift")
        scaler = FeatureScaler.from_dict(selected["feature_scaler"])
        scaled = apply_scaler(arrays, scaler)
        test_indexes = np.where(arrays.outer_folds == int(outer_fold))[0]
        direct_rows = []
        action_rows = []
        for checkpoint in selected["checkpoints"]:
            checkpoint_path = ROOT / checkpoint["path"]
            if _sha256(checkpoint_path) != checkpoint["sha256"]:
                raise RuntimeError("V4 head checkpoint checksum mismatch")
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if payload.get("frozen_backbone_trainable") is not False:
                raise RuntimeError("V4 checkpoint authorizes backbone training")
            if payload.get("candidate_binary_population") != "ALL_VALID_CANDIDATES":
                raise RuntimeError("V4 checkpoint does not use all-group candidates")
            state_dict = payload["state_dict"]
            if any("backbone" in name or "tabular_expert" in name for name in state_dict):
                raise RuntimeError("V4 checkpoint contains prediction-backbone parameters")
            direct, action = _predict(
                state_dict,
                scaled,
                test_indexes,
                selected["config"],
                device,
            )
            direct_rows.append(direct)
            action_rows.append(action)
            checkpoint_authority.append(
                {
                    "outer_fold": int(outer_fold),
                    "seed": int(checkpoint["seed"]),
                    "path": checkpoint["path"],
                    "sha256": checkpoint["sha256"],
                    "head_parameter_count": int(
                        sum(value.numel() for value in state_dict.values())
                    ),
                }
            )
        direct_logits = np.mean(direct_rows, axis=0)
        action_logits = np.mean(action_rows, axis=0)
        threshold = _thresholds(selected["thresholds"])
        stages = arrays.stages[test_indexes]
        decision = make_decisions(
            direct_logits,
            action_logits,
            arrays.action_mask[test_indexes],
            stages,
            threshold,
        )
        row = np.arange(len(test_indexes))
        correct = decision.issued & (
            arrays.action_target[test_indexes][row, decision.top_action] > 0
        )
        replay = pd.DataFrame(
            {
                "group_id": arrays.group_ids[test_indexes],
                "direct_gate_logit": direct_logits,
                "direct_gate_probability": decision.direct_gate_probability,
                "action_any_probability": decision.action_any_probability,
                "joint_gate_probability": decision.joint_gate_probability,
                "top_action_index": decision.top_action,
                "top_action_probability": decision.top_probability,
                "top_action_margin": decision.top_margin,
                "issued": decision.issued.astype(int),
                "correct_top1": correct.astype(int),
            }
        )
        for action_id in range(ACTION_COUNT):
            replay[f"action_logit_{action_id}"] = action_logits[:, action_id]
        replay_rows.append(replay)

    replay = pd.concat(replay_rows, ignore_index=True).sort_values(
        "group_id",
        kind="stable",
    )
    official = official.sort_values("group_id", kind="stable").reset_index(drop=True)
    replay = replay.reset_index(drop=True)
    exact_ids = np.array_equal(
        official["group_id"].to_numpy(),
        replay["group_id"].to_numpy(),
    )
    numeric_columns = [
        "direct_gate_logit",
        "direct_gate_probability",
        "action_any_probability",
        "joint_gate_probability",
        "top_action_probability",
        "top_action_margin",
        *[f"action_logit_{index}" for index in range(ACTION_COUNT)],
    ]
    numeric_match = bool(
        np.allclose(
            official[numeric_columns].to_numpy(dtype=np.float64),
            replay[numeric_columns].to_numpy(dtype=np.float64),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    )
    discrete_columns = ["top_action_index", "issued", "correct_top1"]
    discrete_match = np.array_equal(
        official[discrete_columns].to_numpy(),
        replay[discrete_columns].to_numpy(),
    )

    feature_schema = results["feature_schema"]
    runtime_feature_names = set(feature_schema["group_continuous"]) | set(
        feature_schema["action_continuous"]
    )
    prohibited_features = sorted(runtime_feature_names & PROHIBITED_FEATURES)
    training_script = ROOT / "scripts/recommend_hybrid/two_stage_v4/train_and_evaluate.py"
    sklearn_violations = _sklearn_import_audit(training_script)
    gates = {
        "cache_complete": registry.get("status") == "COMPLETE",
        "prediction_backbone_frozen": registry.get("backbone_trainable") is False,
        "candidate_binary_all_groups": results.get("candidate_binary_population")
        == "ALL_VALID_CANDIDATES",
        "external_ml_ranker_absent": results.get("external_ml_ranker") is False
        and not sklearn_violations,
        "future_and_protected_features_absent": not prohibited_features,
        "group_authority_unchanged": len(official) == 29043,
        "exact_group_replay": exact_ids,
        "numeric_replay": numeric_match,
        "decision_replay": discrete_match,
        "all_head_checkpoints_verified": len(checkpoint_authority) == 9,
    }
    verification = {
        "schema_version": "two_stage_v4_verification_v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "prohibited_features": prohibited_features,
        "sklearn_model_import_violations": sklearn_violations,
        "checkpoint_authority": checkpoint_authority,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = OUT / "final_oof/VERIFICATION.json"
    output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    if verification["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
