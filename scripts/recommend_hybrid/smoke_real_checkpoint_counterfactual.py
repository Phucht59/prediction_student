"""Smoke-test counterfactual scoring with a real frozen release checkpoint.

Raw OULAD tables are intentionally excluded from Git. This smoke constructs one
deterministic contract-valid OULAD tensor, transforms it with the real frozen
fold preprocessor, and runs it through a release Hybrid CNN-BiLSTM checkpoint.
It validates integration only; it is not educational-effect evidence.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD
from src.pipelines.oulad import BASE_CHANNELS, _aggregate, _dynamic
from src.recommend_hybrid.contracts import CheckpointReference, Stage
from src.recommend_hybrid.counterfactual.feature_authority import (
    PreprocessedOULADFeatureAuthority,
)
from src.recommend_hybrid.counterfactual.oulad_tensor import (
    FrozenHybridTensorRiskPredictor,
    OULADCounterfactualScorer,
    OULADTensorCounterfactualSimulator,
    OULADTensorEffectCatalog,
)
from src.recommend_hybrid.prediction_adapter import (
    AGGREGATE_DIMENSION,
    ARCHITECTURE_HASH,
    PARAMETER_COUNT,
    STATIC_DIMENSION,
    HybridPredictionAdapter,
    file_sha256,
)

OUT = ROOT / "artifacts/recommend_hybrid/counterfactual/real_checkpoint_smoke.json"
CLAIM_BOUNDARY = "INTEGRATION_SMOKE_ONLY_NOT_EDUCATIONAL_EFFECT_EVIDENCE"
RELEASE_MODEL_ID = "cnn_bilstm_oulad"
RELEASE_CHECKPOINT = Path(
    "artifacts/final/unified_stage_aware_oulad/checkpoints/"
    "cnn_bilstm_oulad/outer_fold_0/seed_42.pt"
)
RELEASE_MAPPING = Path(
    "artifacts/final/unified_stage_aware_oulad/checkpoint_stage_mapping.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _registered_release_hash() -> str:
    mapping = json.loads((ROOT / RELEASE_MAPPING).read_text(encoding="utf-8"))
    matching = [
        row
        for row in mapping["rows"]
        if row["model_id"] == RELEASE_MODEL_ID
        and int(row["outer_fold"]) == 0
        and int(row["seed"]) == 42
        and row["prediction_stage"] == "E1_EARLY_20PCT"
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "release mapping does not uniquely identify the smoke checkpoint"
        )
    return str(matching[0]["checkpoint_sha256"])


def _decision_threshold() -> float:
    authority = json.loads(
        (
            ROOT / "artifacts/canonical_v3/oulad_h1_training_authority.json"
        ).read_text(encoding="utf-8")
    )
    row = next(
        item
        for item in authority["shared_stage"]
        if int(item["outer_fold"]) == 0
    )
    return float(row["thresholds"]["E1_EARLY_20PCT"])


def _release_adapter() -> HybridPredictionAdapter:
    checkpoint_path = ROOT / RELEASE_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_hash = file_sha256(checkpoint_path)
    expected_hash = _registered_release_hash()
    if actual_hash != expected_hash:
        raise RuntimeError("release checkpoint SHA-256 mismatch")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("release checkpoint payload is not a mapping")
    if payload.get("architecture_hash") != ARCHITECTURE_HASH:
        raise RuntimeError(
            "release checkpoint architecture does not match recommendation authority: "
            f"expected={ARCHITECTURE_HASH} actual={payload.get('architecture_hash')}"
        )
    if int(payload.get("parameter_count", -1)) != PARAMETER_COUNT:
        raise RuntimeError(
            "release checkpoint parameter count does not match authority"
        )
    if int(payload.get("aggregate_dim", -1)) != AGGREGATE_DIMENSION:
        raise RuntimeError("release checkpoint aggregate dimension mismatch")
    if int(payload.get("static_dim", -1)) != STATIC_DIMENSION:
        raise RuntimeError("release checkpoint static dimension mismatch")
    preprocessor = payload.get("preprocessor")
    if not isinstance(preprocessor, Mapping):
        raise RuntimeError("release checkpoint preprocessor is missing")

    model = CNNBiLSTMTabularResidualOULAD(
        47,
        int(payload["aggregate_dim"]),
        int(payload["static_dim"]),
        payload["config"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    reference = CheckpointReference(
        checkpoint_id="release_cnn_bilstm_oulad_outer0_seed42",
        path=str(RELEASE_CHECKPOINT),
        sha256=actual_hash,
        fold=0,
        seed=42,
    )
    return HybridPredictionAdapter(
        (model,),
        (reference,),
        stage=Stage.EARLY_20,
        fold=0,
        decision_threshold=_decision_threshold(),
        aggregate_mean=np.asarray(preprocessor["mean"]),
        aggregate_scale=np.asarray(preprocessor["scale"]),
        static_num_cols=tuple(preprocessor["num_cols"]),
        static_num_mean=np.asarray(preprocessor["num_mean"]),
        static_num_scale=np.asarray(preprocessor["num_scale"]),
        static_categories=dict(preprocessor["categories"]),
    )


def _synthetic_model_inputs(
    adapter: HybridPredictionAdapter,
) -> dict[str, torch.Tensor]:
    channel = {name: index for index, name in enumerate(BASE_CHANNELS)}
    base = np.zeros((1, 4, len(BASE_CHANNELS)), dtype=np.float32)

    base[0, :, channel["total_clicks"]] = [18.0, 12.0, 4.0, 0.0]
    base[0, :, channel["active_days"]] = [3.0, 2.0, 1.0, 0.0]
    base[0, :, channel["unique_sites"]] = [5.0, 4.0, 2.0, 0.0]
    base[0, :, channel["unique_activity_types"]] = [4.0, 3.0, 2.0, 0.0]
    base[0, :, channel["content_clicks"]] = [10.0, 7.0, 2.0, 0.0]
    base[0, :, channel["forum_clicks"]] = [2.0, 1.0, 0.0, 0.0]
    base[0, :, channel["quiz_clicks"]] = [3.0, 2.0, 1.0, 0.0]
    base[0, :, channel["assessment_related_clicks"]] = [3.0, 2.0, 1.0, 0.0]
    base[0, :, channel["submitted_assessment_count"]] = [0.0, 1.0, 0.0, 0.0]
    base[0, :, channel["available_score_count"]] = [0.0, 1.0, 1.0, 1.0]
    base[0, :, channel["cumulative_mean_score"]] = [0.0, 62.0, 62.0, 62.0]
    base[0, :, channel["cumulative_weighted_score"]] = [0.0, 12.4, 12.4, 12.4]
    base[0, :, channel["days_since_last_vle_activity"]] = [0.0, 0.0, 3.0, 10.0]
    base[0, :, channel["weeks_without_activity"]] = [0.0, 0.0, 0.0, 1.0]
    base[0, :, channel["score_missing_mask"]] = [1.0, 0.0, 0.0, 0.0]

    lengths = np.array([4], dtype=np.int64)
    mask = np.ones((1, 4), dtype=bool)
    sequence = _dynamic(base, mask)
    aggregate_temporal = _aggregate(base, lengths)
    raw_context = np.array([[0.20, 4.0, 16.0, 0.25]], dtype=np.float32)
    raw_aggregate = np.column_stack(
        [aggregate_temporal, raw_context]
    ).astype(np.float32)
    transformed_aggregate = adapter.transform_aggregate(raw_aggregate)

    return {
        "sequence": torch.from_numpy(sequence.astype(np.float32)),
        "lengths": torch.from_numpy(lengths),
        "mask": torch.from_numpy(mask.astype(np.float32)),
        "aggregate": torch.from_numpy(transformed_aggregate),
        "static": torch.zeros((1, STATIC_DIMENSION), dtype=torch.float32),
    }


def main() -> int:
    adapter = _release_adapter()
    inputs = _synthetic_model_inputs(adapter)
    baseline = adapter.predict(inputs)
    baseline_risk = float(
        baseline.probabilities[0, 1].detach().cpu().item()
    )
    if not math.isfinite(baseline_risk) or not 0.0 <= baseline_risk <= 1.0:
        raise RuntimeError("real checkpoint produced invalid baseline risk")

    catalog = OULADTensorEffectCatalog.load(
        ROOT / "configs/recommend_hybrid/counterfactual_oulad_tensor.yaml"
    )
    scorer = OULADCounterfactualScorer(
        OULADTensorCounterfactualSimulator(
            catalog,
            PreprocessedOULADFeatureAuthority(adapter),
        ),
        FrozenHybridTensorRiskPredictor(adapter),
    )
    ranking = scorer.score(
        candidate_action_ids=(
            "VLE_ENGAGEMENT",
            "STUDY_SCHEDULE",
            "TARGETED_PRACTICE",
            "ADVISOR_ESCALATION",
        ),
        model_inputs=inputs,
        reference_values={
            "total_clicks_p50": 28.0,
            "total_clicks_p65": 40.0,
            "active_days_p50": 4.0,
            "content_clicks_p50": 18.0,
            "content_clicks_p65": 25.0,
            "unique_sites_p50": 6.0,
            "quiz_clicks_p50": 7.0,
            "quiz_clicks_p65": 10.0,
            "assessment_related_clicks_p50": 8.0,
        },
        workload_minutes={
            "VLE_ENGAGEMENT": 90,
            "STUDY_SCHEDULE": 30,
            "TARGETED_PRACTICE": 120,
            "ADVISOR_ESCALATION": 30,
        },
        evidence_strength={
            "VLE_ENGAGEMENT": 1.0,
            "STUDY_SCHEDULE": 1.0,
            "TARGETED_PRACTICE": 0.8,
            "ADVISOR_ESCALATION": 1.0,
        },
    )
    all_actions = (*ranking.ranked_actions, *ranking.rejected_actions)
    if len(all_actions) != 4:
        raise RuntimeError("real checkpoint smoke lost an action decision")
    if not all(
        math.isfinite(action.counterfactual_risk)
        and 0.0 <= action.counterfactual_risk <= 1.0
        and math.isfinite(action.risk_reduction)
        and math.isfinite(action.utility_score)
        for action in all_actions
    ):
        raise RuntimeError("real checkpoint smoke produced invalid action scores")

    payload = {
        "schema_version": "real_checkpoint_counterfactual_smoke_v2",
        "generated_at": _utc_now(),
        "status": "PASS",
        "claim_boundary": CLAIM_BOUNDARY,
        "release_model_id": RELEASE_MODEL_ID,
        "release_checkpoint_path": str(RELEASE_CHECKPOINT),
        "checkpoint_ids": [
            reference.checkpoint_id
            for reference in adapter.checkpoint_references
        ],
        "architecture_hash": baseline.architecture_hash,
        "fold": adapter.fold,
        "stage": adapter.stage.value,
        "seeds": list(baseline.seeds),
        "decision_threshold": adapter.decision_threshold,
        "frozen_preprocessor_hash": adapter.frozen_preprocessor_hash,
        "baseline_risk": baseline_risk,
        "ranked_action_count": len(ranking.ranked_actions),
        "rejected_action_count": len(ranking.rejected_actions),
        "actions": [action.to_dict() for action in all_actions],
        "scientific_guards": {
            "raw_oulad_data_used": False,
            "synthetic_contract_valid_input_used": True,
            "registered_release_checkpoint_used": True,
            "release_checkpoint_sha_verified": True,
            "architecture_authority_verified": True,
            "real_frozen_preprocessor_used": True,
            "educational_effect_claimed": False,
        },
    }
    _write_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
