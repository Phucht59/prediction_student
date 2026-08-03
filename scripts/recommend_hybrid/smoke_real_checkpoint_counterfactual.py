"""Smoke-test counterfactual scoring with a real frozen checkpoint.

Raw OULAD tables are intentionally excluded from Git. This smoke therefore
constructs one deterministic, contract-valid synthetic OULAD tensor, transforms
it with the real frozen fold preprocessor, and runs it through the real Hybrid
CNN-BiLSTM checkpoint and counterfactual scorer. It validates integration and
scientific plumbing only; it is not an educational-effect evaluation.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.oulad import BASE_CHANNELS, _aggregate, _dynamic
from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.counterfactual.feature_authority import (
    PreprocessedOULADFeatureAuthority,
)
from src.recommend_hybrid.counterfactual.oulad_tensor import (
    FrozenHybridTensorRiskPredictor,
    OULADCounterfactualScorer,
    OULADTensorCounterfactualSimulator,
    OULADTensorEffectCatalog,
)
from src.recommend_hybrid.prediction_adapter import HybridPredictionAdapter

OUT = ROOT / "artifacts/recommend_hybrid/counterfactual/real_checkpoint_smoke.json"
CLAIM_BOUNDARY = "INTEGRATION_SMOKE_ONLY_NOT_EDUCATIONAL_EFFECT_EVIDENCE"


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
        # Zero is the standardized/one-hot reference point and is valid model space.
        "static": torch.zeros((1, 13), dtype=torch.float32),
    }


def main() -> int:
    adapter = HybridPredictionAdapter.from_manifest(
        ROOT,
        stage=Stage.EARLY_20,
        fold=0,
        seeds=(42,),
    )
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
        "schema_version": "real_checkpoint_counterfactual_smoke_v1",
        "generated_at": _utc_now(),
        "status": "PASS",
        "claim_boundary": CLAIM_BOUNDARY,
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
            "real_checkpoint_used": True,
            "real_frozen_preprocessor_used": True,
            "educational_effect_claimed": False,
        },
    }
    _write_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
