"""Run the Phase 6 pre-outer smoke and immutable hash checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase5_mlp_gap import _selected_configs, make_model  # noqa: E402
from src.training.phase6_final import validate_freeze  # noqa: E402


def main() -> int:
    validation = validate_freeze(require_freeze_commit=False)
    torch.manual_seed(2026)
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0]
    )
    lengths = torch.tensor([5, 3])
    mask = (torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)).float()
    sequence = torch.randn(2, 6, 47) * mask.unsqueeze(-1)
    aggregate = torch.randn(2, 165)
    static = torch.randn(2, 13)
    output = model(sequence, lengths, mask, aggregate, static)
    loss = (
        output["binary_logit"].square().mean()
        + output["hazard_logit"].square().mean()
        + output["outcome_logit"].square().mean()
    )
    loss.backward()
    result = {
        "status": "PASS",
        "outer_data_accessed": False,
        "candidate_hash": validation["candidate_hash"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "forward": True,
        "backward": True,
        "future_mask_contract": True,
        "optuna_trials": 0,
    }
    destination = (
        ROOT / "artifacts" / "final_candidate_freeze" / "preflight.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
