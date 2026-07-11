import json
from pathlib import Path

from src.loss_description import describe_effective_loss


def test_selected_config_uses_unweighted_cross_entropy():
    path = Path("artifacts/final/final-5a0b5041-5216-4a48-9e46-b0c16ab14866/selected_config.json")
    params = json.loads(path.read_text(encoding="utf-8"))["best_params"]
    assert describe_effective_loss(params) == "CrossEntropyLoss without class weighting"


def test_weighted_ce_with_balanced_mode_is_described_as_weighted():
    assert describe_effective_loss({"loss": "weighted_ce", "class_weight_mode": "balanced"}) == "CrossEntropyLoss with class weighting"
