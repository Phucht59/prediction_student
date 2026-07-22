import numpy as np
import pytest

from src.studies.v5.common.uci_training import UCIInputs, resample_training


@pytest.mark.parametrize("strategy", ["random_oversampling", "smote", "adasyn"])
def test_v5_uci_resampling_changes_training_only(strategy):
    rng = np.random.default_rng(42)
    target = np.array([0] * 12 + [1] * 30 + [2] * 8)
    inputs = UCIInputs(rng.normal(size=(50, 2, 1)).astype("float32"), rng.normal(size=(50, 6)).astype("float32"), target, rng.uniform(0, 20, 50).astype("float32"))
    result, before, after = resample_training(inputs, strategy, 42)
    assert len(result.target) > len(inputs.target)
    assert before != after
    assert result.sequence.ndim == 3


def test_v5_oulad_protocol_prohibits_smote_on_sequence_tensor():
    import json
    from pathlib import Path
    protocol = json.loads(Path("configs/project_v5_protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["imbalance"]["smote_or_adasyn_on_3d_sequence"] == "PROHIBITED"

