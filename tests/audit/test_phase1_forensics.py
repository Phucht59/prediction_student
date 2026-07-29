from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models._oulad import _OULADCNNBiLSTMBackbone
from src.models._uci import UCICNNBiLSTM
from src.models.oulad_multitask import CNNBiLSTMOULAD
from src.pipelines import oulad
from src.pipelines import uci_support
from src.pipelines.oulad import _DeepPreprocessor


ROOT = Path(__file__).resolve().parents[2]
OULAD = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad"


def _oulad_config(fusion: str = "gated_residual") -> dict[str, object]:
    return {
        "input_projection": 8,
        "conv_channels": 4,
        "kernels": [2, 3],
        "lstm_hidden": 4,
        "lstm_layers": 1,
        "pooling": "masked_mean_max",
        "pooling_projection": 8,
        "aggregate_hidden": 8,
        "static_hidden": 8,
        "fusion_hidden": 8,
        "dropout": 0.0,
        "fusion": fusion,
        "branch_dropout": 0.0,
    }


def test_oulad_masked_future_values_do_not_change_prediction() -> None:
    torch.manual_seed(17)
    model = CNNBiLSTMOULAD(47, 5, 3, _oulad_config()).eval()
    sequence = torch.randn(3, 5, 47)
    changed = sequence.clone()
    changed[:, 2:, :] = torch.randn(3, 3, 47) * 1_000
    lengths = torch.tensor([2, 2, 2])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]] * 3)
    aggregate = torch.randn(3, 5)
    static = torch.randn(3, 3)
    with torch.no_grad():
        first = model(sequence, lengths, mask, aggregate, static)["binary_logit"]
        second = model(changed, lengths, mask, aggregate, static)["binary_logit"]
    assert torch.allclose(first, second, atol=1e-7, rtol=0)


def test_deep_preprocessor_statistics_are_isolated_from_outer_values() -> None:
    import pandas as pd

    train = pd.DataFrame(
        {
            "code_module": ["AAA", "BBB"],
            "presentation_season": ["B", "J"],
            "num_of_prev_attempts": [0, 1],
            "studied_credits": [60, 90],
            "registration_lead_time": [10, 20],
            "module_presentation_length": [240, 260],
        }
    )
    aggregate = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    first = _DeepPreprocessor().fit(train, aggregate)
    extreme_frame = train.copy()
    extreme_frame.loc[:, "studied_credits"] = 10**9
    extreme_aggregate = aggregate * 10**9
    first.transform(extreme_frame, extreme_aggregate)
    second = _DeepPreprocessor().fit(train, aggregate)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.scale, second.scale)
    assert np.array_equal(first.num_mean, second.num_mean)
    assert np.array_equal(first.num_scale, second.num_scale)


def test_each_oulad_run_maps_four_stages_to_one_checkpoint() -> None:
    payload = json.loads((OULAD / "checkpoint_stage_mapping.json").read_text())
    import pandas as pd

    rows = pd.DataFrame(payload["rows"])
    grouped = rows.groupby("training_run_id")
    assert grouped.size().eq(4).all()
    assert grouped["checkpoint"].nunique().eq(1).all()
    assert grouped["checkpoint_sha256"].nunique().eq(1).all()


def test_epoch_one_is_confirmed_metadata_anomaly() -> None:
    path = (
        OULAD
        / "checkpoints"
        / "cnn_bilstm_oulad"
        / "outer_fold_0"
        / "seed_42.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = json.loads((OULAD / "training_run_manifest.json").read_text())
    row = next(
        item
        for item in manifest["runs"]
        if item["model_family"] == "cnn_bilstm"
        and item["outer_fold"] == 0
        and item["seed"] == 42
    )
    assert row["selected_epoch"] == payload["selected_epoch"]
    assert payload["selected_epoch"] == 1
    assert payload["config"]["max_epochs"] == 4


def test_threshold_selection_is_inner_oof_only() -> None:
    import pandas as pd

    protocol = oulad._protocol()
    policies = pd.read_csv(OULAD / "threshold_policies.csv")
    selected = policies.loc[
        policies["threshold_policy"] == "INNER_OOF_STAGE_THRESHOLD"
    ]
    assert protocol["training"]["outer_used_for_tuning"] is False
    assert set(selected["source"]) <= {
        "pooled_inner_oof",
        "amended_pooled_inner_oof_calibrated_probability",
    }


def test_oulad_supported_gated_fusion_forwards() -> None:
    model = CNNBiLSTMOULAD(47, 5, 3, _oulad_config()).eval()
    output = model(
        torch.zeros(2, 3, 47),
        torch.tensor([2, 2]),
        torch.tensor([[1.0, 1.0, 0.0]] * 2),
        torch.zeros(2, 5),
        torch.zeros(2, 3),
    )
    assert output["binary_logit"].shape == (2,)
    assert output["hazard_logit"].shape == (2, 20)
    assert output["outcome_logit"].shape == (2, 3)


def test_oulad_concatenation_has_confirmed_latent_aux_head_bug() -> None:
    model = CNNBiLSTMOULAD(
        47, 5, 3, _oulad_config("concatenation")
    ).eval()
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        model(
            torch.zeros(2, 3, 47),
            torch.tensor([2, 2]),
            torch.tensor([[1.0, 1.0, 0.0]] * 2),
            torch.zeros(2, 5),
            torch.zeros(2, 3),
        )


@pytest.mark.parametrize("fusion", ["concatenation", "gated", "film_residual"])
def test_all_uci_fusion_modes_forward(fusion: str) -> None:
    config = {
        "input_projection": 8,
        "cnn_channels": 8,
        "lstm_hidden": 8,
        "context_hidden": 8,
        "fusion_hidden": 8,
        "dropout": 0.0,
        "fusion": fusion,
    }
    model = UCICNNBiLSTM(7, 4, config).eval()
    output = model(
        torch.zeros(3, 2, 7),
        torch.zeros(3, 4),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
    )
    assert output["classification"].shape == (3, 3)


def test_oulad_backbone_gates_are_two_scalar_gates_per_record() -> None:
    model = _OULADCNNBiLSTMBackbone(47, 5, 3, _oulad_config()).eval()
    _, diagnostics = model(
        torch.zeros(2, 3, 47),
        torch.tensor([2, 2]),
        torch.tensor([[1.0, 1.0, 0.0]] * 2),
        torch.zeros(2, 5),
        torch.zeros(2, 3),
        return_diagnostics=True,
    )
    assert diagnostics["gate"] is not None
    assert diagnostics["gate"].shape == (2, 2)


def test_oulad_inner_split_is_deterministic_and_disjoint() -> None:
    bundle = oulad._build_bundle()
    base = bundle.base[
        ["base_record_id", "id_student", "outer_fold", "target"]
    ].drop_duplicates()
    first = list(oulad._inner_splits(base, 0))
    second = list(oulad._inner_splits(base, 0))
    assert first == second
    for fit, validation in first:
        assert not (fit & validation)


def test_uci_inner_split_is_deterministic_and_group_disjoint() -> None:
    data = uci_support._load_uci("student_mat")
    first = uci_support._inner_splits(data.target, data.groups, seed=42)
    second = uci_support._inner_splits(data.target, data.groups, seed=42)
    for (fit_a, validation_a), (fit_b, validation_b) in zip(
        first, second, strict=True
    ):
        assert np.array_equal(fit_a, fit_b)
        assert np.array_equal(validation_a, validation_b)
        assert not (
            set(data.groups[fit_a]) & set(data.groups[validation_a])
        )
