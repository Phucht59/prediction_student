import numpy as np
import torch

from src.studies.v5.common.uci_model import DualBranchCNNBiLSTM, count_parameters
from src.studies.v5.oulad.models import OULADCNNBiLSTMV5


def test_uci_v5_dual_head_shapes_and_parameter_guard():
    config = {"cnn_channels": 16, "kernel_size": 1, "lstm_hidden": 16, "context_hidden": 24, "fusion_hidden": 24, "fusion": "gated", "dropout": 0.2}
    model = DualBranchCNNBiLSTM(19, config)
    classification, regression = model(torch.zeros(7, 2, 1), torch.zeros(7, 19))
    assert classification.shape == (7, 3)
    assert regression.shape == (7,)
    assert count_parameters(model) < 1_500_000


def test_oulad_v5_variants_mask_padding_and_stay_small():
    config = {"conv_channels": 16, "kernels": [3, 5], "lstm_hidden": 24, "lstm_layers": 1, "pooling": "masked_attention", "pooling_projection": 32, "aggregate_hidden": 48, "static_hidden": 16, "fusion_hidden": 32, "dropout": 0.2}
    sequence = torch.randn(5, 8, 47)
    lengths = torch.tensor([8, 7, 6, 5, 4])
    mask = torch.arange(8)[None, :] < lengths[:, None]
    for variant in ["cnn_bilstm", "cnn_only", "bilstm_only"]:
        model = OULADCNNBiLSTMV5(47, 161, 10, config, variant)
        logits, attention, gates = model(sequence, lengths, mask.float(), torch.randn(5, 161), torch.randn(5, 10), True)
        assert logits.shape == (5,)
        assert torch.isfinite(logits).all()
        assert gates.shape == (5, 3)
        assert torch.allclose(gates.sum(1), torch.ones(5), atol=1e-6)
        if attention is not None:
            assert torch.all(attention.masked_select(~mask) == 0)
        assert sum(p.numel() for p in model.parameters()) < 1_500_000

