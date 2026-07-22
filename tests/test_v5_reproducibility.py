import numpy as np
import torch

from src.studies.v5.common.uci_model import DualBranchCNNBiLSTM
from src.studies.v5.common.uci_training import deterministic_seed


def test_v5_model_initialization_is_deterministic():
    config = {"cnn_channels": 8, "kernel_size": 1, "lstm_hidden": 8, "context_hidden": 8, "fusion_hidden": 8, "fusion": "gated", "dropout": 0.0}
    deterministic_seed(42)
    first = DualBranchCNNBiLSTM(4, config)
    first_state = {name: value.detach().clone() for name, value in first.state_dict().items()}
    deterministic_seed(42)
    second = DualBranchCNNBiLSTM(4, config)
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in first_state.items())

