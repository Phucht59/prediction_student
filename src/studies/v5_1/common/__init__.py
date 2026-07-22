"""Shared V5.1 study components."""

from .uci_data import PRIMARY_CONTEXT_FEATURES, UCIDataV51, build_temporal_features, load_uci_v5_1
from .uci_model import UCIHybridV51, count_parameters, gate_statistics
from .uci_training import UCIInputsV51, fit_uci_model_v5_1, prepare_partition
from .uci_transfer import SharedTrunkSubjectHeadsV51, pretrain_then_finetune

__all__ = [
    "PRIMARY_CONTEXT_FEATURES",
    "UCIDataV51",
    "UCIHybridV51",
    "UCIInputsV51",
    "SharedTrunkSubjectHeadsV51",
    "build_temporal_features",
    "count_parameters",
    "gate_statistics",
    "fit_uci_model_v5_1",
    "load_uci_v5_1",
    "prepare_partition",
    "pretrain_then_finetune",
]
