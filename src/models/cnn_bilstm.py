"""Public model identities for the selected CNN-BiLSTM products.

The wrappers deliberately accept a frozen implementation as their backbone.
They do not construct, train, tune, or alter a historical checkpoint.
"""

from __future__ import annotations

from typing import Any

from src.models._uci import UCICNNBiLSTM


class _FrozenModelFacade:
    model_id: str
    official_name: str

    def __init__(self, backbone: Any) -> None:
        self.backbone = backbone

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        return self.backbone.state_dict()


class CNNBiLSTMStudentMatModel(_FrozenModelFacade):
    model_id = "cnn_bilstm_mat"
    official_name = "CNN-BiLSTM MAT"


class CNNBiLSTMStudentPorModel(_FrozenModelFacade):
    model_id = "cnn_bilstm_por"
    official_name = "CNN-BiLSTM POR"


class CNNBiLSTMOULADModel(_FrozenModelFacade):
    model_id = "cnn_bilstm_oulad"
    official_name = "CNN-BiLSTM OULAD"


__all__ = [
    "UCICNNBiLSTM",
    "CNNBiLSTMStudentMatModel",
    "CNNBiLSTMStudentPorModel",
    "CNNBiLSTMOULADModel",
]
