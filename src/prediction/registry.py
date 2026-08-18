"""Active prediction registry; no dataset-specific model identities."""

from .baselines import ACTIVE_BASELINES


ACTIVE_PREDICTION_REGISTRY = {
    "prediction_model": {
        "model_id": "hybrid",
        "display_name": "Hybrid",
        "task": "binary_student_risk",
        "datasets": ["uci_combined", "oulad"],
    },
    "fitted_instances": ["uci", "oulad_early", "oulad_final"],
    "architecture": "static + aggregate + temporal CNN + BiLSTM + F3 adaptive entropy fusion + one binary logit",
    "joint_training": False,
    "baselines": list(ACTIVE_BASELINES),
}

__all__ = ["ACTIVE_PREDICTION_REGISTRY", "ACTIVE_BASELINES"]
