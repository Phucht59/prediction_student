"""Active prediction registry. One Hybrid, two datasets, information states only."""

from .baselines import ACTIVE_BASELINES


ACTIVE_PREDICTION_REGISTRY = {
    "prediction_model": {
        "model_id": "hybrid",
        "display_name": "Hybrid",
        "public_class": "Hybrid",
        "architecture_id": "C0",
        "authority_source": "Phase4",
        "task": "binary_student_risk",
        "datasets": ["uci_combined", "oulad"],
    },
    "fitted_instances": ["uci", "oulad"],
    "uci_states": ["S0", "S1", "S2"],
    "oulad_states": ["20pct", "35pct", "50pct", "75pct", "100pct"],
    "architecture": "C0 parallel CNN ∥ BiLSTM + corrected availability + 3-way masked softmax + binary logit",
    "joint_training": False,
    "stage_specific_models": False,
    "separate_oulad_100_model": False,
    "baselines": list(ACTIVE_BASELINES),
    "xgboost_active": False,
    "evaluation_status": "robust_inner_finalized",
    "outer_test_used_for_phase4_finalization": False,
}

__all__ = ["ACTIVE_PREDICTION_REGISTRY", "ACTIVE_BASELINES"]
