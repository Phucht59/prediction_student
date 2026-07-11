"""Human-readable descriptions of the effective training loss.

The historical ``weighted_ce`` label is retained in frozen configuration files
for compatibility; it does not by itself mean that class weights were passed.
"""

from __future__ import annotations


def describe_effective_loss(params: dict) -> str:
    """Describe the criterion actually instantiated from a parameter mapping."""
    if params.get("loss") == "focal" or "focal_gamma" in params:
        return "FocalLoss"
    if params.get("loss") == "weighted_ce" and params.get("class_weight_mode", "balanced") == "balanced":
        return "CrossEntropyLoss with class weighting"
    return "CrossEntropyLoss without class weighting"
