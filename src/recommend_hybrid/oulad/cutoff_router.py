"""Past-only routing from arbitrary OULAD cutoff to validated prediction anchor."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from src.recommend_hybrid.common.policy_contracts import PredictionAnchor, RoutingStatus


def route_oulad_cutoff(
    requested_cutoff: float,
    *,
    checkpoint_lineage: tuple[str, ...],
    config: Mapping[str, Any],
) -> PredictionAnchor:
    domain = config["cutoff_domain"]
    if (
        not isfinite(requested_cutoff)
        or requested_cutoff < float(domain["minimum"])
        or requested_cutoff > float(domain["maximum"])
    ):
        raise ValueError("OULAD requested cutoff is outside the 0..100 percent domain")
    anchors = sorted(config["validated_anchors"], key=lambda item: item["cutoff_percent"])
    past = [item for item in anchors if float(item["cutoff_percent"]) <= requested_cutoff]
    if not past:
        return PredictionAnchor(
            requested_cutoff=requested_cutoff,
            anchor_stage=None,
            anchor_cutoff=None,
            prediction_age=None,
            routing_status=RoutingStatus.NO_VALIDATED_PREDICTION_ANCHOR,
            checkpoint_lineage=(),
        )
    selected = past[-1]
    anchor_cutoff = float(selected["cutoff_percent"])
    status = (
        RoutingStatus.EVALUATION_ONLY
        if not selected["intervention"]
        else RoutingStatus.ROUTED
    )
    return PredictionAnchor(
        requested_cutoff=requested_cutoff,
        anchor_stage=selected["stage"],
        anchor_cutoff=anchor_cutoff,
        prediction_age=requested_cutoff - anchor_cutoff,
        routing_status=status,
        checkpoint_lineage=checkpoint_lineage,
    )


__all__ = ["route_oulad_cutoff"]
