"""Route UCI requests only from actual G1/G2 availability."""

from __future__ import annotations

from math import isfinite

from src.recommend_hybrid.common.policy_contracts import PredictionAnchor, RoutingStatus

STAGE_CUTOFF = {"S0": 0.0, "S1": 1.0, "S2": 2.0}


def _valid_grade(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or not 0 <= value <= 20):
        raise ValueError(f"{name} must be in the UCI 0..20 domain")


def route_uci_stage(
    *,
    g1: float | None,
    g2: float | None,
    checkpoint_lineage: tuple[str, ...],
    requested_cutoff: float | None = None,
    stage_evidence_known: bool = True,
) -> PredictionAnchor:
    _valid_grade(g1, "G1")
    _valid_grade(g2, "G2")
    if not stage_evidence_known or g2 is not None and g1 is None:
        cutoff = float(requested_cutoff or 0.0)
        return PredictionAnchor(
            cutoff,
            None,
            None,
            None,
            RoutingStatus.INSUFFICIENT_STAGE_EVIDENCE,
            checkpoint_lineage,
        )
    if requested_cutoff is not None and g1 is None and g2 is None:
        return PredictionAnchor(
            float(requested_cutoff),
            None,
            None,
            None,
            RoutingStatus.INSUFFICIENT_STAGE_EVIDENCE,
            checkpoint_lineage,
        )
    stage = "S2" if g2 is not None else "S1" if g1 is not None else "S0"
    anchor_cutoff = STAGE_CUTOFF[stage]
    requested = anchor_cutoff if requested_cutoff is None else float(requested_cutoff)
    if requested < anchor_cutoff:
        raise ValueError("requested UCI cutoff precedes available assessment evidence")
    return PredictionAnchor(
        requested_cutoff=requested,
        anchor_stage=stage,
        anchor_cutoff=anchor_cutoff,
        prediction_age=None,
        routing_status=RoutingStatus.ROUTED,
        checkpoint_lineage=checkpoint_lineage,
    )


__all__ = ["STAGE_CUTOFF", "route_uci_stage"]
