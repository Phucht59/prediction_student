"""UCI MAT/POR severity evaluation from separate declared configs."""

from __future__ import annotations

from typing import Any, Mapping

from src.recommend_hybrid.common.evidence import apply_severity_rules
from src.recommend_hybrid.common.policy_contracts import EvidenceItem


def evaluate_uci_severity(
    evidence: tuple[EvidenceItem, ...], config: Mapping[str, Any]
) -> tuple[EvidenceItem, ...]:
    return apply_severity_rules(evidence, config["severity_rules"])


__all__ = ["evaluate_uci_severity"]
