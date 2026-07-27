"""Observed pre-cutoff state used by the recommendation policy."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservedState:
    record_id: str
    cutoff: str
    evidence: dict[str, float] = field(default_factory=dict)
