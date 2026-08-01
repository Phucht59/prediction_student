"""Phase 1 contracts for evidence-grounded programmatic weak supervision."""

from .contracts import CandidateActionExample, PredictionContext, SourceRecord
from .labels import LF_ABSTAIN, RelevanceGrade, TargetLabel

__all__ = [
    "CandidateActionExample",
    "LF_ABSTAIN",
    "PredictionContext",
    "RelevanceGrade",
    "SourceRecord",
    "TargetLabel",
]
