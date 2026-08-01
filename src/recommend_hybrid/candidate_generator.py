"""Eligibility-only candidate generation; no scoring, ranking or plan selection."""

from __future__ import annotations

from .action_catalog import ActionCatalog
from .contracts import (
    CandidateEvaluation,
    CandidateStatus,
    ObservedLearningState,
    PredictionContext,
    Stage,
)


class HybridCandidateGenerator:
    def __init__(self, catalog: ActionCatalog) -> None:
        self.catalog = catalog

    def generate(
        self,
        prediction: PredictionContext,
        observed: ObservedLearningState,
        *,
        completed_actions: frozenset[str] = frozenset(),
        contraindications: frozenset[str] = frozenset(),
    ) -> tuple[CandidateEvaluation, ...]:
        if prediction.stage != observed.stage or prediction.cutoff_day != observed.cutoff_day:
            raise ValueError("prediction and observed-state lineage do not align")
        evaluations: list[CandidateEvaluation] = []
        for action in self.catalog.actions:
            if not action.active or prediction.stage not in action.applicable_stages:
                status = CandidateStatus.INELIGIBLE_STAGE
                reasons = ("STAGE_NOT_APPLICABLE",)
            else:
                absent = sorted(set(action.required_evidence) - set(observed.available_evidence))
                unmet = sorted(set(action.prerequisites) - set(completed_actions))
                blocked = sorted(set(action.contraindications) & set(contraindications))
                if absent:
                    status = CandidateStatus.MISSING_REQUIRED_EVIDENCE
                    reasons = tuple(f"MISSING_EVIDENCE_{name.upper()}" for name in absent)
                elif unmet:
                    status = CandidateStatus.PREREQUISITE_NOT_MET
                    reasons = tuple(f"PREREQUISITE_{name}_NOT_MET" for name in unmet)
                elif blocked:
                    status = CandidateStatus.CONTRAINDICATED
                    reasons = tuple(f"CONTRAINDICATION_{name}" for name in blocked)
                elif action.requires_human_review:
                    status = CandidateStatus.REQUIRES_HUMAN_REVIEW
                    reasons = ("ELIGIBLE_WITH_MANDATORY_HUMAN_REVIEW",)
                else:
                    status = CandidateStatus.ELIGIBLE
                    reasons = ("STAGE_AND_EVIDENCE_ELIGIBLE",)
            evaluations.append(CandidateEvaluation(action, status, reasons))
        return tuple(evaluations)

    @staticmethod
    def eligible(
        evaluations: tuple[CandidateEvaluation, ...],
    ) -> tuple[CandidateEvaluation, ...]:
        allowed = {CandidateStatus.ELIGIBLE, CandidateStatus.REQUIRES_HUMAN_REVIEW}
        return tuple(item for item in evaluations if item.status in allowed)


__all__ = ["HybridCandidateGenerator"]
