"""End-to-end stage-aware target-trial evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.recommend_hybrid.final.actions import canonical_action_id

from .aipw import AIPWConfig, AIPWResult, CrossFittedAIPW
from .bootstrap import ClusterBootstrapResult, cluster_bootstrap_mean
from .diagnostics import (
    IdentifiabilityReport,
    IdentifiabilityThresholds,
    assess_identifiability,
)
from .protocol import TargetTrialProtocol, validate_temporal_columns


@dataclass(frozen=True)
class StageActionTrialData:
    stage: str
    action_id: str
    features: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    groups: np.ndarray
    student_ids: np.ndarray
    record_ids: np.ndarray
    maximum_baseline_progress: float
    minimum_treatment_progress: float
    maximum_treatment_progress: float

    def validate(self) -> None:
        protocol = TargetTrialProtocol(stage=self.stage, action_id=self.action_id)
        canonical = canonical_action_id(self.action_id)
        if canonical != self.action_id:
            raise ValueError("trial action_id must be canonical")
        x = np.asarray(self.features, dtype=np.float64)
        t = np.asarray(self.treatment).reshape(-1)
        y = np.asarray(self.outcome).reshape(-1)
        groups = np.asarray(self.groups).reshape(-1)
        students = np.asarray(self.student_ids).reshape(-1)
        records = np.asarray(self.record_ids).reshape(-1)
        row_count = len(t)
        if x.ndim != 2 or len(x) != row_count:
            raise ValueError("features must be [N, F] and align with treatment")
        if any(len(values) != row_count for values in (y, groups, students, records)):
            raise ValueError("trial arrays must align row by row")
        if not np.isfinite(x).all():
            raise ValueError("trial features must be finite")
        if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("treatment and outcome must be binary")
        if not np.array_equal(groups.astype(str), students.astype(str)):
            raise ValueError(
                "groups must equal student_ids so repeated courses and landmarks stay together"
            )
        if len(np.unique(records.astype(str))) != row_count:
            raise ValueError("record_ids must be unique within one stage-action trial")
        validate_temporal_columns(
            stage=protocol.stage,
            maximum_baseline_progress=self.maximum_baseline_progress,
            minimum_treatment_progress=self.minimum_treatment_progress,
            maximum_treatment_progress=self.maximum_treatment_progress,
        )


@dataclass(frozen=True)
class StageActionEvaluation:
    protocol: TargetTrialProtocol
    effect: AIPWResult
    identifiability: IdentifiabilityReport
    bootstrap: ClusterBootstrapResult | None
    retained_mask: np.ndarray
    weights: np.ndarray

    @property
    def status(self) -> str:
        return (
            "CAUSAL_EFFECT_ESTIMATED"
            if self.identifiability.identifiable and self.bootstrap is not None
            else "CAUSAL_EVIDENCE_NOT_IDENTIFIABLE"
        )

    def summary(self) -> dict[str, object]:
        effect_summary = self.effect.summary()
        if self.bootstrap is not None:
            effect_summary.update(
                {
                    "ate": self.bootstrap.estimate,
                    "standard_error": self.bootstrap.bootstrap_standard_error,
                    "confidence_interval": list(self.bootstrap.confidence_interval),
                    "uncertainty_method": "STUDENT_CLUSTER_PERCENTILE_BOOTSTRAP",
                    "bootstrap_iterations": self.bootstrap.iterations,
                }
            )
        else:
            effect_summary["uncertainty_method"] = "NOT_AVAILABLE"
        return {
            "status": self.status,
            "protocol": self.protocol.to_dict(),
            "effect": effect_summary,
            "identifiability": self.identifiability.to_dict(),
            "claim_boundary": (
                "OBSERVATIONAL_CAUSAL_ESTIMATE_UNDER_STATED_ASSUMPTIONS"
                if self.status == "CAUSAL_EFFECT_ESTIMATED"
                else "NO_CAUSAL_EFFECT_CLAIM"
            ),
        }

    def individual_effect_records(
        self,
        student_ids: np.ndarray,
        record_ids: np.ndarray,
    ) -> list[dict[str, Any]]:
        students = np.asarray(student_ids).reshape(-1)
        records = np.asarray(record_ids).reshape(-1)
        if len(students) != len(self.effect.cate) or len(records) != len(self.effect.cate):
            raise ValueError("student_ids and record_ids must align with estimated effects")
        interval = (
            self.bootstrap.confidence_interval if self.bootstrap is not None else (None, None)
        )
        rows: list[dict[str, Any]] = []
        for index, (student_id, record_id) in enumerate(zip(students, records, strict=True)):
            rows.append(
                {
                    "record_id": str(record_id),
                    "student_id": str(student_id),
                    "stage": self.protocol.stage,
                    "action_id": self.protocol.action_id,
                    "cate": float(self.effect.cate[index]),
                    "propensity": float(self.effect.propensity[index]),
                    "outcome_if_control": float(self.effect.outcome_if_control[index]),
                    "outcome_if_treated": float(self.effect.outcome_if_treated[index]),
                    "retained_in_overlap": bool(self.retained_mask[index]),
                    "cross_fit_fold": int(self.effect.fold_id[index]),
                    "stage_action_ate_ci_low": interval[0],
                    "stage_action_ate_ci_high": interval[1],
                    "causal_evidence_identifiable": self.identifiability.identifiable,
                }
            )
        return rows


class StageAwareCausalEvaluator:
    """Evaluate one action at the same landmark that produced its ranking."""

    def __init__(
        self,
        *,
        aipw_config: AIPWConfig = AIPWConfig(),
        identifiability_thresholds: IdentifiabilityThresholds = IdentifiabilityThresholds(),
        bootstrap_iterations: int = 1000,
        random_state: int = 20260806,
    ) -> None:
        self.estimator = CrossFittedAIPW(config=aipw_config)
        self.identifiability_thresholds = identifiability_thresholds
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.random_state = int(random_state)

    def evaluate(self, trial: StageActionTrialData) -> StageActionEvaluation:
        trial.validate()
        protocol = TargetTrialProtocol(stage=trial.stage, action_id=trial.action_id)
        effect = self.estimator.fit_predict(
            trial.features,
            trial.treatment,
            trial.outcome,
            groups=trial.groups,
        )
        report, weights, retained = assess_identifiability(
            features=trial.features,
            treatment=trial.treatment,
            propensity=effect.propensity,
            thresholds=self.identifiability_thresholds,
        )
        bootstrap: ClusterBootstrapResult | None = None
        if report.identifiable:
            bootstrap = cluster_bootstrap_mean(
                effect.doubly_robust_score[retained],
                np.asarray(trial.groups)[retained],
                iterations=self.bootstrap_iterations,
                confidence_level=self.estimator.config.confidence_level,
                random_state=self.random_state,
            )
        return StageActionEvaluation(
            protocol=protocol,
            effect=effect,
            identifiability=report,
            bootstrap=bootstrap,
            retained_mask=retained,
            weights=weights,
        )


__all__ = [
    "StageActionEvaluation",
    "StageActionTrialData",
    "StageAwareCausalEvaluator",
]
