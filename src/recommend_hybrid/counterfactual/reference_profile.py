"""Leakage-safe OULAD action-reference profiles from training-fold behavior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from src.recommend_hybrid.exceptions import ContractValidationError

from .oulad_tensor import BASE_CHANNELS

REFERENCE_SCOPE = "TRAINING_FOLD_COURSE_STAGE_ONLY"
TRAIN_SAMPLE_ROLE = "TRAIN"
REFERENCE_SPECS = (
    ("total_clicks_p50", "total_clicks", 0.50),
    ("total_clicks_p65", "total_clicks", 0.65),
    ("active_days_p50", "active_days", 0.50),
    ("content_clicks_p50", "content_clicks", 0.50),
    ("content_clicks_p65", "content_clicks", 0.65),
    ("unique_sites_p50", "unique_sites", 0.50),
    ("quiz_clicks_p50", "quiz_clicks", 0.50),
    ("quiz_clicks_p65", "quiz_clicks", 0.65),
    (
        "assessment_related_clicks_p50",
        "assessment_related_clicks",
        0.50,
    ),
)


@dataclass(frozen=True)
class ReferenceStatistic:
    reference_key: str
    channel_name: str
    quantile: float
    value: float
    observation_count: int
    positive_only: bool
    fallback_used: bool

    def __post_init__(self) -> None:
        if not self.reference_key or self.channel_name not in BASE_CHANNELS:
            raise ContractValidationError("invalid reference statistic identity")
        if not 0.0 < self.quantile < 1.0:
            raise ContractValidationError("reference quantile must be in (0, 1)")
        if not isfinite(self.value) or self.value < 0.0:
            raise ContractValidationError(
                "reference statistic must be finite and non-negative"
            )
        if self.observation_count <= 0:
            raise ContractValidationError(
                "reference statistic requires observations"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_key": self.reference_key,
            "channel_name": self.channel_name,
            "quantile": self.quantile,
            "value": self.value,
            "observation_count": self.observation_count,
            "positive_only": self.positive_only,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class OULADReferenceProfile:
    version: str
    fold: int
    stage: str
    course_key: str
    sample_role: str
    reference_scope: str
    student_count: int
    observed_week_count: int
    statistics: tuple[ReferenceStatistic, ...]

    def __post_init__(self) -> None:
        if not self.version or self.fold < 0 or not self.stage or not self.course_key:
            raise ContractValidationError("invalid reference profile identity")
        if self.sample_role != TRAIN_SAMPLE_ROLE:
            raise ContractValidationError(
                "reference profile must be built from TRAIN samples"
            )
        if self.reference_scope != REFERENCE_SCOPE:
            raise ContractValidationError("invalid reference profile scope")
        if self.student_count <= 0 or self.observed_week_count <= 0:
            raise ContractValidationError(
                "reference profile requires students and observed weeks"
            )
        keys = [item.reference_key for item in self.statistics]
        expected = [item[0] for item in REFERENCE_SPECS]
        if keys != expected:
            raise ContractValidationError(
                "reference profile statistics are incomplete or unordered"
            )

    @property
    def profile_id(self) -> str:
        payload = json.dumps(
            self.to_dict(include_profile_id=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return "oulad_ref_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def values(self) -> dict[str, float]:
        return {item.reference_key: item.value for item in self.statistics}

    def to_dict(self, *, include_profile_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "fold": self.fold,
            "stage": self.stage,
            "course_key": self.course_key,
            "sample_role": self.sample_role,
            "reference_scope": self.reference_scope,
            "student_count": self.student_count,
            "observed_week_count": self.observed_week_count,
            "statistics": [item.to_dict() for item in self.statistics],
            "values": self.values(),
        }
        if include_profile_id:
            payload["profile_id"] = self.profile_id
        return payload


class OULADReferenceProfileBuilder:
    """Build action targets from observed training-fold weeks only.

    Zero-activity weeks are excluded when enough positive observations exist,
    because intervention targets describe an attainable active week rather than
    the prevalence of inactivity. Small groups fall back to all non-negative
    observed weeks and record that fallback in the profile.
    """

    def __init__(
        self,
        *,
        version: str = "oulad_counterfactual_reference_v1",
        minimum_positive_observations: int = 20,
    ) -> None:
        if not version or minimum_positive_observations <= 0:
            raise ContractValidationError("invalid reference builder config")
        self.version = version
        self.minimum_positive_observations = minimum_positive_observations
        self._channel_index = {
            name: index for index, name in enumerate(BASE_CHANNELS)
        }

    def build(
        self,
        *,
        sequence: np.ndarray,
        lengths: np.ndarray,
        fold: int,
        stage: str,
        course_key: str,
        sample_role: str = TRAIN_SAMPLE_ROLE,
    ) -> OULADReferenceProfile:
        if sample_role.upper() != TRAIN_SAMPLE_ROLE:
            raise ContractValidationError(
                "counterfactual references cannot use validation/test samples"
            )
        array = np.asarray(sequence)
        observed_lengths = np.asarray(lengths, dtype=int)
        self._validate_inputs(array, observed_lengths)
        mask = np.arange(array.shape[1])[None, :] < observed_lengths[:, None]
        statistics: list[ReferenceStatistic] = []

        for key, channel_name, quantile in REFERENCE_SPECS:
            channel_index = self._channel_index[channel_name]
            values = array[:, :, channel_index][mask].astype(np.float64)
            values = values[np.isfinite(values) & (values >= 0.0)]
            if values.size == 0:
                raise ContractValidationError(
                    f"no valid training observations for {channel_name}"
                )
            positive = values[values > 0.0]
            use_positive = positive.size >= self.minimum_positive_observations
            selected = positive if use_positive else values
            statistic = ReferenceStatistic(
                reference_key=key,
                channel_name=channel_name,
                quantile=float(quantile),
                value=float(np.quantile(selected, quantile, method="linear")),
                observation_count=int(selected.size),
                positive_only=use_positive,
                fallback_used=not use_positive,
            )
            statistics.append(statistic)

        return OULADReferenceProfile(
            version=self.version,
            fold=int(fold),
            stage=str(stage),
            course_key=str(course_key),
            sample_role=TRAIN_SAMPLE_ROLE,
            reference_scope=REFERENCE_SCOPE,
            student_count=int(array.shape[0]),
            observed_week_count=int(mask.sum()),
            statistics=tuple(statistics),
        )

    @staticmethod
    def _validate_inputs(sequence: np.ndarray, lengths: np.ndarray) -> None:
        if sequence.ndim != 3 or sequence.shape[0] == 0:
            raise ContractValidationError(
                "reference sequence must be non-empty [N,T,C]"
            )
        if sequence.shape[2] < len(BASE_CHANNELS):
            raise ContractValidationError(
                "reference sequence lacks OULAD base channels"
            )
        if lengths.shape != (sequence.shape[0],):
            raise ContractValidationError(
                "reference lengths do not align with sequence rows"
            )
        if np.any(lengths <= 0) or np.any(lengths > sequence.shape[1]):
            raise ContractValidationError("invalid observed lengths")


__all__ = [
    "OULADReferenceProfile",
    "OULADReferenceProfileBuilder",
    "REFERENCE_SCOPE",
    "REFERENCE_SPECS",
    "ReferenceStatistic",
    "TRAIN_SAMPLE_ROLE",
]
