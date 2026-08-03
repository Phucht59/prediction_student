"""Deterministic counterfactual state simulation with immutable-feature guards."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose

from .contracts import (
    CounterfactualScenario,
    FeatureChange,
    SimulationStatus,
    StateValue,
    canonical_state_items,
)
from .effects import CounterfactualEffectCatalog, EffectOperator, FeatureEffect


class CounterfactualStateSimulator:
    def __init__(self, catalog: CounterfactualEffectCatalog) -> None:
        self.catalog = catalog

    def simulate(
        self,
        action_id: str,
        observed_state: Mapping[str, StateValue],
        reference_values: Mapping[str, float],
    ) -> CounterfactualScenario:
        baseline = dict(observed_state)
        baseline_items = canonical_state_items(baseline)
        spec = self.catalog.by_id(action_id)
        if not spec.scorable:
            return CounterfactualScenario(
                action_id=action_id,
                baseline_state=baseline_items,
                simulated_state=baseline_items,
                changes=(),
                status=SimulationStatus.NOT_SCORABLE,
                reason_codes=(str(spec.non_scorable_reason),),
                simulator_version=self.catalog.version,
            )

        simulated = dict(baseline)
        changes: list[FeatureChange] = []
        missing: list[str] = []
        for effect in spec.effects:
            before = baseline.get(effect.feature_name)
            if before is None:
                missing.append(effect.feature_name)
                continue
            if isinstance(before, bool) or not isinstance(before, int | float):
                missing.append(effect.feature_name)
                continue
            after = self._apply(effect, float(before), reference_values)
            if isinstance(before, int):
                after_value: StateValue = int(round(after))
            else:
                after_value = float(after)
            if self._same_numeric(before, after_value):
                continue
            simulated[effect.feature_name] = after_value
            changes.append(
                FeatureChange(
                    feature_name=effect.feature_name,
                    before=before,
                    after=after_value,
                    operation=effect.operator.value,
                    reference_key=effect.reference_key,
                )
            )

        if changes:
            reasons = ["COUNTERFACTUAL_ACTION_APPLIED"]
            reasons.extend(
                f"MISSING_OPTIONAL_FEATURE:{name}" for name in sorted(missing)
            )
            return CounterfactualScenario(
                action_id=action_id,
                baseline_state=baseline_items,
                simulated_state=canonical_state_items(simulated),
                changes=tuple(changes),
                status=SimulationStatus.SIMULATED,
                reason_codes=tuple(reasons),
                simulator_version=self.catalog.version,
            )

        if missing:
            status = SimulationStatus.MISSING_INPUT
            reasons = tuple(
                f"MISSING_ACTIONABLE_FEATURE:{name}" for name in sorted(missing)
            )
        else:
            status = SimulationStatus.NO_CHANGE
            reasons = ("STATE_ALREADY_MEETS_COUNTERFACTUAL_TARGET",)
        return CounterfactualScenario(
            action_id=action_id,
            baseline_state=baseline_items,
            simulated_state=baseline_items,
            changes=(),
            status=status,
            reason_codes=reasons,
            simulator_version=self.catalog.version,
        )

    @staticmethod
    def _same_numeric(before: float | int, after: StateValue) -> bool:
        if after is None:
            return False
        return isclose(float(before), float(after), rel_tol=0.0, abs_tol=1e-12)

    @staticmethod
    def _apply(
        effect: FeatureEffect,
        current: float,
        reference_values: Mapping[str, float],
    ) -> float:
        if effect.operator is EffectOperator.MAX_REFERENCE:
            if effect.reference_key not in reference_values:
                raise KeyError(f"missing reference value: {effect.reference_key}")
            result = max(current, float(reference_values[effect.reference_key]))
        elif effect.operator is EffectOperator.INCREASE_FRACTION_OF_GAP:
            assert effect.amount is not None and effect.upper_bound is not None
            result = current + effect.amount * (effect.upper_bound - current)
        elif effect.operator is EffectOperator.REDUCE_FRACTION:
            assert effect.amount is not None
            result = current * (1.0 - effect.amount)
        elif effect.operator is EffectOperator.ADD:
            assert effect.amount is not None
            result = current + effect.amount
        elif effect.operator is EffectOperator.SUBTRACT:
            assert effect.amount is not None
            result = current - effect.amount
        elif effect.operator is EffectOperator.MAX_VALUE:
            assert effect.amount is not None
            result = max(current, effect.amount)
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported operator: {effect.operator}")

        if effect.lower_bound is not None:
            result = max(result, effect.lower_bound)
        if effect.upper_bound is not None:
            result = min(result, effect.upper_bound)
        return result


__all__ = ["CounterfactualStateSimulator"]
