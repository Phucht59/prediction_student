"""Validation gates for action feasibility artifacts."""

from __future__ import annotations

import pandas as pd

from .rules import ACTION_IDS, RULE_VERSION, evaluate_feasibility


VALID_STATUSES = {"FEASIBLE", "INFEASIBLE", "UNKNOWN"}


def validate_feasibility(feasibility: pd.DataFrame, state: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {"case_id", "stage", "action_id", "feasibility_status", "reason_code", "rule_version", "source_feature"}
    errors.extend(f"missing:{column}" for column in sorted(required.difference(feasibility.columns)))
    if errors:
        return errors
    if feasibility.duplicated(["case_id", "action_id"]).any(): errors.append("duplicate_case_action")
    if not (feasibility.groupby("case_id").size() == len(ACTION_IDS)).all(): errors.append("not_five_actions_per_case")
    if not feasibility.action_id.isin(ACTION_IDS).all(): errors.append("invalid_action_id")
    if not feasibility.feasibility_status.isin(VALID_STATUSES).all(): errors.append("invalid_status")
    if not (feasibility.rule_version == RULE_VERSION).all(): errors.append("invalid_rule_version")
    if set(feasibility.case_id) != set(state.case_id): errors.append("state_case_coverage_mismatch")
    if "FINAL-100" in set(feasibility.stage.astype(str)): errors.append("final_stage_present")
    forbidden = {"target", "final_result", "score", "date_unregistration"}.intersection(feasibility.columns)
    if forbidden: errors.append(f"forbidden_columns:{','.join(sorted(forbidden))}")
    state_by_case = state.set_index("case_id")
    for row in feasibility.itertuples(index=False):
        expected = evaluate_feasibility(state_by_case.loc[row.case_id].to_dict(), row.action_id)
        actual = (row.feasibility_status, row.reason_code, row.source_feature)
        if actual != expected: errors.append(f"nondeterministic_rule:{row.case_id}:{row.action_id}")
    return errors
