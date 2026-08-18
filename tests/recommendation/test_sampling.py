from __future__ import annotations

import pandas as pd

from src.recommendation.sampling import sample_panel, validate_panels


def _state() -> pd.DataFrame:
    rows = []
    for i in range(120):
        rows.append({
            "case_id": f"c{i}", "student_id": f"s{i}", "enrollment_identity": f"e{i}",
            "stage": ("20pct", "35pct", "50pct", "75pct")[i % 4], "outer_fold": i % 3,
            "risk_band": ("low", "medium", "high")[i % 3], "risk_probability": (i % 100) / 100,
        })
    return pd.DataFrame(rows)


def test_sampling_is_deterministic_and_disjoint():
    state = _state()
    a = sample_panel(state, panel="A", target_size=30, seed=2026)
    b = sample_panel(state, panel="B", target_size=20, seed=2026, excluded_students=set(a.student_id))
    assert a.case_id.tolist() == sample_panel(state, panel="A", target_size=30, seed=2026).case_id.tolist()
    assert validate_panels(a, b, state) == []


def test_sampling_does_not_use_final_stage():
    state = _state()
    state.loc[0, "stage"] = "FINAL-100"
    a = sample_panel(state, panel="A", target_size=30, seed=2026)
    assert "FINAL-100" not in set(a.stage)
