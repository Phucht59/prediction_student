from pathlib import Path

import pandas as pd

from scripts.run_final_repository_closure import (
    PHASE_AB,
    PHASE_C,
    PHASE_D,
    PHASE_E,
    REQUIRED_OUTPUTS,
    database_validation,
    final_metrics,
    historical_registry,
    markdown_link_report,
    recommendation_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_closure_required_outputs_and_official_sources_exist():
    assert len(REQUIRED_OUTPUTS) >= 24
    assert len(REQUIRED_OUTPUTS) == len(set(REQUIRED_OUTPUTS))
    assert all(path.is_dir() for path in [PHASE_AB, PHASE_C, PHASE_E, PHASE_D])


def test_final_metrics_are_derived_from_phase_e_and_roles_remain_frozen():
    metrics = final_metrics().set_index("model")
    assert list(metrics.index) == ["R0", "M1", "M2", "N0", "N1"]
    assert metrics.loc["R0", "macro_f1"] == pytest_approx(0.8988360425446519)
    assert metrics.loc["M1", "macro_f1"] == pytest_approx(0.8999548661053872)
    assert metrics.loc["N0", "role"] == "final thesis hybrid model"
    assert set(metrics["validation_scope"]) == {"nested development OOF; no external confirmation"}


def pytest_approx(value):
    import pytest
    return pytest.approx(value, abs=1e-12)


def test_recommendation_summary_recomputes_phase_d_structural_counts():
    summary = recommendation_summary()
    assert summary["development_cases"] == 316
    assert summary["eligible_for_normal_draft_gate"] == 245
    assert summary["uncertainty_agreement_review_cases"] == 71
    assert summary["generated_actions"] == 1313
    assert summary["expert_validation"] == "PENDING"
    assert summary["effectiveness_validation"] == "NOT_PERFORMED"


def test_historical_registry_preserves_all_nonheadline_classes():
    categories = {entry["category"] for entry in historical_registry()["entries"]}
    assert {"legacy_observed_evidence", "historical_evidence", "diagnostic_evidence", "invalid_protocol_evidence", "smoke_evidence"} <= categories
    assert all(not entry["headline_eligible"] for entry in historical_registry()["entries"])


def test_database_static_validation_and_document_links_pass_without_execution():
    database = database_validation()
    assert database["database_migration_execution"] == "NOT_PERFORMED"
    assert database["database_migration_static_validation"] == "PASS"
    links = markdown_link_report()
    assert links.empty or links["exists"].all()


def test_closure_runner_contains_no_training_entrypoint_or_observed_fetch():
    source = (ROOT / "scripts" / "run_final_repository_closure.py").read_text(encoding="utf-8")
    assert "fit_final_development_estimator(" not in source
    assert "run_strategy_b_phase_c.py" not in source
    assert "legacy_heldout_observed" in (ROOT / "README.md").read_text(encoding="utf-8")
