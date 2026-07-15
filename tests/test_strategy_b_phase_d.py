"""Static and deterministic checks for the authorized Phase D runner."""

from pathlib import Path

import pandas as pd

from scripts.run_strategy_b_phase_d_recommendation import MINIMUM, _casebook, _uncertainty_policy


ROOT = Path(__file__).resolve().parents[1]


def test_phase_d_artifact_contract_contains_every_required_governance_output():
    required = {
        "protocol.json", "phase_e_source_manifest.json", "model_role_contract.json",
        "prediction_snapshot_schema.json", "uncertainty_policy.json", "feature_registry.json",
        "policy_registry.json", "action_catalog.json", "recommendation_case_snapshots.csv",
        "recommendation_instances.csv", "recommendation_goals.csv", "recommendation_actions.csv",
        "technical_safety_metrics.json", "expert_casebook.csv", "expert_validation_status.json",
        "database_migration_report.json", "test_report.json", "artifact_checksums.json",
        "strict_validation.json", "phase_d_conclusion.md",
    }
    assert required <= set(MINIMUM)


def test_uncertainty_policy_uses_only_n0_scores_and_casebook_is_deterministic_stratified():
    oof = pd.DataFrame([
        {"candidate_id": "N0", "source_row_number": row, "prob_0": .6, "prob_1": .3, "prob_2": .1}
        for row in range(1, 13) for _ in range(5)
    ])
    policy = _uncertainty_policy(oof)
    instances = pd.DataFrame([
        {"recommendation_instance_id": f"id-{row:02d}", "predicted_class_name": ["Low", "Medium", "High"][row % 3], "n0_r0_agreement": row % 2 == 0, "predictive_entropy": .9 + .02 * (row % 4), "trajectory": [-1, 0, 1][row % 3]}
        for row in range(48)
    ])
    first = _casebook(instances, policy)
    second = _casebook(instances, policy)
    assert first.equals(second)
    assert len(first) == 60
    assert set(first["evidence_availability"]) == {"sufficient", "insufficient"}
    assert policy["source"] == "phase_e_development_oof_only"


def test_phase_d_has_no_legacy_model_training_or_phase_e_write_path():
    runner = (ROOT / "scripts" / "run_strategy_b_phase_d_recommendation.py").read_text(encoding="utf-8")
    loader = (ROOT / "src" / "postgres_data_source.py").read_text(encoding="utf-8")
    migration = (ROOT / "database" / "migrations" / "004_governed_recommendation_phase_d.sql").read_text(encoding="utf-8")
    assert "fit_final_development_estimator" not in runner
    assert "legacy_heldout_observed" not in runner
    feature_loader = loader.split("def load_development_feature_subset_from_postgres", 1)[1].split("\ndef load_dataset_version", 1)[0]
    assert "source_record_targets" not in feature_loader
    for table in ("recommendation_policies", "prediction_snapshots", "recommendation_revisions", "advisor_decisions", "recommendation_follow_ups", "expert_review_ratings"):
        assert table in migration
    assert "BEFORE UPDATE OR DELETE" in migration
