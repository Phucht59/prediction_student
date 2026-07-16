from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "oulad" / "final"
V3 = ROOT / "artifacts" / "oulad" / "temporal"
PROTOCOL = ROOT / "configs/oulad_v3_fair_db_closure_protocol.yaml"
DECLARED_SEEDS = {42, 2026, 3407}
ENSEMBLE_MAP = {
    "V3-A0F-ENS": "V3-A0F", "V3-H2TF-ENS": "V3-H2TF", "V3-H3CF-ENS": "V3-H3CF",
    "V3-P0-ENS": "V3-P0", "V3-D0-ENS": "V3-D0", "V3-A1-ENS": "V3-A1",
}


def load_json(name: str):
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_fair_ensemble_registry_uses_all_and_only_declared_seeds():
    single = pd.read_csv(ARTIFACT / "single_seed_metrics.csv")
    for source in ENSEMBLE_MAP.values():
        assert set(single.loc[single.candidate_id == source, "seed"]) == DECLARED_SEEDS
    coverage = pd.read_csv(ARTIFACT / "ensemble_prediction_coverage.csv")
    assert set(coverage.candidate_id) == set(ENSEMBLE_MAP)
    assert (coverage.record_alignment == "PASS").all()
    assert (coverage.label_alignment == "PASS").all()
    assert (coverage.declared_seed_count == 3).all()
    assert (coverage.seed_rows == coverage.inner_records * 3).all()


def test_probability_ensemble_is_record_aligned_arithmetic_mean():
    source = pd.read_parquet(V3 / "oof_predictions.parquet")
    observed = pd.read_parquet(ARTIFACT / "ensemble_oof_predictions.parquet")
    for ensemble_id, source_id in ENSEMBLE_MAP.items():
        selected = source.loc[source.candidate_id == source_id]
        assert set(selected.seed.astype(int)) == DECLARED_SEEDS
        assert not selected.duplicated(["record_id", "seed"]).any()
        expected = selected.groupby("record_id", as_index=False).probability.mean().sort_values("record_id")
        actual = observed.loc[observed.candidate_id == ensemble_id, ["record_id", "probability"]].sort_values("record_id")
        assert expected.record_id.tolist() == actual.record_id.tolist()
        assert np.allclose(expected.probability, actual.probability, atol=1e-15, rtol=0)


def test_prediction_contracts_are_permanently_separated():
    registry = load_json("prediction_contract_registry.json")
    mean_metrics = pd.read_csv(ARTIFACT / "mean_seed_metrics.csv")
    ensemble_metrics = pd.read_csv(ARTIFACT / "ensemble_metrics.csv")
    assert set(mean_metrics.prediction_contract) == {"mean_of_seed_metrics"}
    assert "probability_ensemble" in set(ensemble_metrics.prediction_contract)
    assert "never an ensemble" in registry["mean_of_seed_metrics"]
    assert "arithmetic mean of probabilities" in registry["probability_ensemble"]


def test_thresholds_are_fold_specific_inner_oof_only():
    thresholds = pd.read_csv(ARTIFACT / "ensemble_thresholds.csv")
    assert set(thresholds.prediction_contract) == {"pooled_inner_oof_three_seed_probability_ensemble"}
    assert thresholds.groupby("candidate_id").outer_fold.nunique().eq(3).all()
    audit = load_json("fairness_audit.json")
    assert audit["inner_threshold_coverage"] == "PASS"
    assert audit["outer_labels_used_for_threshold"] is False
    assert audit["future_access"] is False
    assert audit["replay_label"] == "threshold-reconstruction replay"


def test_fair_bootstrap_is_paired_by_student_and_never_mixes_contracts():
    rows = pd.read_csv(ARTIFACT / "grouped_bootstrap_fair.csv")
    required = {
        ("V3-D0-ENS", "V3-A0F-ENS"), ("V3-D0-ENS", "V3-P0-ENS"),
        ("V3-D0-ENS", "V3-H3CF-ENS"), ("V3-D0-ENS", "V3-A1-ENS"),
        ("V3-D0-ENS", "V3-MLD"), ("V3-D0-ENS", "V3-MLF"),
    }
    assert required <= set(zip(rows.left_candidate, rows.right_candidate))
    assert (rows.resamples == 5000).all()
    assert (rows.students == 14687).all()
    assert (rows.records == 15378).all()
    assert set(rows.prediction_contract) == {"fair_probability_ensemble_or_registered_deterministic"}
    assert not rows.left_candidate.str.fullmatch(r"V3-(A0F|H2TF|H3CF|P0|D0|A1)").any()


def test_old_mixed_contract_results_are_superseded_and_not_used_for_verdict():
    superseded = load_json("superseded_v3_comparisons.json")
    verdict = load_json("verdict.json")
    assert superseded["status"] == "historical_v3_mixed_contract_result"
    assert verdict["verdict"] == "PRACTICAL_TIE"
    assert verdict["strongest_fair_comparator"] == "V3-A0F-ENS"
    assert abs(verdict["delta"] - (verdict["d0_ensemble_macro_f1"] - verdict["comparator_macro_f1"])) < 1e-15


def test_v3_frozen_comparator_checksums_match_protocol():
    checksums = load_json("v3_artifact_checksums.json")
    values = {
        "protocol": V3 / "resolved_protocol.yaml",
        "oof": V3 / "oof_predictions.parquet",
        "selected_configs": V3 / "selected_configs.json",
        "metrics": V3 / "metrics_summary.csv",
    }
    for key, path in values.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksums[key]["sha256"]
        assert checksums[key]["status"] == "PASS"


def test_database_reproduction_and_permission_artifacts_pass():
    reproduction = load_json("postgres_reproduction_validation.json")
    permission = load_json("postgres_permission_audit.json")
    registration = load_json("postgres_evidence_registration.json")
    assert reproduction["status"] == "PASS"
    assert reproduction["artifact_rows"] == reproduction["database_rows"] == 123024
    assert reproduction["max_probability_absolute_difference"] <= 1e-12
    assert reproduction["max_metric_absolute_difference"] <= 1e-12
    assert permission["status"] == "PASS"
    assert permission["application_profile"]["current_user"] == "student_predict_app_local"
    assert all(item["status"] == "PASS" for item in permission["tests"])
    assert registration["prediction_rows"] == 123024
    assert registration["completed_runs_registered"] == 8


def test_migrations_are_transactional_non_cascade_and_set_based():
    migration_006 = (ROOT / "database/migrations/006_oulad_v3_fair_evidence_registry.sql").read_text(encoding="utf-8")
    migration_007 = (ROOT / "database/migrations/007_optimize_bulk_lineage_integrity_triggers.sql").read_text(encoding="utf-8")
    for text in (migration_006, migration_007):
        assert re.search(r"\bBEGIN\s*;", text, re.I)
        assert re.search(r"\bCOMMIT\s*;", text, re.I)
        assert "pg_advisory_xact_lock" in text
        assert "CASCADE" not in text.upper()
    assert "REFERENCING NEW TABLE" in migration_007
    assert "FOR EACH STATEMENT" in migration_007
    assert "FOR EACH ROW EXECUTE FUNCTION require_running_run_by_run_id" not in migration_007


def test_no_credential_is_present_in_closure_artifacts_or_source_diff():
    forbidden_dsn = re.compile(r"postgresql://(?!<redacted>)[^\s/@:]+:[^\s/@]+@", re.I)
    paths = [path for path in ARTIFACT.rglob("*") if path.is_file() and path.suffix.lower() not in {".parquet", ".png"}]
    paths += [PROTOCOL, ROOT / "scripts/database_register_evidence.py"]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not forbidden_dsn.search(text), path


@pytest.mark.skipif(not os.getenv("POSTGRES_TEST_DSN") or not os.getenv("POSTGRES_TEST_APP_DSN"), reason="live closure database DSNs unavailable")
def test_live_postgres_closure_uses_real_least_privileged_app_role():
    psycopg2 = pytest.importorskip("psycopg2")
    with psycopg2.connect(os.environ["POSTGRES_TEST_APP_DSN"]) as app, app.cursor() as cursor:
        cursor.execute("SELECT current_user,rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=current_user")
        role, superuser, createdb, createrole = cursor.fetchone()
        assert role == "student_predict_app_local"
        assert not any((superuser, createdb, createrole))
        cursor.execute("SELECT count(*) FROM ml_predictions p JOIN ml_experiment_runs r ON r.run_id=p.run_id WHERE r.model_name LIKE 'V3-%'")
        assert cursor.fetchone()[0] == 123024
    with psycopg2.connect(os.environ["POSTGRES_TEST_DSN"]) as admin, admin.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM ml_schema_migrations WHERE migration_id IN ('005_oulad_lineage_and_snapshot_registry.sql','006_oulad_v3_fair_evidence_registry.sql','007_optimize_bulk_lineage_integrity_triggers.sql')")
        assert cursor.fetchone()[0] == 3
        cursor.execute("SELECT count(*) FROM ml_evidence_bundles WHERE study_id='study_c_oulad'")
        assert cursor.fetchone()[0] == 3
        cursor.execute("SELECT count(*) FROM ml_experiment_runs WHERE model_name LIKE 'V3-%' AND status='completed'")
        assert cursor.fetchone()[0] == 8
