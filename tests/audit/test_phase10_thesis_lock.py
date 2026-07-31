from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "final" / "thesis"
REPORT = ROOT / "reports" / "final" / "thesis"


def test_final_model_authority_is_dual_track() -> None:
    registry = yaml.safe_load((ROOT / "configs/final/final_model_authority.yaml").read_text(encoding="utf-8"))
    assert registry["student_mat"]["macro_f1"] == 0.9014601961315334
    assert registry["student_por"]["macro_f1"] == 0.8622587167738002
    assert registry["oulad"]["legacy_endpoint"]["macro_f1"] == 0.8280835945631038
    assert registry["oulad"]["strict_endpoint"]["macro_f1"] == 0.7984000886272689
    assert registry["oulad"]["early_warning"]["status"] == "FROZEN_VALID_DO_NOT_MODIFY"


def test_stage_tables_are_complete_without_interpolation() -> None:
    uci = pd.read_csv(ART / "uci_stage_metrics.csv")
    oulad = pd.read_csv(ART / "oulad_early_warning_h1.csv")
    comparators = pd.read_csv(ART / "oulad_stage_comparators.csv")
    assert len(uci) == 48
    assert set(oulad.observation_percentage) == {20, 35, 50, 75}
    assert len(oulad) == 4
    assert len(comparators) == 32
    assert set(comparators.protocol_id) == {"unified_stage_aware_oulad_v2", "h1_final_outer_v1"}


def test_endpoint_authorities_are_not_mixed() -> None:
    endpoint = pd.read_csv(ART / "oulad_endpoint_authority.csv")
    h0 = endpoint.loc[endpoint.model_id.eq("cnn_bilstm")].iloc[0]
    h1 = endpoint.loc[endpoint.model_id.eq("h1_tabular_residual")].iloc[0]
    assert h0.authority_status == "LEGACY_ENDPOINT_AUTHORITY_WITH_SCORE_PROXY_CAVEAT"
    assert h1.authority_status == "STRICT_NO_UNVERIFIED_SCORE_ENDPOINT_RESULT"
    assert h0.feature_availability_protocol != h1.feature_availability_protocol


def test_claim_guardrails_in_final_thesis_reports() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in REPORT.glob("*.md"))
    forbidden = [
        "H1 beats MLP overall",
        "H1 beats H0 endpoint",
        "H1 endpoint >= 0.83",
        "H0 is definitely leakage",
    ]
    assert not any(claim in text for claim in forbidden)
    assert "0.828084" in text
    assert "0.798400" in text
    assert "legacy endpoint" in text.lower()
    assert "strict" in text.lower()


def test_checksum_manifest_replays() -> None:
    manifest = json.loads((ART / "thesis_evidence_checksums.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        data = path.read_bytes()
        if path.suffix.lower() in {".csv", ".json", ".md", ".yaml", ".yml"}:
            data = data.replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == expected


def test_phase10_has_zero_compute_mutation() -> None:
    gate = json.loads((ART / "phase10_gate.json").read_text(encoding="utf-8"))
    assert gate["training_runs"] == 0
    assert gate["optuna_trials"] == 0
    assert gate["architecture_searches"] == 0
    assert gate["outer_evaluations"] == 0
    assert gate["threshold_tuning"] == 0
    assert gate["early_warning_modified"] is False
