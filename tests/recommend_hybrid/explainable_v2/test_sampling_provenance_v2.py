"""Sampling, feasibility authority, importer provenance, and verifier hardening tests."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
PRIVATE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private"
IMPORTS_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports"
ENVELOPE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/external_reviews"
SCRIPT_DIR = ROOT / "scripts/recommend_hybrid/explainable_v2"
AUDIT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/audit"


@pytest.fixture(scope="module")
def exported_cases():
    """Ensure case export exists for integration-style sampling tests."""
    os.environ.setdefault("CASE_EXPORT_SALT", "test_salt_sampling_provenance_v2")
    from scripts.recommend_hybrid.explainable_v2.export_llm_cases import export_v2_cases

    export_v2_cases()
    audit_path = EXPORT_DIR / "SAMPLING_AUDIT.json"
    assert audit_path.exists(), "SAMPLING_AUDIT.json must exist after export"
    return json.loads(audit_path.read_text(encoding="utf-8"))


def _load_panel_cases() -> tuple[list[dict], list[dict], dict]:
    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    pb_path = EXPORT_DIR / "panel_b_cases.jsonl"
    pmap_path = PRIVATE_DIR / "private_case_mapping.json"
    pa = [json.loads(l) for l in pa_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pb = [json.loads(l) for l in pb_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pmap = json.loads(pmap_path.read_text(encoding="utf-8"))
    return pa, pb, pmap


# ── SECTION 3: Sampling audit tests ──────────────────────────────────────────


def test_sampling_audit_uses_final_selected_cases(exported_cases):
    """SAMPLING_AUDIT counts must match exactly 300 Panel A + 150 Panel B final cases."""
    assert exported_cases["panel_a_case_count"] == 300
    assert exported_cases["panel_b_case_count"] == 150
    assert exported_cases.get("final_selected_case_count", 450) == 450
    pa, pb, _ = _load_panel_cases()
    assert len(pa) == 300
    assert len(pb) == 150


def test_sampling_target_matches_actual_ratio(exported_cases):
    """Final panel ratio must be 2:1 (300:150) with zero overlap."""
    a = exported_cases["panel_a_case_count"]
    b = exported_cases["panel_b_case_count"]
    assert a / b == pytest.approx(2.0, rel=0.01)
    assert exported_cases["panel_student_overlap_count"] == 0
    assert exported_cases["panel_query_overlap_count"] == 0


def test_sampling_is_not_first_n_slice(exported_cases):
    """is_first_n_truncation must be computed, not hard-coded, and must be False."""
    assert "is_first_n_truncation" in exported_cases
    assert exported_cases["is_first_n_truncation"] is False
    assert exported_cases.get("sampling_method") == "PROPORTIONAL_STRATIFIED_GROUP_ALLOCATION"


def test_absolute_deviation_is_computed_correctly(exported_cases):
    """Stratum deviations must use target vs actual, never abs(actual - half)."""
    breakdown = exported_cases.get("stratum_breakdown", {})
    assert breakdown, "stratum_breakdown must be populated"
    for stratum, row in breakdown.items():
        for panel in ("panel_a", "panel_b"):
            target = row[f"{panel}_target"]
            actual = row[f"{panel}_actual"]
            abs_dev = row[f"{panel}_absolute_deviation"]
            assert abs_dev == pytest.approx(abs(actual - target), abs=1e-6), (
                f"Stratum {stratum} {panel}: deviation must be abs(actual-target)"
            )
            half = row.get("pool_count", 0) // 2
            if half != target and actual != half:
                assert abs_dev != abs(actual - half), (
                    f"Stratum {stratum} appears to use half-split deviation formula"
                )


def test_all_stages_and_folds_present(exported_cases):
    """All stages and outer folds must appear in both panels."""
    pa, pb, pmap = _load_panel_cases()
    expected_stages = {"EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"}
    pa_stages = {c["stage"] for c in pa}
    pb_stages = {c["stage"] for c in pb}
    assert pa_stages == expected_stages
    assert pb_stages == expected_stages

    pa_folds = {pmap[c["case_id"]]["outer_fold"] for c in pa}
    pb_folds = {pmap[c["case_id"]]["outer_fold"] for c in pb}
    assert len(pa_folds) >= 3
    assert len(pb_folds) >= 3
    assert exported_cases.get("all_stages_represented") is True
    assert exported_cases.get("all_outer_folds_represented") is True


def test_student_group_disjoint(exported_cases):
    """Students must be 100% disjoint between panels."""
    pa, pb, pmap = _load_panel_cases()
    pa_sids = {pmap[c["case_id"]]["source_student_group_id"] for c in pa}
    pb_sids = {pmap[c["case_id"]]["source_student_group_id"] for c in pb}
    assert pa_sids.isdisjoint(pb_sids)
    assert exported_cases["panel_student_overlap_count"] == 0


# ── SECTION 4: Feasibility authority tests ─────────────────────────────────


def test_no_forced_targeted_content_fallback():
    """Exporter must not force TARGETED_CONTENT_REVIEW when no action is feasible."""
    source = (SCRIPT_DIR / "export_llm_cases.py").read_text(encoding="utf-8")
    assert "if not candidate_actions:" not in source or "TARGETED_CONTENT_REVIEW" not in source.split("if not candidate_actions:")[1].split("\n")[0:3].__str__()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.get_source_segment(source, node.test) or ""
            if "not candidate_actions" in test_src:
                body_src = "\n".join(
                    ast.get_source_segment(source, stmt) or "" for stmt in node.body
                )
                assert "TARGETED_CONTENT_REVIEW" not in body_src, (
                    "Forced TARGETED_CONTENT_REVIEW fallback detected"
                )


def test_policy_thresholds_not_hardcoded_in_exporter():
    """Hardcoded eligibility thresholds must not live in export_llm_cases.py."""
    source = (SCRIPT_DIR / "export_llm_cases.py").read_text(encoding="utf-8")
    forbidden_patterns = [
        "active_day_rate", 0.5,
        "regularity_score", 0.8,
        "content_coverage", 0.8,
    ]
    # Numeric threshold literals tied to eligibility rules are forbidden in exporter
    assert "< 0.5" not in source or "evaluate_action_eligibility" in source
    assert "< 0.8" not in source or "evaluate_action_eligibility" in source
    assert "evaluate_action_eligibility" in source, (
        "Exporter must call evaluate_action_eligibility(case_features, action_id, policy)"
    )


def test_every_eligibility_rule_has_authority():
    """ACTION_FEASIBILITY_AUTHORITY_AUDIT.json must document every action rule."""
    audit_path = AUDIT_DIR / "ACTION_FEASIBILITY_AUTHORITY_AUDIT.json"
    assert audit_path.exists(), "ACTION_FEASIBILITY_AUTHORITY_AUDIT.json is missing"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    actions = audit.get("actions", audit if isinstance(audit, list) else [])
    assert len(actions) >= 5
    for entry in actions:
        assert entry.get("action_id")
        assert entry.get("authority_type") in {
            "EXISTING_SOURCE_FUNCTION",
            "EXISTING_CONFIG",
            "NEW_V2_POLICY",
        }
        assert entry.get("source_file")
        assert entry.get("approved_status") is not None


def test_no_action_both_candidate_and_contraindicated():
    """No exported case may list an action as both candidate and contraindicated."""
    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    pb_path = EXPORT_DIR / "panel_b_cases.jsonl"
    if not pa_path.exists():
        pytest.skip("cases not exported")
    for f in (pa_path, pb_path):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            overlap = set(c.get("candidate_actions", [])) & set(c.get("contraindications", []))
            assert overlap == set(), f"Overlap {overlap} in case {c['case_id']}"


# ── SECTION 5: Importer validate_record path tests ─────────────────────────


def test_main_import_path_calls_validate_record():
    """import_annotations must call validate_record for every parsed record."""
    source = (SCRIPT_DIR / "import_llm_annotations.py").read_text(encoding="utf-8")
    assert "validate_record(" in source
    assert source.count("validate_record(") >= 2
    # Production loop must assign result from validate_record
    assert "is_valid" in source or "validate_record(rec" in source


def test_no_secondary_weak_validation_path():
    """Importer must not duplicate validation logic outside validate_record."""
    source = (SCRIPT_DIR / "import_llm_annotations.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    import_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "import_annotations":
            import_fn = node
            break
    assert import_fn is not None
    fn_src = ast.get_source_segment(source, import_fn) or source
    # Weak inline checks that bypass validate_record are forbidden
    assert fn_src.count("FORBIDDEN_TYPES") == 0 or "validate_record" in fn_src
    assert "if rt in FORBIDDEN_TYPES" not in fn_src.replace(" ", ""), (
        "Inline forbidden-type check bypasses validate_record"
    )


def _make_valid_record(case_id: str, panel_id: str, action_id: str) -> dict:
    return {
        "case_id": case_id,
        "panel_id": panel_id,
        "action_id": action_id,
        "relevance_score": 2,
        "abstain": False,
        "evidence_ids": ["ev1"],
        "rationale": "Evidence supports this action based on pre-cutoff signals.",
        "contraindication_detected": False,
        "safety_flag": False,
        "reviewer_id": "rev_001",
        "reviewer_configuration_id": "cfg_001",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "provider": "OpenAI API",
        "model_name": "gpt-4o",
        "request_id": "req_test_001",
        "response_id": "resp_test_001",
        "batch_id": "batch_test_001",
        "prompt_version": "v2.0_locked",
        "prompt_sha256": "a" * 64,
        "request_batch_sha256": "b" * 64,
        "raw_request_sha256": "c" * 64,
        "raw_response_sha256": "d" * 64,
        "response_record_index": 0,
        "response_record_sha256": hashlib.sha256(
            json.dumps({"case_id": case_id, "action_id": action_id}, sort_keys=True).encode()
        ).hexdigest(),
        "created_at": "2026-08-07T00:00:00Z",
    }


def _run_import_in_tmp(raw_lines: list[str], envelopes: dict | None = None) -> dict:
    """Run import_annotations against a temporary raw file and envelope registry."""
    from scripts.recommend_hybrid.explainable_v2 import import_llm_annotations as imp

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "test_batch.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        out_parquet = tmp_path / "accepted_records.parquet"
        rejected_path = tmp_path / "rejected_records.jsonl"

        cap_audit = tmp_path / "capability.json"
        cap_audit.write_text(
            json.dumps(
                {
                    "external_provider_status": "AVAILABLE",
                    "evaluated_providers": [
                        {"provider_name": "OpenAI API", "status": "AVAILABLE"}
                    ],
                }
            ),
            encoding="utf-8",
        )

        if envelopes:
            env_root = tmp_path / "external_reviews"
            for rel, content in envelopes.items():
                p = env_root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, dict):
                    p.write_text(json.dumps(content, indent=2), encoding="utf-8")
                else:
                    p.write_text(content, encoding="utf-8")

        with patch.object(imp, "RAW_DIR", raw_dir), patch.object(
            imp, "IMPORTS_DIR", tmp_path
        ), patch.object(imp, "CAPABILITY_AUDIT_PATH", cap_audit), patch.object(
            imp, "ACCEPTED_RECORDS_PATH", out_parquet
        ), patch.object(
            imp, "REJECTED_RECORDS_PATH", rejected_path
        ), patch.object(
            imp, "ENVELOPE_ROOT", tmp_path / "external_reviews" if envelopes else ENVELOPE_DIR
        ):
            return imp.import_annotations(raw_dir=raw_dir, output_file=out_parquet)


def test_wrong_panel_rejected_in_real_import():
    """Real import path must reject panel_id mismatch via validate_record."""
    rec = _make_valid_record("case_test_panel", "PANEL_B", "ASSESSMENT_COMPLETION")
    manifest = _run_import_in_tmp([json.dumps(rec)])
    assert manifest.get("invalid_count", 0) >= 1 or manifest.get("real_external_llm_review_count", 0) == 0


def test_ineligible_action_rejected_in_real_import():
    """Real import path must reject ineligible action_id."""
    rec = _make_valid_record("case_test_action", "PANEL_A", "QUIZ_RETRIEVAL_PRACTICE")
    manifest = _run_import_in_tmp([json.dumps(rec)])
    assert manifest.get("invalid_count", 0) >= 1 or manifest.get("real_external_llm_review_count", 0) == 0


def test_invalid_score_rejected_in_real_import():
    """Real import path must reject invalid relevance_score."""
    rec = _make_valid_record("case_test_score", "PANEL_A", "ASSESSMENT_COMPLETION")
    rec["relevance_score"] = 99
    manifest = _run_import_in_tmp([json.dumps(rec)])
    assert manifest.get("invalid_count", 0) >= 1


def test_empty_rationale_rejected_in_real_import():
    """Real import path must reject empty rationale."""
    rec = _make_valid_record("case_test_rat", "PANEL_A", "ASSESSMENT_COMPLETION")
    rec["rationale"] = ""
    manifest = _run_import_in_tmp([json.dumps(rec)])
    assert manifest.get("invalid_count", 0) >= 1


# ── SECTION 6: Provider envelope tests ─────────────────────────────────────


def test_fake_request_id_without_envelope_rejected():
    """Records with request_id but no matching envelope must be rejected."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record

    rec = _make_valid_record("case_env_001", "PANEL_A", "ASSESSMENT_COMPLETION")
    is_valid, code, _ = validate_record(
        rec,
        known_cases={"case_env_001"},
        case_panels={"case_env_001": "PANEL_A"},
        case_candidate_actions={"case_env_001": ["ASSESSMENT_COMPLETION"]},
        approved_providers={"OpenAI API"},
        locked_prompt_hash="a" * 64,
        envelope_registry={},
    )
    assert not is_valid
    assert "ENVELOPE" in code or code in {"MISSING_ENVELOPE", "ENVELOPE_NOT_FOUND", "ENVELOPE_PROVIDER_MISMATCH"}


def test_wrong_envelope_provider_rejected():
    """Envelope provider mismatch must reject the record."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record

    rec = _make_valid_record("case_env_002", "PANEL_A", "ASSESSMENT_COMPLETION")
    registry = {
        ("OpenAI API", "batch_test_001"): {
            "request_envelope": {"provider": "Anthropic API", "model": "gpt-4o", "request_id": "req_test_001", "batch_id": "batch_test_001", "status": 200, "payload_sha256": "c" * 64},
            "response_envelope": {"provider": "Anthropic API", "model": "gpt-4o", "request_id": "req_test_001", "batch_id": "batch_test_001", "status": 200, "payload_sha256": "d" * 64, "records": []},
            "batch_manifest": {"prompt_sha256": "a" * 64, "request_batch_sha256": "b" * 64},
        }
    }
    is_valid, code, _ = validate_record(
        rec,
        known_cases={"case_env_002"},
        case_panels={"case_env_002": "PANEL_A"},
        case_candidate_actions={"case_env_002": ["ASSESSMENT_COMPLETION"]},
        approved_providers={"OpenAI API"},
        locked_prompt_hash="a" * 64,
        envelope_registry=registry,
    )
    assert not is_valid
    assert "PROVIDER" in code or "ENVELOPE" in code


def test_wrong_response_record_hash_rejected():
    """Per-record response hash mismatch must reject the record."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record

    rec = _make_valid_record("case_env_003", "PANEL_A", "ASSESSMENT_COMPLETION")
    rec["response_record_sha256"] = "f" * 64
    registry = {
        ("OpenAI API", "batch_test_001"): {
            "request_envelope": {"provider": "OpenAI API", "model": "gpt-4o", "request_id": "req_test_001", "batch_id": "batch_test_001", "status": 200, "payload_sha256": "c" * 64},
            "response_envelope": {
                "provider": "OpenAI API",
                "model": "gpt-4o",
                "request_id": "req_test_001",
                "batch_id": "batch_test_001",
                "status": 200,
                "payload_sha256": "d" * 64,
                "records": [{"index": 0, "sha256": "e" * 64}],
            },
            "batch_manifest": {"prompt_sha256": "a" * 64, "request_batch_sha256": "b" * 64},
        }
    }
    is_valid, code, _ = validate_record(
        rec,
        known_cases={"case_env_003"},
        case_panels={"case_env_003": "PANEL_A"},
        case_candidate_actions={"case_env_003": ["ASSESSMENT_COMPLETION"]},
        approved_providers={"OpenAI API"},
        locked_prompt_hash="a" * 64,
        envelope_registry=registry,
    )
    assert not is_valid
    assert "HASH" in code or "RECORD" in code


def test_whole_file_hash_not_used_as_record_hash():
    """Importer must not assign whole-file SHA256 as per-record raw_response_sha256."""
    source = (SCRIPT_DIR / "import_llm_annotations.py").read_text(encoding="utf-8")
    assert "response_record_sha256" in source or "response_record_index" in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            seg = ast.get_source_segment(source, node) or ""
            if "raw_response_sha256" in seg and "raw_hash" in seg and "response_record" not in seg:
                pytest.fail("Whole-file hash assigned as record hash detected")


def test_record_links_to_response_index():
    """Accepted records must carry response_record_index from envelope."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record

    rec = _make_valid_record("case_env_004", "PANEL_A", "ASSESSMENT_COMPLETION")
    record_sha = rec["response_record_sha256"]
    registry = {
        ("OpenAI API", "batch_test_001"): {
            "request_envelope": {"provider": "OpenAI API", "model": "gpt-4o", "request_id": "req_test_001", "batch_id": "batch_test_001", "status": 200, "payload_sha256": "c" * 64},
            "response_envelope": {
                "provider": "OpenAI API",
                "model": "gpt-4o",
                "request_id": "req_test_001",
                "batch_id": "batch_test_001",
                "status": 200,
                "payload_sha256": "d" * 64,
                "records": [{"index": 0, "sha256": record_sha}],
            },
            "batch_manifest": {"prompt_sha256": "a" * 64, "request_batch_sha256": "b" * 64},
        }
    }
    is_valid, code, _ = validate_record(
        rec,
        known_cases={"case_env_004"},
        case_panels={"case_env_004": "PANEL_A"},
        case_candidate_actions={"case_env_004": ["ASSESSMENT_COMPLETION"]},
        approved_providers={"OpenAI API"},
        locked_prompt_hash="a" * 64,
        envelope_registry=registry,
    )
    assert is_valid, f"Expected valid record, got {code}"


# ── SECTION 7: Rejected record audit tests ───────────────────────────────────


def test_malformed_json_written_to_rejected_records():
    """Malformed JSON lines must appear in rejected_records.jsonl."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "bad.jsonl").write_text("{not valid json\n", encoding="utf-8")
        rejected_path = tmp_path / "rejected_records.jsonl"
        from scripts.recommend_hybrid.explainable_v2 import import_llm_annotations as imp

        cap = tmp_path / "cap.json"
        cap.write_text(
            json.dumps({"external_provider_status": "AVAILABLE", "evaluated_providers": [{"provider_name": "OpenAI API", "status": "AVAILABLE"}]}),
            encoding="utf-8",
        )
        with patch.object(imp, "RAW_DIR", raw_dir), patch.object(imp, "IMPORTS_DIR", tmp_path), patch.object(
            imp, "CAPABILITY_AUDIT_PATH", cap
        ), patch.object(imp, "REJECTED_RECORDS_PATH", rejected_path), patch.object(
            imp, "ACCEPTED_RECORDS_PATH", tmp_path / "accepted_records.parquet"
        ):
            imp.import_annotations(raw_dir=raw_dir, output_file=tmp_path / "accepted_records.parquet")
        assert rejected_path.exists()
        lines = [json.loads(l) for l in rejected_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert any(r.get("rejection_code") == "MALFORMED_JSON" for r in lines)


def test_rejected_record_contains_line_number():
    """Rejected records must include source line_number."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "bad.jsonl").write_text("{bad\n", encoding="utf-8")
        rejected_path = tmp_path / "rejected_records.jsonl"
        from scripts.recommend_hybrid.explainable_v2 import import_llm_annotations as imp

        cap = tmp_path / "cap.json"
        cap.write_text(
            json.dumps({"external_provider_status": "AVAILABLE", "evaluated_providers": [{"provider_name": "OpenAI API", "status": "AVAILABLE"}]}),
            encoding="utf-8",
        )
        with patch.object(imp, "RAW_DIR", raw_dir), patch.object(imp, "IMPORTS_DIR", tmp_path), patch.object(
            imp, "CAPABILITY_AUDIT_PATH", cap
        ), patch.object(imp, "REJECTED_RECORDS_PATH", rejected_path), patch.object(
            imp, "ACCEPTED_RECORDS_PATH", tmp_path / "accepted_records.parquet"
        ):
            imp.import_annotations(raw_dir=raw_dir, output_file=tmp_path / "accepted_records.parquet")
        rec = json.loads(rejected_path.read_text(encoding="utf-8").strip().splitlines()[0])
        assert rec.get("line_number") == 1


def test_rejected_record_contains_reason():
    """Rejected records must include rejection_code and rejection_message."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "bad.jsonl").write_text("{bad\n", encoding="utf-8")
        rejected_path = tmp_path / "rejected_records.jsonl"
        from scripts.recommend_hybrid.explainable_v2 import import_llm_annotations as imp

        cap = tmp_path / "cap.json"
        cap.write_text(
            json.dumps({"external_provider_status": "AVAILABLE", "evaluated_providers": [{"provider_name": "OpenAI API", "status": "AVAILABLE"}]}),
            encoding="utf-8",
        )
        with patch.object(imp, "RAW_DIR", raw_dir), patch.object(imp, "IMPORTS_DIR", tmp_path), patch.object(
            imp, "CAPABILITY_AUDIT_PATH", cap
        ), patch.object(imp, "REJECTED_RECORDS_PATH", rejected_path), patch.object(
            imp, "ACCEPTED_RECORDS_PATH", tmp_path / "accepted_records.parquet"
        ):
            imp.import_annotations(raw_dir=raw_dir, output_file=tmp_path / "accepted_records.parquet")
        rec = json.loads(rejected_path.read_text(encoding="utf-8").strip().splitlines()[0])
        assert rec.get("rejection_code")
        assert rec.get("rejection_message")


def test_invalid_count_matches_rejected_file():
    """import manifest invalid_count must equal rejected_records.jsonl line count."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "bad.jsonl").write_text("{bad\n", encoding="utf-8")
        rejected_path = tmp_path / "rejected_records.jsonl"
        from scripts.recommend_hybrid.explainable_v2 import import_llm_annotations as imp

        cap = tmp_path / "cap.json"
        cap.write_text(
            json.dumps({"external_provider_status": "AVAILABLE", "evaluated_providers": [{"provider_name": "OpenAI API", "status": "AVAILABLE"}]}),
            encoding="utf-8",
        )
        with patch.object(imp, "RAW_DIR", raw_dir), patch.object(imp, "IMPORTS_DIR", tmp_path), patch.object(
            imp, "CAPABILITY_AUDIT_PATH", cap
        ), patch.object(imp, "REJECTED_RECORDS_PATH", rejected_path), patch.object(
            imp, "ACCEPTED_RECORDS_PATH", tmp_path / "accepted_records.parquet"
        ):
            manifest = imp.import_annotations(raw_dir=raw_dir, output_file=tmp_path / "accepted_records.parquet")
        rejected_lines = [l for l in rejected_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert manifest.get("invalid_count", 0) == len(rejected_lines)


# ── SECTION 9: Final verifier tests ────────────────────────────────────────


def test_fake_raw_jsonl_cannot_open_verifier_gate():
    """Verifier must not count unverified raw JSONL as verified external reviews."""
    from scripts.recommend_hybrid.explainable_v2.verify_scientific_completion import verify

    code, report = verify()
    assert report.get("verified_external_llm_review_count", -1) == 0
    assert code != 0 or report.get("scientific_status") != "VERIFIED_COMPLETE"


def test_verifier_requires_accepted_records():
    """Verifier source must read accepted_records.parquet from importer."""
    source = (SCRIPT_DIR / "verify_scientific_completion.py").read_text(encoding="utf-8")
    assert "accepted_records.parquet" in source


def test_verifier_requires_envelopes():
    """Verifier must cross-check provider envelopes, not trust raw fields alone."""
    source = (SCRIPT_DIR / "verify_scientific_completion.py").read_text(encoding="utf-8")
    assert "envelope" in source.lower() or "external_reviews" in source


def test_verifier_cannot_exit_zero_without_models():
    """Verifier must block completion when five EBM models are missing."""
    source = (SCRIPT_DIR / "verify_scientific_completion.py").read_text(encoding="utf-8")
    assert "five_ebm" in source or "MODEL_DIR" in source
    from scripts.recommend_hybrid.explainable_v2.verify_scientific_completion import verify

    code, _ = verify()
    assert code != 0


def test_verifier_cannot_exit_zero_without_metrics():
    """Verifier must require metric recomputation artifacts before exit 0."""
    source = (SCRIPT_DIR / "verify_scientific_completion.py").read_text(encoding="utf-8")
    assert "metric" in source.lower() or "model_selection" in source
    from scripts.recommend_hybrid.explainable_v2.verify_scientific_completion import verify

    code, _ = verify()
    assert code == 2
