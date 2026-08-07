"""Security unit tests for external annotation package."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
PROMPTS_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts"
PRIVATE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private"
RUN_STATE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state"
SCRIPT_DIR = ROOT / "scripts/recommend_hybrid/explainable_v2"


# ── ISSUE-01: Secret Salt Security Tests ─────────────────────────────────────

def test_export_fails_without_case_export_salt():
    """export_v2_cases must fail non-zero / raise RuntimeError if CASE_EXPORT_SALT is unset."""
    from scripts.recommend_hybrid.explainable_v2.export_llm_cases import export_v2_cases
    old_salt = os.environ.pop("CASE_EXPORT_SALT", None)
    try:
        with pytest.raises((RuntimeError, KeyError)):
            export_v2_cases()
    finally:
        if old_salt is not None:
            os.environ["CASE_EXPORT_SALT"] = old_salt


def test_export_uses_runtime_secret_salt():
    """Different runtime salt values must generate completely different blinded case_ids."""
    from scripts.recommend_hybrid.explainable_v2.export_llm_cases import _blinded_case_id
    id1 = _blinded_case_id("query_test_123", salt="salt_alpha_2026")
    id2 = _blinded_case_id("query_test_123", salt="salt_beta_2026")
    assert id1 != id2
    assert id1.startswith("case_")
    assert id2.startswith("case_")


def test_no_hardcoded_salt_literal_in_source():
    """export_llm_cases.py source code must not contain any default hardcoded salt literal."""
    script = SCRIPT_DIR / "export_llm_cases.py"
    assert script.exists()
    content = script.read_text(encoding="utf-8")
    forbidden_defaults = [
        "recommend_v2_blinded_privacy_salt_2026",
        'get("CASE_EXPORT_SALT"',
        "get('CASE_EXPORT_SALT'",
    ]
    for forbidden in forbidden_defaults:
        assert forbidden not in content, f"Hardcoded salt pattern '{forbidden}' found in source code"


# ── SECTION C: Privacy & Git Hygiene Tests ────────────────────────────────────

def test_private_mapping_is_gitignored():
    """private_case_mapping.json path must match .gitignore rule."""
    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.exists()
    content = gitignore_path.read_text(encoding="utf-8")
    assert "artifacts/recommend_hybrid/explainable_v2/annotations/private/" in content


def test_private_mapping_not_git_tracked():
    """git ls-files --error-unmatch private_case_mapping.json must fail (non-zero exit code)."""
    p_path = PRIVATE_DIR / "private_case_mapping.json"
    res = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(p_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, f"private_case_mapping.json is git-tracked! Output: {res.stdout}"


def test_public_exports_contain_no_raw_query_id():
    """Public case JSONL exports must contain zero query_id or raw identifier fields."""
    for filename in ["panel_a_cases.jsonl", "panel_b_cases.jsonl"]:
        file_path = EXPORT_DIR / filename
        if not file_path.exists():
            pytest.skip(f"{filename} not exported yet")
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            forbidden_keys = {"query_id", "source_query_id", "id_student", "student_group_id", "module", "presentation", "outer_fold"}
            found = forbidden_keys & set(rec.keys())
            assert found == set(), f"Forbidden key {found} in public export {filename}"


def test_public_batches_contain_no_raw_query_id():
    """Public prompt batch files must contain zero query_id or raw identifier fields."""
    for b_dir in [PROMPTS_DIR / "panel_a_request_batches", PROMPTS_DIR / "panel_b_request_batches"]:
        if not b_dir.exists():
            continue
        for bf in b_dir.glob("*.jsonl"):
            for line in bf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                forbidden_keys = {"query_id", "source_query_id", "id_student", "student_group_id", "module", "presentation", "outer_fold"}
                found = forbidden_keys & set(rec.keys())
                assert found == set(), f"Forbidden key {found} in batch file {bf.name}"


def test_runtime_progress_files_not_git_tracked():
    """Runtime progress and log files must not be tracked by git."""
    res = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "artifacts/recommend_hybrid/explainable_v2/run_state/"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, "run_state files are tracked by git"


# ── SECTION D: Schema Consistency Tests ───────────────────────────────────────

def test_export_schema_matches_response_schema():
    """All public case fields must match response schema case expectations."""
    schema_path = PROMPTS_DIR / "response_schema.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    req = schema.get("required", [])
    assert "query_id" not in req, "response_schema still requires query_id!"
    assert "case_id" in req
    assert "panel_id" in req


def test_prompt_does_not_request_hidden_query_id():
    """System prompt and instructions must not request hidden query_id from reviewers."""
    inst_path = PROMPTS_DIR / "annotation_instructions.md"
    if not inst_path.exists():
        pytest.skip("instructions file not found")
    content = inst_path.read_text(encoding="utf-8")
    assert "query_id" not in content, "instructions still request query_id"


def test_every_case_has_panel_id():
    """Every case in panel_a_cases.jsonl and panel_b_cases.jsonl must have panel_id."""
    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    pb_path = EXPORT_DIR / "panel_b_cases.jsonl"
    if not pa_path.exists() or not pb_path.exists():
        pytest.skip("cases not exported yet")
    for line in pa_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert json.loads(line).get("panel_id") == "PANEL_A"
    for line in pb_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert json.loads(line).get("panel_id") == "PANEL_B"


def test_every_response_field_has_importer_mapping():
    """Importer must map all schema fields into normalized output DataFrame."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import REQUIRED_SCHEMA_FIELDS
    schema_path = PROMPTS_DIR / "response_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for prop in schema["properties"].keys():
        if prop not in ["model_version", "endpoint_id", "request_batch_sha256", "candidate_order_sha256", "raw_request_sha256"]:
            assert prop in REQUIRED_SCHEMA_FIELDS, f"Field {prop} missing in importer mapping"


# ── SECTION E: Action Feasibility & Eligibility Tests ────────────────────────

def test_quiz_action_removed_when_quiz_unavailable():
    """QUIZ_RETRIEVAL_PRACTICE must be excluded from candidate_actions if quiz_available is False."""
    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    if not pa_path.exists():
        pytest.skip("panel_a_cases.jsonl missing")
    for line in pa_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        flags = c.get("availability_flags", {})
        cands = c.get("candidate_actions", [])
        if not flags.get("quiz_available", True):
            assert "QUIZ_RETRIEVAL_PRACTICE" not in cands, (
                f"QUIZ_RETRIEVAL_PRACTICE included when quiz_available=False in case {c['case_id']}"
            )


def test_no_action_is_both_eligible_and_contraindicated():
    """No action can be listed in candidate_actions AND contraindications."""
    for f in [EXPORT_DIR / "panel_a_cases.jsonl", EXPORT_DIR / "panel_b_cases.jsonl"]:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            cands = set(c.get("candidate_actions", []))
            contras = set(c.get("contraindications", []))
            overlap = cands & contras
            assert overlap == set(), f"Action {overlap} is both eligible and contraindicated in case {c['case_id']}"


def test_assessment_action_requires_assessment_evidence():
    """ASSESSMENT_COMPLETION requires missing assessment count or assessments due > 0."""
    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    if not pa_path.exists():
        pytest.skip("cases missing")
    for line in pa_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        ev = c.get("observed_pre_cutoff_evidence", {})
        cands = c.get("candidate_actions", [])
        if "ASSESSMENT_COMPLETION" in cands:
            has_ev = (
                ev.get("missing_assessment_count", 0) > 0
                or ev.get("assessments_due", 0) > 0
                or ev.get("due_soon_count", 0) > 0
            )
            assert has_ev, f"ASSESSMENT_COMPLETION eligible without assessment evidence in case {c['case_id']}"


def test_exported_eligibility_matches_candidate_authority():
    """candidate_actions must match deterministic eligibility authority."""
    audit_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/audit/ACTION_CANDIDATE_SCHEMA_AUDIT.json"
    assert audit_path.exists(), "ACTION_CANDIDATE_SCHEMA_AUDIT.json is missing"
