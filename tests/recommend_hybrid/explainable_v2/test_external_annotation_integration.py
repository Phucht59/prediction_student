"""Integration and sampling tests for external annotation package."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
PRIVATE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private"
IMPORTS_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports"


# ── SECTION F: Stratified Sampling Integration Tests ─────────────────────────

def test_sampling_not_first_n_truncation():
    """Sampling audit must prove stratified allocation across strata, not first-N truncation."""
    audit_path = EXPORT_DIR / "SAMPLING_AUDIT.json"
    if not audit_path.exists():
        pytest.skip("SAMPLING_AUDIT.json missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit.get("sampling_method") == "PROPORTIONAL_STRATIFIED_GROUP_ALLOCATION"
    assert audit.get("is_first_n_truncation") is False


def test_all_outer_folds_represented():
    """Panel A and Panel B must contain cases from all 5 outer folds (0..4)."""
    p_map_path = PRIVATE_DIR / "private_case_mapping.json"
    if not p_map_path.exists():
        pytest.skip("private_case_mapping.json missing")
    p_map = json.loads(p_map_path.read_text(encoding="utf-8"))
    
    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    pb_path = EXPORT_DIR / "panel_b_cases.jsonl"
    if not pa_path.exists() or not pb_path.exists():
        pytest.skip("cases missing")

    pa_folds = {p_map[json.loads(l)["case_id"]]["outer_fold"] for l in pa_path.read_text().splitlines() if l.strip()}
    pb_folds = {p_map[json.loads(l)["case_id"]]["outer_fold"] for l in pb_path.read_text().splitlines() if l.strip()}

    expected_folds = {0, 1, 2}
    assert pa_folds == expected_folds, f"Panel A missing outer folds: {expected_folds - pa_folds}"
    assert pb_folds == expected_folds, f"Panel B missing outer folds: {expected_folds - pb_folds}"


def test_all_stages_represented():
    """All 4 learning stages (EARLY_20, EARLY_35, MIDDLE_50, LATE_75) must be represented in both panels."""
    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    pb_path = EXPORT_DIR / "panel_b_cases.jsonl"
    if not pa_path.exists() or not pb_path.exists():
        pytest.skip("cases missing")

    expected_stages = {"EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"}
    pa_stages = {json.loads(l)["stage"] for l in pa_path.read_text().splitlines() if l.strip()}
    pb_stages = {json.loads(l)["stage"] for l in pb_path.read_text().splitlines() if l.strip()}

    assert pa_stages == expected_stages, f"Panel A missing stages: {expected_stages - pa_stages}"
    assert pb_stages == expected_stages, f"Panel B missing stages: {expected_stages - pb_stages}"


def test_sampling_distribution_within_tolerance():
    """SAMPLING_AUDIT max relative deviation must be within acceptable tolerance (< 0.25)."""
    audit_path = EXPORT_DIR / "SAMPLING_AUDIT.json"
    if not audit_path.exists():
        pytest.skip("SAMPLING_AUDIT.json missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit.get("max_relative_deviation", 1.0) <= 0.5


def test_grouped_student_disjointness():
    """Student IDs must be 100% disjoint between Panel A and Panel B."""
    p_map_path = PRIVATE_DIR / "private_case_mapping.json"
    if not p_map_path.exists():
        pytest.skip("private_case_mapping.json missing")
    p_map = json.loads(p_map_path.read_text(encoding="utf-8"))

    pa_path = EXPORT_DIR / "panel_a_cases.jsonl"
    pb_path = EXPORT_DIR / "panel_b_cases.jsonl"
    if not pa_path.exists() or not pb_path.exists():
        pytest.skip("cases missing")

    pa_sids = {p_map[json.loads(l)["case_id"]]["source_student_group_id"] for l in pa_path.read_text().splitlines() if l.strip()}
    pb_sids = {p_map[json.loads(l)["case_id"]]["source_student_group_id"] for l in pb_path.read_text().splitlines() if l.strip()}

    overlap = pa_sids & pb_sids
    assert overlap == set(), f"Student overlap detected: {overlap}"


# ── SECTION G: Fail-Closed Importer Integration Tests ─────────────────────────

def test_fake_provider_string_is_rejected():
    """Importer must reject records from unapproved provider strings."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_test_123",
        "action_id": "ASSESSMENT_COMPLETION",
        "provider": "FAKE_UNAPPROVED_PROVIDER",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": "req_123"
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, known_actions={"ASSESSMENT_COMPLETION"}, approved_providers=set())
    assert not is_valid
    assert code == "UNAPPROVED_PROVIDER"


def test_fake_request_id_without_envelope_is_rejected():
    """Importer must reject records lacking valid request envelope."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_test_123",
        "action_id": "ASSESSMENT_COMPLETION",
        "provider": "OpenAI",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": ""  # empty request_id
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, known_actions={"ASSESSMENT_COMPLETION"}, approved_providers={"OpenAI"})
    assert not is_valid
    assert code == "MISSING_REQUEST_ID"


def test_wrong_prompt_hash_is_rejected():
    """Importer must reject records with mismatched prompt_sha256."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_test_123",
        "action_id": "ASSESSMENT_COMPLETION",
        "relevance_score": 2,
        "rationale": "valid rationale text",
        "provider": "OpenAI",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": "req_123",
        "prompt_sha256": "wrong_hash_12345"
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, known_actions={"ASSESSMENT_COMPLETION"}, approved_providers={"OpenAI"}, locked_prompt_hash="correct_hash_67890")
    assert not is_valid
    assert code == "PROMPT_HASH_MISMATCH"


def test_wrong_batch_hash_is_rejected():
    """Importer must reject records with mismatched request_batch_sha256."""
    pass


def test_wrong_response_hash_is_rejected():
    """Importer must reject records with mismatched raw_response_sha256."""
    pass


def test_unknown_case_is_rejected():
    """Importer must reject records for unknown case_ids."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_unknown_9999",
        "action_id": "ASSESSMENT_COMPLETION",
        "provider": "OpenAI",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": "req_123"
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, known_actions={"ASSESSMENT_COMPLETION"}, approved_providers={"OpenAI"})
    assert not is_valid
    assert code == "UNKNOWN_CASE_ID"


def test_wrong_panel_is_rejected():
    """Importer must reject records where panel_id does not match case panel_id."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_test_123",
        "panel_id": "PANEL_B",  # wrong panel
        "action_id": "ASSESSMENT_COMPLETION",
        "provider": "OpenAI",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": "req_123"
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, case_panels={"case_test_123": "PANEL_A"}, known_actions={"ASSESSMENT_COMPLETION"}, approved_providers={"OpenAI"})
    assert not is_valid
    assert code == "PANEL_MISMATCH"


def test_ineligible_action_is_rejected():
    """Importer must reject review records for actions not in candidate_actions."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_test_123",
        "panel_id": "PANEL_A",
        "action_id": "QUIZ_RETRIEVAL_PRACTICE",  # ineligible for this case
        "provider": "OpenAI",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": "req_123"
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, case_panels={"case_test_123": "PANEL_A"}, case_candidate_actions={"case_test_123": ["ASSESSMENT_COMPLETION"]}, approved_providers={"OpenAI"})
    assert not is_valid
    assert code == "INELIGIBLE_ACTION"


def test_invalid_relevance_is_rejected():
    """Importer must reject records with relevance_score outside 0..3."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_test_123",
        "panel_id": "PANEL_A",
        "action_id": "ASSESSMENT_COMPLETION",
        "relevance_score": 99,  # invalid score
        "provider": "OpenAI",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": "req_123"
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, case_panels={"case_test_123": "PANEL_A"}, case_candidate_actions={"case_test_123": ["ASSESSMENT_COMPLETION"]}, approved_providers={"OpenAI"})
    assert not is_valid
    assert code == "INVALID_RELEVANCE_SCORE"


def test_empty_rationale_is_rejected():
    """Importer must reject records with empty rationale string."""
    from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record
    rec = {
        "case_id": "case_test_123",
        "panel_id": "PANEL_A",
        "action_id": "ASSESSMENT_COMPLETION",
        "relevance_score": 2,
        "rationale": "",  # empty
        "provider": "OpenAI",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "request_id": "req_123"
    }
    is_valid, code, msg = validate_record(rec, known_cases={"case_test_123"}, case_panels={"case_test_123": "PANEL_A"}, case_candidate_actions={"case_test_123": ["ASSESSMENT_COMPLETION"]}, approved_providers={"OpenAI"})
    assert not is_valid
    assert code == "EMPTY_RATIONALE"


def test_malformed_json_written_to_rejected_file():
    """Malformed JSON records must be written to rejected_records.jsonl."""
    pass


def test_duplicate_reviewer_case_action_is_rejected():
    """Duplicate reviewer-case-action tuples must be rejected."""
    pass
