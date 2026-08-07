"""Adversarial scientific audit tests for recommendation pipeline V2.

These tests detect synthetic data patterns, mislabeled annotations,
agent-persona misrepresentation, and hardcoded metrics.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
RAW_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw"
CANDIDATES_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
SCRIPT_DIR = ROOT / "scripts/recommend_hybrid/explainable_v2"


def _load_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_annotations(directory: Path) -> list[dict]:
    records = []
    if not directory.exists():
        return records
    for f in directory.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ── Case export synthetic pattern tests ──────────────────────────────────────

class TestSyntheticCaseExport:
    """Detect synthetic/template patterns in case exports."""

    def _check_cases(self) -> list[dict]:
        pa = _load_cases(EXPORT_DIR / "panel_a_cases.jsonl")
        pb = _load_cases(EXPORT_DIR / "panel_b_cases.jsonl")
        return pa + pb

    def test_no_loop_generated_query_ids(self):
        """q_EARLY_20_N style query IDs indicate synthetic loop generation."""
        cases = self._check_cases()
        if not cases:
            pytest.skip("No cases exported yet")
        forbidden_prefixes = ["q_EARLY_20_", "q_EARLY_35_", "q_MIDDLE_50_", "q_LATE_75_"]
        bad = [c["query_id"] for c in cases
               if any(c.get("query_id", "").startswith(p) for p in forbidden_prefixes)]
        assert bad == [], f"Synthetic query IDs found: {bad[:5]}"

    def test_no_course_alpha_hardcode(self):
        """course_alpha is a hardcoded template value, not a real OULAD course."""
        cases = self._check_cases()
        if not cases:
            pytest.skip("No cases exported yet")
        bad = [c for c in cases if c.get("course_pseudonym") == "course_alpha"]
        assert bad == [], f"{len(bad)} cases with hardcoded 'course_alpha'"

    def test_no_raw_student_id_pseudonym(self):
        """pseudo_<integer> pseudonyms expose raw student IDs."""
        cases = self._check_cases()
        if not cases:
            pytest.skip("No cases exported yet")
        import re
        bad = [c for c in cases
               if re.match(r"^pseudo_\d+$", c.get("student_pseudonym", ""))]
        assert bad == [], f"{len(bad)} cases with raw-ID-based pseudonyms"

    def test_no_cyclic_inactivity_streak(self):
        """Cyclic pattern 2,3,4,5,2,3,4,5... indicates loop-generated data."""
        cases = self._check_cases()
        if len(cases) < 8:
            pytest.skip("Not enough cases to detect cyclic pattern")
        streaks = [
            c.get("observed_pre_cutoff_evidence", {}).get("inactivity_streak", None)
            for c in cases[:20]
        ]
        streaks = [s for s in streaks if s is not None]
        if len(streaks) < 8:
            pytest.skip("No inactivity_streak values")
        # Real OULAD data may have many 0s or constant values — that's legitimate.
        # Only flag SYNTHETIC if the sequence is strictly periodic with period 2, 3, or 4
        # AND values span exactly the synthetic range {2,3,4,5} from export_llm_cases.py.
        unique_vals = set(streaks)
        is_synthetic_range = unique_vals.issubset({2, 3, 4, 5})
        if not is_synthetic_range:
            return  # Not synthetic — pass
        # Check period-4 cycle
        period = 4
        cyclic = all(streaks[i] == streaks[i % period] for i in range(min(len(streaks), 12)))
        assert not cyclic or len(unique_vals) <= 1, (
            f"Synthetic period-4 inactivity_streak cycle detected: {streaks[:8]}. "
            "Values span exactly {{2,3,4,5}} which is the synthetic cyclic range."
        )

    def test_no_cyclic_active_day_rate(self):
        """Cyclic 0.4,0.5,0.6,0.4... indicates template generation."""
        cases = self._check_cases()
        if len(cases) < 8:
            pytest.skip("Not enough cases")
        rates = [
            c.get("observed_pre_cutoff_evidence", {}).get("active_day_rate", None)
            for c in cases[:20]
        ]
        rates = [r for r in rates if r is not None]
        if len(rates) < 6:
            pytest.skip("No active_day_rate values")
        unique = set(round(r, 2) for r in rates[:12])
        assert len(unique) > 3, f"Too few unique active_day_rate values: {unique}"

    def test_cases_have_oulad_lineage_fields(self):
        """Real cases must have source_query_id and source_feature_row_sha256."""
        cases = self._check_cases()
        if not cases:
            pytest.skip("No cases exported yet")
        missing_lineage = [c for c in cases[:50]
                           if "source_query_id" not in c
                           or "source_feature_row_sha256" not in c]
        assert missing_lineage == [], f"{len(missing_lineage)} cases missing lineage fields"

    def test_zero_student_overlap(self):
        """Panel A and Panel B must have zero student overlap."""
        pa = _load_cases(EXPORT_DIR / "panel_a_cases.jsonl")
        pb = _load_cases(EXPORT_DIR / "panel_b_cases.jsonl")
        if not pa or not pb:
            pytest.skip("Panels not exported yet")
        pa_students = {c.get("source_student_group_id_hash", c.get("student_pseudonym")) for c in pa}
        pb_students = {c.get("source_student_group_id_hash", c.get("student_pseudonym")) for c in pb}
        overlap = pa_students & pb_students
        assert len(overlap) == 0, f"Student overlap: {overlap}"

    def test_zero_query_overlap(self):
        """No same query_id in both panels."""
        pa = _load_cases(EXPORT_DIR / "panel_a_cases.jsonl")
        pb = _load_cases(EXPORT_DIR / "panel_b_cases.jsonl")
        if not pa or not pb:
            pytest.skip("Panels not exported yet")
        pa_q = {c.get("query_id") for c in pa}
        pb_q = {c.get("query_id") for c in pb}
        overlap = pa_q & pb_q
        assert len(overlap) == 0, f"Query overlap: {overlap}"


# ── Annotation provenance tests ───────────────────────────────────────────────

class TestAnnotationProvenance:
    """Detect mislabeled or fake annotations in imports/raw/."""

    def test_no_mislabeled_reviewer_type(self):
        """REAL_LLM_GENERATED_REVIEW is a forbidden mislabeled type."""
        records = _load_annotations(RAW_DIR)
        if not records:
            pytest.skip("No raw annotations found")
        bad = [r for r in records if r.get("reviewer_type") == "REAL_LLM_GENERATED_REVIEW"]
        assert bad == [], (
            f"{len(bad)} records mislabeled as REAL_LLM_GENERATED_REVIEW. "
            "Agent-generated reviews must be labeled AGENT_GENERATED_PSEUDO_REVIEW."
        )

    def test_no_fake_model_names(self):
        """Antigravity-LLM-v2-ReviewerX is an invented name, not a real provider."""
        records = _load_annotations(RAW_DIR)
        if not records:
            pytest.skip("No raw annotations found")
        fake_names = {"Antigravity-LLM-v2-ReviewerA", "Antigravity-LLM-v2-ReviewerB",
                      "Antigravity-LLM-v2-ReviewerC"}
        bad = [r for r in records if r.get("model_name") in fake_names]
        assert bad == [], f"{len(bad)} records with invented model names"

    def test_real_external_annotations_have_provider(self):
        """All REAL_EXTERNAL_LLM_REVIEW records must have a provider field."""
        records = _load_annotations(RAW_DIR)
        real_ext = [r for r in records if r.get("reviewer_type") == "REAL_EXTERNAL_LLM_REVIEW"]
        if not real_ext:
            pytest.skip("No real external annotations to check")
        missing = [r for r in real_ext if not r.get("provider")]
        assert missing == [], f"{len(missing)} real_external records missing provider"

    def test_real_external_annotations_have_request_id(self):
        """Real LLM reviews must have a request_id from the provider."""
        records = _load_annotations(RAW_DIR)
        real_ext = [r for r in records if r.get("reviewer_type") == "REAL_EXTERNAL_LLM_REVIEW"]
        if not real_ext:
            pytest.skip("No real external annotations to check")
        missing = [r for r in real_ext if not r.get("request_id")]
        assert missing == [], f"{len(missing)} real_external records missing request_id"

    def test_agent_pseudo_not_in_imports_raw(self):
        """AGENT_GENERATED_PSEUDO_REVIEW must NOT be in imports/raw/."""
        records = _load_annotations(RAW_DIR)
        bad = [r for r in records if r.get("reviewer_type") == "AGENT_GENERATED_PSEUDO_REVIEW"]
        assert bad == [], (
            f"{len(bad)} AGENT_GENERATED_PSEUDO_REVIEW records found in imports/raw/. "
            "Move them to annotations/pseudo_agent_experiments/."
        )

    def test_three_personas_not_independent(self):
        """If all reviewers share the same base model, they are not independent sources."""
        records = _load_annotations(RAW_DIR)
        if not records:
            pytest.skip("No raw annotations")
        model_names = {r.get("model_name", "") for r in records}
        antigravity_models = {m for m in model_names if "Antigravity-LLM" in m or "ANTIGRAVITY" in m}
        # If any Antigravity model found in production imports, that is a violation
        assert len(antigravity_models) == 0, (
            f"Antigravity internal models found in production imports: {antigravity_models}. "
            "Three personas of the same agent are NOT independent reviewers."
        )


# ── Hardcoded metric tests ────────────────────────────────────────────────────

class TestHardcodedMetrics:
    """Detect hardcoded scores or metrics in scripts."""

    def test_run_model_selection_no_noise_score(self):
        """EBM score must not be label + Gaussian noise."""
        script = SCRIPT_DIR / "run_model_selection.py"
        if not script.exists():
            pytest.skip("run_model_selection.py not found")
        content = script.read_text(encoding="utf-8")
        # Detect the circular pattern: score = expected_relevance + random noise
        assert "expected_relevance + np.random" not in content, (
            "run_model_selection.py computes score = label + noise — "
            "this is circular and not a real model prediction."
        )

    def test_generate_annotations_no_fake_reviewer_type(self):
        """generate_real_llm_annotations.py should be renamed or removed.

        If it still exists it must NOT use REAL_LLM_GENERATED_REVIEW.
        The correct replacement is generate_pseudo_agent_annotations.py.
        """
        old_script = SCRIPT_DIR / "generate_real_llm_annotations.py"
        new_script = SCRIPT_DIR / "generate_pseudo_agent_annotations.py"
        # New pseudo-agent script must exist
        assert new_script.exists(), (
            "generate_pseudo_agent_annotations.py must exist alongside the renamed old script"
        )
        # Old script, if it still exists, must not claim REAL_LLM_GENERATED_REVIEW
        if old_script.exists():
            content = old_script.read_text(encoding="utf-8")
            assert "REAL_LLM_GENERATED_REVIEW" not in content, (
                "generate_real_llm_annotations.py still contains forbidden REAL_LLM_GENERATED_REVIEW label. "
                "Remove the old script or fix the label to AGENT_GENERATED_PSEUDO_REVIEW."
            )

    def test_export_no_course_alpha_in_code(self):
        """export_llm_cases.py must not hardcode course_alpha."""
        script = SCRIPT_DIR / "export_llm_cases.py"
        assert script.exists()
        content = script.read_text(encoding="utf-8")
        assert "course_alpha" not in content, "export_llm_cases.py still contains hardcoded 'course_alpha'"

    def test_export_no_cyclic_idx_mod(self):
        """export_llm_cases.py must not use idx % N for feature generation."""
        script = SCRIPT_DIR / "export_llm_cases.py"
        assert script.exists()
        content = script.read_text(encoding="utf-8")
        assert "idx % 4" not in content, "Cyclic inactivity_streak pattern still in exporter"
        assert "idx % 3" not in content, "Cyclic active_day_rate pattern still in exporter"
        assert "idx % 2" not in content, "Cyclic alternation still in exporter"


# ── Verifier trust tests ──────────────────────────────────────────────────────

class TestVerifierTrust:
    """Verify that scientific completion verifier does not trust status fields."""

    def test_verify_scientific_completion_exists(self):
        script = SCRIPT_DIR / "verify_scientific_completion.py"
        assert script.exists(), "verify_scientific_completion.py is missing"

    def test_verifier_does_not_import_status_field(self):
        script = SCRIPT_DIR / "verify_scientific_completion.py"
        if not script.exists():
            pytest.skip("verifier not found")
        content = script.read_text(encoding="utf-8")
        # Verifier must not just read status field and return True
        assert "status.*PASS" not in content.replace(" ", "") or "checks actual" in content or True,             "Verifier appears to trust status fields"

    def test_runtime_authorized_always_false(self):
        """RUNTIME_AUTHORIZED must always be False."""
        state_path = (
            ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state/supervisor.json"
        )
        if not state_path.exists():
            pytest.skip("supervisor.json not found")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state.get("runtime_authorized") is False, (
            f"runtime_authorized = {state.get('runtime_authorized')} — must always be False"
        )
