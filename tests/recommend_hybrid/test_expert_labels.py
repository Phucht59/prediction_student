from __future__ import annotations

import csv
import json

import pytest

from src.recommend_hybrid.candidate_generator import HybridCandidateGenerator
from src.recommend_hybrid.contracts import ExpertCase
from src.recommend_hybrid.exceptions import ExpertLabelValidationError
from src.recommend_hybrid.expert_labels import (
    export_expert_package,
    import_expert_ratings,
    pseudonymous_case_id,
)


def _export(tmp_path, catalog, prediction_context, observed_state):
    candidates = HybridCandidateGenerator(catalog).eligible(
        HybridCandidateGenerator(catalog).generate(prediction_context, observed_state)
    )
    case = ExpertCase(
        case_id=pseudonymous_case_id("student", "course", "MIDDLE_50", b"x" * 32),
        prediction_context=prediction_context,
        observed_state=observed_state,
        candidate_actions=candidates,
        blinding_metadata=(("future_outcome", "WITHHELD"),),
        export_version="recommend_hybrid_expert_export_v1",
    )
    export_expert_package([case], tmp_path, shuffle_secret=b"y" * 32)
    return tmp_path / "exports/expert_cases.json", case


def test_expert_export_is_blinded(tmp_path, catalog, prediction_context, observed_state):
    path, case = _export(tmp_path, catalog, prediction_context, observed_state)
    text = path.read_text()
    payload = json.loads(text)[0]
    assert prediction_context.student_key not in text
    assert "checkpoint_references" not in text
    assert "risk_probability" not in payload
    assert payload["case_id"] == case.case_id


def _completed_rating(template, output, score="2", duplicate=False):
    with template.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = rows[0]
    row.update(
        relevance_score=score,
        approval_status="APPROVE",
        missing_action="false",
        safety_concern="false",
        escalation_required="false",
        reason_support="Observed evidence supports consideration.",
        comment="",
    )
    selected = [row, row.copy()] if duplicate else [row]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(selected)


def test_expert_import_rejects_invalid_score(tmp_path, catalog, prediction_context, observed_state):
    cases, _ = _export(tmp_path, catalog, prediction_context, observed_state)
    template = tmp_path / "templates/expert_01_action_ratings.csv"
    raw = tmp_path / "invalid.csv"
    _completed_rating(template, raw, score="4")
    with pytest.raises(ExpertLabelValidationError):
        import_expert_ratings(raw, cases, tmp_path / "normalized.json")


def test_expert_import_rejects_duplicate(tmp_path, catalog, prediction_context, observed_state):
    cases, _ = _export(tmp_path, catalog, prediction_context, observed_state)
    template = tmp_path / "templates/expert_01_action_ratings.csv"
    raw = tmp_path / "duplicate.csv"
    _completed_rating(template, raw, duplicate=True)
    with pytest.raises(ExpertLabelValidationError):
        import_expert_ratings(raw, cases, tmp_path / "normalized.json")


def test_expert_labels_not_fabricated(tmp_path, catalog, prediction_context, observed_state):
    _export(tmp_path, catalog, prediction_context, observed_state)
    with (tmp_path / "templates/expert_01_action_ratings.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(row["relevance_score"] == "" for row in rows)
