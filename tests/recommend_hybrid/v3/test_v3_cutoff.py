"""Cutoff safety for V3 evidence helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommend_hybrid.v3.evidence_builder import assessment_evidence
from src.prediction.data.oulad_features import events_strictly_before_cutoff


def test_prediction_cutoff_excludes_on_or_after():
    assert events_strictly_before_cutoff(20, 0, 21) is True
    assert events_strictly_before_cutoff(21, 0, 21) is False


def test_assessment_evidence_uses_strict_before_cutoff(tmp_path):
    pd.DataFrame(
        [{"code_module": "AAA", "code_presentation": "2013J", "id_assessment": 1, "date": 20}]
    ).to_csv(tmp_path / "assessments.csv", index=False)
    pd.DataFrame(
        [{"id_assessment": 1, "id_student": 1, "date_submitted": 20}]
    ).to_csv(tmp_path / "studentAssessment.csv", index=False)
    queries = pd.DataFrame(
        [
            {
                "query_id": "1::AAA::2013J::EARLY_20",
                "id_student": 1,
                "code_module": "AAA",
                "code_presentation": "2013J",
                "cutoff_day": 20,
            }
        ]
    )
    metrics = assessment_evidence(queries, tmp_path)
    assert int(metrics.assessments_due.iloc[0]) == 0
    assert int(metrics.due_soon_count.iloc[0]) == 1
