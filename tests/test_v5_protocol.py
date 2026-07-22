import json
from pathlib import Path

from src.studies.v5.common.protocol import ROOT, load_project_protocol, load_study_protocol, verify_declared_sources


def test_v5_protocol_is_frozen_before_results_and_v4_is_immutable():
    protocol = load_project_protocol()
    assert protocol["protocol_status"] == "frozen_before_v5_results"
    assert protocol["v4_frozen_commit"] == "ce79aa0b8f7444ac47ae9ae3ba6e72f997c5dd0a"
    assert protocol["v4_evidence"]["immutable"] is True
    assert protocol["completion_gates"]["cnn_bilstm_must_win"] is False


def test_v5_declared_sources_match_local_files():
    for study in ["student-mat", "student-por", "oulad"]:
        assert all(row["status"] == "PASS" for row in verify_declared_sources(load_study_protocol(study)))


def test_v5_outputs_are_separate_from_v4():
    protocol = load_project_protocol()
    assert protocol["artifacts"]["root"] == "artifacts/v5"
    assert protocol["artifacts"]["report_root"] == "reports/v5"
    assert protocol["artifacts"]["v4_namespace_prohibited"] is True

