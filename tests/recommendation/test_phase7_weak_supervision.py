from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.recommendation.weak_supervision.diagnostics import assign_quality_status, pre_snorkel_diagnostics
from src.recommendation.weak_supervision.label_model import (
    fit_label_models,
    majority_vote,
    probe_label_model_stochasticity,
    two_source_consensus,
)
from src.recommendation.weak_supervision.matrix import (
    FINAL_ACTIONS,
    SOURCES_BY_ACTION,
    build_matrices,
    load_canonical_sources,
    load_matrices,
    validate_phase7_authority,
    validate_source_manifest,
)
from src.recommendation.weak_supervision.silver import (
    apply_action_review_status,
    jsonable,
    validate_silver,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
PANEL_A = ROOT / "artifacts/recommendation/panels/panel_a.parquet"
PANEL_B = ROOT / "artifacts/recommendation/panels/panel_b.parquet"
MANIFEST = ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json"
PHASE7_INPUT = ROOT / "configs/recommendation/phase7_input.yaml"
CONFIG = ROOT / "configs/recommendation/weak_supervision.yaml"
LLM = ROOT / "artifacts/recommendation/labeling/normalized/phase6_llm_labels.parquet"
BEHAVIOR = ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet"
MATRIX_DIR = ROOT / "artifacts/recommendation/weak_supervision/matrices"
SILVER = ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet"
PHASE7_MANIFEST = ROOT / "artifacts/recommendation/weak_supervision/phase7_manifest.json"


def _case_ids(n: int = 500) -> list[str]:
    return [f"c{i:03d}" for i in range(n)]


def _write_panel(path: Path, case_ids: list[str], *, stage: str = "20pct") -> None:
    pd.DataFrame({"case_id": case_ids, "stage": stage}).to_parquet(path, index=False)


def _synthetic_sources(case_ids: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {}
    for action_index, action_id in enumerate(FINAL_ACTIONS):
        for source_index, source in enumerate(SOURCES_BY_ACTION[action_id]):
            labels = []
            for i, _case_id in enumerate(case_ids):
                if source == "LF_BEHAVIOR" and i % 5 == 0:
                    labels.append(-1)
                else:
                    labels.append((i + action_index + source_index) % 4)
            frame = pd.DataFrame({"case_id": case_ids, "action_id": action_id, "label": labels, "lf_name": source})
            frames.setdefault(source, []).append(frame)
    return {source: pd.concat(items, ignore_index=True) for source, items in frames.items()}


def test_source_manifest_authority_enforced(tmp_path):
    validate_phase7_authority(MANIFEST, PHASE7_INPUT, PANEL_A, PANEL_B, weak_supervision_path=CONFIG)
    broken = yaml.safe_load(PHASE7_INPUT.read_text(encoding="utf-8"))
    broken["actions"]["progress_monitoring"] = ["LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"]
    path = tmp_path / "phase7_input.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="phase7_input.yaml"):
        validate_phase7_authority(MANIFEST, path, PANEL_A, PANEL_B, weak_supervision_path=CONFIG)


def test_variable_lf_counts_and_shapes(tmp_path):
    case_ids = _case_ids()
    panel = tmp_path / "panel_a.parquet"
    _write_panel(panel, case_ids)
    built = build_matrices(_synthetic_sources(case_ids), panel, tmp_path / "matrices")
    assert [len(SOURCES_BY_ACTION[action]) for action in FINAL_ACTIONS] == [3, 3, 3, 2, 3]
    assert built["assessment_recovery"].shape == (500, 4)
    assert built["re_engagement"].shape == (500, 4)
    assert built["study_planning"].shape == (500, 4)
    assert built["progress_monitoring"].shape == (500, 3)
    assert built["retrieval_practice"].shape == (500, 4)
    assert list(built["progress_monitoring"].columns) == ["case_id", "LF_GEMINI35", "LF_GEMINI31"]


def test_abstain_maps_to_minus_one(tmp_path):
    case_ids = _case_ids()
    llm_rows = []
    behavior_rows = []
    for action_id, sources in SOURCES_BY_ACTION.items():
        for source in sources:
            if source == "LF_BEHAVIOR":
                behavior_rows.extend(
                    {"case_id": case_id, "action_id": action_id, "lf_name": "LF_BEHAVIOR_X", "label": "ABSTAIN"}
                    for case_id in case_ids
                )
            else:
                llm_rows.extend(
                    {"case_id": case_id, "action_id": action_id, "lf_name": source, "label": "ABSTAIN" if i == 0 else str(i % 4)}
                    for i, case_id in enumerate(case_ids)
                )
    for extra_action in FINAL_ACTIONS:
        if extra_action == "progress_monitoring":
            continue
        behavior_rows.extend(
            {"case_id": case_id, "action_id": extra_action, "lf_name": "LF_BEHAVIOR_X", "label": "ABSTAIN"}
            for case_id in case_ids
        )
    # A4 has no effective behavior source, but the behavioral table still has 500 rows per action.
    behavior_rows.extend(
        {"case_id": case_id, "action_id": "progress_monitoring", "lf_name": "LF_BEHAVIOR_A4", "label": "ABSTAIN"}
        for case_id in case_ids
    )
    llm = pd.DataFrame(llm_rows).drop_duplicates(["case_id", "action_id", "lf_name"])
    behavior = pd.DataFrame(behavior_rows).drop_duplicates(["case_id", "action_id"])
    assert len(llm) == 5000
    assert len(behavior) == 2500
    llm_path = tmp_path / "llm.parquet"
    behavior_path = tmp_path / "behavior.parquet"
    panel_a = tmp_path / "panel_a.parquet"
    panel_b = tmp_path / "panel_b.parquet"
    llm.to_parquet(llm_path, index=False)
    behavior.to_parquet(behavior_path, index=False)
    _write_panel(panel_a, case_ids)
    _write_panel(panel_b, [f"b{i:03d}" for i in range(150)])
    sources = load_canonical_sources(llm_path, behavior_path, panel_a, panel_b)
    matrices = build_matrices(sources, panel_a, tmp_path / "matrices")
    assert (matrices["assessment_recovery"]["LF_GEMINI35"].iloc[0] == -1)
    assert set(matrices["assessment_recovery"]["LF_BEHAVIOR"].unique()) == {-1}


def test_all_abstain_is_not_mapped_to_zero():
    case_ids = _case_ids()
    matrices = {}
    for action_id in FINAL_ACTIONS:
        matrix = pd.DataFrame({"case_id": case_ids})
        for source in SOURCES_BY_ACTION[action_id]:
            matrix[source] = [-1 if i == 0 else (i % 4) for i in range(500)]
        matrices[action_id] = matrix
    silver, _diagnostics = fit_label_models(matrices, seeds=(42,), train_config={"n_epochs": 20, "lr": 0.01, "optimizer": "sgd"})
    empty = silver[(silver["case_id"] == "c000")]
    assert (empty["silver_status"] == "NO_WEAK_EVIDENCE").all()
    assert empty[["p_r0", "p_r1", "p_r2", "p_r3", "expected_relevance", "confidence", "entropy"]].isna().all().all()
    assert empty["hard_label"].isna().all()
    assert not (empty["hard_label"].fillna(0) == 0).all() or empty["hard_label"].isna().all()


def test_no_fake_a4_lf_and_gemma_a4_excluded(tmp_path):
    case_ids = _case_ids()
    panel = tmp_path / "panel_a.parquet"
    _write_panel(panel, case_ids)
    sources = _synthetic_sources(case_ids)
    gemma = sources["LF_GEMMA4"]
    extra = pd.DataFrame({"case_id": case_ids, "action_id": "progress_monitoring", "label": 1, "lf_name": "LF_GEMMA4"})
    sources["LF_GEMMA4"] = pd.concat([gemma, extra], ignore_index=True)
    with pytest.raises(ValueError, match="Gemma4"):
        build_matrices(sources, panel, tmp_path / "matrices")


def test_content_review_academic_help_and_robustness_excluded(tmp_path):
    case_ids = _case_ids()
    panel = tmp_path / "panel_a.parquet"
    _write_panel(panel, case_ids)
    sources = _synthetic_sources(case_ids)
    sources["LF_GEMINI35"] = pd.concat([
        sources["LF_GEMINI35"],
        pd.DataFrame({"case_id": case_ids[:1], "action_id": "content_review", "label": 1, "lf_name": "LF_GEMINI35"}),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="retired or rejected"):
        build_matrices(sources, panel, tmp_path / "bad_content")
    sources = _synthetic_sources(case_ids)
    sources["LF_ACADEMIC_HELP_SEEKING"] = pd.DataFrame({
        "case_id": case_ids,
        "action_id": "academic_help_seeking",
        "label": 1,
        "lf_name": "LF_ACADEMIC_HELP_SEEKING",
    })
    with pytest.raises(ValueError, match="forbidden labeling function"):
        build_matrices(sources, panel, tmp_path / "bad_help")
    sources = _synthetic_sources(case_ids)
    sources["GEMINI_ROBUSTNESS"] = pd.DataFrame({
        "case_id": case_ids,
        "action_id": "assessment_recovery",
        "label": 1,
        "lf_name": "GEMINI_ROBUSTNESS",
    })
    with pytest.raises(ValueError, match="forbidden labeling function"):
        build_matrices(sources, panel, tmp_path / "bad_robust")


def test_panel_b_excluded_from_canonical_sources(tmp_path):
    panel_a_ids = pd.read_parquet(PANEL_A, columns=["case_id"])["case_id"].astype(str).tolist()
    panel_b_ids = pd.read_parquet(PANEL_B, columns=["case_id"])["case_id"].astype(str).tolist()
    llm = pd.read_parquet(LLM)
    behavior = pd.read_parquet(BEHAVIOR)
    leaked = llm.copy()
    leaked.loc[leaked.index[0], "case_id"] = panel_b_ids[0]
    llm_path = tmp_path / "llm.parquet"
    behavior_path = tmp_path / "behavior.parquet"
    leaked.to_parquet(llm_path, index=False)
    behavior.to_parquet(behavior_path, index=False)
    with pytest.raises(ValueError, match="Panel B"):
        load_canonical_sources(llm_path, behavior_path, PANEL_A, PANEL_B)
    assert panel_b_ids[0] not in set(panel_a_ids)


def test_probabilities_expected_relevance_and_entropy():
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]], dtype=float)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    expected = probabilities @ np.arange(4)
    assert expected.min() >= 0 and expected.max() <= 3
    entropy = [-float(np.sum(row[row > 0] * np.log(row[row > 0]))) for row in probabilities]
    assert all(value >= 0 and np.isfinite(value) for value in entropy)


def test_majority_baseline_is_deterministic():
    case_ids = _case_ids()
    matrix = pd.DataFrame({
        "case_id": case_ids,
        "LF_GEMINI35": [0 if i % 7 else -1 for i in range(500)],
        "LF_GEMMA4": [1 if i % 5 else 0 for i in range(500)],
        "LF_BEHAVIOR": [1 if i % 3 else 2 for i in range(500)],
    })
    first = majority_vote(matrix, ("LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"))
    second = majority_vote(matrix, ("LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"))
    assert np.array_equal(first, second)
    assert set(first).issubset({-1, 0, 1, 2, 3})


def test_seed_reproducibility_and_two_source_consensus():
    two_source = np.vstack([
        np.array([[0, 0], [1, 1], [2, 3], [-1, 1], [3, 3]], dtype=int),
        np.tile(np.array([[0, 1], [2, 2], [3, 1], [1, 1]], dtype=int), (124, 1))[:495],
    ])
    values = np.column_stack([two_source, two_source[:, 0]])
    first = probe_label_model_stochasticity(values, seeds=(42, 1201, 2026), train_config={"n_epochs": 20, "lr": 0.01}, threshold=1e-6)
    second = probe_label_model_stochasticity(values, seeds=(42, 1201, 2026), train_config={"n_epochs": 20, "lr": 0.01}, threshold=1e-6)
    assert first["same_seed_max_abs_deviation"] < 1e-8
    assert second["same_seed_max_abs_deviation"] < 1e-8
    assert np.allclose(first["reference_probabilities"], second["reference_probabilities"], atol=1e-8)
    matrix = pd.DataFrame(two_source, columns=["LF_GEMINI35", "LF_GEMINI31"])
    consensus = two_source_consensus(matrix, ("LF_GEMINI35", "LF_GEMINI31"))
    assert np.allclose(consensus[0], [1, 0, 0, 0])
    assert np.allclose(consensus[2], [0, 0, 0.5, 0.5])
    assert np.allclose(consensus[3], [0, 1, 0, 0])


def test_a5_review_is_preserved_without_upgrade():
    pairwise = {
        "LF_GEMINI35_vs_LF_GEMMA4": {"quadratic_weighted_kappa": 0.044},
        "LF_GEMINI35_vs_LF_BEHAVIOR": {"quadratic_weighted_kappa": 0.148},
        "LF_GEMMA4_vs_LF_BEHAVIOR": {"quadratic_weighted_kappa": -0.098},
    }
    status, reasons = assign_quality_status(
        pairwise=pairwise,
        collapse={"flags": []},
        aggregator_type="SNORKEL",
        correlated_family=False,
        a5_config={"remain_review_unless_strong_upgrade": True, "upgrade_min_quadratic_kappa": 0.40, "conflict_kappa_threshold": 0.20},
        usable_count=311,
    )
    assert status == "REVIEW"
    assert "high_source_conflict" in reasons
    frame = pd.DataFrame({
        "action_id": ["retrieval_practice", "retrieval_practice", "assessment_recovery"],
        "silver_status": ["VALID", "NO_WEAK_EVIDENCE", "VALID"],
    })
    updated = apply_action_review_status(frame, {"retrieval_practice"})
    assert updated["silver_status"].tolist() == ["REVIEW", "NO_WEAK_EVIDENCE", "VALID"]


def test_phase7_manifest_helper_is_deterministic(tmp_path):
    payload = {"version": "recommendation.phase7_manifest.v1", "status": {"assessment_recovery": "PASS"}, "count": 2500}
    path = tmp_path / "manifest.json"
    write_json(path, payload)
    write_json(path, payload)
    first = path.read_text(encoding="utf-8")
    write_json(path, jsonable(payload))
    assert path.read_text(encoding="utf-8") == first


def test_real_phase7_artifacts_when_present():
    if not (SILVER.exists() and PHASE7_MANIFEST.exists()):
        pytest.skip("Phase 7 artifacts have not been generated yet")
    validate_source_manifest(MANIFEST, PANEL_A, PANEL_B)
    matrices = load_matrices(MATRIX_DIR)
    assert matrices["assessment_recovery"].shape == (500, 4)
    assert matrices["re_engagement"].shape == (500, 4)
    assert matrices["study_planning"].shape == (500, 4)
    assert matrices["progress_monitoring"].shape == (500, 3)
    assert matrices["retrieval_practice"].shape == (500, 4)
    assert "LF_GEMMA4" not in matrices["progress_monitoring"].columns
    assert "content_review" not in {action for action in matrices}
    silver = pd.read_parquet(SILVER)
    panel_a = set(pd.read_parquet(PANEL_A, columns=["case_id"])["case_id"].astype(str))
    panel_b = set(pd.read_parquet(PANEL_B, columns=["case_id"])["case_id"].astype(str))
    validate_silver(silver, panel_a, panel_b)
    assert set(silver["case_id"].astype(str)).isdisjoint(panel_b)
    assert (silver["silver_status"] == "REVIEW").any()
    manifest = json.loads(PHASE7_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["panel_b_overlap"] == 0
    assert manifest["status_by_action"]["retrieval_practice"] == "REVIEW"
    assert manifest["lf_names_by_action"]["progress_monitoring"] == ["LF_GEMINI35", "LF_GEMINI31"]
    again = json.loads(PHASE7_MANIFEST.read_text(encoding="utf-8"))
    assert json.dumps(manifest, sort_keys=True) == json.dumps(again, sort_keys=True)
    pre = pre_snorkel_diagnostics(matrices)
    assert pre["progress_monitoring"]["effective_lf_count"] == 2
    assert pre["assessment_recovery"]["effective_lf_count"] == 3
