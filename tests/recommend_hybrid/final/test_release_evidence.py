"""Frozen final-evidence regression tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "artifacts" / "recommend_hybrid" / "final"
LINEAGE = ROOT / "artifacts" / "recommend_hybrid" / "explainable_v2"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_release_manifest_is_byte_identical_to_frozen_lineage() -> None:
    canonical = CANONICAL / "release" / "FINAL_RELEASE_MANIFEST.json"
    lineage = LINEAGE / "frozen" / "final_release_v1" / "FINAL_RELEASE_MANIFEST.json"
    assert canonical.read_bytes() == lineage.read_bytes()


def test_canonical_panel_b_metrics_are_byte_identical_to_frozen_lineage() -> None:
    canonical = CANONICAL / "heldout" / "PANEL_B_FINAL_HELDOUT_METRICS.json"
    lineage = LINEAGE / "final_heldout" / "panel_b_v1" / "PANEL_B_FINAL_HELDOUT_METRICS.json"
    assert canonical.read_bytes() == lineage.read_bytes()


def test_final_release_contract_and_metrics() -> None:
    release = _json(CANONICAL / "release" / "FINAL_RELEASE_MANIFEST.json")
    metrics = _json(CANONICAL / "heldout" / "PANEL_B_FINAL_HELDOUT_METRICS.json")

    assert release["status"] == "PASS"
    assert release["runtime_authorized"] is True
    assert release["final_metrics_scope"] == "PANEL_B_FINAL_HELDOUT"
    assert release["post_panel_b_tuning_permitted"] is False
    assert release["simulator_language"] == "model-implied risk delta"

    ranker = metrics["frozen_five_ebm_ranker"]
    baseline = metrics["panel_a_action_stage_only_baseline"]
    bootstrap = metrics["paired_case_bootstrap"]

    assert metrics["panel_b_case_count"] == 150
    assert metrics["real_external_review_record_count"] == 557
    assert metrics["abstained_review_record_count"] == 0
    assert metrics["evidence_complete_case_coverage"] == 1.0
    assert ranker["ndcg_at_3"] == 0.9526603067902532
    assert ranker["exact_best_top1_agreement"] == 0.92
    assert ranker["precision_at_1_relevance_ge_1"] == 0.9733333333333334
    assert ranker["mrr_relevance_ge_1"] == 0.9855555555555556
    assert ranker["recall_at_3_relevance_ge_1"] == 0.8247777777777778
    assert ranker["pairwise_accuracy"] == 0.8353552859618717
    assert ranker["invalid_action_rate"] == 0.0
    assert baseline["ndcg_at_3"] == 0.8275943281032121
    assert bootstrap["mean_full_minus_baseline_ndcg_at_3"] == 0.12466302441561493
    assert bootstrap["ci_low_95"] == 0.09508467988207753
    assert bootstrap["ci_high_95"] == 0.15361541252930452
