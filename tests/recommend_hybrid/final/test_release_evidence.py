"""Frozen final recommendation evidence regression tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "artifacts" / "recommend_hybrid" / "final"
SOURCE_COMMIT = "17b519b22e8b69c875d27547d097e6d3b76bc404"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_release_contract() -> None:
    release = _json(FINAL / "release" / "FINAL_RELEASE_MANIFEST.json")
    migration = _json(FINAL / "MIGRATION_MANIFEST.json")

    assert release["status"] == "PASS"
    assert release["runtime_authorized"] is True
    assert release["final_metrics_scope"] == "PANEL_B_FINAL_HELDOUT"
    assert release["post_panel_b_tuning_permitted"] is False
    assert release["simulator_language"] == "model-implied risk delta"
    assert release["router_statuses"] == [
        "RECOMMEND",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW",
        "NO_FEASIBLE_ACTION",
    ]
    assert migration["scientific_source_commit"] == SOURCE_COMMIT
    assert migration["development_experiment_paths_copied_to_main"] is False


def test_panel_b_evidence_hashes_match_frozen_release() -> None:
    release = _json(FINAL / "release" / "FINAL_RELEASE_MANIFEST.json")
    lineage = release["lineage_sha256"]
    heldout = FINAL / "heldout"

    assert _sha256(heldout / "PANEL_B_FINAL_HELDOUT_METRICS.json") == lineage["panel_b_final_metrics"]
    assert _sha256(heldout / "PANEL_B_FINAL_HELDOUT_MANIFEST.json") == lineage["panel_b_final_manifest"]
    assert _sha256(heldout / "panel_b_real_external_reviews_frozen.jsonl") == lineage["panel_b_frozen_reviews"]
    assert _sha256(heldout / "panel_b_final_heldout_scores.parquet") == lineage["panel_b_scores"]


def test_panel_b_final_metrics_are_locked() -> None:
    metrics = _json(FINAL / "heldout" / "PANEL_B_FINAL_HELDOUT_METRICS.json")
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
