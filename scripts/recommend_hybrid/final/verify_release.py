"""Verify the canonical final recommendation evidence without rerunning Panel B."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "artifacts" / "recommend_hybrid" / "final"
LINEAGE = ROOT / "artifacts" / "recommend_hybrid" / "explainable_v2"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    canonical_release = CANONICAL / "release" / "FINAL_RELEASE_MANIFEST.json"
    lineage_release = LINEAGE / "frozen" / "final_release_v1" / "FINAL_RELEASE_MANIFEST.json"
    canonical_metrics = CANONICAL / "heldout" / "PANEL_B_FINAL_HELDOUT_METRICS.json"
    lineage_metrics = LINEAGE / "final_heldout" / "panel_b_v1" / "PANEL_B_FINAL_HELDOUT_METRICS.json"

    if canonical_release.read_bytes() != lineage_release.read_bytes():
        raise RuntimeError("canonical final release manifest differs from frozen lineage")
    if canonical_metrics.read_bytes() != lineage_metrics.read_bytes():
        raise RuntimeError("canonical Panel-B metrics differ from frozen lineage")

    release = _load_json(canonical_release)
    metrics = _load_json(canonical_metrics)

    if release.get("status") != "PASS":
        raise RuntimeError("final release status is not PASS")
    if release.get("runtime_authorized") is not True:
        raise RuntimeError("runtime is not authorized by the frozen release")
    if release.get("final_metrics_scope") != "PANEL_B_FINAL_HELDOUT":
        raise RuntimeError("unexpected final metric scope")
    if release.get("post_panel_b_tuning_permitted") is not False:
        raise RuntimeError("post-Panel-B tuning must remain forbidden")

    ranker = metrics["frozen_five_ebm_ranker"]
    baseline = metrics["panel_a_action_stage_only_baseline"]
    bootstrap = metrics["paired_case_bootstrap"]

    expected = {
        "panel_b_case_count": 150,
        "real_external_review_record_count": 557,
        "ndcg_at_3": 0.9526603067902532,
        "baseline_ndcg_at_3": 0.8275943281032121,
        "mean_delta": 0.12466302441561493,
        "ci_low": 0.09508467988207753,
        "ci_high": 0.15361541252930452,
        "invalid_action_rate": 0.0,
    }

    observed = {
        "panel_b_case_count": metrics["panel_b_case_count"],
        "real_external_review_record_count": metrics["real_external_review_record_count"],
        "ndcg_at_3": ranker["ndcg_at_3"],
        "baseline_ndcg_at_3": baseline["ndcg_at_3"],
        "mean_delta": bootstrap["mean_full_minus_baseline_ndcg_at_3"],
        "ci_low": bootstrap["ci_low_95"],
        "ci_high": bootstrap["ci_high_95"],
        "invalid_action_rate": ranker["invalid_action_rate"],
    }
    if observed != expected:
        raise RuntimeError(f"final evidence mismatch: {observed!r}")

    print("FINAL_RECOMMENDATION_RELEASE=PASS")
    print("PANEL_B_RERUN=NOT_PERFORMED")


if __name__ == "__main__":
    main()
