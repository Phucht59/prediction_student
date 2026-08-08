"""Verify the frozen final recommendation evidence without rerunning Panel B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "artifacts" / "recommend_hybrid" / "final"
EXPECTED_SOURCE_COMMIT = "17b519b22e8b69c875d27547d097e6d3b76bc404"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    release = _json(FINAL / "release" / "FINAL_RELEASE_MANIFEST.json")
    metrics_path = FINAL / "heldout" / "PANEL_B_FINAL_HELDOUT_METRICS.json"
    reviews_path = FINAL / "heldout" / "panel_b_real_external_reviews_frozen.jsonl"
    scores_path = FINAL / "heldout" / "panel_b_final_heldout_scores.parquet"
    panel_b_manifest_path = FINAL / "heldout" / "PANEL_B_FINAL_HELDOUT_MANIFEST.json"
    metrics = _json(metrics_path)
    migration = _json(FINAL / "MIGRATION_MANIFEST.json")

    if release.get("status") != "PASS":
        raise RuntimeError("frozen final release status is not PASS")
    if release.get("runtime_authorized") is not True:
        raise RuntimeError("frozen final release does not authorize runtime")
    if release.get("final_metrics_scope") != "PANEL_B_FINAL_HELDOUT":
        raise RuntimeError("unexpected final metric scope")
    if release.get("post_panel_b_tuning_permitted") is not False:
        raise RuntimeError("post-Panel-B tuning must remain forbidden")
    if release.get("simulator_language") != "model-implied risk delta":
        raise RuntimeError("simulator language contract changed")
    if migration.get("scientific_source_commit") != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("unexpected scientific source commit")

    lineage = release["lineage_sha256"]
    observed_hashes = {
        "panel_b_final_metrics": _sha256(metrics_path),
        "panel_b_frozen_reviews": _sha256(reviews_path),
        "panel_b_scores": _sha256(scores_path),
        "panel_b_final_manifest": _sha256(panel_b_manifest_path),
    }
    for name, observed in observed_hashes.items():
        if observed != lineage[name]:
            raise RuntimeError(f"frozen evidence hash mismatch: {name}")

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
        raise RuntimeError(f"final metric mismatch: {observed!r}")

    print("FINAL_RECOMMENDATION_RELEASE=PASS")
    print(f"SCIENTIFIC_SOURCE_COMMIT={EXPECTED_SOURCE_COMMIT}")
    print("PANEL_B_RERUN=NOT_PERFORMED")


if __name__ == "__main__":
    main()
