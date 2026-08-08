"""Select and freeze the final four-status safety router on Panel A only."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/recommend_hybrid/explainable_v2.yaml"
LABEL_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
    / "probabilistic_relevance_labels.parquet"
)
OOF_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
    / "panel_a_ebm_oof_predictions.parquet"
)
CANDIDATE_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
)
RELEASE_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/release_gates/panel_a_v1"
    / "PANEL_A_RELEASE_GATES.json"
)
RANKER_FREEZE_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/frozen/ranker_panel_a_v2"
    / "RANKER_PANEL_A_FREEZE_MANIFEST.json"
)
OUTPUT_DIR = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/frozen/router_panel_a_v1"
)
MINIMUM_RECOMMEND_PRECISION = 0.98
FINAL_STATUSES = (
    "RECOMMEND",
    "INSUFFICIENT_EVIDENCE",
    "HUMAN_REVIEW",
    "NO_FEASIBLE_ACTION",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    if release.get("status") != "PASS":
        raise RuntimeError("PANEL_A_RELEASE_GATES_NOT_PASS")
    if release.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_CONTAMINATION_DETECTED")

    labels = pd.read_parquet(LABEL_PATH)
    oof = pd.read_parquet(OOF_PATH)
    candidates = pd.read_parquet(CANDIDATE_PATH)
    complete = labels.groupby("query_id")["retained_for_training"].all()
    query_ids = complete[complete].index
    data = labels[labels["query_id"].isin(query_ids)].merge(
        oof[["query_id", "action_id", "ebm_oof_score"]],
        on=["query_id", "action_id"],
        validate="one_to_one",
    )
    data = data.merge(
        candidates[["query_id", "action_id", "hybrid_uncertainty"]],
        on=["query_id", "action_id"],
        validate="one_to_one",
    )
    data = data[data["eligible"].astype(bool)].copy()
    data["public_score"] = np.clip(
        data["ebm_oof_score"].to_numpy(dtype=float) / 3.0,
        0.0,
        1.0,
    )
    data["label_conflict"] = np.clip(
        data["label_entropy"].to_numpy(dtype=float) / np.log(4.0),
        0.0,
        1.0,
    )

    query_rows = []
    for query_id, group in data.groupby("query_id", sort=True):
        ranked = group.sort_values(
            ["public_score", "action_id"],
            ascending=[False, True],
        )
        top1 = ranked.iloc[0]
        top2_score = float(ranked.iloc[1]["public_score"]) if len(ranked) > 1 else 0.0
        query_rows.append(
            {
                "query_id": str(query_id),
                "top1_score": float(top1["public_score"]),
                "margin": float(top1["public_score"] - top2_score),
                "hybrid_uncertainty": float(top1["hybrid_uncertainty"]),
                "label_conflict": float(top1["label_conflict"]),
                "top1_relevant": bool(float(top1["expected_relevance"]) >= 1.0),
            }
        )
    queries = pd.DataFrame(query_rows)
    if len(queries) != 299:
        raise RuntimeError(f"ROUTER_QUERY_COUNT={len(queries)} expected=299")

    router = config["safety_router"]
    grid = itertools.product(
        router["minimum_top1_score_grid"],
        router["minimum_margin_grid"],
        router["maximum_hybrid_uncertainty_grid"],
        router["maximum_label_conflict_grid"],
    )
    candidates_evaluated = []
    for minimum_score, minimum_margin, maximum_uncertainty, maximum_conflict in grid:
        insufficient = queries["top1_score"] < float(minimum_score)
        ambiguous = (
            (queries["margin"] < float(minimum_margin))
            | (queries["hybrid_uncertainty"] > float(maximum_uncertainty))
            | (queries["label_conflict"] > float(maximum_conflict))
        ) & ~insufficient
        recommend = ~(insufficient | ambiguous)
        recommend_count = int(recommend.sum())
        precision = (
            float(queries.loc[recommend, "top1_relevant"].mean())
            if recommend_count
            else 0.0
        )
        candidates_evaluated.append(
            {
                "minimum_top1_score": float(minimum_score),
                "minimum_top1_margin": float(minimum_margin),
                "maximum_hybrid_uncertainty": float(maximum_uncertainty),
                "maximum_label_conflict": float(maximum_conflict),
                "recommend_count": recommend_count,
                "recommend_coverage": float(recommend.mean()),
                "recommend_precision_relevance_ge_1": precision,
                "insufficient_evidence_count": int(insufficient.sum()),
                "human_review_count": int(ambiguous.sum()),
            }
        )

    admissible = [
        item
        for item in candidates_evaluated
        if item["recommend_count"] > 0
        and item["recommend_precision_relevance_ge_1"]
        >= MINIMUM_RECOMMEND_PRECISION
    ]
    if not admissible:
        raise RuntimeError("NO_ROUTER_CONFIG_MEETS_PANEL_A_PRECISION_CONSTRAINT")
    selected = max(
        admissible,
        key=lambda item: (
            item["recommend_coverage"],
            item["recommend_precision_relevance_ge_1"],
            item["minimum_top1_score"],
            item["minimum_top1_margin"],
            -item["maximum_hybrid_uncertainty"],
            -item["maximum_label_conflict"],
        ),
    )
    thresholds = {
        key: selected[key]
        for key in (
            "minimum_top1_score",
            "minimum_top1_margin",
            "maximum_hybrid_uncertainty",
            "maximum_label_conflict",
        )
    } | {
        "maximum_seed_disagreement": None,
        "maximum_ood_score": float(max(router["maximum_ood_score_grid"])),
    }
    threshold_hash = hashlib.sha256(
        json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    manifest = {
        "schema_version": "recommend_v2_router_freeze_v1",
        "status": "PASS",
        "scope": "PANEL_A_DEVELOPMENT",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "final_metrics_claimed": False,
        "public_route_statuses": list(FINAL_STATUSES),
        "semantics": {
            "NO_FEASIBLE_ACTION": "hard feasibility leaves zero actions",
            "INSUFFICIENT_EVIDENCE": "top feasible evidence is below the frozen sufficiency threshold",
            "HUMAN_REVIEW": "evidence is sufficient but ambiguity or uncertainty is too high",
            "RECOMMEND": "feasible, sufficient, and acceptably unambiguous",
        },
        "selection_panel": "A",
        "development_query_count": len(queries),
        "minimum_recommend_precision_constraint": MINIMUM_RECOMMEND_PRECISION,
        "selection_objective": "maximum_recommend_coverage_then_precision_within_locked_yaml_grids",
        "selected_thresholds": thresholds,
        "selected_thresholds_sha256": threshold_hash,
        "selected_development_operating_point": selected,
        "seed_disagreement_evidence": {
            "status": "UNAVAILABLE_IN_FROZEN_SOURCE_ARTIFACT",
            "threshold_applied": False,
            "zero_imputed": False,
        },
        "ood_evidence": {
            "selection_status": "UNAVAILABLE_FOR_PANEL_A_THRESHOLD_TUNING",
            "fail_closed_runtime_ceiling": thresholds["maximum_ood_score"],
        },
        "lineage": {
            "config_sha256": _sha256(CONFIG_PATH),
            "labels_sha256": _sha256(LABEL_PATH),
            "oof_sha256": _sha256(OOF_PATH),
            "candidate_sha256": _sha256(CANDIDATE_PATH),
            "release_gate_sha256": _sha256(RELEASE_PATH),
            "ranker_freeze_sha256": _sha256(RANKER_FREEZE_PATH),
        },
        "grid_candidate_count": len(candidates_evaluated),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    output_path = OUTPUT_DIR / "ROUTER_FREEZE_MANIFEST.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "checksums.sha256").write_text(
        f"{_sha256(output_path)}  {output_path.name}\n",
        encoding="utf-8",
    )
    print("ROUTER_STATUS_CONTRACT=" + ",".join(FINAL_STATUSES))
    print("SELECTED_THRESHOLDS=" + json.dumps(thresholds, sort_keys=True))
    print(
        "PANEL_A_DEVELOPMENT_RECOMMEND_COVERAGE="
        f"{selected['recommend_coverage']:.6f}"
    )
    print(
        "PANEL_A_DEVELOPMENT_RECOMMEND_PRECISION="
        f"{selected['recommend_precision_relevance_ge_1']:.6f}"
    )
    print("PANEL_B_TOUCHED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
