"""Audit Recommendation V2 action coverage and redundancy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.final.actions import canonical_action_id  # noqa: E402
from src.recommend_hybrid.v2.taxonomy import (  # noqa: E402
    audit_taxonomy,
    taxonomy_manifest,
)

DEFAULT_LABELS = ROOT / "artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/v2/taxonomy_audit.json"
DEFAULT_REPORT = ROOT / "reports/recommend_hybrid/v2/ACTION_TAXONOMY_AUDIT.md"
STAGES = ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")


def run(labels_path: Path, output_path: Path, report_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(labels_path)
    required = {
        "dataset",
        "student_key",
        "stage",
        "action_id",
        "silver_label",
        "silver_status",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"silver labels are missing columns: {missing}")
    frame = frame.loc[
        frame["dataset"].eq("oulad")
        & frame["stage"].isin(STAGES)
        & frame["silver_status"].eq("RETAINED")
    ].copy()
    canonical: list[str | None] = []
    for value in frame["action_id"]:
        try:
            canonical.append(canonical_action_id(value))
        except ValueError:
            canonical.append(None)
    frame["action_id"] = canonical
    frame = frame.loc[frame["action_id"].notna()].copy()
    frame["record_id"] = frame["student_key"].astype(str)
    frame = frame.sort_values(
        ["record_id", "stage", "action_id", "silver_confidence"],
        ascending=[True, True, True, False],
        kind="stable",
    ).drop_duplicates(["record_id", "stage", "action_id"], keep="first")
    audit = audit_taxonomy(frame, stages=STAGES)
    payload = {
        "status": audit["status"],
        "taxonomy": taxonomy_manifest(),
        "audit": audit,
        "source": str(labels_path.relative_to(ROOT)),
        "expert_activation_required_for_research_candidates": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Recommendation V2 Action Taxonomy Audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        "The five learned slots represent observable behavioural families. Governance routes and human escalation remain outside the learned ranker.",
        "",
        "| Action | Rows | Positive rows | Positive rate | All stages | Support gate |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in audit["actions"]:
        lines.append(
            "| {action_id} | {rows} | {positive_rows} | {positive_rate:.4f} | {all_stages_represented} | {minimum_positive_pass} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Maximum pairwise absolute phi: **{audit['maximum_absolute_phi']:.4f}**.",
            "",
            "`ASSESSMENT_TIMELINESS` remains a research candidate and is not activated as a sixth learned action without expert review.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run(args.labels, args.output, args.report)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
