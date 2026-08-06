"""Audit whether assessment timeliness is distinct enough for future taxonomy work."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_v3.oulad_data import build_canonical_bundle  # noqa: E402
from src.pipelines import oulad  # noqa: E402
from src.recommend_hybrid.final.actions import canonical_action_id  # noqa: E402

DEFAULT_LABELS = ROOT / "artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/v2/assessment_timeliness_audit.json"
DEFAULT_REPORT = ROOT / "reports/recommend_hybrid/v2/ASSESSMENT_TIMELINESS_AUDIT.md"
STAGE_SOURCE = {
    "EARLY_20": "E1_EARLY_20PCT",
    "EARLY_35": "E2_EARLY_35PCT",
    "MIDDLE_50": "M1_MIDDLE_50PCT",
    "LATE_75": "L1_LATE_75PCT",
}


def _assessment_completion_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame.loc[
        frame["dataset"].eq("oulad")
        & frame["silver_status"].eq("RETAINED")
        & frame["stage"].isin(STAGE_SOURCE)
    ].copy()
    canonical: list[str | None] = []
    for value in frame["action_id"]:
        try:
            canonical.append(canonical_action_id(value))
        except ValueError:
            canonical.append(None)
    frame["canonical_action"] = canonical
    frame = frame.loc[frame["canonical_action"].eq("ASSESSMENT_COMPLETION")].copy()
    frame["record_id"] = frame["student_key"].astype(str)
    frame["completion_positive"] = pd.to_numeric(
        frame["silver_label"],
        errors="coerce",
    ).fillna(0).gt(0).astype(np.int8)
    return frame.sort_values(
        ["record_id", "stage", "silver_confidence"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates(["record_id", "stage"], keep="first")


def run(labels_path: Path, output_path: Path, report_path: Path) -> dict[str, object]:
    bundle = build_canonical_bundle()
    base_index = {name: index for index, name in enumerate(oulad.BASE_CHANNELS)}
    rows: list[pd.DataFrame] = []
    for stage, source in STAGE_SOURCE.items():
        data = bundle.stages[source]
        base = data.sequence[:, :, : len(oulad.BASE_CHANNELS)]
        observed = np.arange(base.shape[1])[None, :] < data.lengths[:, None]
        submissions = (base[:, :, base_index["submitted_assessment_count"]] * observed).sum(axis=1)
        late = (base[:, :, base_index["late_submission_count"]] * observed).sum(axis=1)
        rate = np.divide(late, np.maximum(submissions, 1.0))
        rows.append(
            pd.DataFrame(
                {
                    "record_id": data.frame["base_record_id"].astype(str),
                    "student_id": data.frame["id_student"].astype(str),
                    "stage": stage,
                    "submission_count": submissions,
                    "late_submission_count": late,
                    "late_submission_rate": rate,
                    "timeliness_candidate_positive": (
                        (submissions > 0) & (rate >= 0.25)
                    ).astype(np.int8),
                }
            )
        )
    candidate = pd.concat(rows, ignore_index=True)
    completion = _assessment_completion_labels(labels_path)
    merged = candidate.merge(
        completion.loc[:, ["record_id", "stage", "completion_positive"]],
        on=["record_id", "stage"],
        how="left",
        validate="one_to_one",
    )
    merged["completion_positive"] = merged["completion_positive"].fillna(0).astype(np.int8)
    if merged["timeliness_candidate_positive"].nunique() < 2 or merged["completion_positive"].nunique() < 2:
        phi = 0.0
    else:
        phi = float(
            abs(
                np.corrcoef(
                    merged["timeliness_candidate_positive"],
                    merged["completion_positive"],
                )[0, 1]
            )
        )
    stage_rows: list[dict[str, object]] = []
    for stage, frame in merged.groupby("stage", sort=True):
        positive = int(frame["timeliness_candidate_positive"].sum())
        stage_rows.append(
            {
                "stage": str(stage),
                "rows": int(len(frame)),
                "with_any_submission": int(frame["submission_count"].gt(0).sum()),
                "candidate_positive": positive,
                "candidate_positive_rate": float(positive / len(frame)),
                "minimum_support_pass": positive >= 30,
            }
        )
    data_support = sum(bool(row["minimum_support_pass"]) for row in stage_rows) >= 3 and phi < 0.90
    payload = {
        "status": "DATA_SUPPORT_PRESENT" if data_support else "INSUFFICIENT_DATA_SUPPORT",
        "candidate": "ASSESSMENT_TIMELINESS",
        "definition": "at least one observed submission and late-submission rate >= 0.25 by cutoff",
        "stages": stage_rows,
        "absolute_phi_with_assessment_completion_label": phi,
        "maximum_nonredundancy_phi": 0.90,
        "data_support_present": data_support,
        "activated_as_learned_action": False,
        "activation_requirements": [
            "expert interpretation review",
            "independent label specification",
            "nonredundancy confirmation",
            "validation-only model selection",
        ],
        "claim_boundary": "TAXONOMY_CANDIDATE_AUDIT_NOT_ACTION_EFFECT_EVIDENCE",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Assessment Timeliness Research-Candidate Audit",
        "",
        f"Status: **{payload['status']}**",
        "",
        "Completion and on-time submission are different behaviours. This audit checks prevalence and redundancy only; it does not activate a sixth learned action.",
        "",
        "| Stage | Rows | Any submission | Candidate positive | Positive rate | Support gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in stage_rows:
        lines.append(
            "| {stage} | {rows} | {with_any_submission} | {candidate_positive} | {candidate_positive_rate:.4f} | {minimum_support_pass} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Absolute phi with `ASSESSMENT_COMPLETION`: **{phi:.4f}**.",
            "",
            "Activation remains false until expert review and an independent label protocol are available.",
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
    payload = run(args.labels, args.output, args.report)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
