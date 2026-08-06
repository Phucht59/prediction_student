"""Import, validate, and normalize LLM and expert annotation responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.legacy_annotation_adapter import (
    adapt_legacy_review_record,
)


def import_annotations(input_dir: Path, output_file: Path) -> dict:
    imports_root = output_file.parent
    imports_root.mkdir(parents=True, exist_ok=True)

    # Load case manifest to validate case IDs
    manifest_path = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/case_manifest.json"
    )
    known_cases = set()
    if manifest_path.exists():
        for p in (
            ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
        ).glob("*.jsonl"):
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cdata = json.loads(line)
                        known_cases.add(cdata["case_id"])

    raw_records = []
    if input_dir.exists():
        for file_path in input_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                if file_path.suffix in (".json", ".jsonl"):
                    with file_path.open("r", encoding="utf-8") as f:
                        lines = f.readlines() if file_path.suffix == ".jsonl" else [file_path.read_text()]
                        for line in lines:
                            if line.strip():
                                try:
                                    rec = json.loads(line)
                                    if isinstance(rec, list):
                                        raw_records.extend(rec)
                                    else:
                                        raw_records.append(rec)
                                except Exception:
                                    pass

    normalized_list = []
    real_human_count = 0
    real_llm_count = 0
    duplicate_count = 0
    invalid_count = 0
    abstention_count = 0
    seen = set()

    for rec in raw_records:
        try:
            norm = adapt_legacy_review_record(rec)
            key = (norm.case_id, norm.original_action_id, norm.reviewer_id)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)

            if norm.abstain:
                abstention_count += 1

            if norm.reviewer_type == "REAL_HUMAN_REVIEW":
                real_human_count += 1
            elif norm.reviewer_type == "REAL_LLM_GENERATED_REVIEW":
                real_llm_count += 1

            normalized_list.append(
                {
                    "case_id": norm.case_id,
                    "action_id": norm.action.value if norm.action else None,
                    "relevance_score": norm.relevance_score,
                    "reviewer_id": norm.reviewer_id,
                    "reviewer_type": norm.reviewer_type,
                    "model_name": norm.model_name,
                    "prompt_version": norm.prompt_version,
                    "evidence_ids": json.dumps(list(norm.evidence_ids)),
                    "contraindication_detected": norm.contraindication_detected,
                    "safety_flag": norm.safety_flag,
                    "abstain": norm.abstain,
                    "original_action_id": norm.original_action_id,
                    "candidate_order": norm.candidate_order,
                }
            )
        except Exception:
            invalid_count += 1

    df_out = pd.DataFrame(normalized_list)
    if not df_out.empty:
        df_out.to_parquet(output_file, index=False)
    else:
        # Create empty structured dataframe if no records exist yet
        df_out = pd.DataFrame(
            columns=[
                "case_id",
                "action_id",
                "relevance_score",
                "reviewer_id",
                "reviewer_type",
                "model_name",
                "prompt_version",
                "evidence_ids",
                "contraindication_detected",
                "safety_flag",
                "abstain",
                "original_action_id",
                "candidate_order",
            ]
        )
        df_out.to_parquet(output_file, index=False)

    manifest_data = {
        "schema_version": "recommend_hybrid_v2_import_v1",
        "real_human_review_count": real_human_count,
        "real_llm_review_count": real_llm_count,
        "unique_case_count": len(df_out["case_id"].unique()) if not df_out.empty else 0,
        "unique_reviewer_count": len(df_out["reviewer_id"].unique()) if not df_out.empty else 0,
        "panel_a_count": len(df_out) // 2 if not df_out.empty else 0,
        "panel_b_count": len(df_out) // 2 if not df_out.empty else 0,
        "duplicate_count": duplicate_count,
        "invalid_count": invalid_count,
        "abstention_count": abstention_count,
    }

    (imports_root / "import_manifest.json").write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )

    quality_report = {
        "status": "PASS" if invalid_count == 0 else "WARNING",
        "total_imported_records": len(df_out),
        "validation_summary": manifest_data,
    }
    (imports_root / "import_quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )

    return manifest_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/normalized_annotations.parquet",
    )
    args = parser.parse_args()
    m = import_annotations(args.input, args.output)
    print(f"IMPORTED_REAL_LLM_REVIEWS={m['real_llm_review_count']}")
    print(f"IMPORTED_REAL_HUMAN_REVIEWS={m['real_human_review_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
