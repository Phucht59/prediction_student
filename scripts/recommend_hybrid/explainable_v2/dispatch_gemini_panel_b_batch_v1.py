"""One-shot, provenance-preserving Gemini dispatcher for held-out Panel B."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from scripts.recommend_hybrid.explainable_v2 import (
    dispatch_gemini_panel_a_batch01_v3 as core,
)


FROZEN_PANEL_A_MANIFEST_SHA256 = (
    "4a9af5a21ace08f13bfdc09504f19c1a9b5616d85df4151379815116d28eb5db"
)
FROZEN_PANEL_A_REVIEWS_SHA256 = (
    "4a4871426880bdcd1257dc15c29a36c23de34481f07be68d8e5095dc20efefb9"
)
DEVELOPMENT_FREEZE_PATH = (
    core.ROOT
    / "artifacts/recommend_hybrid/explainable_v2/frozen/development_v2"
    / "DEVELOPMENT_FREEZE_MANIFEST.json"
)
PANEL_A_FREEZE_DIR = (
    core.ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/frozen/panel_a_v1"
)


def validate_development_authority() -> None:
    freeze = json.loads(DEVELOPMENT_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS":
        raise RuntimeError("Development freeze is not PASS")
    if freeze.get("panel_b_touched") is not False:
        raise RuntimeError("Development freeze records Panel B as touched")
    if freeze.get("runtime_authorized") is not False:
        raise RuntimeError("Development freeze runtime_authorized must be false")

    manifest_path = PANEL_A_FREEZE_DIR / "PANEL_A_FREEZE_MANIFEST.json"
    reviews_path = PANEL_A_FREEZE_DIR / "panel_a_external_reviews_frozen.jsonl"
    if core.sha256_bytes(manifest_path.read_bytes()) != FROZEN_PANEL_A_MANIFEST_SHA256:
        raise RuntimeError("Frozen Panel-A review manifest changed")
    if core.sha256_bytes(reviews_path.read_bytes()) != FROZEN_PANEL_A_REVIEWS_SHA256:
        raise RuntimeError("Frozen Panel-A review records changed")
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise RuntimeError("GEMINI_API_KEY is missing")


def validate_panel_b_source_gate(cases: list[dict[str, Any]]) -> None:
    manifest = json.loads(core.CASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("case_export_classification") != core.EXPECTED_CASE_EXPORT_CLASSIFICATION:
        raise RuntimeError("Panel B source is not verified V4 query-level lineage")
    if manifest.get("query_level_evidence_invariant_across_actions") is not True:
        raise RuntimeError("Query-level evidence invariance is not verified")
    if int(manifest.get("panel_b_case_count", -1)) != 150:
        raise RuntimeError("V4 case manifest Panel B count is not 150")
    if manifest.get("zero_student_overlap") is not True:
        raise RuntimeError("V4 Panel A/B student overlap gate failed")
    if manifest.get("zero_query_overlap") is not True:
        raise RuntimeError("V4 Panel A/B query overlap gate failed")
    if manifest.get("runtime_authorized") is not False:
        raise RuntimeError("V4 runtime_authorized must remain false")
    if len(cases) != 50:
        raise RuntimeError("Each Panel B provider batch must contain exactly 50 cases")
    if any(case.get("panel_id") != "PANEL_B" for case in cases):
        raise RuntimeError("Panel B provider batch contains a non-Panel-B case")


def configure_batch(batch_number: int) -> None:
    if batch_number < 1 or batch_number > 3:
        raise ValueError("Panel B batch_number must be between 1 and 3")

    suffix = f"{batch_number:02d}"
    batch_id = f"panel_b_batch_{suffix}"
    source = (
        core.ROOT
        / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts"
        / "panel_b_request_batches"
        / f"batch_{suffix}.jsonl"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Panel B source batch does not exist: {source}")

    batch_dir = core.ENVELOPE_ROOT / core.PROVIDER / batch_id
    core.EXPECTED_PANEL_ID = "PANEL_B"
    core.EXPECTED_PANEL_CASE_COUNT = 150
    core.BATCH_ID = batch_id
    core.SOURCE_BATCH_PATH = source
    core.BATCH_DIR = batch_dir
    core.RAW_REQUEST_DIR = batch_dir / "raw_requests"
    core.RAW_RESPONSE_DIR = batch_dir / "raw_responses"
    core.CASE_STATE_DIR = batch_dir / "case_state"
    core.REQUEST_ENVELOPE_PATH = batch_dir / "request_envelope.json"
    core.RESPONSE_ENVELOPE_PATH = batch_dir / "response_envelope.json"
    core.BATCH_MANIFEST_PATH = batch_dir / "batch_manifest.json"
    core.NORMALIZED_PATH = batch_dir / "normalized_records.jsonl"
    core.CHECKSUMS_PATH = batch_dir / "checksums.sha256"
    core.BATCH_SNAPSHOT_PATH = batch_dir / "request_batch_snapshot.jsonl"
    core.IMPORT_RAW_PATH = core.IMPORT_RAW_DIR / f"{batch_id}_gemini.jsonl"
    core.validate_v4_source_gate = validate_panel_b_source_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-number", type=int, required=True, choices=range(1, 4))
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument("--inter-request-delay", type=float, default=5.1)
    args = parser.parse_args()

    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.base_delay < 0 or args.inter_request_delay < 0:
        raise SystemExit("Delay values must be >= 0")

    configure_batch(args.batch_number)
    if args.execute:
        validate_development_authority()
    result = core.dispatch(
        model=args.model,
        execute=args.execute,
        limit=None,
        max_attempts=args.max_attempts,
        base_delay=args.base_delay,
        inter_request_delay=args.inter_request_delay,
    )
    print(f"PANEL_B_BATCH_NUMBER={args.batch_number}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
