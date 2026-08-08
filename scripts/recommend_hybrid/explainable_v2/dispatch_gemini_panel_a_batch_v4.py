from __future__ import annotations

import argparse
import os

from scripts.recommend_hybrid.explainable_v2 import (
    dispatch_gemini_panel_a_batch01_v3 as core,
)


def configure_batch(batch_number: int) -> None:
    if batch_number < 1 or batch_number > 6:
        raise ValueError("Panel A batch_number must be between 1 and 6")

    batch_suffix = f"{batch_number:02d}"
    batch_id = f"panel_a_batch_{batch_suffix}"

    source_batch_path = (
        core.ROOT
        / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts"
        / "panel_a_request_batches"
        / f"batch_{batch_suffix}.jsonl"
    )
    if not source_batch_path.is_file():
        raise FileNotFoundError(
            f"Panel A source batch does not exist: {source_batch_path}"
        )

    batch_dir = core.ENVELOPE_ROOT / core.PROVIDER / batch_id

    core.BATCH_ID = batch_id
    core.SOURCE_BATCH_PATH = source_batch_path
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-number",
        type=int,
        required=True,
        choices=range(1, 7),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", core.DEFAULT_MODEL),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument("--inter-request-delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.base_delay < 0 or args.inter_request_delay < 0:
        raise SystemExit("Delay values must be >= 0")

    configure_batch(args.batch_number)
    print(f"GENERIC_PANEL_A_BATCH_NUMBER={args.batch_number}")

    return core.dispatch(
        model=args.model,
        execute=args.execute,
        limit=args.limit,
        max_attempts=args.max_attempts,
        base_delay=args.base_delay,
        inter_request_delay=args.inter_request_delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
