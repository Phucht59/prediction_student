"""Run exact retrained controls only against the current completed full-grid authority."""
from __future__ import annotations

import json

import run_exact_negative_controls as exact
from postsearch_authority import atomic_json, current_model_authority, prepare_namespace


def main() -> None:
    authority = current_model_authority()
    prepare_namespace(
        exact.CONTROL_OUT,
        exact.MARKER,
        "negative_controls_stale_model_archive",
        authority,
    )
    # Namespace preparation has already archived incompatible evidence and
    # created the authority-bound marker. Preserve partial batches for resume.
    exact.archive_reduced_budget_controls_once = lambda: None
    exact.main()
    payload = json.loads(exact.MARKER.read_text(encoding="utf-8"))
    payload.update(authority)
    atomic_json(exact.MARKER, payload)


if __name__ == "__main__":
    main()
