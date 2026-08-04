"""Run exact ablations only against the current completed full-grid authority."""
from __future__ import annotations

import json

import run_exact_ablation as exact
from postsearch_authority import atomic_json, current_model_authority, prepare_namespace


def main() -> None:
    authority = current_model_authority()
    prepare_namespace(
        exact.ABLATION_OUT,
        exact.MARKER,
        "ablations_stale_model_archive",
        authority,
    )
    exact.archive_reduced_budget_ablations_once = lambda: None
    exact.main()
    payload = json.loads(exact.MARKER.read_text(encoding="utf-8"))
    payload.update(authority)
    atomic_json(exact.MARKER, payload)


if __name__ == "__main__":
    main()
