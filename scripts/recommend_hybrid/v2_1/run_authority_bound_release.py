"""Run the fail-closed release gate only with evidence bound to the current full-grid model."""
from __future__ import annotations

import corrected_release
import run_exact_ablation as ablation
import run_exact_negative_controls as controls
from postsearch_authority import assert_bound, current_model_authority


def main() -> None:
    authority = current_model_authority()
    assert_bound(controls.MARKER, authority, "Negative controls")
    assert_bound(ablation.MARKER, authority, "Ablations")
    corrected_release.main()


if __name__ == "__main__":
    main()
