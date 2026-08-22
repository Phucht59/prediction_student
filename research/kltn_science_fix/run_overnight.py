"""CPU analyses first, then GPU ablation. Research folder only."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.kltn_science_fix.paths import REP, ensure


def _run(name, fn) -> None:
    print("=" * 60, name, flush=True)
    try:
        fn()
        print("OK", name, flush=True)
    except Exception:
        print("FAIL", name, flush=True)
        traceback.print_exc()
        (REP / f"FAIL_{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")


def main() -> None:
    ensure()
    from research.kltn_science_fix.run_stats import main as stats
    from research.kltn_science_fix.run_label_split import main as labels
    from research.kltn_science_fix.run_fairness import main as fair
    from research.kltn_science_fix.run_gates import main as gates
    from research.kltn_science_fix.run_curves import main as curves
    from research.kltn_science_fix.run_spearman import main as spearman
    from research.kltn_science_fix.run_survival import main as survival
    from research.kltn_science_fix.run_ablation import main as ablation

    for name, fn in [
        ("survival", survival),
        ("ablation", ablation),
    ]:
        _run(name, fn)


if __name__ == "__main__":
    main()
