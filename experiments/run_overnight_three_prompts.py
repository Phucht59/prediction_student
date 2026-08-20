"""Run the three evaluation/improvement prompts in order. Fast I/O, CUDA training, logs to disk."""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = ROOT / "artifacts" / "experiments" / "overnight.log"
OUT = ROOT / "artifacts" / "experiments"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).strftime('%H:%M:%SZ')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def step(name: str, fn) -> None:
    log(f"START {name}")
    try:
        fn()
        log(f"OK {name}")
    except Exception:
        log(f"FAIL {name}\n{traceback.format_exc()}")
        raise


def prompt1_and_2() -> None:
    from experiments.validation.leakage_overfit import main as leak
    from experiments.validation.render_figures import main as figs

    leak()
    figs()


def prompt3_uci() -> None:
    from experiments.cv5 import run_cv5

    sys.argv = ["run_cv5.py", "--dataset", "uci", "--folds", "0,1,2,3,4"]
    run_cv5.main()


def prompt3_oulad() -> None:
    from experiments.cv5 import run_cv5

    sys.argv = ["run_cv5.py", "--dataset", "oulad", "--folds", "0,1,2,3,4"]
    run_cv5.main()


def final_index() -> None:
    report = ROOT / "reports" / "prediction" / "experiments" / "OVERNIGHT_THREE_PROMPTS.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    leak = OUT / "validation" / "LEAKAGE_OVERFIT_AUDIT.json"
    figs = list((OUT / "validation" / "figures").glob("*.png")) if (OUT / "validation" / "figures").is_dir() else []
    cv = OUT / "cv5" / "hybrid_vs_baselines.csv"
    lines = [
        "# Overnight run: three prompts",
        "",
        "1. Scientific evaluation + leakage/overfit (frozen splits from `codex/backup-hybrid-phase8-2026-08-17`).",
        "2. Ablation / error / subgroup / fairness / SHAP-fusion figures.",
        "3. 5-fold CV Hybrid CNN–BiLSTM vs LR/RF; outer fold 0 never in train/STOP.",
        "",
        f"- leakage audit exists: {leak.is_file()}",
        f"- figures: {len(figs)}",
        f"- 5-fold table exists: {cv.is_file()}",
        "",
    ]
    if leak.is_file():
        payload = json.loads(leak.read_text(encoding="utf-8"))
        lines += ["## Leakage verdict", "", "```", json.dumps(payload.get("verdict"), indent=2), "```", ""]
    if cv.is_file():
        lines += ["## Hybrid vs baselines (5-fold)", "", "```", cv.read_text(encoding="utf-8"), "```", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {report}")


def main() -> int:
    log("OVERNIGHT begin")
    step("P1+P2 leakage+figures", prompt1_and_2)
    step("P3 UCI 5-fold", prompt3_uci)
    step("P3 OULAD 5-fold", prompt3_oulad)
    step("index", final_index)
    log("OVERNIGHT done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
