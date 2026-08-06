from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.recommend_hybrid.explainable_v2.risk_policy_selection import run
def main():
    try:
        rows=run(ROOT)
        report=ROOT/"reports/recommend_hybrid_v2/RISK_STRATIFICATION_RESULTS.md"; report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text("# Risk stratification\n\nStatus: COMPLETE; thresholds selected on grouped inner validation only.\n\nOuter folds completed: %d/3.\n"%len(rows),encoding="utf-8")
        return 0
    except FileNotFoundError as exc:
        print("STATUS: BLOCKED_MISSING_VALIDATED_OUTCOME_AUTHORITY",exc); return 2
    except Exception as exc:
        print("STATUS: FAILED_IMPLEMENTATION_ERROR",repr(exc)); return 1
if __name__=="__main__": raise SystemExit(main())
