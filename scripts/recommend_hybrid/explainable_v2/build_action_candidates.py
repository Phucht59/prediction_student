from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from src.recommend_hybrid.explainable_v2.candidate_builder import build
def main():
    try: build(ROOT); print("STATUS: COMPLETE"); return 0
    except FileNotFoundError as exc: print("STATUS: BLOCKED",exc); return 2
    except RuntimeError as exc: print(str(exc)); return 2
    except Exception as exc: print("STATUS: FAILED_IMPLEMENTATION_ERROR",repr(exc)); return 1
if __name__=="__main__": raise SystemExit(main())
