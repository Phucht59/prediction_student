import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.recommend_hybrid.weak_supervision.candidate_generation import build_candidates
from src.recommend_hybrid.weak_supervision.lf_registry import write_registry
if __name__ == "__main__":
    frame = build_candidates()
    write_registry(ROOT / "artifacts/recommend_hybrid/scientific_labeling/lf_registry.yaml")
    print(f"SCIENTIFIC_CANDIDATES_BUILT rows={len(frame)}")
