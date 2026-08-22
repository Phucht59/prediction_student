"""Research-only roots. Serving code is not imported for writes."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KLTN_SPLITS = Path(r"C:\hufit\kltn\artifacts\hybrid\phase1\splits")
PHASE2_CACHE = ROOT / "test_lab" / "artifacts" / "hybrid_vnext" / "phase2" / "cache"
PHASE4 = ROOT / "test_lab" / "artifacts" / "hybrid_vnext" / "phase4"
RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts" / "research" / "kltn_science_fix"
REP = ROOT / "reports" / "research" / "hybrid_superiority_v2"
FIG = REP / "figures"
CH = REP / "chapters"
RUN = ART / "runs"
CKPT = ART / "checkpoints"
OOF = ART / "oof"
SPLIT = ART / "splits"
SERVING_CKPT = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data"

EXPECTED_SPLIT = {
    "uci_inner": "ad8f44e5931d652e353d9d9ebe7b0e840eca3d895243b92d57deb0b3b6e02ae8",
    "oulad_inner": "8559efcf156bcb05eb0a2bdf9e945d54f3989358d8f15064dab1204cd872650c",
}


def ensure() -> None:
    for path in (ART, REP, FIG, CH, RUN, CKPT, OOF, SPLIT):
        path.mkdir(parents=True, exist_ok=True)
