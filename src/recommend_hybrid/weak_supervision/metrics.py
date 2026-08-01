"""Non-performance diagnostics for LF and silver-label quality."""
from __future__ import annotations

import numpy as np


def quality_metrics(frame) -> dict:
    retained = frame[frame.silver_status == "RETAINED"]
    return {"candidate_rows": int(len(frame)), "retained": int(len(retained)), "abstained": int((frame.silver_status == "ABSTAIN").sum()), "coverage": float(len(retained) / len(frame)) if len(frame) else 0.0, "mean_confidence": float(frame.silver_confidence.mean()), "conflict_rate": float(frame.lf_conflict.mean()), "label_distribution": {str(label): int((retained.silver_label == label).sum()) for label in (0,1,2)}, "human_review_action_rate": float(frame.human_review_required.mean()), "evidence_gap_rate": float((frame.action_status == "INSUFFICIENT_EVIDENCE").mean())}


__all__ = ["quality_metrics"]
