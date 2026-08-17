"""Phase 6B STOP-only threshold diagnostics; no model-selection side effects."""
from __future__ import annotations

import numpy as np

from src.hybrid.training.evaluation import binary_classification_metrics

THRESHOLD_GRID=tuple(round(value/100,2) for value in range(5,96))


def select_stop_threshold(target, score, grid=THRESHOLD_GRID):
    """Select threshold using STOP labels/scores only, with frozen tie breaks."""
    ranked=[]
    for threshold in grid:
        metrics=binary_classification_metrics(target,score,threshold=threshold)
        ranked.append((metrics["risk_f1"],metrics["risk_recall"],-abs(threshold-.5),-threshold,threshold))
    return max(ranked)[-1]


def stage_threshold_metrics(stop_target,stop_score,valid_target,valid_score):
    threshold=select_stop_threshold(stop_target,stop_score)
    return {"fixed_0_5":binary_classification_metrics(valid_target,valid_score,threshold=.5),
            "stop_selected":binary_classification_metrics(valid_target,valid_score,threshold=threshold),
            "selected_threshold":threshold}
