"""Convert Snorkel soft probabilities into conservative silver-label records."""
from __future__ import annotations

import numpy as np

from .labels import TargetLabel


def apply_silver_policy(frame, model, *, confidence_threshold: float, minimum_families: int):
    from .label_model import vote_matrix
    from .lf_registry import registry
    matrix = vote_matrix(frame); probabilities = model.predict_proba(matrix)
    families = [lf.lf_family for lf in registry()]
    rows = frame.copy(); labels=[]; statuses=[]; coverage=[]; conflict=[]
    for source, votes, probs in zip(rows.to_dict("records"), matrix, probabilities):
        active_families = {families[i] for i, vote in enumerate(votes) if vote != -1}
        counts = np.bincount(votes[votes >= 0], minlength=3)
        has_conflict = int((counts > 0).sum() > 1)
        cap = source["action_status"] == "INSUFFICIENT_EVIDENCE"
        label = int(np.argmax(probs))
        if source["human_review_required"] and label != TargetLabel.INAPPROPRIATE:
            label = TargetLabel.CONDITIONAL
        if cap and label != TargetLabel.INAPPROPRIATE:
            label = TargetLabel.CONDITIONAL
        accepted = len(active_families) >= minimum_families and float(np.max(probs)) >= confidence_threshold
        labels.append(label if accepted else None); statuses.append("RETAINED" if accepted else "ABSTAIN"); coverage.append(len(active_families)); conflict.append(has_conflict)
    rows["lf_votes"] = [list(map(int, item)) for item in matrix]; rows["lf_family_coverage"] = coverage; rows["lf_conflict"] = conflict
    rows["silver_prob_0"] = probabilities[:,0]; rows["silver_prob_1"] = probabilities[:,1]; rows["silver_prob_2"] = probabilities[:,2]
    rows["silver_expected_relevance"] = probabilities @ np.asarray([0.0,1.0,2.0]); rows["silver_label"] = labels; rows["silver_confidence"] = probabilities.max(axis=1); rows["silver_status"] = statuses
    return rows


__all__ = ["apply_silver_policy"]
