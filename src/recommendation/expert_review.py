"""Expert review state; missing labels remain explicit."""

PENDING_EXPERT_LABELS = "PENDING_EXPERT_LABELS"


def empty_expert_metrics() -> dict[str, object]:
    return {
        "status": PENDING_EXPERT_LABELS,
        "action_precision": None,
        "action_recall": None,
        "action_f1": None,
    }
