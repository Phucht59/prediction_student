"""Non-causal safety checks for recommendation plans."""


def validate_actions(actions: list[str]) -> dict[str, object]:
    normalized = [action.strip() for action in actions if action.strip()]
    return {
        "actions": normalized,
        "duplicates": len(normalized) - len(set(normalized)),
        "valid": len(normalized) == len(set(normalized)),
    }
