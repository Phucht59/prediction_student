from pathlib import Path


ADVICE_RULES = {
    "absences": "Reduce absences and attend class more regularly.",
    "studytime": "Increase weekly study time.",
    "failures": "Review subjects that were failed before.",
    "G1": "Improve early semester scores.",
    "G2": "Improve mid semester scores.",
    "raisedhands": "Participate more in class activities.",
    "VisitedResources": "Use learning materials more often.",
    "StudentAbsenceDays": "Improve attendance.",
}


def create_study_advice(important_features: list[str]) -> list[str]:
    advice = []

    for feature in important_features:
        for keyword, message in ADVICE_RULES.items():
            if keyword.lower() in feature.lower() and message not in advice:
                advice.append(message)

    if not advice:
        advice.append("No specific advice rule matched the most important features.")

    return advice


def save_study_advice(advice_lines: list[str], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as file:
        for line in advice_lines:
            file.write(f"- {line}\n")

