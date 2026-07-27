"""Corrected final recommendation validation constants."""

RECORDS = 15378
GENERATED = 10953
PARTIAL_EVIDENCE = 1209
ABSTAINED = 3216


def generated_or_partial_rate() -> float:
    return (GENERATED + PARTIAL_EVIDENCE) / RECORDS
