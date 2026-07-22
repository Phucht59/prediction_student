"""Formatting helpers shared by final reports."""


def display(value: object, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"
