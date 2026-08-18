"""Produce agreement diagnostics after both normalized tables exist."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _dist(frame: pd.DataFrame) -> dict:
    counts = frame["label"].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _agreement(merged: pd.DataFrame) -> tuple[int, int, float | None]:
    exact = int((merged["label_gemma"] == merged["label_gemini"]).sum())
    disagreement = int(len(merged) - exact)
    numeric = merged[~merged["label_gemma"].eq("ABSTAIN") & ~merged["label_gemini"].eq("ABSTAIN")].copy()
    kappa = None
    numeric["label_gemma"] = pd.to_numeric(numeric["label_gemma"], errors="coerce")
    numeric["label_gemini"] = pd.to_numeric(numeric["label_gemini"], errors="coerce")
    numeric = numeric.dropna(subset=["label_gemma", "label_gemini"])
    if len(numeric) and numeric["label_gemma"].nunique() > 1 and numeric["label_gemini"].nunique() > 1:
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = float(cohen_kappa_score(numeric["label_gemma"], numeric["label_gemini"], weights="quadratic"))
        except Exception:
            kappa = None
    return exact, disagreement, kappa


def build(gemma_path: Path, gemini_path: Path, state_path: Path, output: Path) -> None:
    gemma = pd.read_parquet(gemma_path).rename(columns={"label": "label_gemma"})
    gemini = pd.read_parquet(gemini_path).rename(columns={"label": "label_gemini"})
    merged = gemma.merge(gemini, on=["case_id", "action_id"], how="outer", suffixes=("_gemma", "_gemini"), validate="one_to_one")
    if merged["label_gemma"].isna().any() or merged["label_gemini"].isna().any():
        raise ValueError("Gemma/Gemini normalized tables do not have the same grain")
    state = pd.read_parquet(state_path)[["case_id", "stage", "risk_band"]].drop_duplicates("case_id")
    merged = merged.merge(state, on="case_id", how="left", validate="many_to_one")
    exact, disagreement, kappa = _agreement(merged)
    lines = ["# Weak-label quality report", "", "This report measures **LLM weak-source agreement**, not human inter-rater agreement.", "",
             f"- Gemma coverage: {gemma['case_id'].nunique()} cases / {gemma['case_id'].nunique() / 500:.3f} of Panel A",
             f"- Gemini coverage: {gemini['case_id'].nunique()} cases / {gemini['case_id'].nunique() / 500:.3f} of Panel A",
             f"- Gemma ABSTAIN rate: {gemma['abstain'].mean():.4f}", f"- Gemini ABSTAIN rate: {gemini['abstain'].mean():.4f}",
             f"- Gemma distribution: `{_dist(gemma)}`", f"- Gemini distribution: `{_dist(gemini)}`",
             f"- Exact agreement: {exact}/{len(merged)} ({exact / len(merged):.4f})",
             f"- Disagreement: {disagreement}/{len(merged)} ({disagreement / len(merged):.4f})",
             f"- Quadratic weighted kappa on numeric non-ABSTAIN pairs: {kappa if kappa is not None else 'UNAVAILABLE'}", "", "## Agreement by action"]
    for action, group in merged.groupby("action_id", sort=True):
        a, d, _ = _agreement(group)
        lines.append(f"- {action}: exact={a}, disagreement={d}, n={len(group)}")
    for column, title in (("stage", "stage"), ("risk_band", "risk band")):
        lines += ["", f"## Agreement by {title}"]
        for value, group in merged.groupby(column, sort=True):
            a, d, _ = _agreement(group)
            lines.append(f"- {value}: exact={a}, disagreement={d}, n={len(group)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/recommendation/states/oulad_student_states.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recommendation/WEAK_LABEL_QUALITY.md")
    args = parser.parse_args()
    build(args.gemma, args.gemini, args.state, args.output)


if __name__ == "__main__":
    main()
