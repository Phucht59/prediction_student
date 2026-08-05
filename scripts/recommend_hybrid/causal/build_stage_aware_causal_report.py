"""Build a thesis-ready Markdown report from validated local artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAUSAL = ROOT / "artifacts/recommend_hybrid/causal/target_trials/stage_action_effects.json"
DEFAULT_IMBALANCE = ROOT / "artifacts/recommend_hybrid/causal/imbalance/metrics.json"
DEFAULT_VALIDATION = ROOT / "reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_VALIDATION.json"
DEFAULT_OUTPUT = ROOT / "reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_RESULTS.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def build(
    causal_path: Path,
    imbalance_path: Path,
    validation_path: Path,
    output_path: Path,
) -> str:
    causal = _load(causal_path)
    imbalance = _load(imbalance_path)
    validation = _load(validation_path)
    if validation.get("status") != "PASS":
        raise RuntimeError("report generation requires a PASS validation artifact")

    lines = [
        "# Stage-Aware Causal Recommendation Evidence",
        "",
        "## Claim boundary",
        "",
        "This report contains observational target-trial estimates under measured-confounding, positivity, consistency, and model assumptions. It does not prove randomized or deployed recommendation effectiveness.",
        "",
        "## Protocol",
        "",
        f"- Stages: {', '.join(causal['stage_order'])}",
        f"- Actions: {', '.join(causal['action_order'])}",
        f"- Cross-fit folds: {causal['cross_fit_splits']}",
        f"- Student-cluster bootstrap iterations: {causal['bootstrap_iterations']}",
        "- Recommendation lifecycle: latest valid recommendation wins.",
        "",
        "## Stage-action effects",
        "",
        "| Stage | Action | Status | N | Treated | Control | ATE | 95% CI | Max SMD | ESS fraction |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for trial in causal["trials"]:
        protocol = trial.get("protocol", {})
        effect = trial.get("effect", {})
        diagnostic = trial.get("identifiability", {})
        interval = effect.get("confidence_interval") or [None, None]
        lines.append(
            "| {stage} | {action} | {status} | {n} | {treated} | {control} | {ate} | [{low}, {high}] | {smd} | {ess} |".format(
                stage=protocol.get("stage", "NA"),
                action=protocol.get("action_id", "NA"),
                status=trial.get("status", "NA"),
                n=diagnostic.get("retained_count", effect.get("sample_count", "NA")),
                treated=diagnostic.get("treated_count", "NA"),
                control=diagnostic.get("control_count", "NA"),
                ate=_number(effect.get("ate")),
                low=_number(interval[0]),
                high=_number(interval[1]),
                smd=_number(diagnostic.get("maximum_smd")),
                ess=_number(diagnostic.get("ess_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Frozen Hybrid imbalance sensitivity",
            "",
            "These experiments retrain only an identical linear head over frozen Hybrid embeddings. They do not replace the canonical checkpoint.",
            "",
            "| Mode | Train rows fitted | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced accuracy | Specificity | Brier |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in imbalance["results"]:
        metrics = row["metrics"]
        lines.append(
            "| {mode} | {count} | {threshold} | {roc} | {pr} | {precision} | {recall} | {f1} | {balanced} | {specificity} | {brier} |".format(
                mode=row["mode"],
                count=row["fitted_train_count"],
                threshold=_number(row["threshold"]),
                roc=_number(metrics.get("roc_auc")),
                pr=_number(metrics.get("pr_auc")),
                precision=_number(metrics.get("precision")),
                recall=_number(metrics.get("recall")),
                f1=_number(metrics.get("f1")),
                balanced=_number(metrics.get("balanced_accuracy")),
                specificity=_number(metrics.get("specificity")),
                brier=_number(metrics.get("brier_score")),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- `CAUSAL_EFFECT_ESTIMATED` means the preregistered overlap, count, balance, ESS, and bootstrap gates passed.",
            "- `CAUSAL_EVIDENCE_NOT_IDENTIFIABLE` means no causal-effect claim is allowed for that action-stage pair.",
            "- Positive ATE/CATE estimates describe the overlap population represented by the observational data.",
            "- The absence of expert labels, deployment, and randomized assignment remains a limitation.",
            "",
            f"Validation status: **{validation['status']}**",
            "",
        ]
    )
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--imbalance", type=Path, default=DEFAULT_IMBALANCE)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.causal, args.imbalance, args.validation, args.output)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output)}))


if __name__ == "__main__":
    main()
