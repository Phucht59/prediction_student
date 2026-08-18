"""Pre-Snorkel, collapse, majority, and per-action quality diagnostics."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from .label_model import A4_ACTION, A5_ACTION
from .matrix import FINAL_ACTIONS, SOURCES_BY_ACTION
from .silver import PROBABILITY_COLUMNS, VALID_STATUSES

QUALITY_DOMAIN = frozenset({"PASS", "PASS_WITH_WARNING", "REVIEW", "FAIL"})


def _kappa(left: pd.Series, right: pd.Series, weights: str) -> float | None:
    mask = (left >= 0) & (right >= 0)
    if not mask.any():
        return None
    try:
        from sklearn.metrics import cohen_kappa_score

        value = float(cohen_kappa_score(left[mask], right[mask], labels=[0, 1, 2, 3], weights=weights))
        return None if math.isnan(value) else value
    except Exception:
        return None


def _class_distribution(values: np.ndarray) -> dict[str, int]:
    return {
        "0": int((values == 0).sum()),
        "1": int((values == 1).sum()),
        "2": int((values == 2).sum()),
        "3": int((values == 3).sum()),
        "ABSTAIN": int((values < 0).sum()),
    }


def lf_source_diagnostics(matrix: pd.DataFrame, sources: tuple[str, ...] | list[str]) -> dict[str, dict]:
    report = {}
    for source in sources:
        values = matrix[source].to_numpy(dtype=int)
        abstain = values < 0
        report[source] = {
            "coverage": float((~abstain).mean()),
            "abstain_rate": float(abstain.mean()),
            "labeled_count": int((~abstain).sum()),
            "abstain_count": int(abstain.sum()),
            "class_distribution": _class_distribution(values),
        }
    return report


def pairwise_diagnostics(matrix: pd.DataFrame, sources: tuple[str, ...] | list[str]) -> dict[str, dict]:
    report = {}
    for left_name, right_name in combinations(sources, 2):
        left = matrix[left_name].to_numpy(dtype=int)
        right = matrix[right_name].to_numpy(dtype=int)
        overlap = (left >= 0) & (right >= 0)
        exact_all = left == right
        conflict = overlap & (left != right)
        report[f"{left_name}_vs_{right_name}"] = {
            "overlap": int(overlap.sum()),
            "overlap_rate": float(overlap.mean()),
            "exact_agreement": int(exact_all.sum()),
            "exact_agreement_rate": float(exact_all.mean()),
            "overlap_exact_agreement_rate": float((left[overlap] == right[overlap]).mean()) if overlap.any() else None,
            "conflict_count": int(conflict.sum()),
            "conflict_rate": float(conflict.mean()),
            "overlap_conflict_rate": float((left[overlap] != right[overlap]).mean()) if overlap.any() else None,
            "linear_weighted_kappa": _kappa(pd.Series(left), pd.Series(right), "linear"),
            "quadratic_weighted_kappa": _kappa(pd.Series(left), pd.Series(right), "quadratic"),
        }
    return report


def pre_snorkel_diagnostics(matrices: dict[str, pd.DataFrame]) -> dict:
    report = {}
    for action_id in FINAL_ACTIONS:
        sources = SOURCES_BY_ACTION[action_id]
        matrix = matrices[action_id]
        values = matrix[list(sources)].to_numpy(dtype=int)
        report[action_id] = {
            "effective_lf_count": len(sources),
            "sources": list(sources),
            "all_abstain_count": int((values < 0).all(axis=1).sum()),
            "usable_weak_label_coverage": float((values >= 0).any(axis=1).mean()),
            "labeling_functions": lf_source_diagnostics(matrix, sources),
            "pairwise": pairwise_diagnostics(matrix, sources),
        }
    return report


def majority_comparison(frame: pd.DataFrame) -> dict[str, dict]:
    report = {}
    for action_id in FINAL_ACTIONS:
        group = frame[frame["action_id"] == action_id]
        comparable = group[group["majority_label"].isin([0, 1, 2, 3]) & group["silver_status"].isin(VALID_STATUSES)]
        if comparable.empty:
            agreement = None
            disagreement = None
        else:
            same = comparable["aggregator_majority_same"].to_numpy(dtype=bool)
            agreement = float(same.mean())
            disagreement = float((~same).mean())
        hard = group.loc[group["silver_status"].isin(VALID_STATUSES), "hard_label"].dropna().astype(int)
        majority = group.loc[group["majority_label"].isin([0, 1, 2, 3]), "majority_label"].astype(int)
        report[action_id] = {
            "comparable_rows": int(len(comparable)),
            "agreement_rate": agreement,
            "disagreement_rate": disagreement,
            "aggregator_class_distribution": _class_distribution(hard.to_numpy()) if len(hard) else _class_distribution(np.array([], dtype=int)),
            "majority_class_distribution": _class_distribution(majority.to_numpy()) if len(majority) else _class_distribution(np.array([], dtype=int)),
        }
    return report


def collapse_flags(frame: pd.DataFrame, pairwise: dict, seed_deviation: float, stochastic: bool) -> dict:
    evidence = frame[frame["silver_status"].isin(VALID_STATUSES)].copy()
    flags = []
    mode_share = None
    expected_std = None
    mean_confidence = None
    high_confidence_conflict = False
    if not evidence.empty:
        hard = evidence["hard_label"].astype(int).to_numpy()
        counts = np.bincount(hard, minlength=4)
        mode_share = float(counts.max() / len(hard))
        if mode_share >= 0.95:
            flags.append("hard_label_collapse")
        expected_std = float(evidence["expected_relevance"].std(ddof=0))
        if expected_std < 1e-3:
            flags.append("probability_collapse")
        mean_confidence = float(evidence["confidence"].mean())
        max_conflict = 0.0
        for item in pairwise.values():
            rate = item.get("overlap_conflict_rate")
            if rate is not None:
                max_conflict = max(max_conflict, float(rate))
        high_confidence_conflict = bool(mean_confidence >= 0.95 and max_conflict >= 0.30)
        if high_confidence_conflict:
            flags.append("overconfident_despite_conflict")
    if stochastic and seed_deviation > 0.05:
        flags.append("unstable_across_seeds")
    return {
        "flags": flags,
        "collapsed": bool(flags),
        "hard_label_mode_share": mode_share,
        "expected_relevance_std": expected_std,
        "mean_confidence": mean_confidence,
        "high_confidence_despite_conflict": high_confidence_conflict,
        "seed_max_abs_deviation": float(seed_deviation),
    }


def _kappas(pairwise: dict, *, exclude_behavior: bool = False) -> list[float]:
    values = []
    for name, item in pairwise.items():
        if exclude_behavior and "LF_BEHAVIOR" in str(name):
            continue
        kappa = item.get("quadratic_weighted_kappa")
        if kappa is not None:
            values.append(float(kappa))
    return values


def assign_quality_status(
    *,
    pairwise: dict,
    collapse: dict,
    aggregator_type: str,
    correlated_family: bool,
    a5_config: dict | None = None,
    usable_count: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if usable_count <= 0:
        return "FAIL", ["no_usable_weak_labels"]
    collapse_flags_present = set(collapse.get("flags") or [])
    if "probability_collapse" in collapse_flags_present:
        return "FAIL", ["probability_collapse"]

    primary = _kappas(pairwise, exclude_behavior=True) or _kappas(pairwise)
    all_kappas = _kappas(pairwise)
    settings = dict(a5_config or {})
    conflict_threshold = float(settings.get("conflict_kappa_threshold", 0.20))
    upgrade_threshold = float(settings.get("upgrade_min_quadratic_kappa", 0.40))
    any_negative = any(value < 0 for value in all_kappas)
    high_conflict = bool(primary) and (min(primary) < conflict_threshold or any_negative)
    strong_upgrade = bool(primary) and min(primary) >= upgrade_threshold and not any_negative

    if settings.get("remain_review_unless_strong_upgrade") and not strong_upgrade:
        if high_conflict:
            reasons.append("high_source_conflict")
        else:
            reasons.append("insufficient_agreement_to_leave_review")
        if "overconfident_despite_conflict" in collapse_flags_present:
            reasons.append("overconfident_despite_conflict")
        return "REVIEW", reasons
    if high_conflict and not strong_upgrade:
        reasons.append("high_source_conflict")
        if "overconfident_despite_conflict" in collapse_flags_present:
            reasons.append("overconfident_despite_conflict")
        return "REVIEW", reasons
    if correlated_family or aggregator_type == "TWO_SOURCE_CONSENSUS":
        reasons.append("correlated_gemini_family")
        if aggregator_type == "TWO_SOURCE_CONSENSUS":
            reasons.append("two_source_consensus_fallback")
        return "PASS_WITH_WARNING", reasons
    if "hard_label_collapse" in collapse_flags_present:
        reasons.append("prevalence_hard_label_concentration")
    if "unstable_across_seeds" in collapse_flags_present:
        reasons.append("seed_averaged_label_model")
    return "PASS", reasons


def action_quality_report(
    silver: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    run_diagnostics: dict,
    *,
    a5_config: dict | None = None,
) -> tuple[dict, dict]:
    pre = pre_snorkel_diagnostics(matrices)
    majority = majority_comparison(silver)
    report = {}
    for action_id in FINAL_ACTIONS:
        group = silver[silver["action_id"] == action_id].copy()
        evidence = group[group["silver_status"].isin(VALID_STATUSES)]
        run = run_diagnostics[action_id]
        collapse = collapse_flags(
            group,
            pre[action_id]["pairwise"],
            float(run["cross_seed_max_abs_deviation"]),
            bool(run["meaningfully_stochastic"]),
        )
        correlated_family = bool(action_id == A4_ACTION)
        status, reasons = assign_quality_status(
            pairwise=pre[action_id]["pairwise"],
            collapse=collapse,
            aggregator_type=run["aggregator_type"],
            correlated_family=correlated_family,
            a5_config=a5_config if action_id == A5_ACTION else None,
            usable_count=int(run["usable_count"]),
        )
        hard = evidence["hard_label"].dropna().astype(int) if not evidence.empty else pd.Series(dtype=int)
        report[action_id] = {
            "effective_lf_count": pre[action_id]["effective_lf_count"],
            "sources": pre[action_id]["sources"],
            "usable_weak_label_coverage": pre[action_id]["usable_weak_label_coverage"],
            "usable_count": int(run["usable_count"]),
            "all_abstain_count": int(run["all_abstain_count"]),
            "review_count": int(group["silver_status"].eq("REVIEW").sum()),
            "valid_count": int(group["silver_status"].eq("VALID").sum()),
            "no_weak_evidence_count": int(group["silver_status"].eq("NO_WEAK_EVIDENCE").sum()),
            "aggregator_type": run["aggregator_type"],
            "class_distribution": _class_distribution(hard.to_numpy()) if len(hard) else _class_distribution(np.array([], dtype=int)),
            "mean_expected_relevance": None if evidence.empty else float(evidence["expected_relevance"].mean()),
            "mean_confidence": None if evidence.empty else float(evidence["confidence"].mean()),
            "median_confidence": None if evidence.empty else float(evidence["confidence"].median()),
            "mean_entropy": None if evidence.empty else float(evidence["entropy"].mean()),
            "aggregator_vs_majority_agreement": majority[action_id]["agreement_rate"],
            "aggregator_vs_majority_disagreement": majority[action_id]["disagreement_rate"],
            "seed_policy": run["seed_policy"],
            "seeds_used": list(run["seeds_used"]),
            "same_seed_max_abs_deviation": float(run["same_seed_max_abs_deviation"]),
            "cross_seed_max_abs_deviation": float(run["cross_seed_max_abs_deviation"]),
            "meaningfully_stochastic": bool(run["meaningfully_stochastic"]),
            "estimated_lf_reliability": run["estimated_lf_reliability"],
            "collapse": collapse,
            "quality_status": status,
            "quality_reasons": reasons,
            "labeling_functions": pre[action_id]["labeling_functions"],
            "pairwise": pre[action_id]["pairwise"],
            "majority": majority[action_id],
        }
        if status not in QUALITY_DOMAIN:
            raise ValueError(f"invalid quality status for {action_id}")
    return report, pre
