"""Independent per-action Snorkel aggregation and majority diagnostics."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .matrix import FINAL_ACTIONS, SOURCES, SOURCES_BY_ACTION

LABEL_MODEL_FIT_KEYS = ("n_epochs", "lr", "l2", "optimizer", "log_freq", "mu_eps")
A4_ACTION = "progress_monitoring"
A5_ACTION = "retrieval_practice"
DEFAULT_SEEDS = (42, 1201, 2026)


def _ensure_username() -> None:
    # Torch's Windows cache path calls getpass; this keeps local execution
    # independent of an optional password-database module.
    if not any(os.environ.get(name) for name in ("USERNAME", "USER", "LOGNAME", "LNAME")):
        os.environ["USERNAME"] = "recommendation"


def _matrix_values(matrix: pd.DataFrame, sources: tuple[str, ...] | list[str]) -> np.ndarray:
    return matrix[list(sources)].to_numpy(dtype=int)


def _all_abstain_mask(values: np.ndarray) -> np.ndarray:
    return (values < 0).all(axis=1)


def _train_kwargs(train_config: dict | None) -> dict:
    config = dict(train_config or {})
    return {key: config[key] for key in LABEL_MODEL_FIT_KEYS if key in config}


def majority_vote(matrix: pd.DataFrame, sources: tuple[str, ...] | list[str] | None = None) -> np.ndarray:
    """Deterministic MajorityLabelVoter hard labels; all-abstain and ties stay -1.

    Ties use Snorkel's abstain policy so the baseline does not invent a class.
    """
    from snorkel.labeling.model import MajorityLabelVoter

    resolved = tuple(sources) if sources is not None else tuple(column for column in matrix.columns if str(column).startswith("LF_"))
    if not resolved:
        resolved = SOURCES
    values = _matrix_values(matrix, resolved)
    output = np.full(len(values), -1, dtype=int)
    keep = ~_all_abstain_mask(values)
    if not keep.any():
        return output
    voter = MajorityLabelVoter(cardinality=4)
    output[keep] = np.asarray(voter.predict(values[keep], tie_break_policy="abstain"), dtype=int)
    return output


def two_source_consensus(matrix: pd.DataFrame, sources: tuple[str, ...] | list[str]) -> np.ndarray:
    """Transparent two-source probabilities: one-hot on agreement, 0.5/0.5 on conflict."""
    values = _matrix_values(matrix, sources)
    probabilities = np.full((len(values), 4), np.nan, dtype=float)
    for index, row in enumerate(values):
        labels = row[row >= 0]
        if len(labels) == 0:
            continue
        counts = np.bincount(labels, minlength=4).astype(float)
        probabilities[index] = counts / counts.sum()
    return probabilities


def _fit_snorkel_once(values: np.ndarray, seed: int, train_config: dict) -> tuple[np.ndarray, np.ndarray]:
    _ensure_username()
    from snorkel.labeling.model import LabelModel

    model = LabelModel(cardinality=4, verbose=False)
    model.fit(values, progress_bar=False, seed=int(seed), **train_config)
    probabilities = np.asarray(model.predict_proba(values), dtype=float)
    weights = np.asarray(model.get_weights(), dtype=float)
    return probabilities, weights


def probe_label_model_stochasticity(
    values: np.ndarray,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    train_config: dict | None = None,
    threshold: float = 1e-6,
) -> dict:
    config = _train_kwargs(train_config)
    first_seed = int(seeds[0])
    repeat_a, _ = _fit_snorkel_once(values, first_seed, config)
    repeat_b, _ = _fit_snorkel_once(values, first_seed, config)
    same_seed = float(np.max(np.abs(repeat_a - repeat_b)))
    stacked = []
    weights = []
    for seed in seeds:
        probabilities, lf_weights = _fit_snorkel_once(values, int(seed), config)
        stacked.append(probabilities)
        weights.append(lf_weights)
    stacked_array = np.stack(stacked, axis=0)
    mean = stacked_array.mean(axis=0)
    cross_seed = float(np.max(np.abs(stacked_array - mean)))
    stochastic = cross_seed > float(threshold)
    return {
        "same_seed_max_abs_deviation": same_seed,
        "cross_seed_max_abs_deviation": cross_seed,
        "meaningfully_stochastic": bool(stochastic),
        "policy": "average_three_seeds" if stochastic else "deterministic_single_seed",
        "seeds_used": list(seeds) if stochastic else [first_seed],
        "threshold": float(threshold),
        "seed_probabilities": stacked_array,
        "seed_weights": weights,
        "mean_probabilities": mean,
        "reference_probabilities": repeat_a,
        "reference_weights": weights[0],
    }


def snorkel_output_usable(
    probabilities: np.ndarray,
    *,
    collapse_hard_label_share: float = 0.95,
    collapse_expected_relevance_std: float = 1e-3,
    seed_deviation: float = 0.0,
    unstable_seed_deviation: float = 0.05,
    stochastic: bool = False,
    reject_hard_label_collapse: bool = False,
    reject_seed_instability: bool = False,
) -> tuple[bool, str | None]:
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        return False, "unexpected_probability_shape"
    if not np.isfinite(probabilities).all():
        return False, "non_finite_probabilities"
    if (probabilities < -1e-8).any() or (probabilities > 1 + 1e-8).any():
        return False, "probabilities_out_of_range"
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        return False, "probabilities_not_normalized"
    expected = probabilities @ np.arange(4)
    if float(np.std(expected)) < collapse_expected_relevance_std:
        return False, "collapsed_probabilities"
    if reject_hard_label_collapse:
        hard = probabilities.argmax(axis=1)
        mode_share = float(np.max(np.bincount(hard, minlength=4)) / len(hard))
        if mode_share >= collapse_hard_label_share:
            return False, "collapsed_hard_labels"
    if reject_seed_instability and stochastic and seed_deviation > unstable_seed_deviation:
        return False, "unstable_across_seeds"
    return True, None


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return float("nan")
    return float(-np.sum(positive * np.log(positive)))


def _select_probabilities(probe: dict, *, average_if_stochastic: bool) -> tuple[np.ndarray, np.ndarray]:
    if probe["meaningfully_stochastic"] and average_if_stochastic:
        weights = np.mean(np.stack(probe["seed_weights"], axis=0), axis=0)
        return probe["mean_probabilities"], weights
    return probe["reference_probabilities"], probe["reference_weights"]


def decide_a4_aggregator(
    matrix: pd.DataFrame,
    sources: tuple[str, ...],
    probe: dict,
    *,
    config: dict | None = None,
    average_if_stochastic: bool = True,
) -> tuple[str, np.ndarray, np.ndarray, dict]:
    settings = dict(config or {})
    probabilities, weights = _select_probabilities(probe, average_if_stochastic=average_if_stochastic)
    usable, reason = snorkel_output_usable(
        probabilities,
        collapse_hard_label_share=float(settings.get("collapse_hard_label_share", 0.95)),
        collapse_expected_relevance_std=float(settings.get("collapse_expected_relevance_std", 1e-3)),
        seed_deviation=float(probe["cross_seed_max_abs_deviation"]),
        unstable_seed_deviation=float(settings.get("unstable_seed_deviation", 0.05)),
        stochastic=bool(probe["meaningfully_stochastic"]),
        reject_hard_label_collapse=True,
        reject_seed_instability=True,
    )
    decision = {
        "correlated_family": True,
        "family": settings.get("family", "gemini"),
        "snorkel_usable": usable,
        "fallback_reason": None if usable else reason,
    }
    if usable and settings.get("allow_snorkel_if_usable", True):
        decision["aggregator_type"] = "SNORKEL"
        return "SNORKEL", probabilities, weights, decision
    consensus = two_source_consensus(matrix, sources)
    decision["aggregator_type"] = "TWO_SOURCE_CONSENSUS"
    if decision["fallback_reason"] is None:
        decision["fallback_reason"] = "two_source_family_not_independent"
    return "TWO_SOURCE_CONSENSUS", consensus, weights, decision


def fit_label_models(
    matrices: dict[str, pd.DataFrame],
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    train_config: dict | None = None,
    stochasticity_threshold: float = 1e-6,
    average_if_stochastic: bool = True,
    a4_config: dict | None = None,
    label_model_version: str = "recommendation.weak_supervision.v5",
    phase6_source_manifest_version: str = "recommendation.phase6_source_manifest.v1",
) -> tuple[pd.DataFrame, dict]:
    rows = []
    diagnostics = {}
    for action_id in FINAL_ACTIONS:
        matrix = matrices[action_id]
        sources = SOURCES_BY_ACTION[action_id]
        values = _matrix_values(matrix, sources)
        if values.shape != (500, len(sources)):
            raise ValueError(f"unexpected matrix shape for {action_id}: {values.shape}")
        majority = majority_vote(matrix, sources)
        if len(sources) < 3:
            if action_id != A4_ACTION:
                raise ValueError(f"{action_id} has fewer than 3 LFs and no documented fallback")
            probabilities = two_source_consensus(matrix, sources)
            weights = np.full(len(sources), np.nan, dtype=float)
            aggregator = "TWO_SOURCE_CONSENSUS"
            probe = {
                "policy": "not_applicable_two_source_consensus",
                "seeds_used": [],
                "same_seed_max_abs_deviation": 0.0,
                "cross_seed_max_abs_deviation": 0.0,
                "meaningfully_stochastic": False,
            }
            a4_decision = {
                "correlated_family": True,
                "family": (a4_config or {}).get("family", "gemini"),
                "snorkel_usable": False,
                "fallback_reason": "snorkel_label_model_requires_at_least_3_labeling_functions",
                "aggregator_type": "TWO_SOURCE_CONSENSUS",
            }
        else:
            probe = probe_label_model_stochasticity(
                values,
                seeds=seeds,
                train_config=train_config,
                threshold=stochasticity_threshold,
            )
            if action_id == A4_ACTION:
                aggregator, probabilities, weights, a4_decision = decide_a4_aggregator(
                    matrix,
                    sources,
                    probe,
                    config=a4_config,
                    average_if_stochastic=average_if_stochastic,
                )
            else:
                probabilities, weights = _select_probabilities(probe, average_if_stochastic=average_if_stochastic)
                usable, reason = snorkel_output_usable(
                    probabilities,
                    seed_deviation=float(probe["cross_seed_max_abs_deviation"]),
                    stochastic=bool(probe["meaningfully_stochastic"]),
                    reject_hard_label_collapse=False,
                    reject_seed_instability=False,
                )
                if not usable:
                    raise ValueError(f"Snorkel output unusable for {action_id}: {reason}")
                aggregator = "SNORKEL"
                a4_decision = None
        abstain = _all_abstain_mask(values)
        for index, case_id in enumerate(matrix["case_id"].astype(str)):
            if abstain[index]:
                row = {
                    "case_id": case_id,
                    "action_id": action_id,
                    "p_r0": np.nan,
                    "p_r1": np.nan,
                    "p_r2": np.nan,
                    "p_r3": np.nan,
                    "expected_relevance": np.nan,
                    "hard_label": pd.NA,
                    "confidence": np.nan,
                    "entropy": np.nan,
                    "majority_label": int(majority[index]),
                    "aggregator_majority_same": False,
                    "silver_status": "NO_WEAK_EVIDENCE",
                    "aggregator_type": aggregator,
                    "label_model_version": label_model_version,
                    "phase6_source_manifest_version": phase6_source_manifest_version,
                }
            else:
                probs = np.asarray(probabilities[index], dtype=float)
                hard_label = int(np.argmax(probs))
                majority_label = int(majority[index])
                row = {
                    "case_id": case_id,
                    "action_id": action_id,
                    "p_r0": float(probs[0]),
                    "p_r1": float(probs[1]),
                    "p_r2": float(probs[2]),
                    "p_r3": float(probs[3]),
                    "expected_relevance": float(np.dot(probs, np.arange(4))),
                    "hard_label": hard_label,
                    "confidence": float(np.max(probs)),
                    "entropy": _entropy(probs),
                    "majority_label": majority_label,
                    "aggregator_majority_same": bool(majority_label >= 0 and hard_label == majority_label),
                    "silver_status": "VALID",
                    "aggregator_type": aggregator,
                    "label_model_version": label_model_version,
                    "phase6_source_manifest_version": phase6_source_manifest_version,
                }
            rows.append(row)
        diagnostics[action_id] = {
            "sources": list(sources),
            "aggregator_type": aggregator,
            "estimated_lf_reliability": {
                source: None if not np.isfinite(float(weights[index])) else float(weights[index])
                for index, source in enumerate(sources)
            },
            "all_abstain_count": int(abstain.sum()),
            "usable_count": int((~abstain).sum()),
            "seed_policy": probe["policy"],
            "seeds_used": list(probe["seeds_used"]),
            "same_seed_max_abs_deviation": float(probe["same_seed_max_abs_deviation"]),
            "cross_seed_max_abs_deviation": float(probe["cross_seed_max_abs_deviation"]),
            "meaningfully_stochastic": bool(probe["meaningfully_stochastic"]),
            "a4_decision": a4_decision,
        }
    output = pd.DataFrame(rows).sort_values(["case_id", "action_id"]).reset_index(drop=True)
    output["hard_label"] = output["hard_label"].astype("Int64")
    output["majority_label"] = output["majority_label"].astype("Int64")
    if len(output) != 2500 or output.duplicated(["case_id", "action_id"]).any():
        raise ValueError("silver output must contain 2,500 unique case-action rows")
    return output, diagnostics
