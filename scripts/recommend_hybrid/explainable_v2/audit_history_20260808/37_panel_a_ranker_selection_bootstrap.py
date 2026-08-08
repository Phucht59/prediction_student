from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 2026
NDCG_NONINFERIORITY_MARGIN = 0.002


def _query_contributions(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for query_id, group in frame.groupby("query_id", sort=False):
        group = group.sort_values(
            [score_col, "action_id"],
            ascending=[False, True],
        )
        y = group["expected_relevance"].to_numpy(dtype=float)
        s = group[score_col].to_numpy(dtype=float)
        k = min(3, len(group))

        order = np.argsort(-s)[:k]
        ideal = np.argsort(-y)[:k]
        gains = np.power(2.0, y) - 1.0
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum(gains[order] * discounts))
        idcg = float(np.sum(gains[ideal] * discounts))
        ndcg = 0.0 if idcg <= 0.0 else dcg / idcg

        top_idx = int(np.argmax(s))
        exact_top1 = float(
            y[top_idx] >= float(np.max(y)) - 1e-12
        )

        pair_correct = 0
        pair_total = 0
        for left in range(len(y)):
            for right in range(left + 1, len(y)):
                if y[left] == y[right] or s[left] == s[right]:
                    continue
                pair_total += 1
                pair_correct += int(
                    (y[left] > y[right]) == (s[left] > s[right])
                )

        positives = y >= 1.0
        ranked_idx = np.argsort(-s)
        ranked_pos = positives[ranked_idx]
        if positives.any():
            precision1 = float(ranked_pos[0])
            positive_ranks = np.flatnonzero(ranked_pos)
            mrr = 1.0 / float(positive_ranks[0] + 1)
            recall3 = float(ranked_pos[:k].sum()) / float(positives.sum())
        else:
            precision1 = np.nan
            mrr = np.nan
            recall3 = np.nan

        rows.append(
            {
                "query_id": query_id,
                "ndcg_at_3": ndcg,
                "exact_best_top1": exact_top1,
                "pair_correct": pair_correct,
                "pair_total": pair_total,
                "precision_at_1": precision1,
                "mrr": mrr,
                "recall_at_3": recall3,
                "top1_action": str(group.iloc[0]["action_id"]),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(contrib: pd.DataFrame) -> dict[str, float | int]:
    pair_total = int(contrib["pair_total"].sum())
    pair_correct = int(contrib["pair_correct"].sum())
    return {
        "query_count": int(len(contrib)),
        "ndcg_at_3": float(contrib["ndcg_at_3"].mean()),
        "exact_best_top1_agreement": float(
            contrib["exact_best_top1"].mean()
        ),
        "pairwise_accuracy": (
            float(pair_correct / pair_total) if pair_total else 0.0
        ),
        "precision_at_1_relevance_ge_1": float(
            contrib["precision_at_1"].dropna().mean()
        ),
        "mrr_relevance_ge_1": float(
            contrib["mrr"].dropna().mean()
        ),
        "recall_at_3_relevance_ge_1": float(
            contrib["recall_at_3"].dropna().mean()
        ),
        "unique_top1_actions": int(
            contrib["top1_action"].nunique()
        ),
    }


def _crossfit_isotonic(frame: pd.DataFrame) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    folds = sorted(int(v) for v in frame["outer_fold"].unique())

    for hold_fold in folds:
        train = frame[frame["outer_fold"].astype(int) != hold_fold]
        hold = frame[frame["outer_fold"].astype(int) == hold_fold]

        for action in sorted(frame["action_id"].unique()):
            train_a = train[train["action_id"] == action]
            hold_a = hold[hold["action_id"] == action]
            if hold_a.empty:
                continue
            if len(train_a) < 30:
                raise RuntimeError(
                    f"INSUFFICIENT_CALIBRATION_ROWS "
                    f"action={action} fold={hold_fold} "
                    f"rows={len(train_a)}"
                )

            model = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
            )
            model.fit(
                train_a["raw_score_01"].to_numpy(dtype=float),
                train_a["expected_relevance"].to_numpy(dtype=float) / 3.0,
            )
            output.loc[hold_a.index] = model.predict(
                hold_a["raw_score_01"].to_numpy(dtype=float)
            )

    if output.isna().any():
        raise RuntimeError(
            "CROSSFIT_ISOTONIC_INCOMPLETE="
            f"{int(output.isna().sum())}"
        )
    return output


def _paired_bootstrap(
    raw_contrib: pd.DataFrame,
    iso_contrib: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    raw = raw_contrib.set_index("query_id")
    iso = iso_contrib.set_index("query_id")

    if set(raw.index) != set(iso.index):
        raise RuntimeError("BOOTSTRAP_QUERY_SETS_DO_NOT_MATCH")

    query_ids = np.array(sorted(raw.index.astype(str)), dtype=object)
    raw = raw.loc[query_ids]
    iso = iso.loc[query_ids]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ndcg_delta = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    top1_delta = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    pair_delta = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)

    raw_ndcg = raw["ndcg_at_3"].to_numpy(dtype=float)
    iso_ndcg = iso["ndcg_at_3"].to_numpy(dtype=float)
    raw_top1 = raw["exact_best_top1"].to_numpy(dtype=float)
    iso_top1 = iso["exact_best_top1"].to_numpy(dtype=float)
    raw_pc = raw["pair_correct"].to_numpy(dtype=float)
    raw_pt = raw["pair_total"].to_numpy(dtype=float)
    iso_pc = iso["pair_correct"].to_numpy(dtype=float)
    iso_pt = iso["pair_total"].to_numpy(dtype=float)

    n = len(query_ids)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, size=n)

        ndcg_delta[iteration] = float(
            iso_ndcg[idx].mean() - raw_ndcg[idx].mean()
        )
        top1_delta[iteration] = float(
            iso_top1[idx].mean() - raw_top1[idx].mean()
        )

        raw_total = float(raw_pt[idx].sum())
        iso_total = float(iso_pt[idx].sum())
        raw_pair = (
            float(raw_pc[idx].sum() / raw_total)
            if raw_total > 0.0
            else 0.0
        )
        iso_pair = (
            float(iso_pc[idx].sum() / iso_total)
            if iso_total > 0.0
            else 0.0
        )
        pair_delta[iteration] = iso_pair - raw_pair

    def summarize(values: np.ndarray) -> dict[str, float | int]:
        return {
            "iterations": BOOTSTRAP_ITERATIONS,
            "mean_delta": float(values.mean()),
            "ci_low_95": float(np.quantile(values, 0.025)),
            "ci_high_95": float(np.quantile(values, 0.975)),
            "probability_positive": float(np.mean(values > 0.0)),
        }

    return {
        "ndcg_at_3": summarize(ndcg_delta),
        "exact_best_top1_agreement": summarize(top1_delta),
        "pairwise_accuracy": summarize(pair_delta),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()
    sys.path.insert(0, str(root))

    oof_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
        / "panel_a_ebm_oof_predictions.parquet"
    )
    manifest_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
        / "FIVE_EBM_MANIFEST.json"
    )
    models_dir = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
        / "final_models"
    )

    for path in (oof_path, manifest_path, models_dir):
        if not path.exists():
            raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT={path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("FIVE_EBM_MANIFEST_NOT_PASS")
    if manifest.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_CONTAMINATION_DETECTED")
    if manifest.get("runtime_authorized") is not False:
        raise RuntimeError("RUNTIME_AUTHORIZED_MUST_REMAIN_FALSE")

    train_mod = importlib.import_module(
        "scripts.recommend_hybrid.explainable_v2.train_five_ebm_models"
    )
    ranker_mod = importlib.import_module(
        "src.recommend_hybrid.explainable_v2.ranker"
    )

    train_features = tuple(train_mod.FEATURES)
    runtime_features = tuple(ranker_mod.FEATURE_COLUMNS)
    if train_features != runtime_features:
        raise RuntimeError(
            "TRAIN_RUNTIME_FEATURE_SCHEMA_MISMATCH "
            f"train={train_features} runtime={runtime_features}"
        )
    if "seed_disagreement" in train_features:
        raise RuntimeError("SEED_DISAGREEMENT_STILL_IN_EBM_SCHEMA")
    if len(train_features) != 16:
        raise RuntimeError(
            f"UNEXPECTED_FEATURE_COUNT={len(train_features)} expected=16"
        )

    model_feature_checks = {}
    for action in train_mod.ACTIONS:
        model_path = models_dir / f"{action}.joblib"
        if not model_path.exists():
            raise RuntimeError(f"MISSING_EBM_MODEL={model_path}")
        model = joblib.load(model_path)
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            names = getattr(model, "feature_names", None)
        if names is not None:
            observed = tuple(str(v) for v in list(names))
            if observed != train_features:
                raise RuntimeError(
                    f"MODEL_FEATURE_SCHEMA_MISMATCH action={action} "
                    f"observed={observed}"
                )
            model_feature_checks[action] = "EXACT_MATCH"
        else:
            model_feature_checks[action] = "ATTRIBUTE_UNAVAILABLE"

    frame = pd.read_parquet(oof_path)
    if len(frame) != 1500 or frame["case_id"].nunique() != 300:
        raise RuntimeError(
            f"OOF_SHAPE_INVALID rows={len(frame)} "
            f"cases={frame['case_id'].nunique()}"
        )

    eligible = frame[frame["eligible"].astype(bool)].copy()
    if len(eligible) != 1117:
        raise RuntimeError(
            f"ELIGIBLE_ROWS={len(eligible)} expected=1117"
        )

    eligible["raw_score_01"] = np.clip(
        eligible["ebm_oof_score"].to_numpy(dtype=float) / 3.0,
        0.0,
        1.0,
    )
    eligible["isotonic_score_01"] = _crossfit_isotonic(eligible)

    raw_contrib = _query_contributions(eligible, "raw_score_01")
    iso_contrib = _query_contributions(
        eligible,
        "isotonic_score_01",
    )
    raw_metrics = _aggregate(raw_contrib)
    iso_metrics = _aggregate(iso_contrib)
    bootstrap = _paired_bootstrap(raw_contrib, iso_contrib)

    ndcg_ci_low = float(
        bootstrap["ndcg_at_3"]["ci_low_95"]
    )
    top1_ci_low = float(
        bootstrap["exact_best_top1_agreement"]["ci_low_95"]
    )
    point_ndcg_delta = float(
        iso_metrics["ndcg_at_3"] - raw_metrics["ndcg_at_3"]
    )
    point_top1_delta = float(
        iso_metrics["exact_best_top1_agreement"]
        - raw_metrics["exact_best_top1_agreement"]
    )

    # Predeclared Panel-A development selection rule:
    # 1) calibrated NDCG must be non-inferior within 0.002 absolute at 95% CI;
    # 2) exact top-1 improvement must have 95% CI strictly above zero.
    ndcg_noninferior = (
        ndcg_ci_low >= -NDCG_NONINFERIORITY_MARGIN
    )
    top1_significantly_better = top1_ci_low > 0.0

    if ndcg_noninferior and top1_significantly_better:
        decision = "SELECT_ISOTONIC_CALIBRATED_RANKER"
    else:
        decision = "KEEP_RAW_EBM_RANKER"

    report = {
        "schema_version": "panel_a_ranker_selection_bootstrap_v1",
        "status": "PASS",
        "panel": "A",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "final_metrics_claimed": False,
        "development_only": True,
        "feature_schema": list(train_features),
        "feature_count": len(train_features),
        "seed_disagreement_used_by_ebm": False,
        "model_feature_checks": model_feature_checks,
        "raw_eligible_metrics": raw_metrics,
        "crossfit_isotonic_eligible_metrics": iso_metrics,
        "point_deltas_isotonic_minus_raw": {
            "ndcg_at_3": point_ndcg_delta,
            "exact_best_top1_agreement": point_top1_delta,
            "pairwise_accuracy": float(
                iso_metrics["pairwise_accuracy"]
                - raw_metrics["pairwise_accuracy"]
            ),
        },
        "paired_query_bootstrap": bootstrap,
        "selection_rule": {
            "primary_metric": "ndcg_at_3",
            "ndcg_noninferiority_margin_absolute": (
                NDCG_NONINFERIORITY_MARGIN
            ),
            "require_ndcg_ci_low_gte_negative_margin": True,
            "secondary_metric": "exact_best_top1_agreement",
            "require_top1_delta_ci_low_gt_zero": True,
        },
        "selection_gates": {
            "ndcg_noninferior": ndcg_noninferior,
            "top1_significantly_better": top1_significantly_better,
        },
        "decision": decision,
    }

    out_dir = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/ranker_development/panel_a_v1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "RANKER_SELECTION_BOOTSTRAP.json"
    out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison = eligible[
        [
            "query_id",
            "case_id",
            "outer_fold",
            "stage",
            "action_id",
            "expected_relevance",
            "eligible",
            "raw_score_01",
            "isotonic_score_01",
        ]
    ].copy()
    comparison.to_parquet(
        out_dir / "ranker_selection_oof_scores.parquet",
        index=False,
    )

    print("=== PANEL-A RANKER SELECTION WITH PAIRED BOOTSTRAP ===")
    print("PANEL_A_CASES=300")
    print("ELIGIBLE_ACTION_ROWS=1117")
    print("FEATURE_SCHEMA_MATCH=TRUE")
    print("FEATURE_COUNT=16")
    print("SEED_DISAGREEMENT_USED_BY_EBM=FALSE")
    print("PANEL_B_TOUCHED=FALSE")
    print("FINAL_METRICS_CLAIMED=FALSE")

    print("\nRAW_ELIGIBLE")
    for key, value in raw_metrics.items():
        print(f"{key}={value}")

    print("\nCROSSFIT_ISOTONIC_ELIGIBLE")
    for key, value in iso_metrics.items():
        print(f"{key}={value}")

    print("\nPAIRED_BOOTSTRAP_DELTA_ISOTONIC_MINUS_RAW")
    for metric, values in bootstrap.items():
        print(
            f"{metric}: "
            f"mean={values['mean_delta']:.6f} "
            f"ci95=[{values['ci_low_95']:.6f},"
            f"{values['ci_high_95']:.6f}] "
            f"p_positive={values['probability_positive']:.6f}"
        )

    print("\nSELECTION_GATES")
    print(
        "NDCG_NONINFERIOR_0P002="
        + str(ndcg_noninferior).upper()
    )
    print(
        "TOP1_SIGNIFICANTLY_BETTER="
        + str(top1_significantly_better).upper()
    )
    print(f"RANKER_SELECTION={decision}")
    print(f"AUDIT_PATH={out_json}")
    print("RANKER_SELECTION_AUDIT=PASS")
    if decision == "SELECT_ISOTONIC_CALIBRATED_RANKER":
        print("NEXT_ACTION=FIT_FINAL_PANEL_A_ISOTONIC_AND_FREEZE_RANKER")
    else:
        print("NEXT_ACTION=FREEZE_RAW_EBM_RANKER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
