from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


TARGET = {
    "query_id": "414696::GGG::2014B::EARLY_20",
    "case_id": "case_3bed45903f5da99df23a2022",
    "outer_fold": 0,
    "stage": "EARLY_20",
    "action_id": "RECOVER_ENGAGEMENT",
}

KEYS = ["query_id", "case_id", "outer_fold", "stage", "action_id"]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("query_id", "case_id", "stage", "action_id"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "outer_fold" in out.columns:
        out["outer_fold"] = out["outer_fold"].astype(int)
    return out


def select_target(df: pd.DataFrame) -> pd.DataFrame:
    d = normalize_keys(df)
    mask = pd.Series(True, index=d.index)
    for key, value in TARGET.items():
        if key not in d.columns:
            continue
        if key == "outer_fold":
            mask &= d[key].astype(int).eq(int(value))
        else:
            mask &= d[key].astype(str).eq(str(value))
    return d.loc[mask].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    args = ap.parse_args()
    root = args.repo.resolve()
    sys.path.insert(0, str(root))

    paths = {
        "votes": root
        / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
        / "weak_vote_matrix.parquet",
        "labels": root
        / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
        / "probabilistic_relevance_labels.parquet",
        "candidates": root
        / "artifacts/recommend_hybrid/explainable_v2/features"
        / "action_candidates.parquet",
        "oof": root
        / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
        / "panel_a_ebm_oof_predictions.parquet",
        "manifest": root
        / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
        / "label_model_manifest.json",
    }
    for name, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"MISSING_{name.upper()}={path}")

    votes = normalize_keys(pd.read_parquet(paths["votes"]))
    labels = normalize_keys(pd.read_parquet(paths["labels"]))
    candidates = normalize_keys(pd.read_parquet(paths["candidates"]))
    oof = normalize_keys(pd.read_parquet(paths["oof"]))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    from scripts.recommend_hybrid.explainable_v2 import train_five_ebm_models as runner

    training, _, _ = runner._load_inputs()
    training = normalize_keys(training)

    print("=== SINGLE-SOURCE ELIGIBLE ROW TRACE ===")
    print("TARGET=" + json.dumps(TARGET, sort_keys=True))

    for name, df in (
        ("VOTES", votes),
        ("LABELS", labels),
        ("CANDIDATES", candidates),
        ("OOF", oof),
        ("TRAINING_INPUT", training),
    ):
        hit = select_target(df)
        print(f"{name}_MATCH_COUNT={len(hit)}")
        if len(hit):
            print(f"{name}_ROW=")
            print(hit.to_string(index=False))

    label_cols = list(labels.columns)
    print("LABEL_COLUMNS=" + json.dumps(label_cols))

    status_col = "label_status" if "label_status" in labels.columns else None
    family_col = (
        "independent_source_families"
        if "independent_source_families" in labels.columns
        else None
    )
    confidence_col = (
        "label_confidence" if "label_confidence" in labels.columns else None
    )

    if status_col:
        print(
            "LABEL_STATUS_COUNTS="
            + json.dumps(
                {
                    str(k): int(v)
                    for k, v in labels[status_col]
                    .astype(str)
                    .value_counts(dropna=False)
                    .to_dict()
                    .items()
                },
                sort_keys=True,
            )
        )
    else:
        print("LABEL_STATUS_COLUMN_PRESENT=FALSE")

    if family_col:
        print(
            "LABEL_FAMILY_COUNT_DISTRIBUTION="
            + json.dumps(
                {
                    str(k): int(v)
                    for k, v in labels[family_col]
                    .value_counts(dropna=False)
                    .sort_index()
                    .to_dict()
                    .items()
                },
                sort_keys=True,
            )
        )
    else:
        print("LABEL_FAMILY_COUNT_COLUMN_PRESENT=FALSE")

    target_label = select_target(labels)
    target_train = select_target(training)
    target_candidate = select_target(candidates)

    if len(target_label) != 1:
        raise RuntimeError(f"TARGET_LABEL_COUNT={len(target_label)}")
    if len(target_candidate) != 1:
        raise RuntimeError(f"TARGET_CANDIDATE_COUNT={len(target_candidate)}")

    row = target_label.iloc[0]
    candidate = target_candidate.iloc[0]

    label_status = str(row[status_col]) if status_col else "UNKNOWN"
    family_count = (
        int(row[family_col]) if family_col and pd.notna(row[family_col]) else None
    )
    confidence = (
        float(row[confidence_col])
        if confidence_col and pd.notna(row[confidence_col])
        else None
    )

    eligible = bool(candidate.get("eligible", False))
    training_present = len(target_train) == 1

    print(f"TARGET_ELIGIBLE={str(eligible).upper()}")
    print(f"TARGET_LABEL_STATUS={label_status}")
    print(f"TARGET_INDEPENDENT_SOURCE_FAMILIES={family_count}")
    print(f"TARGET_LABEL_CONFIDENCE={confidence}")
    print(f"TARGET_PRESENT_IN_EBM_TRAINING_INPUT={str(training_present).upper()}")

    # Query-level context: determine whether this row affected the frozen top-3.
    query_oof = oof[oof["query_id"].eq(TARGET["query_id"])].copy()
    query_candidates = candidates[candidates["query_id"].eq(TARGET["query_id"])].copy()
    if "eligible" in query_candidates.columns:
        query_oof = query_oof.merge(
            query_candidates[KEYS + ["eligible"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    if "ebm_oof_score" in query_oof.columns:
        query_oof = query_oof.sort_values(
            ["ebm_oof_score", "action_id"], ascending=[False, True]
        )
    print("QUERY_FROZEN_RANKING=")
    print(query_oof.to_string(index=False))

    # Protocol interpretation.
    minimum_families = 2
    protocol_violation = (
        training_present
        and family_count is not None
        and family_count < minimum_families
    )
    correctly_abstained = (
        family_count is not None
        and family_count < minimum_families
        and label_status.upper() in {"ABSTAINED", "ABSTAIN", "NOT_RETAINED"}
    )

    print("=== INTERPRETATION ===")
    print(
        "LABEL_MINIMUM_SOURCE_FAMILIES_PROTOCOL="
        f"{minimum_families}"
    )
    print(
        "LOW_SUPPORT_LABEL_CORRECTLY_ABSTAINED="
        + str(correctly_abstained).upper()
    )
    print(
        "LOW_SUPPORT_ROW_PRESENT_IN_TRAINING="
        + str(protocol_violation).upper()
    )

    if protocol_violation:
        print("AUDIT_STATUS=FAIL_LOW_SUPPORT_LABEL_USED_FOR_EBM_TRAINING")
        print(
            "NEXT_ACTION=EXCLUDE_NONRETAINED_LABEL_ROWS_FROM_EBM_TRAINING_"
            "AND_REPEAT_PANEL_A_MODEL_SELECTION"
        )
        return 2

    if correctly_abstained:
        print("AUDIT_STATUS=PASS_LOW_SUPPORT_ROW_NOT_A_TRAINING_LABEL_VIOLATION")
        print(
            "NEXT_ACTION=REALIGN_RELEASE_GATE_TO_LABEL_RETENTION_SCOPE_"
            "THEN_FREEZE_ROUTER"
        )
        return 0

    print("AUDIT_STATUS=REVIEW_REQUIRED")
    print("NEXT_ACTION=INSPECT_LABEL_RETENTION_CONTRACT")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
