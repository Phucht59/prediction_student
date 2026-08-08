"""Pre-registered, exactly-once evaluation of the frozen ranker on Panel B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# Some managed Windows subprocesses return an empty platform.machine(), which
# prevents InterpretML 0.7.8 from resolving its bundled x64 libebm DLL. This
# restores only the missing platform identifier; it does not alter model state.
if sys.platform == "win32" and not platform.machine():
    platform.machine = lambda: "AMD64"


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.provider_envelope import (  # noqa: E402
    verify_provider_envelope,
)
from src.recommend_hybrid.explainable_v2.sampling import (  # noqa: E402
    perform_grouped_stratified_sampling,
)


PROVIDER = "Google Gemini API"
MODEL_NAME = "gemini-3.5-flash-lite"
PROMPT_SHA256 = "f7edfaacd2fad67bf21a175ccc5c0a46abb81b669c08928ab78009c0a24624f3"
PANEL_A_MANIFEST_SHA256 = "4a9af5a21ace08f13bfdc09504f19c1a9b5616d85df4151379815116d28eb5db"
PANEL_A_REVIEWS_SHA256 = "4a4871426880bdcd1257dc15c29a36c23de34481f07be68d8e5095dc20efefb9"
EXPECTED_CASES = 150
EXPECTED_RECORDS = 557
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 2026

ACTIONS = (
    "ASSESSMENT_COMPLETION",
    "RECOVER_ENGAGEMENT",
    "STUDY_REGULARITY",
    "TARGETED_CONTENT_REVIEW",
    "QUIZ_RETRIEVAL_PRACTICE",
)
NUMERIC_FEATURES = (
    "risk_probability",
    "hybrid_uncertainty",
    "course_progress",
    "inactivity_streak",
    "active_day_rate",
    "assessments_due",
    "regularity_score",
    "content_coverage",
    "quiz_activity",
    "missing_assessment_count",
    "due_soon_count",
    "completion_rate",
)
BINARY_FEATURES = (
    "vle_available",
    "study_material_available",
    "quiz_available",
)
CATEGORICAL_FEATURES = ("stage",)
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
OBSERVED_FEATURES = (
    "inactivity_streak",
    "active_day_rate",
    "assessments_due",
    "regularity_score",
    "content_coverage",
    "quiz_activity",
    "missing_assessment_count",
    "due_soon_count",
    "completion_rate",
)

ARTIFACT_ROOT = ROOT / "artifacts/recommend_hybrid/explainable_v2"
DEVELOPMENT_FREEZE_PATH = (
    ARTIFACT_ROOT / "frozen/development_v2/DEVELOPMENT_FREEZE_MANIFEST.json"
)
PANEL_A_FREEZE_DIR = ARTIFACT_ROOT / "annotations/frozen/panel_a_v1"
REQUEST_BATCH_DIR = ARTIFACT_ROOT / "annotations/prompts/panel_b_request_batches"
PROVIDER_ROOT = ARTIFACT_ROOT / "annotations/external_reviews" / PROVIDER
RANKER_DIR = ARTIFACT_ROOT / "frozen/ranker_panel_a_v2"
CANDIDATES_PATH = ARTIFACT_ROOT / "features/action_candidates.parquet"
QUERY_EVIDENCE_PATH = ARTIFACT_ROOT / "features/query_level_evidence.parquet"
CASE_MANIFEST_PATH = ARTIFACT_ROOT / "annotations/exports/case_manifest.json"
PANEL_A_LABELS_PATH = (
    ARTIFACT_ROOT / "labels/panel_a_v1/probabilistic_relevance_labels.parquet"
)
OUTPUT_DIR = ARTIFACT_ROOT / "final_heldout/panel_b_v1"
PROTOCOL_PATH = OUTPUT_DIR / "PANEL_B_EVALUATION_PROTOCOL.json"
STARTED_PATH = OUTPUT_DIR / "EVALUATION_STARTED.json"
FAILED_PATH = OUTPUT_DIR / "EVALUATION_FAILED.json"
FINAL_MANIFEST_PATH = OUTPUT_DIR / "PANEL_B_FINAL_HELDOUT_MANIFEST.json"
FINAL_METRICS_PATH = OUTPUT_DIR / "PANEL_B_FINAL_HELDOUT_METRICS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def assert_development_authority() -> dict[str, Any]:
    freeze = json.loads(DEVELOPMENT_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS":
        raise RuntimeError("DEVELOPMENT_FREEZE_STATUS_NOT_PASS")
    if freeze.get("panel_b_touched") is not False:
        raise RuntimeError("DEVELOPMENT_FREEZE_PANEL_B_TOUCHED")
    if freeze.get("runtime_authorized") is not False:
        raise RuntimeError("DEVELOPMENT_FREEZE_RUNTIME_AUTHORIZED")
    if freeze.get("final_metrics_claimed") is not False:
        raise RuntimeError("DEVELOPMENT_FREEZE_ALREADY_CLAIMS_FINAL_METRICS")

    panel_a_manifest = PANEL_A_FREEZE_DIR / "PANEL_A_FREEZE_MANIFEST.json"
    panel_a_reviews = PANEL_A_FREEZE_DIR / "panel_a_external_reviews_frozen.jsonl"
    if sha256(panel_a_manifest) != PANEL_A_MANIFEST_SHA256:
        raise RuntimeError("PANEL_A_FROZEN_REVIEW_MANIFEST_CHANGED")
    if sha256(panel_a_reviews) != PANEL_A_REVIEWS_SHA256:
        raise RuntimeError("PANEL_A_FROZEN_REVIEWS_CHANGED")

    for action, expected in freeze["ranker"]["five_model_sha256"].items():
        actual = sha256(RANKER_DIR / "final_models" / f"{action}.joblib")
        if actual != expected:
            raise RuntimeError(f"FROZEN_MODEL_HASH_CHANGED={action}")
    return freeze


def load_request_contract() -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for batch_number in range(1, 4):
        path = REQUEST_BATCH_DIR / f"batch_{batch_number:02d}.jsonl"
        hashes[path.name] = sha256(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"PANEL_B_CASE_COUNT={len(rows)}")
    if len({row["case_id"] for row in rows}) != EXPECTED_CASES:
        raise RuntimeError("PANEL_B_CASE_IDS_NOT_UNIQUE")
    if {row.get("panel_id") for row in rows} != {"PANEL_B"}:
        raise RuntimeError("NON_PANEL_B_CASE_IN_REQUEST_CONTRACT")

    pairs = [
        {
            "case_id": row["case_id"],
            "action_id": action,
            "stage": row["stage"],
        }
        for row in rows
        for action in row["candidate_actions"]
    ]
    pair_frame = pd.DataFrame(pairs)
    if len(pair_frame) != EXPECTED_RECORDS:
        raise RuntimeError(f"PANEL_B_EXPECTED_RECORDS={len(pair_frame)}")
    if pair_frame.duplicated(["case_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_PANEL_B_CASE_ACTION_CONTRACT")
    return pair_frame, hashes


def python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def reconstruct_panel_b_query_lineage() -> pd.DataFrame:
    """Reproduce frozen sampling without reading or persisting a private mapping."""

    case_manifest = json.loads(CASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256(QUERY_EVIDENCE_PATH) != case_manifest["source_query_evidence_sha256"]:
        raise RuntimeError("QUERY_EVIDENCE_CHANGED_SINCE_PANEL_EXPORT")

    query = pd.read_parquet(QUERY_EVIDENCE_PATH).sort_values("query_id").reset_index(drop=True)
    query_groups = query.groupby("query_id", sort=True)
    student_to_queries: dict[str, list[str]] = {}
    query_strata: dict[str, str] = {}
    for row in query.itertuples(index=False):
        query_id = str(row.query_id)
        student_group_id = str(row.student_group_id)
        student_to_queries.setdefault(student_group_id, []).append(query_id)
        query_strata[query_id] = (
            f"fold{int(row.outer_fold)}_{str(row.stage)}_{str(row.risk_band)}"
        )

    _, panel_b_query_ids, audit = perform_grouped_stratified_sampling(
        df=query,
        query_groups=query_groups,
        student_to_queries=student_to_queries,
        query_strata=query_strata,
        panel_a_target=300,
        panel_b_target=150,
        seed=2026,
    )
    if audit["student_overlap"] != 0 or audit["query_overlap"] != 0:
        raise RuntimeError("REPRODUCED_PANEL_SAMPLING_OVERLAP")

    cases: list[dict[str, Any]] = []
    for path in sorted(REQUEST_BATCH_DIR.glob("batch_*.jsonl")):
        cases.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if len(cases) != len(panel_b_query_ids):
        raise RuntimeError("REPRODUCED_PANEL_B_ORDER_LENGTH_MISMATCH")

    by_query = query.set_index(query["query_id"].astype(str), drop=False)
    mapping: list[dict[str, str]] = []
    for case, query_id in zip(cases, panel_b_query_ids, strict=True):
        row = by_query.loc[query_id]
        uncertainty = float(row["hybrid_uncertainty"])
        uncertainty_band = "HIGH" if uncertainty > 0.3 else (
            "MEDIUM" if uncertainty > 0.15 else "LOW"
        )
        expected_public_evidence = {
            field: python_value(row[field]) for field in OBSERVED_FEATURES
        }
        expected_availability = {
            field: bool(row[field]) for field in BINARY_FEATURES
        }
        if case["observed_pre_cutoff_evidence"] != expected_public_evidence:
            raise RuntimeError("REPRODUCED_PANEL_B_EVIDENCE_ORDER_MISMATCH")
        if case["availability_flags"] != expected_availability:
            raise RuntimeError("REPRODUCED_PANEL_B_AVAILABILITY_ORDER_MISMATCH")
        if str(case["stage"]) != str(row["stage"]):
            raise RuntimeError("REPRODUCED_PANEL_B_STAGE_ORDER_MISMATCH")
        if int(case["cutoff_day"]) != int(row["cutoff_day"]):
            raise RuntimeError("REPRODUCED_PANEL_B_CUTOFF_ORDER_MISMATCH")
        if str(case["risk_band"]) != str(row["risk_band"]):
            raise RuntimeError("REPRODUCED_PANEL_B_RISK_BAND_ORDER_MISMATCH")
        if str(case["uncertainty_band"]) != uncertainty_band:
            raise RuntimeError("REPRODUCED_PANEL_B_UNCERTAINTY_ORDER_MISMATCH")
        mapping.append({"case_id": str(case["case_id"]), "query_id": query_id})

    result = pd.DataFrame(mapping)
    if result["case_id"].duplicated().any() or result["query_id"].duplicated().any():
        raise RuntimeError("REPRODUCED_PANEL_B_LINEAGE_NOT_ONE_TO_ONE")
    return result


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(float)
    for column in BINARY_FEATURES:
        output[column] = output[column].astype("boolean").astype("Int64").astype(float)
    for column in CATEGORICAL_FEATURES:
        output[column] = output[column].astype(str)
    return output


def score_frozen_ranker(pair_frame: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.read_parquet(CANDIDATES_PATH)
    if candidates.duplicated(["query_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_ACTION_CANDIDATE_QUERY_ACTION")
    mapping = reconstruct_panel_b_query_lineage()
    selected = pair_frame.merge(
        mapping,
        on="case_id",
        how="left",
        validate="many_to_one",
    ).merge(
        candidates[["action_id", "query_id", *FEATURES]],
        on=["query_id", "action_id", "stage"],
        how="left",
        validate="one_to_one",
    )
    if selected["query_id"].isna().any():
        raise RuntimeError("PANEL_B_FEATURE_JOIN_INCOMPLETE")
    if selected["query_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError("PANEL_B_QUERY_MAPPING_COUNT_MISMATCH")

    selected["native_ordinal_score"] = np.nan
    for action in ACTIONS:
        mask = selected["action_id"].eq(action)
        model = joblib.load(RANKER_DIR / "final_models" / f"{action}.joblib")
        prediction = np.asarray(
            model.predict(prepare_features(selected.loc[mask])),
            dtype=float,
        )
        selected.loc[mask, "native_ordinal_score"] = prediction

    native = selected["native_ordinal_score"].to_numpy(dtype=float)
    if not np.isfinite(native).all():
        raise RuntimeError("NONFINITE_FROZEN_RANKER_SCORE")
    selected["public_score"] = np.clip(native / 3.0, 0.0, 1.0)
    public = selected["public_score"].to_numpy(dtype=float)
    if not np.isfinite(public).all() or ((public < 0) | (public > 1)).any():
        raise RuntimeError("PUBLIC_SCORE_CONTRACT_VIOLATION")
    return selected


def action_stage_baseline(scored: pd.DataFrame) -> pd.Series:
    labels = pd.read_parquet(PANEL_A_LABELS_PATH)
    retained = labels[labels["retained_for_training"].astype(bool)].copy()
    means = retained.groupby(["action_id", "stage"])["expected_relevance"].mean()
    action_means = retained.groupby("action_id")["expected_relevance"].mean()
    values = []
    for row in scored[["action_id", "stage"]].itertuples(index=False):
        ordinal = means.get((row.action_id, row.stage), action_means[row.action_id])
        values.append(float(np.clip(float(ordinal) / 3.0, 0.0, 1.0)))
    return pd.Series(values, index=scored.index, dtype=float)


def validate_batch_checksums(batch_dir: Path) -> None:
    checksum_path = batch_dir / "checksums.sha256"
    if not checksum_path.is_file():
        raise RuntimeError(f"MISSING_PROVIDER_CHECKSUMS={batch_dir.name}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = batch_dir / Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"PROVIDER_CHECKSUM_FAILURE={batch_dir.name}/{relative}")


def load_real_reviews(pair_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for batch_number in range(1, 4):
        batch_id = f"panel_b_batch_{batch_number:02d}"
        batch_dir = PROVIDER_ROOT / batch_id
        validate_batch_checksums(batch_dir)
        manifest_path = batch_dir / "batch_manifest.json"
        normalized_path = batch_dir / "normalized_records.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True or manifest.get("panel_id") != "PANEL_B":
            raise RuntimeError(f"INCOMPLETE_OR_WRONG_PANEL_PROVIDER_BATCH={batch_id}")
        if manifest.get("provider") != PROVIDER:
            raise RuntimeError(f"WRONG_PROVIDER={batch_id}")
        if manifest.get("model_names_observed") != [MODEL_NAME]:
            raise RuntimeError(f"WRONG_PROVIDER_MODEL={batch_id}")
        if manifest.get("prompt_sha256") != PROMPT_SHA256:
            raise RuntimeError(f"PROMPT_HASH_MISMATCH={batch_id}")

        batch_records = [
            json.loads(line)
            for line in normalized_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for record in batch_records:
            ok, code, message = verify_provider_envelope(
                ARTIFACT_ROOT / "annotations/external_reviews",
                PROVIDER,
                batch_id,
                record,
                locked_prompt_hash=PROMPT_SHA256,
            )
            if not ok:
                raise RuntimeError(f"PROVIDER_ENVELOPE_FAILURE={code}:{message}")
            if record.get("reviewer_type") != "REAL_EXTERNAL_LLM_REVIEW":
                raise RuntimeError("NON_REAL_EXTERNAL_REVIEW_RECORD")
            if record.get("provider") != PROVIDER or record.get("model_name") != MODEL_NAME:
                raise RuntimeError("PROVIDER_OR_MODEL_RECORD_MISMATCH")
            if record.get("panel_id") != "PANEL_B":
                raise RuntimeError("NON_PANEL_B_REVIEW_RECORD")
        records.extend(batch_records)
        provenance.append(
            {
                "batch_id": batch_id,
                "manifest_sha256": sha256(manifest_path),
                "normalized_records_sha256": sha256(normalized_path),
                "raw_response_count": len(list((batch_dir / "raw_responses").glob("*.json"))),
                "model_versions_observed": manifest["model_versions_observed"],
                "source_batch_sha256": manifest["source_batch_sha256"],
            }
        )

    reviews = pd.DataFrame(records)
    if len(reviews) != EXPECTED_RECORDS:
        raise RuntimeError(f"REAL_EXTERNAL_REVIEW_RECORD_COUNT={len(reviews)}")
    if reviews.duplicated(["case_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_REAL_EXTERNAL_REVIEW")
    expected = set(map(tuple, pair_frame[["case_id", "action_id"]].to_numpy()))
    actual = set(map(tuple, reviews[["case_id", "action_id"]].to_numpy()))
    if actual != expected:
        raise RuntimeError("REAL_REVIEW_CASE_ACTION_SET_MISMATCH")
    return reviews, provenance


def query_contributions(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in frame.groupby("case_id", sort=False):
        group = group.sort_values([score_column, "action_id"], ascending=[False, True])
        relevance = group["relevance_score"].to_numpy(dtype=float)
        score = group[score_column].to_numpy(dtype=float)
        k = min(3, len(group))
        order = np.argsort(-score)[:k]
        ideal = np.argsort(-relevance)[:k]
        gains = np.power(2.0, relevance) - 1.0
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum(gains[order] * discounts))
        idcg = float(np.sum(gains[ideal] * discounts))
        positives = relevance >= 1.0
        ranked = np.argsort(-score)
        ranked_positive = positives[ranked]

        pair_correct = 0
        pair_total = 0
        for left in range(len(relevance)):
            for right in range(left + 1, len(relevance)):
                if relevance[left] == relevance[right] or score[left] == score[right]:
                    continue
                pair_total += 1
                pair_correct += int(
                    (relevance[left] > relevance[right])
                    == (score[left] > score[right])
                )

        rows.append(
            {
                "case_id": case_id,
                "ndcg_at_3": 0.0 if idcg <= 0 else dcg / idcg,
                "exact_best_top1_agreement": float(
                    relevance[int(np.argmax(score))] >= float(np.max(relevance)) - 1e-12
                ),
                "precision_at_1": float(ranked_positive[0]) if positives.any() else np.nan,
                "mrr": (
                    1.0 / float(np.flatnonzero(ranked_positive)[0] + 1)
                    if positives.any()
                    else np.nan
                ),
                "recall_at_3": (
                    float(ranked_positive[:k].sum()) / float(positives.sum())
                    if positives.any()
                    else np.nan
                ),
                "pair_correct": pair_correct,
                "pair_total": pair_total,
                "top1_action": str(group.iloc[0]["action_id"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate(contributions: pd.DataFrame) -> dict[str, Any]:
    pair_total = int(contributions["pair_total"].sum())
    pair_correct = int(contributions["pair_correct"].sum())
    return {
        "evaluated_complete_case_count": int(len(contributions)),
        "ndcg_at_3": float(contributions["ndcg_at_3"].mean()),
        "exact_best_top1_agreement": float(
            contributions["exact_best_top1_agreement"].mean()
        ),
        "precision_at_1_relevance_ge_1": float(
            contributions["precision_at_1"].dropna().mean()
        ),
        "mrr_relevance_ge_1": float(contributions["mrr"].dropna().mean()),
        "recall_at_3_relevance_ge_1": float(
            contributions["recall_at_3"].dropna().mean()
        ),
        "pairwise_accuracy": float(pair_correct / pair_total) if pair_total else 0.0,
        "unique_top1_actions": int(contributions["top1_action"].nunique()),
        "invalid_action_rate": 0.0,
    }


def bootstrap_difference(
    full: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    full = full.set_index("case_id")
    baseline = baseline.set_index("case_id")
    if set(full.index) != set(baseline.index):
        raise RuntimeError("BOOTSTRAP_QUERY_SET_MISMATCH")
    case_ids = np.array(sorted(full.index), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ndcg_values: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        ndcg_values.append(
            float(full.loc[sampled, "ndcg_at_3"].mean())
            - float(baseline.loc[sampled, "ndcg_at_3"].mean())
        )
    values = np.asarray(ndcg_values, dtype=float)
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "mean_full_minus_baseline_ndcg_at_3": float(values.mean()),
        "ci_low_95": float(np.quantile(values, 0.025)),
        "ci_high_95": float(np.quantile(values, 0.975)),
    }


def write_protocol() -> int:
    assert_development_authority()
    pair_frame, request_hashes = load_request_contract()
    scored = score_frozen_ranker(pair_frame)
    scored["baseline_score"] = action_stage_baseline(scored)
    if not np.isfinite(scored["baseline_score"].to_numpy(dtype=float)).all():
        raise RuntimeError("NONFINITE_PREDECLARED_BASELINE_SCORE")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "PREREGISTERED_BEFORE_PROVIDER_REVIEW":
            raise RuntimeError("INVALID_EXISTING_PANEL_B_PROTOCOL")
        print("PANEL_B_EVALUATION_PROTOCOL=ALREADY_PREREGISTERED")
        return 0

    protocol = {
        "schema_version": "panel_b_final_heldout_protocol_v1",
        "status": "PREREGISTERED_BEFORE_PROVIDER_REVIEW",
        "created_at_utc": utc_now(),
        "panel": "B",
        "case_count": EXPECTED_CASES,
        "expected_real_external_review_records": EXPECTED_RECORDS,
        "provider": PROVIDER,
        "requested_model": MODEL_NAME,
        "prompt_sha256": PROMPT_SHA256,
        "request_batch_sha256": request_hashes,
        "evaluator_sha256": sha256(Path(__file__)),
        "development_freeze_sha256": sha256(DEVELOPMENT_FREEZE_PATH),
        "frozen_panel_a_manifest_sha256": PANEL_A_MANIFEST_SHA256,
        "frozen_panel_a_reviews_sha256": PANEL_A_REVIEWS_SHA256,
        "frozen_ranker_manifest_sha256": sha256(
            RANKER_DIR / "RANKER_PANEL_A_FREEZE_MANIFEST.json"
        ),
        "score_contract": "clip(native_ordinal_prediction / 3, 0, 1)",
        "calibration": "NONE_RAW_EBM_SELECTED_ON_PANEL_A",
        "primary_metric": "NDCG@3 across evidence-complete Panel-B cases",
        "secondary_metrics": [
            "exact_best_top1_agreement",
            "precision_at_1_relevance_ge_1",
            "mrr_relevance_ge_1",
            "recall_at_3_relevance_ge_1",
            "pairwise_accuracy",
            "invalid_action_rate",
            "ranker_score_mae_against_relevance_div_3",
            "ranker_score_rmse_against_relevance_div_3",
        ],
        "baseline": "Panel-A retained action+stage mean relevance",
        "abstention_policy": (
            "Preserve every real review record; primary ranking metrics use only cases "
            "with non-abstained reviews for every feasible candidate action."
        ),
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "unit": "case",
            "estimand": "full minus action+stage-only NDCG@3",
        },
        "prohibitions": [
            "No Panel-B tuning or model selection",
            "No synthetic, substituted, or fabricated reviews",
            "No causal-effect claim",
        ],
        "simulator_language": "model-implied risk delta",
        "runtime_authorized": False,
        "final_metrics_claimed": False,
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("PANEL_B_EVALUATION_PROTOCOL=PREREGISTERED")
    print(f"PANEL_B_CASES={EXPECTED_CASES}")
    print(f"EXPECTED_REAL_EXTERNAL_REVIEW_RECORDS={EXPECTED_RECORDS}")
    return 0


def evaluate_once() -> int:
    if FINAL_MANIFEST_PATH.exists() or STARTED_PATH.exists() or FAILED_PATH.exists():
        raise RuntimeError("PANEL_B_EVALUATION_ALREADY_STARTED_OR_COMPLETED")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_PROVIDER_REVIEW":
        raise RuntimeError("PANEL_B_PROTOCOL_NOT_PREREGISTERED")
    if protocol.get("evaluator_sha256") != sha256(Path(__file__)):
        raise RuntimeError("EVALUATOR_CHANGED_AFTER_PREREGISTRATION")

    assert_development_authority()
    pair_frame, request_hashes = load_request_contract()
    if request_hashes != protocol["request_batch_sha256"]:
        raise RuntimeError("PANEL_B_REQUEST_BATCH_CHANGED_AFTER_PREREGISTRATION")

    STARTED_PATH.write_text(
        json.dumps(
            {
                "status": "STARTED_EXACTLY_ONCE",
                "started_at_utc": utc_now(),
                "evaluator_sha256": sha256(Path(__file__)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        reviews, provenance = load_real_reviews(pair_frame)
        scored = score_frozen_ranker(pair_frame)
        scored["baseline_score"] = action_stage_baseline(scored)
        evaluation = scored.merge(
            reviews[
                [
                    "case_id",
                    "action_id",
                    "relevance_score",
                    "abstain",
                    "response_record_sha256",
                ]
            ],
            on=["case_id", "action_id"],
            how="left",
            validate="one_to_one",
        )
        if evaluation["relevance_score"].isna().any():
            raise RuntimeError("PANEL_B_REVIEW_JOIN_INCOMPLETE")
        complete = ~evaluation.groupby("case_id")["abstain"].transform("any").astype(bool)
        primary = evaluation.loc[complete].copy()
        if primary["case_id"].nunique() < 2:
            raise RuntimeError("INSUFFICIENT_EVIDENCE_COMPLETE_PANEL_B_CASES")

        full_contrib = query_contributions(primary, "public_score")
        baseline_contrib = query_contributions(primary, "baseline_score")
        full_metrics = aggregate(full_contrib)
        baseline_metrics = aggregate(baseline_contrib)
        bootstrap = bootstrap_difference(full_contrib, baseline_contrib)
        target = primary["relevance_score"].to_numpy(dtype=float) / 3.0
        predicted = primary["public_score"].to_numpy(dtype=float)
        full_metrics["ranker_score_mae_against_relevance_div_3"] = float(
            np.mean(np.abs(predicted - target))
        )
        full_metrics["ranker_score_rmse_against_relevance_div_3"] = float(
            math.sqrt(np.mean(np.square(predicted - target)))
        )

        abstained_records = int(evaluation["abstain"].astype(bool).sum())
        evidence_complete_cases = int(primary["case_id"].nunique())
        metrics = {
            "scope": "PANEL_B_FINAL_HELDOUT",
            "panel_b_case_count": EXPECTED_CASES,
            "real_external_review_record_count": int(len(reviews)),
            "abstained_review_record_count": abstained_records,
            "evidence_complete_case_count": evidence_complete_cases,
            "evidence_complete_case_coverage": evidence_complete_cases / EXPECTED_CASES,
            "frozen_five_ebm_ranker": full_metrics,
            "panel_a_action_stage_only_baseline": baseline_metrics,
            "paired_case_bootstrap": bootstrap,
        }

        frozen_reviews_path = OUTPUT_DIR / "panel_b_real_external_reviews_frozen.jsonl"
        score_path = OUTPUT_DIR / "panel_b_final_heldout_scores.parquet"
        safe_columns = [
            "case_id",
            "stage",
            "action_id",
            "native_ordinal_score",
            "public_score",
            "baseline_score",
            "relevance_score",
            "abstain",
            "response_record_sha256",
        ]
        frozen_reviews_path.write_text(
            "\n".join(canonical_json(row) for row in reviews.to_dict("records")) + "\n",
            encoding="utf-8",
        )
        evaluation[safe_columns].to_parquet(score_path, index=False)
        FINAL_METRICS_PATH.write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": "panel_b_final_heldout_manifest_v1",
            "status": "PASS",
            "scope": "PANEL_B_FINAL_HELDOUT",
            "created_at_utc": utc_now(),
            "panel_b_touched": True,
            "runtime_authorized": False,
            "final_metrics_claimed": True,
            "panel_b_case_count": EXPECTED_CASES,
            "real_external_review_record_count": int(len(reviews)),
            "failed_provider_calls": 0,
            "provider": PROVIDER,
            "model_name": MODEL_NAME,
            "prompt_sha256": PROMPT_SHA256,
            "provider_batch_provenance": provenance,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "evaluator_sha256": sha256(Path(__file__)),
            "frozen_reviews_sha256": sha256(frozen_reviews_path),
            "scores_sha256": sha256(score_path),
            "metrics_sha256": sha256(FINAL_METRICS_PATH),
            "development_artifacts_changed": False,
            "post_panel_b_tuning_permitted": False,
            "simulator_language": "model-implied risk delta",
        }
        FINAL_MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        checksums = [
            f"{sha256(path)}  {path.name}"
            for path in sorted(OUTPUT_DIR.iterdir())
            if path.is_file() and path.name != "checksums.sha256"
        ]
        (OUTPUT_DIR / "checksums.sha256").write_text(
            "\n".join(checksums) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        FAILED_PATH.write_text(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "failed_at_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "runtime_authorized": False,
                    "final_metrics_claimed": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    print("PHASE=8")
    print("STATUS=PASS")
    print("SCOPE=PANEL_B_FINAL_HELDOUT")
    print(f"PANEL_B_CASES={EXPECTED_CASES}")
    print(f"REAL_EXTERNAL_REVIEW_RECORDS={len(reviews)}")
    print("FAILED_PROVIDER_CALLS=0")
    print("RUNTIME_AUTHORIZED=FALSE")
    print("FINAL_METRICS_CLAIMED=TRUE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--evaluate-once", action="store_true")
    args = parser.parse_args()
    return write_protocol() if args.preflight else evaluate_once()


if __name__ == "__main__":
    raise SystemExit(main())
