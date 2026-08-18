"""Idempotent load of frozen recommendation results into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.finalization import BUNDLE_VERSION, FREEZE_VERSION, STATE_VERSION  # noqa: E402
from src.recommendation.persistence.runtime import ensure_bundle, upsert_enrollment_external  # noqa: E402
from src.database.connection import engine, transaction  # noqa: E402
from src.database.repository import _json  # noqa: E402


def _payload(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    return _json(value)


def _counts() -> dict:
    queries = {
        "students": "SELECT count(*) FROM catalog.student",
        "courses": "SELECT count(*) FROM catalog.course",
        "enrollments": "SELECT count(*) FROM catalog.enrollment",
        "state_snapshots": "SELECT count(*) FROM recommendation.state_snapshot",
        "runs": "SELECT count(*) FROM recommendation.run",
        "scores": "SELECT count(*) FROM recommendation.score",
        "explanations": "SELECT count(*) FROM recommendation.explanation",
        "plans": "SELECT count(*) FROM recommendation.plan",
    }
    with engine.connect() as connection:
        return {name: int(connection.execute(text(sql)).scalar()) for name, sql in queries.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "artifacts/recommendation/final/oulad_recommendation_scores.parquet")
    parser.add_argument("--plans", type=Path, default=ROOT / "artifacts/recommendation/final/oulad_recommendation_plans.parquet")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.scores.exists() or not args.plans.exists():
        print(json.dumps({"status": "BLOCKED_MISSING_INFERENCE_ARTIFACTS"}))
        return 2
    plans = pd.read_parquet(args.plans)
    scores = pd.read_parquet(args.scores)
    if args.limit:
        keep = set(plans.sort_values("case_id").head(args.limit)["case_id"].astype(str))
        plans = plans[plans["case_id"].astype(str).isin(keep)].copy()
        scores = scores[scores["case_id"].astype(str).isin(keep)].copy()
    print(json.dumps({"dry_run": args.dry_run, "cases": int(len(plans)), "score_rows": int(len(scores)), "before": _counts()}, indent=2))
    if args.dry_run:
        return 0
    bundle_id = ensure_bundle(BUNDLE_VERSION, FREEZE_VERSION, {"freeze_version": FREEZE_VERSION})
    enrollments = plans[["student_id", "module", "presentation", "enrollment_identity"]].drop_duplicates()
    enrollment_map = {}
    for row in enrollments.itertuples(index=False):
        enrollment_map[str(row.enrollment_identity)] = upsert_enrollment_external(
            str(row.student_id), str(row.module), str(row.presentation), str(row.enrollment_identity)
        )
    with transaction() as connection:
        action_ids = {row.action_key: str(row.action_id) for row in connection.execute(text("SELECT action_id, action_key FROM recommendation.action"))}
    loaded = 0
    for start in range(0, len(plans), args.batch_size):
        batch = plans.iloc[start:start + args.batch_size]
        case_ids = set(batch["case_id"].astype(str))
        score_batch = scores[scores["case_id"].astype(str).isin(case_ids)]
        snap_rows, run_rows, score_rows, expl_rows, plan_rows = [], [], [], [], []
        for plan in batch.itertuples(index=False):
            enrollment_id = enrollment_map[str(plan.enrollment_identity)]
            request_key = f"{plan.enrollment_identity}|{plan.stage}|{BUNDLE_VERSION}|{STATE_VERSION}"
            snap_rows.append({
                "e": enrollment_id, "g": plan.stage, "v": STATE_VERSION, "c": str(plan.case_id),
                "r": float(plan.risk_probability), "s": _json({"bundle_version": BUNDLE_VERSION}),
            })
            run_rows.append({
                "e": enrollment_id, "b": bundle_id, "g": plan.stage, "k": request_key,
                "p": plan.plan_status, "r": float(plan.risk_probability),
            })
        with transaction() as connection:
            connection.execute(text(
                "INSERT INTO recommendation.state_snapshot(enrollment_id,stage,state_version,case_id,risk_probability,features,source_lineage) "
                "VALUES (:e,:g,:v,:c,:r,'{}'::jsonb,:s) "
                "ON CONFLICT (enrollment_id,stage,state_version) DO UPDATE SET case_id=EXCLUDED.case_id, risk_probability=EXCLUDED.risk_probability"
            ), snap_rows)
            connection.execute(text(
                "INSERT INTO recommendation.run(enrollment_id,bundle_id,stage,request_key,plan_status,risk_probability) "
                "VALUES (:e,:b,:g,:k,:p,:r) "
                "ON CONFLICT (request_key) DO UPDATE SET plan_status=EXCLUDED.plan_status"
            ), run_rows)
            run_map = {
                row.request_key: str(row.run_id)
                for row in connection.execute(
                    text("SELECT request_key, run_id FROM recommendation.run WHERE request_key IN :k").bindparams(bindparam("k", expanding=True)),
                    {"k": [row["k"] for row in run_rows]},
                )
            }
            plan_by_case = {str(plan.case_id): plan for plan in batch.itertuples(index=False)}
            for score in score_batch.itertuples(index=False):
                plan = plan_by_case[str(score.case_id)]
                request_key = f"{plan.enrollment_identity}|{plan.stage}|{BUNDLE_VERSION}|{STATE_VERSION}"
                run_id = run_map[request_key]
                aid = action_ids[score.action_id]
                score_rows.append({
                    "r": run_id, "a": aid, "w": float(score.raw_score), "v": float(score.relevance_score),
                    "n": int(score.rank), "f": score.feasibility_status, "l": score.release_status,
                    "q": score.quality_warning, "m": BUNDLE_VERSION,
                })
                expl_rows.append({
                    "r": run_id, "a": aid, "i": float(score.intercept),
                    "p": _payload(score.top_positive_reasons), "n": _payload(score.top_negative_reasons),
                    "x": _json({"raw_score": float(score.raw_score)}),
                })
            for plan in batch.itertuples(index=False):
                request_key = f"{plan.enrollment_identity}|{plan.stage}|{BUNDLE_VERSION}|{STATE_VERSION}"
                top = list(plan.top_actions) if hasattr(plan.top_actions, "__iter__") and not isinstance(plan.top_actions, str) else []
                plan_rows.append({"r": run_map[request_key], "p": plan.plan_status, "t": top})
            connection.execute(text(
                "INSERT INTO recommendation.score(run_id,action_id,raw_score,relevance_score,rank,feasibility_status,release_status,quality_warning,model_version) "
                "VALUES (:r,:a,:w,:v,:n,:f,:l,:q,:m) ON CONFLICT (run_id,action_id) DO UPDATE SET relevance_score=EXCLUDED.relevance_score, rank=EXCLUDED.rank"
            ), score_rows)
            connection.execute(text(
                "INSERT INTO recommendation.explanation(run_id,action_id,intercept,top_positive,top_negative,payload,explanation_version) "
                "VALUES (:r,:a,:i,:p,:n,:x,'ebm_local_v1') ON CONFLICT (run_id,action_id) DO UPDATE SET payload=EXCLUDED.payload"
            ), expl_rows)
            connection.execute(text(
                "INSERT INTO recommendation.plan(run_id,plan_status,top_action_keys) VALUES (:r,:p,:t) "
                "ON CONFLICT (run_id) DO UPDATE SET plan_status=EXCLUDED.plan_status, top_action_keys=EXCLUDED.top_action_keys"
            ), plan_rows)
        loaded += len(batch)
        print(json.dumps({"loaded_cases": loaded}), flush=True)
    print(json.dumps({"loaded_cases": loaded, "after": _counts()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
