"""CLI: inspect and load the live student_db PostgreSQL database."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from psycopg2.extras import Json, execute_values

from src.database.connection import DatabaseSettings, connect_with_retry, load_dotenv

ROOT = Path(__file__).resolve().parents[2]
LIVE_SQL_DIR = ROOT / "database" / "live"
RAW_SQL = LIVE_SQL_DIR / "001_create_raw_schema.sql"
RELATIONSHIP_SQL = LIVE_SQL_DIR / "003_add_relationships.sql"
RAW_TRUNCATE_SQL = """
TRUNCATE TABLE
    raw.oulad_student_vle,
    raw.oulad_student_assessment,
    raw.oulad_student_registration,
    raw.oulad_student_info,
    raw.oulad_assessments,
    raw.oulad_vle,
    raw.oulad_courses,
    raw.uci_mat,
    raw.uci_por
RESTART IDENTITY
"""
DEFAULT_RAW_CANDIDATES = (
    Path(r"C:\hufit\kltn\data\raw"),
    ROOT / "data" / "raw",
)
STAGE_MAP = {
    "20pct": "20",
    "35pct": "35",
    "50pct": "50",
    "75pct": "75",
    "100pct": "FINAL",
    "EARLY_20": "20",
    "EARLY_35": "35",
    "MIDDLE_50": "50",
    "LATE_75": "75",
}


def settings() -> DatabaseSettings:
    load_dotenv()
    return DatabaseSettings.from_environment()


def raw_dir() -> Path:
    load_dotenv()
    override = Path(os_env("RAW_DATA_DIR")) if os_env("RAW_DATA_DIR") else None
    if override and override.is_dir():
        return override
    for candidate in DEFAULT_RAW_CANDIDATES:
        if (candidate / "studentInfo.csv").is_file() or (candidate / "student-mat.csv").is_file():
            return candidate
    raise FileNotFoundError("raw OULAD/UCI CSVs not found; set RAW_DATA_DIR")


def os_env(name: str) -> str:
    import os

    return os.getenv(name, "").strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_status(_: argparse.Namespace) -> int:
    connection = connect_with_retry(settings())
    try:
        with connection.cursor() as cursor:
            print("database", connection.info.dbname, "user", connection.info.user)
            queries = {
                "catalog.student": "SELECT COUNT(*) FROM catalog.student",
                "catalog.enrollment": "SELECT COUNT(*) FROM catalog.enrollment",
                "prediction.prediction": "SELECT COUNT(*) FROM prediction.prediction",
                "recommendation.recommendation": "SELECT COUNT(*) FROM recommendation.recommendation",
                "recommendation.recommendation_item": "SELECT COUNT(*) FROM recommendation.recommendation_item",
            }
            for label, sql in queries.items():
                try:
                    cursor.execute(sql)
                    print(f"{label}\t{cursor.fetchone()[0]}")
                except Exception as exc:
                    connection.rollback()
                    print(f"{label}\tERROR {exc}")
            cursor.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema='raw' ORDER BY 1
                """
            )
            raw_tables = [row[0] for row in cursor.fetchall()]
            print("raw_tables", raw_tables or "NONE")
            if raw_tables:
                for table in raw_tables:
                    if table == "load_manifest":
                        continue
                    cursor.execute(f"SELECT COUNT(*) FROM raw.{table}")
                    print(f"raw.{table}\t{cursor.fetchone()[0]}")
    finally:
        connection.close()
    return 0


def apply_live_schema(cursor) -> None:
    cursor.execute("SET statement_timeout = 0")
    cursor.execute(RAW_SQL.read_text(encoding="utf-8"))
    if RELATIONSHIP_SQL.is_file():
        cursor.execute(RELATIONSHIP_SQL.read_text(encoding="utf-8"))


def cmd_migrate_raw(_: argparse.Namespace) -> int:
    connection = connect_with_retry(settings())
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = 0")
            cursor.execute("SET maintenance_work_mem = '1GB'")
            apply_live_schema(cursor)
        print("live schema ready")
    finally:
        connection.close()
    return 0


def _copy_csv(cursor, table: str, path: Path, copy_sql: str) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        cursor.copy_expert(copy_sql, handle)
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cursor.fetchone()[0])


def _load_uci_json(cursor, table: str, path: Path) -> int:
    cursor.execute(f"TRUNCATE {table}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            rows.append((Json(row), path.name))
    execute_values(
        cursor,
        f"INSERT INTO {table} (payload, source_file) VALUES %s",
        rows,
        page_size=500,
    )
    return len(rows)


def cmd_load_raw(_: argparse.Namespace) -> int:
    source = raw_dir()
    dest = ROOT / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    connection = connect_with_retry(settings())
    try:
        connection.autocommit = False
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = 0")
            apply_live_schema(cursor)
            cursor.execute(RAW_TRUNCATE_SQL)
            specs = [
                (
                    "raw.uci_mat",
                    source / "student-mat.csv",
                    None,
                    "uci_json",
                ),
                (
                    "raw.uci_por",
                    source / "student-por.csv",
                    None,
                    "uci_json",
                ),
                (
                    "raw.oulad_courses",
                    source / "courses.csv",
                    "COPY raw.oulad_courses (code_module, code_presentation, module_presentation_length) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')",
                    "copy",
                ),
                (
                    "raw.oulad_assessments",
                    source / "assessments.csv",
                    "COPY raw.oulad_assessments (code_module, code_presentation, id_assessment, assessment_type, date, weight) FROM STDIN WITH (FORMAT csv, HEADER true, FORCE_NULL (date, weight))",
                    "copy",
                ),
                (
                    "raw.oulad_vle",
                    source / "vle.csv",
                    "COPY raw.oulad_vle (id_site, code_module, code_presentation, activity_type, week_from, week_to) FROM STDIN WITH (FORMAT csv, HEADER true, FORCE_NULL (week_from, week_to))",
                    "copy",
                ),
                (
                    "raw.oulad_student_info",
                    source / "studentInfo.csv",
                    "COPY raw.oulad_student_info (code_module, code_presentation, id_student, gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability, final_result) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')",
                    "copy",
                ),
                (
                    "raw.oulad_student_registration",
                    source / "studentRegistration.csv",
                    "COPY raw.oulad_student_registration (code_module, code_presentation, id_student, date_registration, date_unregistration) FROM STDIN WITH (FORMAT csv, HEADER true, FORCE_NULL (date_registration, date_unregistration))",
                    "copy",
                ),
                (
                    "raw.oulad_student_assessment",
                    source / "studentAssessment.csv",
                    "COPY raw.oulad_student_assessment (id_assessment, id_student, date_submitted, is_banked, score) FROM STDIN WITH (FORMAT csv, HEADER true, FORCE_NULL (date_submitted, score))",
                    "copy",
                ),
                (
                    "raw.oulad_student_vle",
                    source / "studentVle.csv",
                    "COPY raw.oulad_student_vle (code_module, code_presentation, id_student, id_site, date, sum_click) FROM STDIN WITH (FORMAT csv, HEADER true, FORCE_NULL (date, sum_click))",
                    "copy",
                ),
            ]
            for table, path, copy_sql, kind in specs:
                if not path.is_file():
                    print("SKIP missing", path)
                    continue
                target = dest / path.name
                if path.resolve() != target.resolve():
                    shutil.copy2(path, target)
                print("LOADING", table, path.name, "bytes", path.stat().st_size)
                if kind == "uci_json":
                    count = _load_uci_json(cursor, table, path)
                else:
                    count = _copy_csv(cursor, table, path, copy_sql)
                cursor.execute(
                    """
                    INSERT INTO raw.load_manifest(table_name, source_path, source_sha256, row_count, loaded_at)
                    VALUES (%s,%s,%s,%s, NOW())
                    ON CONFLICT (table_name) DO UPDATE SET
                        source_path=EXCLUDED.source_path,
                        source_sha256=EXCLUDED.source_sha256,
                        row_count=EXCLUDED.row_count,
                        loaded_at=EXCLUDED.loaded_at
                    """,
                    (table, str(path), sha256_file(path), count),
                )
                print("  rows", count)
            connection.commit()
        print("RAW_LOAD_COMPLETE")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return 0


def cmd_load_predictions(_: argparse.Namespace) -> int:
    parquet = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "c0_oof_predictions.parquet"
    frame = pd.read_parquet(parquet)
    connection = connect_with_retry(settings())
    return _insert_predictions(connection, frame, parquet)


def _insert_predictions(connection, frame: pd.DataFrame, parquet: Path) -> int:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT model_id FROM prediction.model WHERE model_key='hybrid' LIMIT 1")
            row = cursor.fetchone()
            if row:
                model_id = row[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO prediction.model
                        (model_key, display_name, version, model_type, artifact_path, config_path, is_active)
                    VALUES ('hybrid', 'Hybrid C0', 'final', 'hybrid_cnn_bilstm', %s, %s, TRUE)
                    RETURNING model_id
                    """,
                    (
                        "artifacts/prediction/reconstructed",
                        "configs/prediction/hybrid_final.json",
                    ),
                )
                model_id = cursor.fetchone()[0]
            cursor.execute("SET statement_timeout = 0")
            cursor.execute(
                """
                DELETE FROM recommendation.recommendation_item
                WHERE recommendation_id IN (
                    SELECT recommendation_id FROM recommendation.recommendation
                    WHERE metadata->>'authority' = 'Five-EBM-C0'
                )
                """
            )
            cursor.execute("DELETE FROM recommendation.recommendation WHERE metadata->>'authority' = 'Five-EBM-C0'")
            cursor.execute(
                """
                DELETE FROM prediction.prediction
                WHERE run_id IN (
                    SELECT run_id FROM prediction.model_run
                    WHERE dataset='oulad' AND task='binary_risk_oof'
                )
                """
            )
            cursor.execute("DELETE FROM prediction.model_run WHERE dataset='oulad' AND task='binary_risk_oof'")
            cursor.execute(
                """
                INSERT INTO prediction.model_run (model_id, dataset, task, started_at, completed_at, status, metadata)
                VALUES (%s, 'oulad', 'binary_risk_oof', NOW(), NOW(), 'completed', %s)
                RETURNING run_id
                """,
                (
                    model_id,
                    Json(
                        {
                            "source": str(parquet.as_posix()),
                            "row_count": int(len(frame)),
                            "authority": "Phase4 Hybrid C0",
                        }
                    ),
                ),
            )
            run_id = cursor.fetchone()[0]
            cursor.execute("DELETE FROM prediction.prediction WHERE run_id=%s", (run_id,))
            cursor.execute(
                """
                SELECT e.enrollment_id, s.external_student_id, c.course_code, c.presentation
                FROM catalog.enrollment e
                JOIN catalog.student s ON s.student_id=e.student_id
                JOIN catalog.course c ON c.course_id=e.course_id
                """
            )
            enroll = {
                (str(sid), str(code), str(pres)): eid
                for eid, sid, code, pres in cursor.fetchall()
            }
            batch = []
            missing = 0
            for row in frame.itertuples(index=False):
                key = (f"OULAD:{int(row.id_student)}", str(row.code_module), str(row.code_presentation))
                enrollment_id = enroll.get(key)
                if enrollment_id is None:
                    missing += 1
                    continue
                stage = STAGE_MAP.get(str(row.stage_or_endpoint), STAGE_MAP.get(str(row.stage), str(row.stage)))
                batch.append(
                    (
                        enrollment_id,
                        run_id,
                        stage,
                        float(row.risk_probability),
                        bool(int(row.predicted_risk)),
                        float(row.prediction_threshold),
                        float(row.uncertainty),
                        Json(
                            {
                                "query_id": str(row.query_id),
                                "record_id": str(row.record_id),
                                "inner_fold": int(row.inner_fold),
                                "split_role": str(row.split_role),
                            }
                        ),
                    )
                )
            execute_values(
                cursor,
                """
                INSERT INTO prediction.prediction
                    (enrollment_id, run_id, stage, risk_probability, predicted_risk, threshold, uncertainty, metadata)
                VALUES %s
                """,
                batch,
                page_size=1000,
            )
            connection.commit()
            print("PREDICTION_LOAD", "run", run_id, "inserted", len(batch), "unmatched", missing)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return 0


def cmd_load_recommendations(_: argparse.Namespace) -> int:
    scores = pd.read_parquet(ROOT / "artifacts" / "recommend_hybrid" / "v3" / "ranker" / "oof_predictions.parquet")
    connection = connect_with_retry(settings())
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT action_id, action_key FROM recommendation.action")
            actions = {str(key): aid for aid, key in cursor.fetchall()}
            for key in [
                "ASSESSMENT_COMPLETION",
                "RECOVER_ENGAGEMENT",
                "STUDY_REGULARITY",
                "TARGETED_CONTENT_REVIEW",
                "QUIZ_RETRIEVAL_PRACTICE",
            ]:
                if key not in actions:
                    cursor.execute(
                        """
                        INSERT INTO recommendation.action (action_key, action_name, description, is_active)
                        VALUES (%s,%s,%s,TRUE) RETURNING action_id
                        """,
                        (key, key.replace("_", " ").title(), "V3 canonical action"),
                    )
                    actions[key] = cursor.fetchone()[0]
            cursor.execute(
                """
                SELECT p.prediction_id, p.metadata->>'query_id'
                FROM prediction.prediction p
                WHERE p.metadata ? 'query_id'
                """
            )
            pred = {qid: pid for pid, qid in cursor.fetchall() if qid}
            cursor.execute("SET statement_timeout = 0")
            cursor.execute(
                """
                DELETE FROM recommendation.recommendation_item
                WHERE recommendation_id IN (
                    SELECT recommendation_id FROM recommendation.recommendation
                    WHERE metadata->>'authority' = 'Five-EBM-C0'
                )
                """
            )
            cursor.execute("DELETE FROM recommendation.recommendation WHERE metadata->>'authority' = 'Five-EBM-C0'")
            inserted = 0
            for query_id, group in scores.groupby("query_id", sort=False):
                prediction_id = pred.get(str(query_id))
                if prediction_id is None:
                    continue
                ordered = group.sort_values(["score", "action_id"], ascending=[False, True])
                feasible = ordered.loc[ordered.eligible.astype(bool)]
                if feasible.empty:
                    route = "NO_FEASIBLE_ACTION"
                    ranked = ordered.head(0)
                else:
                    route = "RECOMMEND"
                    ranked = feasible.head(3)
                cursor.execute(
                    """
                    INSERT INTO recommendation.recommendation (prediction_id, risk_band, route_status, metadata)
                    VALUES (%s,%s,%s,%s) RETURNING recommendation_id
                    """,
                    (
                        prediction_id,
                        "C0",
                        route,
                        Json({"query_id": str(query_id), "authority": "Five-EBM-C0"}),
                    ),
                )
                rec_id = cursor.fetchone()[0]
                for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                    action_id = actions.get(str(row.action_id))
                    if action_id is None:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO recommendation.recommendation_item
                            (recommendation_id, action_id, rank_position, score, explanation, feasible)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            rec_id,
                            action_id,
                            rank,
                            float(row.score),
                            Json({"action_id": str(row.action_id)}),
                            bool(row.eligible),
                        ),
                    )
                inserted += 1
            connection.commit()
            print("V3_RECOMMENDATION_LOAD", "queries", inserted)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return 0


def cmd_load_all(args: argparse.Namespace) -> int:
    cmd_migrate_raw(args)
    cmd_load_raw(args)
    cmd_load_predictions(args)
    cmd_load_recommendations(args)
    return cmd_status(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live PostgreSQL loaders")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(handler=cmd_status)
    sub.add_parser("migrate-raw").set_defaults(handler=cmd_migrate_raw)
    sub.add_parser("load-raw").set_defaults(handler=cmd_load_raw)
    sub.add_parser("load-predictions").set_defaults(handler=cmd_load_predictions)
    sub.add_parser("load-recommendations").set_defaults(handler=cmd_load_recommendations)
    sub.add_parser("load-all").set_defaults(handler=cmd_load_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
