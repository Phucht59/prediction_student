from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from psycopg2.extras import Json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "database" / "schema.sql"


def main() -> None:
    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "skipped", "reason": "DATABASE_URL is not configured"}, indent=2))
        return
    if database_url.startswith("postgresql+psycopg2://"):
        database_url = "postgresql://" + database_url.removeprefix("postgresql+psycopg2://")

    import psycopg2

    demo_payload = {
        "source": "scripts/demo_postgres_schema.py",
        "purpose": "verify paper schema insert/query",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            cur.execute(
                """
                INSERT INTO paper_runs (generated_at, result_rows, summary_rows, postgres_status, postgres_message, run_payload)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING paper_run_id
                """,
                (
                    datetime.now(timezone.utc),
                    0,
                    0,
                    "demo_ok",
                    "Schema demo insert/query succeeded.",
                    Json(demo_payload),
                ),
            )
            run_id = int(cur.fetchone()[0])
            cur.execute(
                """
                INSERT INTO paper_evaluation_metrics
                    (run_id, dataset, model_name, protocol_class, accuracy, precision_macro, recall_macro, f1_macro, metric_payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING metric_id
                """,
                (
                    run_id,
                    "demo",
                    "postgres_schema_demo",
                    "demo",
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    Json({"note": "demo metric row"}),
                ),
            )
            metric_id = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT pr.paper_run_id, pr.postgres_status, pem.dataset, pem.model_name, pem.f1_macro
                FROM paper_runs pr
                JOIN paper_evaluation_metrics pem ON pem.run_id = pr.paper_run_id
                WHERE pr.paper_run_id = %s AND pem.metric_id = %s
                """,
                (run_id, metric_id),
            )
            row = cur.fetchone()
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": int(row[0]),
                "postgres_status": row[1],
                "dataset": row[2],
                "model_name": row[3],
                "f1_macro": float(row[4]),
                "connection": "DATABASE_URL configured; value hidden",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
