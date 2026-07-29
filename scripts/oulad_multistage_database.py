"""Build a disposable PostgreSQL replacement for unified OULAD stage evidence.

The canonical ``student_predict`` database is intentionally never opened for
writes.  This runner creates a separately named review database, loads the
already validated UCI authority, and replaces only the OULAD prediction views
with the four-stage unified authority.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values

from scripts import database_final as db
from scripts import unified_database as uci_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad"
EVIDENCE = ROOT / "artifacts" / "refactor" / "oulad_multistage_database_replacement_validation.json"
REPORT = ROOT / "reports" / "refactor" / "OULAD_MULTISTAGE_DATABASE_MIGRATION_REPORT.md"


def _create(settings: dict[str, str]) -> tuple[str, str]:
    admin = {
        **settings,
        "POSTGRES_USER": settings.get("POSTGRES_ADMIN_USER", "postgres"),
        "POSTGRES_PASSWORD": settings.get("POSTGRES_ADMIN_PASSWORD", settings["POSTGRES_PASSWORD"]),
    }
    connection = psycopg2.connect(uci_db._dsn(admin, "postgres"))
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            base = "student_predict_oulad_multistage_replacement"
            number = 1
            candidate = base
            while True:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (candidate,))
                if cursor.fetchone() is None:
                    break
                number += 1
                candidate = f"{base}_{number}"
            cursor.execute(sql.SQL("CREATE DATABASE {} ").format(sql.Identifier(candidate)))
    finally:
        connection.close()
    return candidate, uci_db._dsn(admin, candidate)


def _new_dataset_version(cursor, eligibility: pd.DataFrame) -> tuple[int, dict[str, int]]:
    cursor.execute("SELECT dataset_id FROM catalog.dataset WHERE slug='oulad'")
    dataset_id = int(cursor.fetchone()[0])
    digest = db._sha256(OUT / "eligibility_manifest.parquet")
    cursor.execute(
        """
        INSERT INTO catalog.dataset_version(
            dataset_id,version_label,source_sha256,row_count,data_schema,source_files,status,sealed_at
        ) VALUES (%s,'unified-stage-v1',%s,%s,%s,%s,'sealed',NOW())
        RETURNING dataset_version_id
        """,
        (dataset_id, digest, int(eligibility.base_record_id.nunique()), Json({"contract": "unified_oulad_four_stage"}), Json(["artifacts/final/unified_stage_aware_oulad/eligibility_manifest.parquet"])),
    )
    version = int(cursor.fetchone()[0])
    base = eligibility.drop_duplicates("base_record_id")
    records = [
        (
            version, row.base_record_id, str(int(row.id_student)), row.code_module,
            row.code_module, row.code_presentation,
            "At-risk" if int(row.target) else "Not-at-risk", float(row.target),
            Json({"unified_stage_record": True}),
        )
        for row in base.itertuples(index=False)
    ]
    execute_values(
        cursor,
        """INSERT INTO catalog.record(
            dataset_version_id,source_record_id,student_key,subject,code_module,
            code_presentation,target_label,target_numeric,attributes
        ) VALUES %s""",
        records,
        page_size=2000,
    )
    cursor.execute("SELECT source_record_id,record_pk FROM catalog.record WHERE dataset_version_id=%s", (version,))
    return version, {str(k): int(v) for k, v in cursor.fetchall()}


def _replace_oulad(cursor, record_map: dict[str, int], version: int) -> dict[str, int]:
    stage = pd.read_csv(OUT / "stage_metrics.csv").query("threshold_policy == 'INNER_OOF_STAGE_THRESHOLD'")
    per_class = pd.read_csv(OUT / "per_class_metrics.csv").query("threshold_policy == 'INNER_OOF_STAGE_THRESHOLD'")
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    source = "artifacts/final/unified_stage_aware_oulad"
    digest = db._sha256(OUT / "predictions.parquet")
    cursor.execute("UPDATE ml.run SET dataset_version_id=%s WHERE run_id LIKE 'final:oulad:%'", (version,))
    cursor.execute("DELETE FROM ml.prediction WHERE run_id LIKE 'final:oulad:%'")
    cursor.execute("DELETE FROM ml.metric WHERE run_id LIKE 'final:oulad:%' AND scope='stage'")
    metric_rows = []
    metric_names = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "risk_precision", "risk_recall", "risk_f1", "pr_auc", "roc_auc", "brier", "nll", "ece")
    for row in stage.itertuples(index=False):
        for name in metric_names:
            metric_rows.append((f"final:oulad:{row.model_id}", name, float(getattr(row, name)), "stage", "ensemble", row.prediction_stage, None, None, None, None, "proportion", Json({"source": f"{source}/stage_metrics.csv", "threshold_policy": row.threshold_policy})))
    for row in per_class.itertuples(index=False):
        for name in ("precision", "recall", "f1", "support"):
            metric_rows.append((f"final:oulad:{row.model_id}", name, float(getattr(row, name)), "stage", "per_class", row.prediction_stage, row.class_name, None, None, None, "count" if name == "support" else "proportion", Json({"source": f"{source}/per_class_metrics.csv"})))
    execute_values(cursor, """INSERT INTO ml.metric(
        run_id,metric_name,metric_value,scope,aggregation,prediction_stage,class_label,budget,fold,seed,unit,detail
    ) VALUES %s ON CONFLICT(run_id,metric_name,scope,aggregation,prediction_stage,class_label,budget,fold,seed)
    DO UPDATE SET metric_value=EXCLUDED.metric_value, unit=EXCLUDED.unit, detail=EXCLUDED.detail""", metric_rows, page_size=2000)
    prediction_rows = [
        (f"final:oulad:{row.model_id}", record_map[row.base_record_id], row.prediction_stage,
         "At-risk" if int(row.predicted_label) else "Not-at-risk",
         Json({"Not-at-risk": float(1-row.probability), "At-risk": float(row.probability)}),
         int(row.outer_fold), "mean_probability_across_fixed_seeds", f"{source}/predictions.parquet", digest,
         Json({"model_id": row.model_id, "cutoff_day": int(row.cutoff_day), "progress_fraction": float(row.progress_fraction), "true_label": int(row.target)}),
         "OPERATIONAL_RISK_SET", "INNER_OOF_STAGE_THRESHOLD", int(row.cutoff_day), float(row.progress_fraction), "At-risk" if int(row.target) else "Not-at-risk")
        for row in predictions.itertuples(index=False)
    ]
    execute_values(cursor, """INSERT INTO ml.prediction(
        run_id,record_pk,prediction_stage,predicted_label,probabilities,outer_fold,aggregation,source_artifact,source_sha256,detail,
        cohort,threshold_policy,cutoff_day,progress_fraction,true_label
    ) VALUES %s ON CONFLICT(run_id,record_pk,prediction_stage,cohort,threshold_policy) DO UPDATE SET
        predicted_label=EXCLUDED.predicted_label, probabilities=EXCLUDED.probabilities, outer_fold=EXCLUDED.outer_fold,
        aggregation=EXCLUDED.aggregation, source_artifact=EXCLUDED.source_artifact, source_sha256=EXCLUDED.source_sha256,
        detail=EXCLUDED.detail, cutoff_day=EXCLUDED.cutoff_day, progress_fraction=EXCLUDED.progress_fraction, true_label=EXCLUDED.true_label
    """, prediction_rows, page_size=2000)
    return {"stage_metrics": len(metric_rows), "predictions": len(prediction_rows)}


def _validate(dsn: str, source_counts: dict[str, int]) -> dict:
    with psycopg2.connect(dsn) as connection:
        connection.set_session(readonly=True)
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM ml.model")
            models = int(cursor.fetchone()["count"])
            cursor.execute("SELECT prediction_stage,COUNT(DISTINCT run_id) AS runs FROM ml.prediction WHERE run_id LIKE 'final:oulad:%' GROUP BY prediction_stage ORDER BY prediction_stage")
            stage_runs = [dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT COUNT(*) AS count FROM ml.prediction WHERE run_id LIKE 'final:oulad:%'")
            predictions = int(cursor.fetchone()["count"])
    counts = uci_db._counts(dsn)
    checks = {
        "models_30": models == 30,
        "four_stages_ten_runs": len(stage_runs) == 4 and all(int(row["runs"]) == 10 for row in stage_runs),
        "recommendation_unchanged": all(counts[name] == source_counts[name] for name in ("risk_profiles", "plans", "actions", "reviews")),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "counts": counts, "oulad_stage_runs": stage_runs, "oulad_prediction_rows": predictions, "credentials": "REDACTED"}


def run() -> dict:
    settings = uci_db._load_local_settings()
    admin = {**settings, "POSTGRES_USER": settings.get("POSTGRES_ADMIN_USER", "postgres"), "POSTGRES_PASSWORD": settings.get("POSTGRES_ADMIN_PASSWORD", settings["POSTGRES_PASSWORD"])}
    source_dsn = uci_db._dsn(admin, settings["POSTGRES_DB"])
    source_counts = uci_db._counts(source_dsn)
    name, dsn = _create(settings)
    db.apply_migrations(dsn)
    db.load_canonical(dsn)
    uci_db._load_stage_authority(dsn)
    eligibility = pd.read_parquet(OUT / "eligibility_manifest.parquet")
    with psycopg2.connect(dsn) as connection:
        with connection.cursor() as cursor:
            version, record_map = _new_dataset_version(cursor, eligibility)
            loaded = _replace_oulad(cursor, record_map, version)
        connection.commit()
    validation = _validate(dsn, source_counts)
    result = {"schema_version": "oulad_multistage_database_replacement_v1", "status": "READY_FOR_DATABASE_CUTOVER" if validation["status"] == "PASS" else "FAIL", "replacement_database": name, "canonical_database_modified": False, "source_opened_read_only": True, "loaded": loaded, "validation": validation, "future_oulad": "LOCKED_NOT_EXECUTED", "recommendation_modified": False, "credentials": "REDACTED", "generated_at": datetime.now(timezone.utc).isoformat()}
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("# OULAD Multi-stage Database Migration Report\n\n- Status: `{}'\n- Replacement database: `{}`\n- Canonical database modified: NO\n- OULAD uses E1/E2/M1/L1 unified predictions; F2 legacy rows remain archived evidence only.\n- Recommendation generation was not run.\n".format(result["status"], name), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
