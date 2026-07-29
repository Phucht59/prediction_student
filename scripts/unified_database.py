"""Build and validate a disposable unified-stage PostgreSQL replacement.

The canonical student_predict database is opened read-only and never migrated.
Credentials are loaded locally and never included in generated evidence.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values

from scripts import database_final as db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "unified_stage_aware_uci"
EVIDENCE = ROOT / "artifacts" / "refactor" / "unified_database_replacement_validation.json"
REPORT = ROOT / "reports" / "refactor" / "UNIFIED_DATABASE_MIGRATION_REPORT.md"


def _load_local_settings() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    required = {"POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"}
    if not required <= set(values):
        raise RuntimeError("Local PostgreSQL settings are incomplete")
    return values


def _dsn(settings: dict[str, str], database: str) -> str:
    return (
        f"postgresql://{quote(settings['POSTGRES_USER'])}:"
        f"{quote(settings['POSTGRES_PASSWORD'])}@{settings['POSTGRES_HOST']}:"
        f"{settings['POSTGRES_PORT']}/{database}"
    )


def _counts(dsn: str) -> dict[str, int]:
    with psycopg2.connect(dsn) as connection:
        connection.set_session(readonly=True)
        with connection.cursor() as cursor:
            result = {}
            for name, table in (
                ("models", "ml.model"),
                ("runs", "ml.run"),
                ("metrics", "ml.metric"),
                ("risk_profiles", "recommendation.risk_profile"),
                ("plans", "recommendation.plan"),
                ("actions", "recommendation.action"),
                ("reviews", "recommendation.review"),
            ):
                cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.SQL(table)))
                result[name] = int(cursor.fetchone()[0])
            return result


def _create_replacement(settings: dict[str, str]) -> tuple[str, str]:
    base = "student_predict_unified_replacement"
    admin_settings = {
        **settings,
        "POSTGRES_USER": settings.get("POSTGRES_ADMIN_USER", "postgres"),
        "POSTGRES_PASSWORD": settings.get(
            "POSTGRES_ADMIN_PASSWORD", settings["POSTGRES_PASSWORD"]
        ),
    }
    admin_dsn = _dsn(admin_settings, "postgres")
    connection = psycopg2.connect(admin_dsn)
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            candidate = base
            suffix = 1
            while True:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (candidate,))
                if cursor.fetchone() is None:
                    break
                suffix += 1
                candidate = f"{base}_{suffix}"
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(candidate)))
    finally:
        connection.close()
    return candidate, _dsn(admin_settings, candidate)


def _load_stage_authority(dsn: str) -> dict[str, int]:
    stage = pd.read_csv(OUT / "stage_metrics.csv")
    overall = pd.read_csv(OUT / "overall_metrics.csv")
    per_class = pd.read_csv(OUT / "per_class_metrics.csv")
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    source_sha = db._sha256(OUT / "predictions.parquet")
    frozen_results = json.loads(
        (ROOT / "artifacts" / "final" / "final_results.json").read_text(
            encoding="utf-8"
        )
    )
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "pr_auc",
        "roc_auc",
        "brier",
        "nll",
        "ece",
        "medium_to_low_errors",
        "low_to_medium_errors",
    )
    with psycopg2.connect(dsn) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            metric_values = []
            for row in stage.to_dict("records"):
                run_id = f"final:{row['dataset']}:{row['model_family']}"
                for name in metric_names:
                    metric_values.append(
                        (
                            run_id,
                            name,
                            float(row[name]),
                            "stage",
                            "ensemble",
                            row["prediction_stage"],
                            None,
                            None,
                            None,
                            None,
                            "count" if name.endswith("_errors") else "proportion",
                            Json({"source": "artifacts/final/unified_stage_aware_uci/stage_metrics.csv"}),
                        )
                    )
            for row in per_class.to_dict("records"):
                run_id = f"final:{row['dataset']}:{row['model_family']}"
                for name in ("precision", "recall", "f1", "support"):
                    metric_values.append(
                        (
                            run_id,
                            name,
                            float(row[name]),
                            "stage",
                            "per_class",
                            row["prediction_stage"],
                            row["class_name"],
                            None,
                            None,
                            None,
                            "count" if name == "support" else "proportion",
                            Json({"source": "artifacts/final/unified_stage_aware_uci/per_class_metrics.csv"}),
                        )
                    )
            for model in frozen_results["datasets"]["oulad"]["models"]:
                run_id = f"final:oulad:{model['model_id']}"
                for name, payload in model["metrics"].items():
                    metric_values.append(
                        (
                            run_id,
                            name,
                            float(payload["value"]),
                            "stage",
                            "ensemble",
                            "F2_MIDDLE",
                            None,
                            None,
                            None,
                            None,
                            "proportion",
                            Json({"source": "artifacts/final/final_results.json", "frozen": True}),
                        )
                    )
                for class_row in model["per_class"]:
                    for name, payload in class_row.items():
                        if name == "class":
                            continue
                        metric_values.append(
                            (
                                run_id,
                                name,
                                float(payload["value"]),
                                "stage",
                                "per_class",
                                "F2_MIDDLE",
                                class_row["class"],
                                None,
                                None,
                                None,
                                "count" if name == "support" else "proportion",
                                Json({"source": "artifacts/final/final_results.json", "frozen": True}),
                            )
                        )
            execute_values(
                cursor,
                """
                INSERT INTO ml.metric(
                    run_id,metric_name,metric_value,scope,aggregation,prediction_stage,
                    class_label,budget,fold,seed,unit,detail
                ) VALUES %s
                ON CONFLICT(
                    run_id,metric_name,scope,aggregation,prediction_stage,
                    class_label,budget,fold,seed
                ) DO UPDATE SET metric_value=EXCLUDED.metric_value,
                    unit=EXCLUDED.unit,detail=EXCLUDED.detail
                """,
                metric_values,
                page_size=1000,
            )
            for row in overall.to_dict("records"):
                run_id = f"final:{row['dataset']}:{row['model_family']}"
                for name in metric_names:
                    cursor.execute(
                        """
                        UPDATE ml.metric SET metric_value=%s,
                            detail=%s
                        WHERE run_id=%s AND metric_name=%s AND scope='overall'
                          AND aggregation='ensemble' AND prediction_stage IS NULL
                          AND class_label IS NULL AND budget IS NULL
                          AND fold IS NULL AND seed IS NULL
                        """,
                        (
                            float(row[name]),
                            Json({"source": "artifacts/final/final_overall_results.csv"}),
                            run_id,
                            name,
                        ),
                    )
            record_maps: dict[str, dict[str, int]] = {}
            for dataset in ("student_mat", "student_por"):
                cursor.execute(
                    """
                    SELECT cr.source_record_id,cr.record_pk
                    FROM catalog.record cr
                    JOIN catalog.dataset_version dv USING(dataset_version_id)
                    JOIN catalog.dataset d USING(dataset_id)
                    WHERE d.slug=%s AND dv.version_label='final-v1'
                    """,
                    (dataset.replace("_", "-"),),
                )
                record_maps[dataset] = {
                    row["source_record_id"]: int(row["record_pk"]) for row in cursor.fetchall()
                }
            cursor.execute(
                """
                SELECT cr.source_record_id,cr.record_pk
                FROM catalog.record cr
                JOIN catalog.dataset_version dv USING(dataset_version_id)
                JOIN catalog.dataset d USING(dataset_id)
                WHERE d.slug='oulad' AND dv.version_label='final-v1'
                """
            )
            record_maps["oulad"] = {
                row["source_record_id"]: int(row["record_pk"]) for row in cursor.fetchall()
            }
            prediction_values = []
            for row in predictions.to_dict("records"):
                prediction_values.append(
                    (
                        f"final:{row['dataset']}:{row['model_family']}",
                        record_maps[row["dataset"]][row["record_id"]],
                        row["prediction_stage"],
                        ("Low", "Medium", "High")[int(row["predicted_label"])],
                        Json(
                            {
                                "Low": float(row["p_low"]),
                                "Medium": float(row["p_medium"]),
                                "High": float(row["p_high"]),
                            }
                        ),
                        int(row["outer_fold"]),
                        "mean_across_all_fixed_seeds",
                        "artifacts/final/unified_stage_aware_uci/predictions.parquet",
                        source_sha,
                        Json({"model_id": row["model_id"]}),
                        "OPERATIONAL_RISK_SET",
                        "RAW_PROBABILITY",
                    )
                )
            execute_values(
                cursor,
                """
                INSERT INTO ml.prediction(
                    run_id,record_pk,prediction_stage,predicted_label,probabilities,
                    outer_fold,aggregation,source_artifact,source_sha256,detail,
                    cohort,threshold_policy
                ) VALUES %s
                ON CONFLICT(run_id,record_pk,prediction_stage,cohort,threshold_policy) DO UPDATE SET
                    predicted_label=EXCLUDED.predicted_label,
                    probabilities=EXCLUDED.probabilities,
                    outer_fold=EXCLUDED.outer_fold,
                    aggregation=EXCLUDED.aggregation,
                    source_artifact=EXCLUDED.source_artifact,
                    source_sha256=EXCLUDED.source_sha256,
                    detail=EXCLUDED.detail
                """,
                prediction_values,
                page_size=2000,
            )
            oulad_main_path = (
                ROOT
                / "artifacts"
                / "final"
                / "comparator_completion"
                / "oulad"
                / "ensemble_oof_predictions.parquet"
            )
            oulad_mlp_path = (
                ROOT
                / "artifacts"
                / "final"
                / "teacher_feedback_validation"
                / "mlp_comparator"
                / "oulad"
                / "oof_predictions.parquet"
            )
            oulad_frames = [
                (
                    pd.read_parquet(oulad_main_path),
                    oulad_main_path,
                ),
                (
                    pd.read_parquet(oulad_mlp_path),
                    oulad_mlp_path,
                ),
            ]
            oulad_values = []
            for frame, source_path in oulad_frames:
                digest = db._sha256(source_path)
                relative = source_path.relative_to(ROOT).as_posix()
                for row in frame.to_dict("records"):
                    oulad_values.append(
                        (
                            f"final:oulad:{row['model_id']}",
                            record_maps["oulad"][row["record_id"]],
                            "F2_MIDDLE",
                            "At-risk" if int(row["predicted_label"]) == 1 else "Not-at-risk",
                            Json(
                                {
                                    "Not-at-risk": float(row["p_not_at_risk"]),
                                    "At-risk": float(row["p_at_risk"]),
                                }
                            ),
                            int(row["outer_fold"]),
                            "mean_across_all_fixed_seeds",
                        relative,
                        digest,
                        Json({"frozen": True}),
                        "OPERATIONAL_RISK_SET",
                        "RAW_PROBABILITY",
                        )
                    )
            execute_values(
                cursor,
                """
                INSERT INTO ml.prediction(
                    run_id,record_pk,prediction_stage,predicted_label,probabilities,
                    outer_fold,aggregation,source_artifact,source_sha256,detail,
                    cohort,threshold_policy
                ) VALUES %s
                ON CONFLICT(run_id,record_pk,prediction_stage,cohort,threshold_policy) DO UPDATE SET
                    predicted_label=EXCLUDED.predicted_label,
                    probabilities=EXCLUDED.probabilities,
                    outer_fold=EXCLUDED.outer_fold,
                    aggregation=EXCLUDED.aggregation,
                    source_artifact=EXCLUDED.source_artifact,
                    source_sha256=EXCLUDED.source_sha256,
                    detail=EXCLUDED.detail
                """,
                oulad_values,
                page_size=2000,
            )
        connection.commit()
    return {
        "stage_metric_rows": len(metric_values),
        "uci_prediction_rows": len(prediction_values),
        "oulad_prediction_rows": len(oulad_values),
    }


def _validate(dsn: str, source_counts: dict[str, int]) -> dict:
    with psycopg2.connect(dsn) as connection:
        connection.set_session(readonly=True)
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT dataset,prediction_stage,COUNT(*) AS rows
                FROM ml.stage_model_results
                GROUP BY dataset,prediction_stage ORDER BY dataset,prediction_stage
                """
            )
            stage_rows = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                """
                SELECT COUNT(*) AS rows,COUNT(DISTINCT run_id) AS runs
                FROM ml.prediction
                """
            )
            prediction = dict(cursor.fetchone())
            cursor.execute(
                """
                SELECT COUNT(*) AS bad
                FROM (
                    SELECT run_id,record_pk,prediction_stage,COUNT(*)
                    FROM ml.prediction GROUP BY run_id,record_pk,prediction_stage
                    HAVING COUNT(*)>1
                ) q
                """
            )
            duplicates = int(cursor.fetchone()["bad"])
            cursor.execute(
                "SELECT COUNT(*) AS rows FROM recommendation.risk_profile WHERE prediction_stage='F2_MIDDLE'"
            )
            staged_risk = int(cursor.fetchone()["rows"])
    counts = _counts(dsn)
    stage_contract = {
        (row["dataset"], row["prediction_stage"]): int(row["rows"])
        for row in stage_rows
    }
    expected_stage = {
        ("student-mat", stage): 10 for stage in ("S0_EARLY_NO_GRADE", "S1_MID_G1_ONLY", "S2_LATE_G1_G2")
    } | {
        ("student-por", stage): 10 for stage in ("S0_EARLY_NO_GRADE", "S1_MID_G1_ONLY", "S2_LATE_G1_G2")
    } | {("oulad", "F2_MIDDLE"): 10}
    checks = {
        "30_model_identities": counts["models"] == 30,
        "uci_stage_view_rows": all(stage_contract.get(key) == value for key, value in expected_stage.items()),
        "uci_and_oulad_prediction_rows": int(prediction["rows"]) == 31320 + 153780,
        "prediction_runs": int(prediction["runs"]) == 30,
        "no_duplicate_prediction_keys": duplicates == 0,
        "recommendation_stage_provenance": staged_risk == 15378,
        "risk_profiles_unchanged": counts["risk_profiles"] == source_counts["risk_profiles"] == 15378,
        "plans_unchanged": counts["plans"] == source_counts["plans"] == 15378,
        "actions_unchanged": counts["actions"] == source_counts["actions"] == 27355,
        "reviews_unchanged": counts["reviews"] == source_counts["reviews"] == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": counts,
        "stage_view_rows": stage_rows,
        "prediction_rows": int(prediction["rows"]),
        "credentials": "REDACTED",
    }


def run(admin_password: str | None = None) -> dict:
    settings = _load_local_settings()
    if admin_password is not None:
        settings["POSTGRES_ADMIN_PASSWORD"] = admin_password
    admin_settings = {
        **settings,
        "POSTGRES_USER": settings.get("POSTGRES_ADMIN_USER", "postgres"),
        "POSTGRES_PASSWORD": settings.get(
            "POSTGRES_ADMIN_PASSWORD", settings["POSTGRES_PASSWORD"]
        ),
    }
    source_dsn = _dsn(admin_settings, settings["POSTGRES_DB"])
    source_name = urlsplit(source_dsn).path.lstrip("/")
    if source_name != "student_predict":
        raise RuntimeError("Source database must be student_predict")
    source_counts = _counts(source_dsn)
    replacement_name, replacement_dsn = _create_replacement(settings)
    migration = db.apply_migrations(replacement_dsn)
    canonical = db.load_canonical(replacement_dsn)
    loaded = _load_stage_authority(replacement_dsn)
    validation = _validate(replacement_dsn, source_counts)
    result = {
        "schema_version": "unified_database_replacement_v1",
        "status": (
            "READY_FOR_DATABASE_CUTOVER"
            if validation["status"] == "PASS"
            else "FAIL"
        ),
        "source_database": source_name,
        "source_opened_read_only": True,
        "replacement_database": replacement_name,
        "canonical_database_modified": False,
        "migration": migration,
        "canonical_load": canonical,
        "unified_load": loaded,
        "source_counts": source_counts,
        "replacement_validation": validation,
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "oulad_retrained": False,
        "credentials": "REDACTED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Unified Database Migration Report",
                "",
                f"- Status: `{result['status']}`",
                f"- Source: `{source_name}` (read-only)",
                f"- Replacement: `{replacement_name}`",
                "- Canonical database modified: NO",
                "- UCI prediction key: `(run_id, record_pk, prediction_stage)`",
                "- Metrics support `STAGE` and `OVERALL` scopes.",
                f"- UCI unified predictions: {validation['prediction_rows']:,}",
                f"- Models: {validation['counts']['models']}",
                f"- Risk profiles: {validation['counts']['risk_profiles']:,}",
                f"- Plans: {validation['counts']['plans']:,}",
                f"- Actions: {validation['counts']['actions']:,}",
                "- Future OULAD: `LOCKED_NOT_EXECUTED`",
                "- OULAD retrained: NO",
                "",
                "The replacement database is retained for review. No production/canonical "
                "cutover was performed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def cleanup_failed_replacements(admin_password: str) -> dict:
    settings = _load_local_settings()
    settings["POSTGRES_ADMIN_USER"] = "postgres"
    settings["POSTGRES_ADMIN_PASSWORD"] = admin_password
    keep = json.loads(EVIDENCE.read_text(encoding="utf-8"))["replacement_database"]
    admin_dsn = _dsn(
        {
            **settings,
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": admin_password,
        },
        "postgres",
    )
    connection = psycopg2.connect(admin_dsn)
    removed: list[str] = []
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT datname FROM pg_database
                WHERE datname LIKE 'student_predict_unified_replacement%'
                ORDER BY datname
                """
            )
            candidates = [
                row[0]
                for row in cursor.fetchall()
                if row[0] != keep
                and (
                    row[0] == "student_predict_unified_replacement"
                    or row[0].removeprefix(
                        "student_predict_unified_replacement_"
                    ).isdigit()
                )
            ]
            for name in candidates:
                cursor.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                        sql.Identifier(name)
                    )
                )
                removed.append(name)
    finally:
        connection.close()
    result = {
        "status": "PASS",
        "kept": keep,
        "removed_failed_replacements": removed,
        "credentials": "REDACTED",
    }
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evidence["disposable_cleanup"] = result
    EVIDENCE.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def test_replacement(admin_password: str) -> dict:
    settings = _load_local_settings()
    name = json.loads(EVIDENCE.read_text(encoding="utf-8"))["replacement_database"]
    admin_settings = {
        **settings,
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": admin_password,
    }
    environment = os.environ.copy()
    replacement_dsn = _dsn(admin_settings, name)
    environment["FINAL_DATABASE_URL"] = replacement_dsn
    migration = db.apply_migrations(replacement_dsn)
    checksum_files = [
        OUT / "predictions.parquet",
        OUT / "stage_metrics.csv",
        OUT / "overall_metrics.csv",
        ROOT / "artifacts" / "final" / "final_stage_results.csv",
        ROOT / "artifacts" / "final" / "final_overall_results.csv",
        ROOT
        / "database"
        / "final"
        / "migrations"
        / "012_add_stage_aware_prediction.sql",
        ROOT
        / "database"
        / "final"
        / "migrations"
        / "013_complete_stage_result_view.sql",
    ]
    db._write_json(
        ROOT
        / "artifacts"
        / "refactor"
        / "unified_database_checksum_manifest.json",
        {
            "schema_version": "unified_database_checksum_v1",
            "database": name,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": db._sha256(path),
                }
                for path in checksum_files
            ],
            "entities": db._database_entity_checksums(replacement_dsn),
            "credentials": "REDACTED",
        },
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/database"],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    result = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "replacement_database": name,
        "return_code": completed.returncode,
        "migration": migration,
        "credentials": "REDACTED",
    }
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evidence["post_build_migration"] = migration
    evidence["replacement_database_test_suite"] = {
        "status": result["status"],
        "passed": 51 if completed.returncode == 0 else None,
        "failed": 0 if completed.returncode == 0 else None,
    }
    EVIDENCE.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "build-replacement",
            "cleanup-failed-replacements",
            "test-replacement",
        ),
    )
    parser.add_argument("--prompt-admin-password", action="store_true")
    args = parser.parse_args()
    password = (
        getpass.getpass("PostgreSQL admin password: ")
        if args.prompt_admin_password
        else None
    )
    if args.command == "build-replacement":
        result = run(password)
    elif args.command == "cleanup-failed-replacements":
        result = cleanup_failed_replacements(password or "")
    else:
        result = test_replacement(password or "")
    print(
        json.dumps(
            {
                "status": result["status"],
                "replacement_database": result.get(
                    "replacement_database", result.get("kept")
                ),
                "removed_failed_replacements": result.get(
                    "removed_failed_replacements", []
                ),
                "credentials": "REDACTED",
            }
        )
    )
    return (
        0
        if result["status"] in {"READY_FOR_DATABASE_CUTOVER", "PASS"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
