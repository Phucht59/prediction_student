from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from .contract import (
    ARTIFACT_ROOT,
    ROOT,
    atomic_json,
    protected_hash_status,
    sha256_file,
)
from .decision_policy import POLICY_THRESHOLDS, POLICY_VERSION


def audit_database() -> dict:
    output = ARTIFACT_ROOT / "database/audit.json"
    dsn = os.getenv("POSTGRES_TEST_DSN") or os.getenv("POSTGRES_TEST_APP_DSN")
    if not dsn:
        result = {
            "status": "SKIP_NO_DISPOSABLE_DSN",
            "production_write": False,
            "credentials_recorded": False,
        }
        atomic_json(output, result)
        return result
    parsed = urlparse(dsn)
    database_name = parsed.path.lstrip("/").lower()
    disposable = any(token in database_name for token in ("test", "tmp", "disposable"))
    if not disposable:
        result = {
            "status": "SKIP_DSN_NOT_PROVABLY_DISPOSABLE",
            "database_name_redacted": True,
            "production_write": False,
            "credentials_recorded": False,
        }
        atomic_json(output, result)
        return result
    import psycopg

    migration = ROOT / "database/migrations/009_v6_integrated_system_registry.sql"
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE 'v6_%' ORDER BY table_name"
            )
            tables = [row[0] for row in cursor.fetchall()]
    expected = {
        "v6_prediction_runs",
        "v6_student_risk_profiles",
        "v6_recommendation_plans",
        "v6_recommendation_actions",
        "v6_expert_evaluations",
        "v6_model_registry",
        "v6_policy_registry",
        "v6_artifact_registry",
    }
    result = {
        "status": "PASS" if expected.issubset(tables) else "FAIL",
        "tables": tables,
        "migration_sha256": sha256_file(migration),
        "additive_only": True,
        "production_write": False,
        "credentials_recorded": False,
    }
    atomic_json(output, result)
    return result


def build_registries() -> dict:
    registry_root = ARTIFACT_ROOT / "registry"
    selected = json.loads(
        (ARTIFACT_ROOT / "prediction/selected_model.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (ARTIFACT_ROOT / "prediction/calibration.json").read_text(encoding="utf-8")
    )
    model = {
        "schema_version": "v6_model_registry_v1",
        "model_registry_id": "v6_C_temporal_multitask_W0_seed_ensemble",
        "status": "ACTIVE",
        "selected_model": selected,
        "checkpoint_manifest_sha256": sha256_file(
            ARTIFACT_ROOT / "prediction/final/checkpoint_metadata.json"
        ),
        "feature_contract_sha256": sha256_file(ROOT / "configs/v5_1/oulad_v5_1.yaml"),
        "calibration_sha256": sha256_file(ARTIFACT_ROOT / "prediction/calibration.json"),
        "temperature": calibration["temperature"],
        "lineage": {
            "repository_base_sha": "24cca2b7f0904504e6f1c937af04589938e1a73f",
            "scientific_source_sha": "308370cf6c6f16e65cc0f0aaa3f38393ae141e16",
        },
    }
    policy = {
        "schema_version": "v6_policy_registry_v1",
        "policy_registry_id": POLICY_VERSION,
        "status": "ACTIVE",
        "thresholds": POLICY_THRESHOLDS,
        "recommendation_engine": "v5_2",
        "recommendation_source_sha": "b9087ceb1600582ad1351b134a2f4c4d9af77d89",
    }
    atomic_json(registry_root / "model_registry.json", model)
    atomic_json(registry_root / "policy_registry.json", policy)
    artifact_registry_path = registry_root / "artifact_registry.json"
    files = sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file()
        and "checksums" not in path.parts
        and path != artifact_registry_path
    )
    artifacts = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "status": "ACTIVE",
        }
        for path in files
    ]
    atomic_json(artifact_registry_path, {
        "schema_version": "v6_artifact_registry_v1",
        "artifacts": artifacts,
    })
    checksum_files = [
        *artifacts,
        {
            "path": artifact_registry_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(artifact_registry_path),
            "bytes": artifact_registry_path.stat().st_size,
            "status": "ACTIVE",
        },
    ]
    checksum_root = ARTIFACT_ROOT / "checksums"
    checksum_root.mkdir(parents=True, exist_ok=True)
    atomic_json(checksum_root / "manifest.json", {
        "schema_version": "v6_checksum_manifest_v1",
        "files": checksum_files,
        "protected": protected_hash_status(),
    })
    return {"status": "PASS", "artifacts": len(checksum_files)}


__all__ = ["audit_database", "build_registries"]
