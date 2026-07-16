from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_summary(output: str) -> dict[str, int]:
    values = {}
    for name in ("passed", "skipped", "failed", "error", "errors", "deselected"):
        matches = re.findall(rf"(\d+)\s+{name}\b", output)
        values["errors" if name in {"error", "errors"} else name] = max([int(value) for value in matches], default=0)
    values["collected"] = values["passed"] + values["skipped"] + values["failed"] + values["errors"]
    return values


def execute(name: str, arguments: list[str], cwd: Path) -> tuple[dict[str, object], str]:
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *arguments],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        env=os.environ.copy(),
    )
    output = process.stdout or ""
    result = {
        "name": name,
        "command": f"{Path(sys.executable).name} -m pytest -q {' '.join(arguments)}",
        "return_code": process.returncode,
        **parse_summary(output),
    }
    return result, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--report-root", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact = Path(args.artifact_root); report = Path(args.report_root)
    artifact.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    if not os.getenv("POSTGRES_TEST_DSN") or not os.getenv("POSTGRES_TEST_APP_DSN"):
        raise RuntimeError("Both live PostgreSQL test DSNs are required for the closure suite")
    sections = [
        ("unit_and_scientific", ["--ignore=tests/test_postgres_source_ml_integration.py", "-k", "not live_postgres"]),
        ("postgresql_integration", ["tests/test_postgres_source_ml_integration.py"]),
        ("postgresql_permission", [
            "tests/test_postgres_source_ml_integration.py::test_app_role_is_insert_only_for_source_split_prediction_ledgers",
            "tests/test_oulad_v3_fair_db_closure.py::test_live_postgres_closure_uses_real_least_privileged_app_role",
        ]),
        ("full_suite", []),
    ]
    results = []; outputs = []
    for name, arguments in sections:
        result, output = execute(name, arguments, root)
        results.append(result)
        outputs.extend([f"===== {name} =====", output.rstrip(), ""])
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["return_code"] != 0:
            break
    full = next((item for item in results if item["name"] == "full_suite"), None)
    status = "PASS" if len(results) == len(sections) and all(item["return_code"] == 0 for item in results) and full and full["failed"] == full["errors"] == full["skipped"] == 0 else "FAIL"
    payload = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "postgres_admin_dsn_present": True,
        "postgres_application_dsn_present": True,
        "credential_values_recorded": False,
        "sections": results,
        "full_suite": full,
    }
    (report / "test_stdout.txt").write_text("\n".join(outputs), encoding="utf-8")
    (artifact / "test_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
