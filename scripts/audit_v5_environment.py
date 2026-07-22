from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.studies.v5.common.artifacts import atomic_write_json
from src.studies.v5.common.protocol import load_project_protocol, load_study_protocol, verify_declared_sources


def main() -> int:
    import numpy
    import pandas
    import sklearn
    import torch

    source_checks = {
        study: verify_declared_sources(load_study_protocol(study))
        for study in ["student-mat", "student-por", "oulad"]
    }
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for rows in source_checks.values() for row in rows) else "FAIL",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {"torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__},
        "cuda": {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        },
        "docker_available": shutil.which("docker") is not None,
        "git_available": shutil.which("git") is not None,
        "database": {
            "v5_dsn_present": bool(os.getenv("V5_DATABASE_URL")),
            "test_dsn_present": bool(os.getenv("POSTGRES_TEST_DSN")),
            "credentials_logged": False,
        },
        "protocol_status": load_project_protocol()["protocol_status"],
        "source_checks": source_checks,
        "future_benchmark": "LOCKED_NOT_EXECUTED",
    }
    output = ROOT / "reports" / "v5" / "final" / "environment_audit.json"
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

