from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "oulad-v3-fair-db-closure-20260716-v1"


def main() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_oulad_v3_fair_db_closure.py"),
        "--artifact-root",
        str(ROOT / "artifacts" / "study_c_oulad_v3_closure" / RUN_ID),
        "--report-root",
        str(ROOT / "reports" / "study_c_oulad_v3_closure" / RUN_ID),
        "--check-only",
    ]
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
