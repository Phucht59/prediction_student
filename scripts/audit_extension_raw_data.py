from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.common.hashing import sha256_file


REQUIRED = {
    "student_mat": "student-mat.csv",
    "student_por": "student-por.csv",
    "oulad_courses": "courses.csv",
    "oulad_assessments": "assessments.csv",
    "oulad_student_assessment": "studentAssessment.csv",
    "oulad_student_info": "studentInfo.csv",
    "oulad_student_registration": "studentRegistration.csv",
    "oulad_student_vle": "studentVle.csv",
    "oulad_vle": "vle.csv",
}


def find_unique(root: Path, filename: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower() == filename.lower()]
    if not matches:
        raise FileNotFoundError(f"Required raw file not found: {filename}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous raw file {filename}: {[str(path) for path in matches]}")
    return matches[0].resolve()


def inspect_csv(path: Path) -> dict[str, object]:
    delimiter = ";" if path.name.lower().startswith("student-") else ","
    encoding = "utf-8-sig"
    row_count = 0
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        columns = next(reader)
        for _ in reader:
            row_count += 1
    return {
        "row_count": row_count,
        "column_names": columns,
        "column_count": len(columns),
        "encoding": encoding,
        "delimiter": delimiter,
        "duplicate_column_status": "PASS" if len(columns) == len(set(columns)) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifests" / "extension_raw_manifest.json")
    args = parser.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = []
    for logical_dataset, filename in REQUIRED.items():
        path = find_unique(args.raw_root.resolve(), filename)
        profile = inspect_csv(path)
        files.append(
            {
                "logical_dataset": logical_dataset,
                "absolute_source_path": str(path),
                "relative_repository_path": path.relative_to(ROOT).as_posix(),
                "filename": path.name,
                "file_size": path.stat().st_size,
                "sha256": sha256_file(path),
                **profile,
                "load_status": "PASS",
            }
        )

    manifest = {
        "schema_version": "extension_raw_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_git_commit": commit,
        "status": "PASS" if all(item["duplicate_column_status"] == "PASS" for item in files) else "FAIL",
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "files": len(files), "output": str(args.output)}, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
