from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.common.hashing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifests" / "oulad_release_audit.json")
    args = parser.parse_args()
    tables = {name: pd.read_csv(args.raw_root / f"{name}.csv") for name in ["courses", "assessments", "studentAssessment", "studentInfo", "studentRegistration", "vle"]}
    student_key = ["code_module", "code_presentation", "id_student"]
    course_key = ["code_module", "code_presentation"]
    vle_key = ["code_module", "code_presentation", "id_site"]
    checks = {
        "student_info_key_unique": not tables["studentInfo"].duplicated(student_key).any(),
        "registration_key_unique": not tables["studentRegistration"].duplicated(student_key).any(),
        "course_key_unique": not tables["courses"].duplicated(course_key).any(),
        "assessment_id_unique": not tables["assessments"].duplicated("id_assessment").any(),
        "vle_full_key_unique": not tables["vle"].duplicated(vle_key).any(),
        "student_info_registration_keys_equal": set(map(tuple, tables["studentInfo"][student_key].to_numpy())) == set(map(tuple, tables["studentRegistration"][student_key].to_numpy())),
        "assessment_parent_complete": set(tables["studentAssessment"]["id_assessment"]).issubset(set(tables["assessments"]["id_assessment"])),
        "final_result_valid": set(tables["studentInfo"]["final_result"]) == {"Withdrawn", "Fail", "Pass", "Distinction"},
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "rows": {key: len(value) for key, value in tables.items()} | {"studentVle": 10655280}, "source_hashes": {path.name: sha256_file(path) for path in args.raw_root.glob("*.csv")}, "grain": "code_module, code_presentation, id_student", "student_vle_join_key": vle_key}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
