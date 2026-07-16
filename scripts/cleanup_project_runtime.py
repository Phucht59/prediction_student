from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "reports" / "project_cleanup_inventory.csv"
SEARCH_ROOTS = [ROOT / "scripts", ROOT / "src", ROOT / "tests"]
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", "htmlcov"}


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def is_tracked(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def removable_directories() -> list[Path]:
    candidates = [ROOT / ".pytest_cache"]
    for search_root in SEARCH_ROOTS:
        if search_root.exists():
            candidates.extend(path for path in search_root.rglob("*") if path.is_dir() and path.name in CACHE_NAMES)
    candidates.extend(ROOT / name for name in ["logs", "tmp", "temp"])
    return sorted({path.resolve() for path in candidates if path.exists()}, key=lambda path: path.as_posix())


def assert_safe(path: Path) -> None:
    resolved = path.resolve()
    relative = resolved.relative_to(ROOT)
    if not relative.parts:
        raise RuntimeError("Refusing to remove repository root")
    if relative.parts[0] in {".git", "artifacts", "reports", "data"} or relative.parts[0].startswith(".venv"):
        raise RuntimeError(f"Refusing to remove protected path: {resolved}")
    if resolved.name not in CACHE_NAMES | {"logs", "tmp", "temp"}:
        raise RuntimeError(f"Path is outside the cleanup allowlist: {resolved}")


def inventory_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in removable_directories():
        assert_safe(path)
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "category": "runtime_cache",
                "tracked": is_tracked(path),
                "size_bytes": directory_bytes(path),
                "reason": "Generated Python/test cache outside immutable evidence",
                "action": "delete",
            }
        )
    preserved = [
        ("artifacts/", "scientific_evidence", "Immutable scientific artifacts and checksum manifests"),
        ("reports/study_c_oulad_v3_closure/", "scientific_evidence", "Immutable scientific report mirror"),
        ("artifacts/study_c_oulad_v3_closure/oulad-v3-fair-db-closure-20260716-v1/threshold_replay_cache/", "scientific_evidence_cache", "Replay cache belongs to immutable closure evidence"),
        (".venv-oulad-v2/", "environment", "Working CUDA/Python environment is not runtime clutter"),
        (".env", "local_configuration", "Ignored local configuration; never printed or committed"),
        ("MODEL_IMPROVEMENT_PLAN_V3.md", "historical_document", "Referenced by historical audit; preserve in place"),
        ("SCIENTIFIC_PROTOCOL_V2.md", "frozen_protocol", "Referenced by immutable source provenance; preserve in place"),
    ]
    for relative, category, reason in preserved:
        path = ROOT / relative
        if path.exists():
            rows.append(
                {
                    "path": relative,
                    "category": category,
                    "tracked": is_tracked(path),
                    "size_bytes": directory_bytes(path) if path.is_dir() else path.stat().st_size,
                    "reason": reason,
                    "action": "preserve",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and remove allowlisted runtime caches without touching evidence.")
    parser.add_argument("--apply", action="store_true", help="Delete only inventory rows marked delete after writing the inventory.")
    args = parser.parse_args()

    rows = inventory_rows()
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "category", "tracked", "size_bytes", "reason", "action"])
        writer.writeheader()
        writer.writerows(rows)

    removed = []
    if args.apply:
        for row in rows:
            if row["action"] != "delete":
                continue
            path = (ROOT / str(row["path"])).resolve()
            assert_safe(path)
            shutil.rmtree(path)
            removed.append(str(row["path"]))
    print(f"inventory={INVENTORY.relative_to(ROOT).as_posix()} rows={len(rows)} removed={len(removed)}")


if __name__ == "__main__":
    main()
