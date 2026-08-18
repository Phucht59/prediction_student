"""Apply versioned plain-SQL migrations additively."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


def apply_sql(path: Path) -> None:
    from src.database.connection import engine, transaction

    sql = path.read_text(encoding="utf-8")
    statements = [part.strip() for part in sql.split(";") if part.strip() and not part.strip().startswith("--")]
    with transaction() as connection:
        for statement in statements:
            connection.execute(text(statement))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrations", type=Path, default=ROOT / "database/migrations")
    args = parser.parse_args()
    files = sorted(args.migrations.glob("*.sql"))
    for path in files:
        apply_sql(path)
        print(f"applied {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
