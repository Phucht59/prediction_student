import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.final_release.build import main as build  # noqa: E402
from src.final_release.reports import generate  # noqa: E402
from src.final_release.validate import main as validate, write_checksum_manifest  # noqa: E402

if __name__ == "__main__":
    build()
    generate()
    write_checksum_manifest()
    raise SystemExit(validate())
