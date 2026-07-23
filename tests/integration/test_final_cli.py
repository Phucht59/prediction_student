import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_final_status_is_read_only_and_ready() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "project.py"), "final", "status"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["training_performed"] is True
    assert payload["comparator_completion_performed"] is True
    assert payload["dataset_model_rows"] == {
        "student_mat": 9,
        "student_por": 9,
        "oulad": 9,
    }
