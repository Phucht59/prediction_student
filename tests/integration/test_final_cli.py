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
    assert payload["status"] == "PASS"
    assert payload["active_model_family"] == "hybrid"
    assert payload["active_public_model_class"] == "src.prediction.model.Hybrid"
    assert payload["training_performed"] is False
    assert payload["hpo_performed"] is False
    assert payload["outer_evaluation_rerun"] is False
