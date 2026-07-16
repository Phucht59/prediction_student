from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from src.common.model_display_names import add_display_name, get_display_name, load_model_display_names


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "oulad" / "final"
REPORT = ROOT / "reports" / "oulad" / "final"
FIGURES = ROOT / "reports" / "thesis_figures"
BANNED_READER_IDS = ["V3-D0-ENS", "V3-A0F-ENS", "H2TF", "H3CF", "P0-ENS", "D0-ENS"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_display_mapping_loads_and_covers_thesis_models():
    mapping = load_model_display_names()
    expected = {
        "V3-MLF": "Logistic Regression",
        "V3-MLD": "Machine Learning with Dynamic Features",
        "V3-A0F-ENS": "MLP",
        "V3-H2TF-ENS": "CNN–BiLSTM",
        "V3-D0-ENS": "CNN–BiLSTM Ensemble",
    }
    for candidate_id, display_name in expected.items():
        assert mapping[candidate_id]["display_name"] == display_name
    assert mapping["C-R0"]["display_name"] == "Random Forest"
    assert mapping["C-S0"]["display_name"] == "SVM"
    assert mapping["C-H0"]["display_name"] == "HistGradientBoosting"


def test_display_helper_preserves_candidate_id_and_has_safe_fallback():
    source = {"candidate_id": "V3-D0-ENS", "macro_f1": 0.8311261008483025}
    displayed = add_display_name(source)
    assert displayed["candidate_id"] == source["candidate_id"]
    assert displayed["display_name"] == "CNN–BiLSTM Ensemble"
    assert get_display_name("unknown-internal-id") == "unknown-internal-id"


def test_readme_uses_thesis_names_and_rounded_frozen_metrics():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not any(candidate_id in readme for candidate_id in BANNED_READER_IDS)
    metrics = pd.read_csv(ARTIFACT / "ensemble_metrics.csv").set_index("candidate_id")
    expected_rows = {
        "Logistic Regression": "V3-MLF",
        "Machine Learning with Dynamic Features": "V3-MLD",
        "MLP": "V3-A0F-ENS",
        "CNN–BiLSTM": "V3-P0-ENS",
        "CNN–BiLSTM Ensemble": "V3-D0-ENS",
    }
    for display_name, candidate_id in expected_rows.items():
        row = metrics.loc[candidate_id]
        expected = (
            f"| {display_name} | {row.macro_f1:.4f} | {row.at_risk_precision:.4f} | "
            f"{row.at_risk_recall:.4f} | {row.pr_auc:.4f} |"
        )
        assert expected in readme


def test_thesis_figures_use_display_labels_and_frozen_source():
    manifest = json.loads((FIGURES / "figure_manifest.json").read_text(encoding="utf-8"))
    source = ROOT / manifest["source"]
    assert sha256(source) == manifest["source_sha256"]
    assert manifest["metrics_copied_manually"] is False
    assert set(manifest["display_labels"]) == {
        "Logistic Regression", "MLP", "CNN–BiLSTM", "CNN–BiLSTM Ensemble"
    }
    assert not any("V3-" in label for label in manifest["display_labels"])
    for name in manifest["figures"]:
        path = FIGURES / name
        assert path.exists() and path.stat().st_size > 10_000
        assert b"V3-" not in path.read_bytes()


def test_frozen_closure_checksums_and_database_reproduction_remain_valid():
    validation = json.loads((ARTIFACT / "validation_report.json").read_text(encoding="utf-8"))
    manifest_path = ARTIFACT / "artifact_checksums.json"
    assert sha256(manifest_path) == validation["artifact_checksums_sha256"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        assert sha256(ARTIFACT / item["path"]) == item["sha256"]
    reproduction = json.loads((ARTIFACT / "postgres_reproduction_validation.json").read_text(encoding="utf-8"))
    assert reproduction["status"] == "PASS"
    assert reproduction["artifact_rows"] == reproduction["database_rows"] == 123024


def test_readme_relative_links_exist():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean_target = target.split("#", 1)[0]
        assert (ROOT / clean_target).exists(), target


def test_no_plaintext_postgres_secret_or_runtime_log_is_added():
    diff = subprocess.check_output(["git", "diff", "--", "."], cwd=ROOT, text=True, errors="replace")
    assert not re.search(r"postgresql://(?!<redacted>)[^\s/@:]+:[^\s/@]+@", diff, re.I)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    runtime_pattern = re.compile(r"(^|/)(logs?|tmp|temp|__pycache__|\.pytest_cache)(/|$)|\.(log|tmp|bak)$", re.I)
    assert not [path for path in tracked if runtime_pattern.search(path)]


def test_check_only_validator_entrypoint_and_report_mirror_exist():
    source = (ROOT / "scripts" / "validate_oulad_final.py").read_text(encoding="utf-8")
    assert "--check-only" in source
    assert (REPORT / "figures").is_dir()
    assert (ROOT / "scripts" / "validate_release.py").is_file()
