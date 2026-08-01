from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_final_authority_has_only_two_hybrid_families() -> None:
    authority = yaml.safe_load(
        (ROOT / "configs/final/final_model_authority.yaml").read_text(encoding="utf-8")
    )
    assert authority["uci"]["architecture_count"] == 1
    assert authority["oulad"]["architecture_count"] == 1
    assert authority["uci"]["student_mat"]["macro_f1"] == 0.9014601961315334
    assert authority["uci"]["student_por"]["macro_f1"] == 0.8622587167738002
    assert authority["oulad"]["final"]["macro_f1"] == 0.8940709888551659
    assert authority["oulad"]["stage_75"]["macro_f1"] == 0.8524909688936928


def test_final_replay_contains_eight_models_and_full_metrics() -> None:
    release = ROOT / "artifacts/final_release"
    uci = pd.read_csv(release / "uci_main_full_metrics.csv")
    oulad = pd.read_csv(release / "oulad_canonical_v3_full_metrics.csv")
    assert uci.groupby("dataset").size().to_dict() == {"student_mat": 8, "student_por": 8}
    assert set(oulad.groupby("stage").size()) == {8}
    required = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1", "pr_auc", "roc_auc",
        "nll", "brier", "ece",
    ]
    assert not uci.loc[:, required].isna().any().any()
    assert not oulad.loc[:, required].isna().any().any()


def test_release_checksums_and_replay_pass() -> None:
    release = ROOT / "artifacts/final_release"
    assert json.loads((release / "FINAL_REPLAY_PASS.json").read_text(encoding="utf-8"))["status"] == "FINAL_REPLAY_PASS"
    assert json.loads((release / "CHECKSUMS.json").read_text(encoding="utf-8"))["status"] == "PASS"
