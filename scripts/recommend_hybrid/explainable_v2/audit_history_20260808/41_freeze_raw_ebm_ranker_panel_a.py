from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EXPECTED_SELECTED_CONFIG = "a70599afad40"
EXPECTED_SELECTED_PARAMS = {
    "interactions": 3,
    "learning_rate": 0.025,
    "max_bins": 64,
    "max_rounds": 2000,
    "min_samples_leaf": 20,
}
EXPECTED_RAW_NDCG = 0.9722541839577713
EXPECTED_CASES = 300
EXPECTED_ELIGIBLE = 1117


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_with_hash(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_hash = sha256_file(src)
    dst_hash = sha256_file(dst)
    if src_hash != dst_hash:
        raise RuntimeError(f"COPY_HASH_MISMATCH={src}")
    return {
        "source": str(src),
        "frozen": str(dst),
        "sha256": dst_hash,
        "size_bytes": dst.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()

    ebm_dir = root / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
    selection_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/ranker_development"
        / "ebm_locked_grid_v1/EBM_GRID_SELECTION.json"
    )
    ranker_selection_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/ranker_development"
        / "panel_a_v1/RANKER_SELECTION_BOOTSTRAP.json"
    )
    frozen_panel_a_manifest = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/annotations/frozen/panel_a_v1"
        / "PANEL_A_FREEZE_MANIFEST.json"
    )
    label_manifest = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
        / "label_model_manifest.json"
    )

    required = [
        ebm_dir / "FIVE_EBM_MANIFEST.json",
        ebm_dir / "panel_a_ebm_oof_predictions.parquet",
        selection_path,
        ranker_selection_path,
        frozen_panel_a_manifest,
        label_manifest,
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT={path}")

    ebm_manifest = json.loads((ebm_dir / "FIVE_EBM_MANIFEST.json").read_text(encoding="utf-8"))
    grid = json.loads(selection_path.read_text(encoding="utf-8"))
    ranker = json.loads(ranker_selection_path.read_text(encoding="utf-8"))
    panel_a_freeze = json.loads(frozen_panel_a_manifest.read_text(encoding="utf-8"))
    label = json.loads(label_manifest.read_text(encoding="utf-8"))

    if ebm_manifest.get("status") != "PASS":
        raise RuntimeError("EBM_MANIFEST_NOT_PASS")
    if ebm_manifest.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_TOUCHED_IN_EBM_MANIFEST")
    if ebm_manifest.get("runtime_authorized") is not False:
        raise RuntimeError("RUNTIME_AUTHORIZED_MUST_BE_FALSE")

    if grid.get("status") != "PASS" or grid.get("search_complete") is not True:
        raise RuntimeError("GRID_SELECTION_NOT_COMPLETE")
    selected = grid.get("selected", {})
    if str(selected.get("config_id")) != EXPECTED_SELECTED_CONFIG:
        raise RuntimeError("GRID_SELECTED_CONFIG_MISMATCH")
    for key, expected in EXPECTED_SELECTED_PARAMS.items():
        if selected.get(key) != expected:
            raise RuntimeError(f"GRID_PARAM_MISMATCH={key}")

    if ranker.get("status") != "PASS":
        raise RuntimeError("RANKER_SELECTION_AUDIT_NOT_PASS")
    if ranker.get("decision") != "KEEP_RAW_EBM_RANKER":
        raise RuntimeError(f"UNEXPECTED_RANKER_DECISION={ranker.get('decision')}")
    raw_ndcg = float(ranker["raw_eligible_metrics"]["ndcg_at_3"])
    if abs(raw_ndcg - EXPECTED_RAW_NDCG) > 1e-12:
        raise RuntimeError(
            f"RAW_NDCG_REPRODUCIBILITY_FAILURE={raw_ndcg}"
        )
    if ranker.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_TOUCHED_IN_RANKER_SELECTION")
    if ranker.get("runtime_authorized") is not False:
        raise RuntimeError("RUNTIME_AUTHORIZED_IN_RANKER_SELECTION")

    # Generic lineage sanity checks.
    if panel_a_freeze.get("training_eligible") is not True:
        raise RuntimeError("FROZEN_PANEL_A_NOT_TRAINING_ELIGIBLE")
    if label.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_TOUCHED_IN_LABEL_MANIFEST")

    final_models_dir = ebm_dir / "final_models"
    if not final_models_dir.is_dir():
        raise RuntimeError(f"MISSING_FINAL_MODELS_DIR={final_models_dir}")
    model_files = sorted(final_models_dir.glob("*.joblib"))
    if len(model_files) != 5:
        raise RuntimeError(f"EXPECTED_5_FINAL_MODELS_GOT={len(model_files)}")

    freeze_dir = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/frozen"
        / "ranker_panel_a_v1"
    )
    if freeze_dir.exists() and any(freeze_dir.iterdir()):
        raise RuntimeError(
            f"FREEZE_DIR_ALREADY_NONEMPTY={freeze_dir}; "
            "do not overwrite an existing freeze"
        )
    freeze_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    copied.append(
        copy_with_hash(
            ebm_dir / "FIVE_EBM_MANIFEST.json",
            freeze_dir / "FIVE_EBM_MANIFEST.json",
        )
    )
    copied.append(
        copy_with_hash(
            ebm_dir / "panel_a_ebm_oof_predictions.parquet",
            freeze_dir / "panel_a_ebm_oof_predictions.parquet",
        )
    )
    copied.append(
        copy_with_hash(
            selection_path,
            freeze_dir / "EBM_GRID_SELECTION.json",
        )
    )
    copied.append(
        copy_with_hash(
            ranker_selection_path,
            freeze_dir / "RANKER_SELECTION_BOOTSTRAP.json",
        )
    )

    frozen_models = freeze_dir / "final_models"
    for model_path in model_files:
        copied.append(
            copy_with_hash(
                model_path,
                frozen_models / model_path.name,
            )
        )

    freeze_manifest = {
        "schema_version": "ranker_panel_a_freeze_v1",
        "status": "PASS",
        "scope": "PANEL_A_DEVELOPMENT_FREEZE_NOT_FINAL_HELDOUT_EVALUATION",
        "panel_a_cases": EXPECTED_CASES,
        "eligible_action_rows": EXPECTED_ELIGIBLE,
        "panel_b_touched": False,
        "runtime_authorized": False,
        "final_metrics_claimed": False,
        "ranker_family": "five_action_ebm",
        "ranker_calibration": "NONE_RAW_EBM_SELECTED",
        "selected_config_id": EXPECTED_SELECTED_CONFIG,
        "selected_params": EXPECTED_SELECTED_PARAMS,
        "development_operational_ndcg_at_3": raw_ndcg,
        "development_exact_best_top1_agreement": float(
            ranker["raw_eligible_metrics"]["exact_best_top1_agreement"]
        ),
        "development_pairwise_accuracy": float(
            ranker["raw_eligible_metrics"]["pairwise_accuracy"]
        ),
        "selection_rule": grid.get("selection_rule"),
        "panel_a_annotation_freeze_manifest_sha256": sha256_file(
            frozen_panel_a_manifest
        ),
        "label_model_manifest_sha256": sha256_file(label_manifest),
        "files": copied,
        "release_status": "NOT_YET_RELEASED",
        "next_required_gate": "RELEASE_GATES_AND_SAFETY_ROUTER_FREEZE",
    }

    manifest_path = freeze_dir / "RANKER_PANEL_A_FREEZE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for item in copied:
        frozen_path = Path(item["frozen"])
        checksum_lines.append(
            f"{item['sha256']}  {frozen_path.relative_to(freeze_dir).as_posix()}"
        )
    checksum_lines.append(
        f"{sha256_file(manifest_path)}  {manifest_path.name}"
    )
    (freeze_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    print("=== PANEL-A RAW EBM RANKER FREEZE ===")
    print("RANKER_SELECTION=KEEP_RAW_EBM_RANKER")
    print(f"SELECTED_CONFIG_ID={EXPECTED_SELECTED_CONFIG}")
    print(f"DEVELOPMENT_OPERATIONAL_NDCG_AT_3={raw_ndcg:.6f}")
    print(
        "DEVELOPMENT_EXACT_TOP1="
        f"{freeze_manifest['development_exact_best_top1_agreement']:.6f}"
    )
    print(
        "DEVELOPMENT_PAIRWISE_ACCURACY="
        f"{freeze_manifest['development_pairwise_accuracy']:.6f}"
    )
    print("FINAL_MODEL_COUNT=5")
    print("RANKER_CALIBRATION=NONE")
    print("PANEL_B_TOUCHED=FALSE")
    print("FINAL_METRICS_CLAIMED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    print(f"FREEZE_DIR={freeze_dir}")
    print("RANKER_PANEL_A_FREEZE=PASS")
    print("NEXT_ACTION=RUN_RELEASE_GATES_AND_FREEZE_SAFETY_ROUTER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
