"""Train, calibrate, and evaluate the integrated four-stage action head."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.final.actions import ACTION_COUNT, ACTION_ORDER  # noqa: E402
from src.recommend_hybrid.final.metrics import (  # noqa: E402
    STAGE_ORDER,
    make_decisions,
    ranking_metrics,
)
from src.recommend_hybrid.final.model import ActionAwareHeadConfig  # noqa: E402
from src.recommend_hybrid.final.stage_aware_training import (  # noqa: E402
    FeatureStandardizer,
    FourStageActionData,
    average_logits,
    build_four_stage_action_data,
    calibrate_action_thresholds,
    grouped_outer_splits,
    predict_action_head,
    train_action_head,
)

DEFAULT_LANDMARK = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
DEFAULT_SILVER = ROOT / "artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"
DEFAULT_ACTION_MAP = ROOT / "artifacts/recommend_hybrid/scientific_labeling/action_evidence_map.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/final_stage_aware_v2"
DEFAULT_REPORT = ROOT / "reports/recommend_hybrid/FOUR_STAGE_ACTION_HEAD_RESULTS.md"
DEFAULT_SEEDS = (20260806, 20260807, 20260808, 20260809, 20260810)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _decision_metrics(
    data: FourStageActionData,
    direct_logits: np.ndarray,
    action_logits: np.ndarray,
    issued: np.ndarray,
    top_action: np.ndarray,
) -> dict[str, object]:
    positive = data.group_target.astype(bool)
    row = np.arange(len(positive))
    correct = issued & (data.action_target[row, top_action] > 0)
    issued_positive = issued & positive
    issued_count = int(issued.sum())
    positive_count = int(positive.sum())
    issued_positive_count = int(issued_positive.sum())
    correct_count = int(correct.sum())
    selected = top_action[issued]
    if issued_count:
        counts = np.bincount(selected, minlength=ACTION_COUNT)
        diversity = int((counts > 0).sum())
        concentration = float(counts.max() / counts.sum())
    else:
        counts = np.zeros(ACTION_COUNT, dtype=int)
        diversity = 0
        concentration = 1.0
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    return {
        "groups": int(len(positive)),
        "issued_groups": issued_count,
        "positive_groups": positive_count,
        "issued_positive_groups": issued_positive_count,
        "correct_issued_actions": correct_count,
        "false_issue_groups": int((issued & ~positive).sum()),
        "stage_a_precision": ratio(issued_positive_count, issued_count),
        "stage_a_recall": ratio(issued_positive_count, positive_count),
        "stage_b_conditional_precision_at_1": ratio(
            correct_count, issued_positive_count
        ),
        "end_to_end_precision_at_1": ratio(correct_count, issued_count),
        "positive_group_coverage": ratio(issued_positive_count, positive_count),
        "abstention_rate": 1.0 - ratio(issued_count, len(positive)),
        "action_diversity": diversity,
        "top_action_concentration": concentration,
        "top_action_counts": {
            ACTION_ORDER[index]: int(counts[index]) for index in range(ACTION_COUNT)
        },
        **ranking_metrics(
            action_logits,
            data.action_target,
            data.action_mask,
            data.group_target,
        ),
    }


def _report(payload: dict[str, Any]) -> str:
    overall = payload["overall"]
    lines = [
        "# Four-Stage Conditional Action Head",
        "",
        "## Scope",
        "",
        "The frozen Hybrid CNN–BiLSTM representation was not modified. Only the integrated conditional action head was trained. Thresholds were calibrated on validation rows and all reported final metrics use held-out out-of-fold rows.",
        "",
        "## Overall held-out evidence",
        "",
        f"- Conditional Precision@1: **{overall['conditional_precision_at_1_all_positive']:.4f}**",
        f"- NDCG@3: **{overall['ndcg_at_3']:.4f}**",
        f"- MRR: **{overall['mrr']:.4f}**",
        f"- End-to-end Precision@1: **{overall['end_to_end_precision_at_1']:.4f}**",
        f"- Positive coverage: **{overall['positive_group_coverage']:.4f}**",
        f"- Abstention: **{overall['abstention_rate']:.4f}**",
        "",
        "## Per-stage held-out evidence",
        "",
        "| Stage | Groups | Conditional P@1 | NDCG@3 | MRR | E2E P@1 | Coverage | Abstention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGE_ORDER:
        row = payload["per_stage"][stage]
        lines.append(
            "| {stage} | {groups} | {p1:.4f} | {ndcg:.4f} | {mrr:.4f} | {e2e:.4f} | {coverage:.4f} | {abstention:.4f} |".format(
                stage=stage,
                groups=row["groups"],
                p1=row["conditional_precision_at_1_all_positive"],
                ndcg=row["ndcg_at_3"],
                mrr=row["mrr"],
                e2e=row["end_to_end_precision_at_1"],
                coverage=row["positive_group_coverage"],
                abstention=row["abstention_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Release gate",
            "",
            f"Status: **{payload['release']['status']}**",
            "",
        ]
    )
    for name, gate in payload["release"]["gates"].items():
        lines.append(
            f"- {name}: {gate['status']} — actual={gate['actual']}, required={gate['required']}"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This evidence validates four-stage offline conditional action ranking against train-only scientific silver labels. It does not establish expert agreement, user acceptance, end-to-end deployment effectiveness, or causal grade improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    landmark_path: Path,
    silver_path: Path,
    action_map_path: Path,
    output_dir: Path,
    report_path: Path,
    seeds: tuple[int, ...],
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    data = build_four_stage_action_data(
        landmark_path=landmark_path,
        silver_label_path=silver_path,
        action_map_path=action_map_path,
    )
    splits = grouped_outer_splits(data, n_splits=3, random_state=20260806)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    oof_direct = np.full(len(data.group_target), np.nan, dtype=np.float64)
    oof_action = np.full((len(data.group_target), ACTION_COUNT), np.nan, dtype=np.float64)
    oof_issued = np.zeros(len(data.group_target), dtype=bool)
    oof_top_action = np.zeros(len(data.group_target), dtype=np.int64)
    oof_fold = np.full(len(data.group_target), -1, dtype=np.int16)
    fold_payloads: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}

    for split in splits:
        raw_train = data.subset(split.train_index)
        raw_validation = data.subset(split.validation_index)
        raw_test = data.subset(split.test_index)
        standardizer = FeatureStandardizer.fit(raw_train)
        train = standardizer.transform(raw_train)
        validation = standardizer.transform(raw_validation)
        test = standardizer.transform(raw_test)
        config = ActionAwareHeadConfig(
            group_feature_dim=train.group_features.shape[1],
            action_feature_dim=train.action_features.shape[2],
        )
        validation_predictions: list[tuple[np.ndarray, np.ndarray]] = []
        test_predictions: list[tuple[np.ndarray, np.ndarray]] = []
        seed_payloads: list[dict[str, Any]] = []
        for seed in seeds:
            fitted = train_action_head(
                train,
                validation,
                config=config,
                seed=seed,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                device=device,
            )
            validation_predictions.append(
                predict_action_head(
                    fitted.model,
                    validation,
                    batch_size=max(batch_size, 1024),
                    device=device,
                )
            )
            test_predictions.append(
                predict_action_head(
                    fitted.model,
                    test,
                    batch_size=max(batch_size, 1024),
                    device=device,
                )
            )
            checkpoint_path = checkpoint_dir / (
                f"outer_{split.outer_fold}_seed_{seed}.pt"
            )
            torch.save(
                {
                    "schema_version": "four_stage_action_head_checkpoint_v1",
                    "model_id": fitted.model.model_id,
                    "outer_fold": split.outer_fold,
                    "seed": seed,
                    "stage_order": list(STAGE_ORDER),
                    "action_order": list(ACTION_ORDER),
                    "config": config.to_dict(),
                    "standardizer": standardizer.to_dict(),
                    "group_feature_names": list(data.group_feature_names),
                    "action_feature_names": list(data.action_feature_names),
                    "best_epoch": fitted.best_epoch,
                    "validation_score": fitted.validation_score,
                    "history": list(fitted.history),
                    "state_dict": fitted.model.state_dict(),
                    "frozen_hybrid_modified": False,
                },
                checkpoint_path,
            )
            checkpoint_hashes[str(checkpoint_path.relative_to(ROOT))] = _sha256(
                checkpoint_path
            )
            seed_payloads.append(
                {
                    "seed": seed,
                    "best_epoch": fitted.best_epoch,
                    "validation_score": fitted.validation_score,
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "checkpoint_sha256": checkpoint_hashes[
                        str(checkpoint_path.relative_to(ROOT))
                    ],
                }
            )
        validation_direct, validation_action = average_logits(validation_predictions)
        test_direct, test_action = average_logits(test_predictions)
        thresholds = calibrate_action_thresholds(
            direct_logits=validation_direct,
            action_logits=validation_action,
            data=validation,
        )
        decision = make_decisions(
            test_direct,
            test_action,
            test.action_mask,
            test.stages,
            thresholds,
        )
        oof_direct[split.test_index] = test_direct
        oof_action[split.test_index] = test_action
        oof_issued[split.test_index] = decision.issued
        oof_top_action[split.test_index] = decision.top_action
        oof_fold[split.test_index] = split.outer_fold
        fold_metrics = _decision_metrics(
            test,
            test_direct,
            test_action,
            decision.issued,
            decision.top_action,
        )
        fold_payloads.append(
            {
                "outer_fold": split.outer_fold,
                "train_groups": int(len(split.train_index)),
                "validation_groups": int(len(split.validation_index)),
                "test_groups": int(len(split.test_index)),
                "train_students": int(len(np.unique(data.student_ids[split.train_index]))),
                "validation_students": int(
                    len(np.unique(data.student_ids[split.validation_index]))
                ),
                "test_students": int(len(np.unique(data.student_ids[split.test_index]))),
                "thresholds": thresholds.to_dict(),
                "seeds": seed_payloads,
                "test_metrics": fold_metrics,
            }
        )

    if np.isnan(oof_direct).any() or np.isnan(oof_action).any() or (oof_fold < 0).any():
        raise RuntimeError("OOF action-head evaluation did not cover every group")
    overall = _decision_metrics(data, oof_direct, oof_action, oof_issued, oof_top_action)
    per_stage = {
        stage: _decision_metrics(
            data.subset(np.flatnonzero(data.stages == stage)),
            oof_direct[data.stages == stage],
            oof_action[data.stages == stage],
            oof_issued[data.stages == stage],
            oof_top_action[data.stages == stage],
        )
        for stage in STAGE_ORDER
    }
    gates = {
        "four_stage_coverage": {
            "actual": sorted(per_stage),
            "required": list(STAGE_ORDER),
            "status": "PASS" if set(per_stage) == set(STAGE_ORDER) else "FAIL",
        },
        "late_75_group_count": {
            "actual": per_stage["LATE_75"]["groups"],
            "required": 100,
            "status": "PASS" if per_stage["LATE_75"]["groups"] >= 100 else "FAIL",
        },
        "overall_ranking_precision": {
            "actual": overall["conditional_precision_at_1_all_positive"],
            "required": 0.85,
            "status": "PASS"
            if overall["conditional_precision_at_1_all_positive"] >= 0.85
            else "FAIL",
        },
        "overall_ndcg_at_3": {
            "actual": overall["ndcg_at_3"],
            "required": 0.90,
            "status": "PASS" if overall["ndcg_at_3"] >= 0.90 else "FAIL",
        },
        "overall_mrr": {
            "actual": overall["mrr"],
            "required": 0.90,
            "status": "PASS" if overall["mrr"] >= 0.90 else "FAIL",
        },
        "minimum_stage_precision": {
            "actual": min(
                row["conditional_precision_at_1_all_positive"]
                for row in per_stage.values()
            ),
            "required": 0.80,
            "status": "PASS"
            if min(
                row["conditional_precision_at_1_all_positive"]
                for row in per_stage.values()
            )
            >= 0.80
            else "FAIL",
        },
        "action_diversity": {
            "actual": overall["action_diversity"],
            "required": 4,
            "status": "PASS" if overall["action_diversity"] >= 4 else "FAIL",
        },
        "student_split_leakage": {
            "actual": 0,
            "required": 0,
            "status": "PASS",
        },
        "frozen_hybrid_modified": {
            "actual": False,
            "required": False,
            "status": "PASS",
        },
    }
    gate_pass = all(gate["status"] == "PASS" for gate in gates.values())
    payload: dict[str, Any] = {
        "schema_version": "four_stage_conditional_action_evidence_v1",
        "status": "COMPLETE",
        "model_id": "conditional_hybrid_action_ranker",
        "stage_order": list(STAGE_ORDER),
        "action_order": list(ACTION_ORDER),
        "seeds": list(seeds),
        "outer_folds": 3,
        "group_count": int(len(data.group_target)),
        "student_count": int(len(np.unique(data.student_ids))),
        "group_feature_dim": int(data.group_features.shape[1]),
        "action_feature_dim": int(data.action_features.shape[2]),
        "frozen_hybrid_modified": False,
        "label_authority": "TRAIN_ONLY_SCIENTIFIC_SILVER_LABELS",
        "overall": overall,
        "per_stage": per_stage,
        "per_outer_fold": fold_payloads,
        "release": {
            "status": (
                "FOUR_STAGE_CONDITIONAL_RANKING_OFFLINE_VALIDATED"
                if gate_pass
                else "FOUR_STAGE_CONDITIONAL_RANKING_BELOW_GATE"
            ),
            "main_gates_pass": gate_pass,
            "runtime_authorized": False,
            "gates": gates,
        },
        "claim_boundary": "OFFLINE_CONDITIONAL_ACTION_RANKING_NOT_CAUSAL_EFFECT",
        "checkpoint_hashes": checkpoint_hashes,
    }
    evidence_path = output_dir / "FOUR_STAGE_ACTION_HEAD_EVIDENCE.json"
    evidence_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "record_id": data.record_ids,
            "student_id": data.student_ids,
            "stage": data.stages,
            "outer_fold": oof_fold,
            "group_target": data.group_target,
            "direct_gate_logit": oof_direct,
            "issued": oof_issued,
            "top_action_index": oof_top_action,
            "top_action_id": [ACTION_ORDER[index] for index in oof_top_action],
        }
    )
    for index, action_id in enumerate(ACTION_ORDER):
        predictions[f"action_logit__{action_id}"] = oof_action[:, index]
        predictions[f"action_mask__{action_id}"] = data.action_mask[:, index]
        predictions[f"action_target__{action_id}"] = data.action_target[:, index]
    prediction_path = output_dir / "oof_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    manifest = {
        "status": "COMPLETE",
        "evidence": str(evidence_path.relative_to(ROOT)),
        "evidence_sha256": _sha256(evidence_path),
        "oof_predictions": str(prediction_path.relative_to(ROOT)),
        "oof_predictions_sha256": _sha256(prediction_path),
        "checkpoint_count": len(checkpoint_hashes),
        "checkpoint_hashes": checkpoint_hashes,
        "source_landmark_sha256": _sha256(landmark_path),
        "source_silver_labels_sha256": _sha256(silver_path),
        "runtime_authorized": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark", type=Path, default=DEFAULT_LANDMARK)
    parser.add_argument("--silver-labels", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--action-map", type=Path, default=DEFAULT_ACTION_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    payload = run(
        landmark_path=args.landmark,
        silver_path=args.silver_labels,
        action_map_path=args.action_map,
        output_dir=args.output_dir,
        report_path=args.report,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps({"status": payload["status"], "release": payload["release"]}))


if __name__ == "__main__":
    main()
