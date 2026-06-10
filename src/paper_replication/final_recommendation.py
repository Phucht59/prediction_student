from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.paper_replication.pipeline import PROJECT_ROOT, resolve_columns
from src.paper_replication.v15_prediction_ensemble import metric_dict
from src.paper_replication.v6_case_sweep import SweepCase, make_model, prepare_split, predict_proba, read_dataset


FINAL_DIR = PROJECT_ROOT / "models" / "final"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MANIFEST = FINAL_DIR / "final_model_manifest.json"
DEFAULT_RESULTS = RESULTS_DIR / "final_predictions_and_recommendations.json"
DEFAULT_REPORT = REPORTS_DIR / "final_learning_recommendations_report.md"

STUDENT_LABELS = ["Very low", "Low", "Medium", "Good", "Excellent"]
XAPI_LABELS = ["Low", "Middle", "High"]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def case_from_params(params: dict[str, Any]) -> SweepCase:
    allowed = {field.name for field in fields(SweepCase)}
    values = {key: value for key, value in params.items() if key in allowed}
    if "datasets" in values:
        values["datasets"] = tuple(values["datasets"])
    return SweepCase(**values)


def resolve_model_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_single_model_prediction(dataset: str, model_file: str) -> dict[str, Any]:
    model_path = resolve_model_path(model_file)
    payload = torch.load(model_path, map_location="cpu")
    case = case_from_params(payload["case_params"])
    seed = int(payload["seed"])
    raw = read_dataset(dataset)
    if payload.get("protocol") == "strict validation" or case.name.startswith("strict") or "strict" in model_file:
        from src.paper_replication.v18_strict_validation import prepare_strict_split
        split = prepare_strict_split(dataset, raw, case, seed)
    else:
        split = prepare_split(dataset, raw, case, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(dataset, split, case).to(device)
    model.load_state_dict(payload["state_dict"])
    probs = predict_proba(model, split.X_test, int(case.batch_size), device)
    pred = probs.argmax(axis=1)
    return {
        "payload": payload,
        "case": case,
        "raw": raw,
        "split": split,
        "probabilities": probs,
        "predictions": pred,
        "model_path": str(model_path),
    }


def evaluate_final_models(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = manifest["selected_models"]
    outputs: dict[str, dict[str, Any]] = {}
    for dataset in ("student-mat", "student-por", "xapi"):
        item = selected[dataset]
        if "members" in item:
            member_outputs = [load_single_model_prediction(dataset, member["model_file"]) for member in item["members"]]
            reference = member_outputs[0]
            weights = np.asarray(item.get("weights", [1.0 / len(member_outputs)] * len(member_outputs)), dtype=np.float64)
            probs = np.tensordot(weights, np.stack([m["probabilities"] for m in member_outputs], axis=0), axes=(0, 0))
            outputs[dataset] = {
                **reference,
                "probabilities": probs,
                "predictions": probs.argmax(axis=1),
                "ensemble_members": member_outputs,
            }
        else:
            prediction = load_single_model_prediction(dataset, item["model_file"])
            outputs[dataset] = prediction
    return outputs


def student_recommendations(raw_row: dict[str, Any], pred_class: int, confidence: float) -> list[str]:
    g1 = float(raw_row.get("G1", 0))
    g2 = float(raw_row.get("G2", 0))
    recs: list[str] = []
    if pred_class <= 1:
        recs.append("Dat lich hoc phu dao hang tuan va theo doi bai tap ngan theo tung chu de.")
    elif pred_class == 2:
        recs.append("Duy tri luyen tap dinh ky va tap trung vao cac chu de bi sai nhieu de day len nhom kha.")
    else:
        recs.append("Duy tri nhip hoc hien tai va giao them bai nang cao de giu on dinh thanh tich.")
    if g2 < 10:
        recs.append("Uu tien on lai kien thuc nen tang vi diem G2 dang duoi nguong dat.")
    if g2 < g1:
        recs.append("Can can thiep som vi diem G2 giam so voi G1.")
    if float(raw_row.get("failures", 0) or 0) > 0:
        recs.append("Lap ke hoach bu lo hong do co lich su rot mon/that bai hoc tap.")
    if float(raw_row.get("absences", 0) or 0) >= 10:
        recs.append("Giam vang hoc va them buoi hoc bu vi so ngay vang cao.")
    if float(raw_row.get("studytime", 2) or 2) <= 1:
        recs.append("Tang thoi gian tu hoc len muc toi thieu 2 buoi moi tuan.")
    if confidence < 0.55:
        recs.append("Du doan chua chac chan cao; nen danh gia them bang bai kiem tra ngan.")
    return recs[:5]


def xapi_recommendations(raw_row: dict[str, Any], pred_class: int, confidence: float) -> list[str]:
    recs: list[str] = []
    if pred_class == 0:
        recs.append("Can can thiep muc cao: theo doi hang tuan va giao muc tieu hoc tap nho.")
    elif pred_class == 1:
        recs.append("Can day tu nhom trung binh len cao bang tang tuong tac va tai nguyen hoc.")
    else:
        recs.append("Duy tri muc tuong tac hien tai va giao nhiem vu nang cao.")
    raised = float(raw_row.get("raisedhands", 0) or 0)
    visited = float(raw_row.get("VisITedResources", raw_row.get("VisitedResources", 0)) or 0)
    discussion = float(raw_row.get("Discussion", 0) or 0)
    if raised < 30:
        recs.append("Tang tan suat phat bieu/dat cau hoi tren lop.")
    if visited < 50:
        recs.append("Tang so lan truy cap tai nguyen hoc tap moi tuan.")
    if str(raw_row.get("StudentAbsenceDays", "")).lower() == "above-7":
        recs.append("Giam vang hoc vi StudentAbsenceDays dang o muc Above-7.")
    if discussion < 30:
        recs.append("Tham gia thao luan nhieu hon de tang muc do tuong tac.")
    if confidence < 0.55:
        recs.append("Du doan chua chac chan cao; can theo doi them hanh vi hoc tap gan nhat.")
    return recs[:5]


def build_dataset_result(dataset: str, output: dict[str, Any]) -> dict[str, Any]:
    split = output["split"]
    probs = output["probabilities"]
    pred = output["predictions"]
    labels = list(range(split.n_classes))
    target_names = STUDENT_LABELS if dataset.startswith("student") else XAPI_LABELS
    metrics = metric_dict(split.y_test, pred)
    report = classification_report(
        split.y_test,
        pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    raw_test = output["raw"].iloc[split.test_indices].reset_index(drop=True)
    low_risk_rows = []
    for local_index, row in raw_test.iterrows():
        pred_class = int(pred[local_index])
        confidence = float(np.max(probs[local_index]))
        true_class = int(split.y_test[local_index])
        if dataset.startswith("student"):
            recs = student_recommendations(row.to_dict(), pred_class, confidence)
            should_include = pred_class <= 2 or true_class <= 2
        else:
            recs = xapi_recommendations(row.to_dict(), pred_class, confidence)
            should_include = pred_class <= 1 or true_class <= 1
        if should_include:
            low_risk_rows.append(
                {
                    "row_index": int(split.test_indices[local_index]),
                    "true_class": true_class,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "recommendations": recs,
                }
            )
    low_risk_rows = sorted(low_risk_rows, key=lambda item: (item["predicted_class"], item["confidence"]))[:15]
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(split.y_test, pred, labels=labels).tolist(),
        "n_recommendation_candidates": int(len(low_risk_rows)),
        "sample_recommendations": low_risk_rows,
    }


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Final Learning Recommendation Report",
        "",
        "This report uses the final three-dataset CNN-BiLSTM prediction models. It does not retrain models.",
        "",
        "## Prediction Metrics Recomputed From Final Models",
        "",
        "| dataset | accuracy | precision_macro | recall_macro | f1_macro | rmse | r2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, result in payload["datasets"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {dataset} | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
            f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} | {metrics['rmse']:.4f} | {metrics['r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation Logic",
            "",
            "- Student datasets: recommendations are based on predicted G3 class, confidence, G1/G2 trend, failures, absences, and studytime.",
            "- xAPI: recommendations are based on predicted class, confidence, raised hands, visited resources, absence days, and discussion activity.",
            "- The rules are deterministic and explainable; they should be reviewed by an academic advisor before real deployment.",
            "",
            "## Sample At-Risk Recommendations",
            "",
        ]
    )
    for dataset, result in payload["datasets"].items():
        lines.append(f"### {dataset}")
        if not result["sample_recommendations"]:
            lines.append("")
            lines.append("No at-risk sample recommendations were selected.")
            lines.append("")
            continue
        for sample in result["sample_recommendations"][:5]:
            lines.append(
                f"- row `{sample['row_index']}` true={sample['true_class']} pred={sample['predicted_class']} "
                f"confidence={sample['confidence']:.3f}: "
                + " / ".join(sample["recommendations"])
            )
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(manifest_path: Path = DEFAULT_MANIFEST, results_path: Path = DEFAULT_RESULTS, report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = evaluate_final_models(manifest)
    dataset_results = {dataset: build_dataset_result(dataset, output) for dataset, output in outputs.items()}
    payload = {
        "manifest": str(manifest_path),
        "protocol_warning": manifest.get("protocol_warning"),
        "datasets": dataset_results,
        "artifacts": {"results": str(results_path), "report": str(report_path)},
    }
    save_json(results_path, payload)
    write_report(payload, report_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final models and create learning recommendations.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_all(Path(args.manifest), Path(args.results), Path(args.report))
    for dataset, result in payload["datasets"].items():
        metrics = result["metrics"]
        print(
            f"{dataset:12s} acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
            f"recommendation_candidates={result['n_recommendation_candidates']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
