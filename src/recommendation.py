"""Neural risk ranking and staged learning-path generation.

The MLP is trained only on the dataset train pool. Its supervision is an
explicit set of domain criteria, so the resulting offline metrics measure
fidelity to that reference policy rather than causal learning outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, ROOT_DIR
from src.utils import set_seed, setup_logger


logger = setup_logger("recommendation")
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}
RECOMMENDATION_MODELS_DIR = ROOT_DIR / "models" / "recommendation"

DATASET_KIND = {
    "student-mat": "student",
    "student-por": "student",
    "xapi": "xapi",
}

RISK_CODES = {
    "student": (
        "attendance",
        "failure_history",
        "grade_gap",
        "study_time",
        "wellbeing",
        "time_management",
    ),
    "xapi": (
        "attendance",
        "resource_usage",
        "class_engagement",
        "course_updates",
        "parent_support",
        "school_support",
    ),
}


def _number(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, default)
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def extract_features(frame: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """Build the compact feature matrix consumed by the recommendation MLP."""

    kind = DATASET_KIND[dataset_name]
    rows: list[list[float]] = []
    for record in frame.to_dict("records"):
        if kind == "student":
            rows.append(
                [
                    _number(record, "absences"),
                    _number(record, "studytime", 1.0),
                    _number(record, "failures"),
                    _number(record, "G1"),
                    _number(record, "G2"),
                    _number(record, "Dalc", 1.0),
                    _number(record, "Walc", 1.0),
                    _number(record, "goout", 1.0),
                ]
            )
        else:
            rows.append(
                [
                    _number(record, "raisedhands"),
                    _number(record, "VisITedResources"),
                    _number(record, "AnnouncementsView"),
                    _number(record, "Discussion"),
                    float(str(record.get("StudentAbsenceDays", "")).strip().lower() == "above-7"),
                    float(str(record.get("ParentAnsweringSurvey", "")).strip().lower() == "no"),
                    float(str(record.get("ParentschoolSatisfaction", "")).strip().lower() == "bad"),
                ]
            )
    return np.asarray(rows, dtype=np.float32)


def reference_risk_targets(frame: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """Create auditable weak-supervision targets from domain risk criteria."""

    kind = DATASET_KIND[dataset_name]
    targets: list[list[float]] = []
    for record in frame.to_dict("records"):
        if kind == "student":
            absences = _number(record, "absences")
            study_time = _number(record, "studytime", 1.0)
            failures = _number(record, "failures")
            g1 = _number(record, "G1")
            g2 = _number(record, "G2")
            alcohol = _number(record, "Dalc", 1.0) + _number(record, "Walc", 1.0)
            goout = _number(record, "goout", 1.0)
            ratio = absences / max(study_time, 0.5)
            targets.append(
                [
                    float(absences >= 10 or ratio >= 5),
                    float(failures > 0),
                    float(g2 < 10 or (g1 > 0 and g2 < g1)),
                    float(study_time <= 1),
                    float(alcohol >= 6),
                    float(goout >= 4),
                ]
            )
        else:
            targets.append(
                [
                    float(str(record.get("StudentAbsenceDays", "")).strip().lower() == "above-7"),
                    float(_number(record, "VisITedResources") < 40),
                    float(_number(record, "raisedhands") < 30 or _number(record, "Discussion") < 30),
                    float(_number(record, "AnnouncementsView") < 30),
                    float(str(record.get("ParentAnsweringSurvey", "")).strip().lower() == "no"),
                    float(str(record.get("ParentschoolSatisfaction", "")).strip().lower() == "bad"),
                ]
            )
    return np.asarray(targets, dtype=np.float32)


class RecommendationMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _training_fingerprint(features: np.ndarray, targets: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(features).tobytes())
    digest.update(np.ascontiguousarray(targets).tobytes())
    return digest.hexdigest()


def train_recommendation_model(
    dataset_name: str,
    train_frame: pd.DataFrame,
    model_path: Path | None = None,
    seed: int = 42,
    max_epochs: int = 800,
) -> dict[str, Any]:
    """Train a deterministic MLP using only the locked experiment train pool."""

    if dataset_name not in DATASET_KIND:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    set_seed(seed)
    features = extract_features(train_frame, dataset_name)
    targets = reference_risk_targets(train_frame, dataset_name)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale

    indices = np.arange(len(normalized))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    x_train = torch.tensor(normalized[train_idx], dtype=torch.float32)
    y_train = torch.tensor(targets[train_idx], dtype=torch.float32)
    x_val = torch.tensor(normalized[val_idx], dtype=torch.float32)
    y_val = torch.tensor(targets[val_idx], dtype=torch.float32)

    positives = y_train.sum(dim=0)
    negatives = len(y_train) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=0.5, max=10.0)

    model = RecommendationMLP(normalized.shape[1], targets.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = None
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= 60:
            break

    if best_state is None:
        raise RuntimeError("Recommendation MLP training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    checkpoint = {
        "schema_version": 2,
        "dataset_name": dataset_name,
        "dataset_kind": DATASET_KIND[dataset_name],
        "risk_codes": list(RISK_CODES[DATASET_KIND[dataset_name]]),
        "input_dim": int(normalized.shape[1]),
        "output_dim": int(targets.shape[1]),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "state_dict": model.state_dict(),
        "training_rows": int(len(train_frame)),
        "training_fingerprint": _training_fingerprint(features, targets),
        "seed": seed,
        "epochs_completed": epoch + 1,
        "best_validation_loss": best_loss,
        "supervision": "domain_criteria_weak_supervision",
    }
    model_path = model_path or (RECOMMENDATION_MODELS_DIR / f"{dataset_name}_mlp.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    logger.info("Saved recommendation MLP for %s to %s", dataset_name, model_path)
    return checkpoint


def load_training_pool(dataset_name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{dataset_name}_3class_train_pool.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing train pool: {path}")
    return pd.read_csv(path)


def load_or_train_recommendation_model(
    dataset_name: str,
    train_frame: pd.DataFrame | None = None,
    force_retrain: bool = False,
) -> tuple[RecommendationMLP, dict[str, Any]]:
    train_frame = load_training_pool(dataset_name) if train_frame is None else train_frame
    model_path = RECOMMENDATION_MODELS_DIR / f"{dataset_name}_mlp.pt"
    features = extract_features(train_frame, dataset_name)
    targets = reference_risk_targets(train_frame, dataset_name)
    expected_fingerprint = _training_fingerprint(features, targets)
    checkpoint = None
    if model_path.exists() and not force_retrain:
        candidate = torch.load(model_path, map_location="cpu", weights_only=False)
        if (
            isinstance(candidate, dict)
            and candidate.get("schema_version") == 2
            and candidate.get("training_fingerprint") == expected_fingerprint
        ):
            checkpoint = candidate
    if checkpoint is None:
        checkpoint = train_recommendation_model(dataset_name, train_frame, model_path=model_path)

    model = RecommendationMLP(checkpoint["input_dim"], checkpoint["output_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@dataclass(frozen=True)
class RiskFactor:
    code: str
    title: str
    evidence: str
    priority: int
    score: float


class MLPLearningPathEngine:
    """Rank observable risks with an MLP and assemble staged interventions."""

    def __init__(self, dataset_name: str, train_frame: pd.DataFrame | None = None):
        if dataset_name not in DATASET_KIND:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.dataset_kind = DATASET_KIND[dataset_name]
        self.model, self.checkpoint = load_or_train_recommendation_model(dataset_name, train_frame)
        self.mean = np.asarray(self.checkpoint["feature_mean"], dtype=np.float32)
        self.scale = np.asarray(self.checkpoint["feature_scale"], dtype=np.float32)

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        features = extract_features(frame, self.dataset_name)
        normalized = (features - self.mean) / self.scale
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(normalized, dtype=torch.float32))).numpy()

    def _risk_mapping(self, features: dict[str, Any], scores: np.ndarray) -> dict[int, RiskFactor]:
        if self.dataset_kind == "student":
            absences = _number(features, "absences")
            study_time = _number(features, "studytime", 1.0)
            ratio = absences / max(study_time, 0.5)
            return {
                0: RiskFactor("attendance", "Nguy cơ chuyên cần", f"Vắng {absences:.0f} buổi; tỷ lệ vắng/học {ratio:.1f}.", 1, float(scores[0])),
                1: RiskFactor("failure_history", "Lỗ hổng kiến thức tích lũy", f"Số lần trượt môn trước đây: {_number(features, 'failures'):.0f}.", 1, float(scores[1])),
                2: RiskFactor("grade_gap", "Kết quả giữa kỳ chưa đạt", f"G1={_number(features, 'G1'):.0f}, G2={_number(features, 'G2'):.0f}.", 1, float(scores[2])),
                3: RiskFactor("study_time", "Thời lượng tự học thấp", f"Mức studytime hiện tại: {study_time:.0f}/4.", 2, float(scores[3])),
                4: RiskFactor("wellbeing", "Thói quen sinh hoạt ảnh hưởng học tập", f"Tổng Dalc + Walc = {_number(features, 'Dalc', 1.0) + _number(features, 'Walc', 1.0):.0f}.", 3, float(scores[4])),
                5: RiskFactor("time_management", "Phân bổ thời gian chưa hợp lý", f"Mức goout hiện tại: {_number(features, 'goout', 1.0):.0f}/5.", 3, float(scores[5])),
            }
        return {
            0: RiskFactor("attendance", "Nguy cơ chuyên cần", "Số ngày vắng học thuộc nhóm Above-7.", 1, float(scores[0])),
            1: RiskFactor("resource_usage", "Khai thác học liệu thấp", f"VisITedResources={_number(features, 'VisITedResources'):.0f}/100.", 1, float(scores[1])),
            2: RiskFactor("class_engagement", "Tương tác lớp học thấp", f"raisedhands={_number(features, 'raisedhands'):.0f}, Discussion={_number(features, 'Discussion'):.0f}.", 2, float(scores[2])),
            3: RiskFactor("course_updates", "Theo dõi thông báo chưa đều", f"AnnouncementsView={_number(features, 'AnnouncementsView'):.0f}/100.", 2, float(scores[3])),
            4: RiskFactor("parent_support", "Thiếu phối hợp gia đình", "Phụ huynh chưa tham gia khảo sát học tập.", 3, float(scores[4])),
            5: RiskFactor("school_support", "Cần tăng kết nối nhà trường", "Mức hài lòng của phụ huynh là Bad.", 3, float(scores[5])),
        }

    @staticmethod
    def _student_actions(risk_codes: set[str]) -> list[dict[str, str]]:
        actions = []
        if "attendance" in risk_codes:
            actions.append({"phase": "Tuần 1", "goal": "Khôi phục chuyên cần", "actions": "Lập lịch đi học đủ; đăng ký lớp bù cho nội dung G1/G2 đã bỏ lỡ; cố vấn kiểm tra chuyên cần mỗi tuần."})
        if {"failure_history", "grade_gap"} & risk_codes:
            actions.append({"phase": "Tuần 1-2", "goal": "Bù lỗ hổng kiến thức", "actions": "Làm bài chẩn đoán theo chủ đề; học lại hai chủ đề yếu nhất; hoàn thành tối thiểu 3 bài luyện tập có phản hồi mỗi tuần."})
        if {"study_time", "time_management"} & risk_codes:
            actions.append({"phase": "Tuần 2-4", "goal": "Ổn định nếp tự học", "actions": "Tăng ít nhất 3 giờ tự học có kế hoạch mỗi tuần; chia thành các phiên 45 phút; giảm một buổi đi chơi trong tuần nếu trùng lịch học."})
        if "wellbeing" in risk_codes:
            actions.append({"phase": "Tuần 2-4", "goal": "Điều chỉnh thói quen sinh hoạt", "actions": "Giảm sử dụng đồ uống có cồn trong ngày học; duy trì giấc ngủ và lịch học cố định; trao đổi với cố vấn khi khó tự điều chỉnh."})
        actions.append({"phase": "Mỗi cuối tuần", "goal": "Theo dõi tiến bộ", "actions": "Cập nhật điểm bài tập và tỷ lệ chuyên cần; nếu điểm luyện tập dưới 60% trong hai tuần liên tiếp, chuyển sang phụ đạo trực tiếp."})
        return actions

    @staticmethod
    def _xapi_actions(risk_codes: set[str]) -> list[dict[str, str]]:
        actions = []
        if "attendance" in risk_codes:
            actions.append({"phase": "Tuần 1", "goal": "Khôi phục chuyên cần", "actions": "Xác nhận nguyên nhân vắng; hoàn thành gói bài bù; giáo viên kiểm tra tiến độ sau từng buổi học."})
        if {"resource_usage", "course_updates"} & risk_codes:
            actions.append({"phase": "Tuần 1-2", "goal": "Tăng sử dụng học liệu", "actions": "Truy cập hệ thống ít nhất 4 ngày/tuần; đọc toàn bộ thông báo; hoàn thành hai tài nguyên trọng tâm trước buổi học tiếp theo."})
        if "class_engagement" in risk_codes:
            actions.append({"phase": "Tuần 2-4", "goal": "Tăng tương tác học tập", "actions": "Đặt ít nhất một câu hỏi hoặc phản hồi trong mỗi buổi; tham gia hai thảo luận học thuật mỗi tuần; giáo viên ghi nhận mức tham gia."})
        if {"parent_support", "school_support"} & risk_codes:
            actions.append({"phase": "Trong 2 tuần", "goal": "Phối hợp gia đình - nhà trường", "actions": "Gửi báo cáo tiến độ ngắn cho phụ huynh; thống nhất một mục tiêu học tập và một lịch kiểm tra hằng tuần."})
        actions.append({"phase": "Mỗi cuối tuần", "goal": "Đánh giá lộ trình", "actions": "So sánh mức truy cập, thảo luận và bài tập với tuần trước; nếu không cải thiện sau hai tuần, bố trí kèm cặp trực tiếp."})
        return actions

    def generate(self, features: dict[str, Any], predicted_class: int, confidence: float) -> dict[str, Any]:
        scores = self.predict_scores(pd.DataFrame([features]))[0]
        selected = np.flatnonzero(scores >= 0.5).tolist()
        if not selected and predicted_class != 2:
            selected = [int(np.argmax(scores))]
        mapping = self._risk_mapping(features, scores)
        risks = sorted((mapping[index] for index in selected), key=lambda item: (item.priority, -item.score))
        risk_codes = {risk.code for risk in risks}

        if predicted_class == 2 and not risks:
            risk_band = "stable"
            headline = "Duy trì lộ trình học tập hiện tại"
        elif predicted_class == 0 or any(risk.priority == 1 for risk in risks):
            risk_band = "high"
            headline = "Lộ trình can thiệp ưu tiên 4 tuần"
        else:
            risk_band = "moderate"
            headline = "Lộ trình củng cố để tiến lên nhóm High"

        actions = self._student_actions(risk_codes) if self.dataset_kind == "student" else self._xapi_actions(risk_codes)
        return {
            "predicted_class": int(predicted_class),
            "predicted_class_name": CLASS_NAMES[int(predicted_class)],
            "confidence": round(float(confidence), 6),
            "risk_band": risk_band,
            "headline": headline,
            "risk_factors": [risk.__dict__ for risk in risks],
            "risk_scores": {code: round(float(score), 6) for code, score in zip(RISK_CODES[self.dataset_kind], scores)},
            "learning_path": actions,
        }


def generate_learning_path_report(
    original_features: pd.DataFrame,
    predictions: np.ndarray,
    confidences: np.ndarray,
    dataset_name: str,
    train_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate one MLP-ranked staged learning path per evaluated student."""

    engine = MLPLearningPathEngine(dataset_name, train_frame=train_frame)
    rows = []
    for row_index, record in enumerate(original_features.reset_index(drop=True).to_dict("records")):
        recommendation = engine.generate(record, int(predictions[row_index]), float(confidences[row_index]))
        rows.append(
            {
                "source_row_index": row_index,
                "predicted_class": recommendation["predicted_class"],
                "predicted_class_name": recommendation["predicted_class_name"],
                "confidence": recommendation["confidence"],
                "risk_band": recommendation["risk_band"],
                "headline": recommendation["headline"],
                "risk_factors": json.dumps(recommendation["risk_factors"], ensure_ascii=False),
                "risk_scores": json.dumps(recommendation["risk_scores"], ensure_ascii=False),
                "learning_path": json.dumps(recommendation["learning_path"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)
