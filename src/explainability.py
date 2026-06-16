"""Model explanation and rule-based learning-path recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.utils import setup_logger

logger = setup_logger("explainability")

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def calculate_permutation_importance(
    model,
    val_loader,
    device,
    numerical_feature_names,
    categorical_feature_names,
):
    """Measure the F1-Macro drop after shuffling each context feature."""

    model.eval()
    sequences = []
    numerical = []
    categorical = []
    labels = []

    with torch.no_grad():
        for batch in val_loader:
            seq_x, num_x, cat_x, batch_labels, _ = batch[:5]
            sequences.append(seq_x.cpu())
            numerical.append(num_x.cpu())
            categorical.append(cat_x.cpu())
            labels.extend(batch_labels.numpy())

    full_seq = torch.cat(sequences, dim=0)
    full_num = torch.cat(numerical, dim=0)
    full_cat = torch.cat(categorical, dim=0)

    def evaluate(num_values, cat_values):
        predictions = []
        with torch.no_grad():
            for start in range(0, len(labels), 32):
                stop = start + 32
                logits = model(
                    full_seq[start:stop].to(device),
                    num_values[start:stop].to(device),
                    cat_values[start:stop].to(device),
                )
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return f1_score(labels, predictions, average="macro", zero_division=0)

    baseline_f1 = evaluate(full_num, full_cat)
    importances = []
    for column, feature_name in enumerate(numerical_feature_names):
        shuffled = full_num.clone()
        shuffled[:, column] = shuffled[torch.randperm(len(shuffled)), column]
        importances.append((feature_name, baseline_f1 - evaluate(shuffled, full_cat)))

    for column, feature_name in enumerate(categorical_feature_names):
        shuffled = full_cat.clone()
        shuffled[:, column] = shuffled[torch.randperm(len(shuffled)), column]
        importances.append((feature_name, baseline_f1 - evaluate(full_num, shuffled)))

    return pd.DataFrame(importances, columns=["Feature", "Importance"]).sort_values(
        "Importance",
        ascending=False,
        ignore_index=True,
    )


def explain_model(
    model,
    val_loader,
    device,
    numerical_feature_names,
    categorical_feature_names,
    out_path,
):
    logger.info("Calculating permutation importance...")
    importance = calculate_permutation_importance(
        model,
        val_loader,
        device,
        numerical_feature_names,
        categorical_feature_names,
    )
    importance.to_csv(out_path, index=False)
    return importance


@dataclass(frozen=True)
class RiskFactor:
    code: str
    title: str
    evidence: str
    priority: int


def extract_student_features(row: dict[str, Any] | pd.Series) -> list[float]:
    def get_num(name: str, default: float = 0.0) -> float:
        val = row.get(name, default) if isinstance(row, dict) else (row[name] if name in row.index else default)
        try:
            return default if pd.isna(val) else float(val)
        except (TypeError, ValueError):
            return default
    return [
        get_num("absences", 0.0),
        get_num("studytime", 1.0),
        get_num("failures", 0.0),
        get_num("G1", 0.0),
        get_num("G2", 0.0),
        get_num("Dalc", 1.0),
        get_num("Walc", 1.0),
        get_num("goout", 1.0),
    ]


def extract_xapi_features(row: dict[str, Any] | pd.Series) -> list[float]:
    def get_num(name: str, default: float = 0.0) -> float:
        val = row.get(name, default) if isinstance(row, dict) else (row[name] if name in row.index else default)
        try:
            return default if pd.isna(val) else float(val)
        except (TypeError, ValueError):
            return default

    def get_str(name: str) -> str:
        val = row.get(name, "") if isinstance(row, dict) else (row[name] if name in row.index else "")
        return str(val).strip().lower()

    return [
        get_num("raisedhands", 0.0),
        get_num("VisITedResources", 0.0),
        get_num("AnnouncementsView", 0.0),
        get_num("Discussion", 0.0),
        1.0 if get_str("StudentAbsenceDays") == "above-7" else 0.0,
        1.0 if get_str("ParentAnsweringSurvey") == "no" else 0.0,
        1.0 if get_str("ParentschoolSatisfaction") == "bad" else 0.0,
    ]


class RecommendationMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RuleBasedLearningPathEngine:
    """Map observable academic risks to a staged learning roadmap."""

    def __init__(self, dataset_kind: str):
        if dataset_kind not in {"student", "xapi"}:
            raise ValueError(f"Unsupported dataset kind: {dataset_kind}")
        self.dataset_kind = dataset_kind

        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]

        if self.dataset_kind == "student":
            self.input_dim = 8
            self.weights_path = project_root / "models" / "mlp_rec_student.pt"
        else:
            self.input_dim = 7
            self.weights_path = project_root / "models" / "mlp_rec_xapi.pt"

        self.model = RecommendationMLP(self.input_dim, 6)

        if not self.weights_path.exists():
            self._auto_train(project_root)

        self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def _auto_train(self, project_root: Path):
        logger.info(f"Auto-training MLP recommendation model for {self.dataset_kind}...")
        
        # Load raw data and compute ground truth
        X_list = []
        Y_list = []

        if self.dataset_kind == "student":
            mat_path = project_root / "data" / "raw" / "student-mat.csv"
            por_path = project_root / "data" / "raw" / "student-por.csv"
            df_mat = pd.read_csv(mat_path, sep=";")
            df_por = pd.read_csv(por_path, sep=";")
            df = pd.concat([df_mat, df_por], ignore_index=True)

            for _, row in df.iterrows():
                X_list.append(extract_student_features(row))
                
                absences = float(row.get("absences", 0.0))
                study_time = float(row.get("studytime", 1.0))
                failures = float(row.get("failures", 0.0))
                g1 = float(row.get("G1", 0.0))
                g2 = float(row.get("G2", 0.0))
                alcohol = float(row.get("Dalc", 1.0)) + float(row.get("Walc", 1.0))
                goout = float(row.get("goout", 1.0))
                ratio = absences / max(study_time, 0.5)

                Y_list.append([
                    float(absences >= 10 or ratio >= 5),
                    float(failures > 0),
                    float(g2 < 10 or (g1 > 0 and g2 < g1)),
                    float(study_time <= 1),
                    float(alcohol >= 6),
                    float(goout >= 4)
                ])
        else:
            xapi_path = project_root / "data" / "raw" / "xAPI-Edu-Data.csv"
            df = pd.read_csv(xapi_path, sep=",")

            for _, row in df.iterrows():
                X_list.append(extract_xapi_features(row))

                raised_hands = float(row.get("raisedhands", 0.0))
                resources = float(row.get("VisITedResources", 0.0))
                announcements = float(row.get("AnnouncementsView", 0.0))
                discussion = float(row.get("Discussion", 0.0))
                absence_days = str(row.get("StudentAbsenceDays", "")).strip().lower()
                survey = str(row.get("ParentAnsweringSurvey", "")).strip().lower()
                satisfaction = str(row.get("ParentschoolSatisfaction", "")).strip().lower()

                Y_list.append([
                    float(absence_days == "above-7"),
                    float(resources < 40),
                    float(raised_hands < 30 or discussion < 30),
                    float(announcements < 30),
                    float(survey == "no"),
                    float(satisfaction == "bad")
                ])

        X_train = torch.tensor(X_list, dtype=torch.float32)
        Y_train = torch.tensor(Y_list, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(150):
            optimizer.zero_grad()
            logits = self.model(X_train)
            loss = criterion(logits, Y_train)
            loss.backward()
            optimizer.step()

        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        logger.info(f"Auto-training complete. Weights saved to {self.weights_path}.")

    @staticmethod
    def _number(features: dict[str, Any], name: str, default: float = 0.0) -> float:
        try:
            value = features.get(name, default)
            return default if pd.isna(value) else float(value)
        except (TypeError, ValueError):
            return default

    def _student_actions(self, risk_codes: set[str]) -> list[dict[str, str]]:
        actions = []
        if "attendance" in risk_codes:
            actions.append({"phase": "Tuần 1", "goal": "Khôi phục chuyên cần", "actions": "Lập lịch đi học đủ; đăng ký lớp bù cho nội dung G1/G2 đã bỏ lỡ; cố vấn kiểm tra chuyên cần mỗi tuần."})
        if "failure_history" in risk_codes or "grade_gap" in risk_codes:
            actions.append({"phase": "Tuần 1-2", "goal": "Bù lỗ hổng kiến thức", "actions": "Làm bài chẩn đoán theo chủ đề; học lại hai chủ đề yếu nhất; hoàn thành tối thiểu 3 bài luyện tập có phản hồi mỗi tuần."})
        if "study_time" in risk_codes or "time_management" in risk_codes:
            actions.append({"phase": "Tuần 2-4", "goal": "Ổn định nếp tự học", "actions": "Tăng ít nhất 3 giờ tự học có kế hoạch mỗi tuần; chia thành các phiên 45 phút; giảm một buổi đi chơi trong tuần nếu trùng lịch học."})
        if "wellbeing" in risk_codes:
            actions.append({"phase": "Tuần 2-4", "goal": "Điều chỉnh thói quen sinh hoạt", "actions": "Giảm sử dụng đồ uống có cồn trong ngày học; duy trì giấc ngủ và lịch học cố định; trao đổi với cố vấn khi khó tự điều chỉnh."})
        actions.append({"phase": "Mỗi cuối tuần", "goal": "Theo dõi tiến bộ", "actions": "Cập nhật điểm bài tập và tỷ lệ chuyên cần; nếu điểm luyện tập dưới 60% trong hai tuần liên tiếp, chuyển sang phụ đạo trực tiếp."})
        return actions

    def _xapi_actions(self, risk_codes: set[str]) -> list[dict[str, str]]:
        actions = []
        if "attendance" in risk_codes:
            actions.append({"phase": "Tuần 1", "goal": "Khôi phục chuyên cần", "actions": "Xác nhận nguyên nhân vắng; hoàn thành gói bài bù; giáo viên kiểm tra tiến độ sau từng buổi học."})
        if "resource_usage" in risk_codes or "course_updates" in risk_codes:
            actions.append({"phase": "Tuần 1-2", "goal": "Tăng sử dụng học liệu", "actions": "Truy cập hệ thống ít nhất 4 ngày/tuần; đọc toàn bộ thông báo; hoàn thành hai tài nguyên trọng tâm trước buổi học tiếp theo."})
        if "class_engagement" in risk_codes:
            actions.append({"phase": "Tuần 2-4", "goal": "Tăng tương tác học tập", "actions": "Đặt ít nhất một câu hỏi hoặc phản hồi trong mỗi buổi; tham gia hai thảo luận học thuật mỗi tuần; giáo viên ghi nhận mức tham gia."})
        if "parent_support" in risk_codes or "school_support" in risk_codes:
            actions.append({"phase": "Trong 2 tuần", "goal": "Phối hợp gia đình - nhà trường", "actions": "Gửi báo cáo tiến độ ngắn cho phụ huynh; thống nhất một mục tiêu học tập và một lịch kiểm tra hằng tuần."})
        actions.append({"phase": "Mỗi cuối tuần", "goal": "Đánh giá lộ trình", "actions": "So sánh mức truy cập, thảo luận và bài tập với tuần trước; nếu không cải thiện sau hai tuần, bố trí kèm cặp trực tiếp."})
        return actions

    def generate(
        self,
        features: dict[str, Any],
        predicted_class: int,
        confidence: float,
    ) -> dict[str, Any]:
        if self.dataset_kind == "student":
            x = extract_student_features(features)
        else:
            x = extract_xapi_features(features)

        x_tensor = torch.tensor([x], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_tensor)[0]
            probs = torch.sigmoid(logits).numpy()

        active_indices = [i for i, p in enumerate(probs) if p > 0.5]

        risks = []
        if self.dataset_kind == "student":
            absences = self._number(features, "absences")
            study_time = self._number(features, "studytime", 1.0)
            failures = self._number(features, "failures")
            g1 = self._number(features, "G1")
            g2 = self._number(features, "G2")
            alcohol = self._number(features, "Dalc") + self._number(features, "Walc")
            goout = self._number(features, "goout")
            ratio = absences / max(study_time, 0.5)

            if 0 in active_indices:
                risks.append(RiskFactor("attendance", "Nguy cơ chuyên cần", f"Vắng {absences:.0f} buổi; tỷ lệ vắng/học {ratio:.1f}.", 1))
            if 1 in active_indices:
                risks.append(RiskFactor("failure_history", "Lỗ hổng kiến thức tích lũy", f"Số lần trượt môn trước đây: {failures:.0f}.", 1))
            if 2 in active_indices:
                risks.append(RiskFactor("grade_gap", "Kết quả giữa kỳ chưa đạt", f"G1={g1:.0f}, G2={g2:.0f}.", 1))
            if 3 in active_indices:
                risks.append(RiskFactor("study_time", "Thời lượng tự học thấp", f"Mức studytime hiện tại: {study_time:.0f}/4.", 2))
            if 4 in active_indices:
                risks.append(RiskFactor("wellbeing", "Thói quen sinh hoạt ảnh hưởng học tập", f"Tổng Dalc + Walc = {alcohol:.0f}.", 3))
            if 5 in active_indices:
                risks.append(RiskFactor("time_management", "Phân bổ thời gian chưa hợp lý", f"Mức goout hiện tại: {goout:.0f}/5.", 3))
        else:
            raised_hands = self._number(features, "raisedhands")
            resources = self._number(features, "VisITedResources")
            announcements = self._number(features, "AnnouncementsView")
            discussion = self._number(features, "Discussion")

            if 0 in active_indices:
                risks.append(RiskFactor("attendance", "Nguy cơ chuyên cần", "Số ngày vắng học thuộc nhóm Above-7.", 1))
            if 1 in active_indices:
                risks.append(RiskFactor("resource_usage", "Khai thác học liệu thấp", f"VisITedResources={resources:.0f}/100.", 1))
            if 2 in active_indices:
                risks.append(RiskFactor("class_engagement", "Tương tác lớp học thấp", f"raisedhands={raised_hands:.0f}, Discussion={discussion:.0f}.", 2))
            if 3 in active_indices:
                risks.append(RiskFactor("course_updates", "Theo dõi thông báo chưa đều", f"AnnouncementsView={announcements:.0f}/100.", 2))
            if 4 in active_indices:
                risks.append(RiskFactor("parent_support", "Thiếu phối hợp gia đình", "Phụ huynh chưa tham gia khảo sát học tập.", 3))
            if 5 in active_indices:
                risks.append(RiskFactor("school_support", "Cần tăng kết nối nhà trường", "Mức hài lòng của phụ huynh là Bad.", 3))

        risks = sorted(risks, key=lambda risk: risk.priority)
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
            "learning_path": actions,
        }



def generate_learning_path_report(
    original_features: pd.DataFrame,
    predictions: np.ndarray,
    confidences: np.ndarray,
    dataset_kind: str,
) -> pd.DataFrame:
    """Generate one staged learning path for every evaluated student."""

    engine = RuleBasedLearningPathEngine(dataset_kind)
    rows = []
    for row_index, (_, feature_row) in enumerate(original_features.reset_index(drop=True).iterrows()):
        feature_snapshot = feature_row.to_dict()
        recommendation = engine.generate(
            feature_snapshot,
            int(predictions[row_index]),
            float(confidences[row_index]),
        )
        rows.append(
            {
                "source_row_index": row_index,
                "predicted_class": recommendation["predicted_class"],
                "predicted_class_name": recommendation["predicted_class_name"],
                "confidence": recommendation["confidence"],
                "risk_band": recommendation["risk_band"],
                "headline": recommendation["headline"],
                "risk_factors": json.dumps(recommendation["risk_factors"], ensure_ascii=False),
                "learning_path": json.dumps(recommendation["learning_path"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)
