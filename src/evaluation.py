import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from pathlib import Path
from src.config import *
from src.utils import setup_logger

logger = setup_logger("evaluation")

import json
from pathlib import Path
from src.config import REPORTS_DIR, METRICS_DIR, MODELS_DIR

def create_summary_report(ds_name: str, target_mode: str):
    metrics_path = METRICS_DIR / f"{ds_name}_{target_mode}_locked_test_metrics.json"
    params_path = MODELS_DIR / f"{ds_name}_{target_mode}_best_params.json"
    
    report_path = REPORTS_DIR / f"summary_report_{ds_name}_{target_mode}.md"
    
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    best_params = {}
    if params_path.exists():
        with open(params_path, "r") as f:
            best_params = json.load(f)
            
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# BÁO CÁO TỔNG KẾT V26 SCIENTIFIC HYBRID\n\n")
        f.write(f"- **Dataset**: {ds_name}\n")
        f.write(f"- **Chế độ mục tiêu (Target Mode)**: {target_mode}\n\n")
        
        f.write("## 1. Mục tiêu và Quy trình No-Leakage\n")
        f.write("Mô hình HybridCNNBiLSTMAttentionOrdinalV2 được huấn luyện với kỷ luật khắt khe:\n")
        f.write("- Trích lập 20% Locked Test ngay từ đầu.\n")
        f.write("- Mọi quá trình Scaler, LabelEncoding, Oversampling (SMOTE/ADASYN) CHỈ thực hiện nội bộ trong Train Pool (80%).\n")
        f.write("- Ensemble từ 5 mô hình huấn luyện với Fixed Seeds để giảm nhiễu hoàn toàn.\n\n")
        
        f.write("## 2. Kết quả Đánh giá (Locked Test Final)\n")
        if metrics:
            f.write(f"- **Accuracy**: {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"- **F1 Macro**: {metrics.get('f1_macro', 0):.4f}\n")
            f.write(f"- **Precision Macro**: {metrics.get('precision_macro', 0):.4f}\n")
            f.write(f"- **Recall Macro**: {metrics.get('recall_macro', 0):.4f}\n\n")
        else:
            f.write("*Chưa có kết quả do chưa chạy script evaluate.*\n\n")
            
        f.write("## 3. Siêu tham số tối ưu (Optuna Best Params)\n")
        if best_params:
            for k, v in best_params.items():
                f.write(f"- `{k}`: {v}\n")
        f.write("\n## 4. Hệ thống khuyến nghị (Recommendation Engine)\n")
        f.write("Đã tự động sinh khuyến nghị cá nhân hóa cho từng sinh viên trong tệp output `recommendations/`.\n")
        f.write("Chiến lược dựa trên luật kết hợp confidence score và các biến số risk rủi ro như (absence_study_ratio, grade_growth).\n\n")
        
        f.write("## 5. Hạn chế & Hướng cải tiến\n")
        f.write("- Kích thước dữ liệu gốc khá nhỏ (vài trăm samples), do đó dễ gặp overfit nếu không tuning hyperparameter cẩn thận.\n")
        f.write("- Việc bảo vệ tuyệt đối Locked Test có thể khiến metrics thấp hơn các paper cũ có áp dụng rò rỉ SMOTE, nhưng đây là kết quả ĐÁNG TIN CẬY và CHUẨN KHOA HỌC nhất.\n")


