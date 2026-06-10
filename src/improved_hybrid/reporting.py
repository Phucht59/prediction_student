from pathlib import Path
import pandas as pd
from datetime import datetime
from .config import RESULTS_DIR

def create_markdown_report(metrics_dict, spec, output_path: Path):
    lines = []
    lines.append(f"# Báo Cáo Kết Quả Mô Hình HybridCNNBiLSTMAttentionOrdinal")
    lines.append(f"**Dataset:** {spec.display_name}")
    lines.append(f"**Thời gian tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## 1. Kết quả Final Locked Test")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    if 'final_test' in metrics_dict:
        for k, v in metrics_dict['final_test'].items():
            lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    
    lines.append("## 2. Kết quả Repeated Stratified CV (Mean ± Std)")
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---|---|")
    if 'cv' in metrics_dict:
        for k, vals in metrics_dict['cv'].items():
            mean = sum(vals) / len(vals)
            std = pd.Series(vals).std()
            lines.append(f"| {k} | {mean:.4f} | {std:.4f} |")
    lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
