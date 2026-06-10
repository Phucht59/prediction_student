import json
from pathlib import Path

def write_json(path: Path, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def write_markdown_report(ablation_results, final_metrics, target_mode, out_path: Path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Báo cáo V26 Scientific Hybrid\n\n")
        f.write(f"Chế độ mục tiêu (Target Mode): {target_mode}\n\n")
        
        f.write("## 1. Kết quả Ablation (Validation F1)\n\n")
        cols = ablation_results.columns
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in ablation_results.iterrows():
            f.write("| " + " | ".join([str(x) for x in row.values]) + " |\n")
        f.write("\n\n")
        
        f.write("## 2. Kết quả Ensemble Final Test\n\n")
        for k, v in final_metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")
