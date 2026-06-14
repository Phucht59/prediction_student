"""Build the final notebook that regenerates every figure used by the report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "tao_toan_bo_hinh_anh_bao_cao.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown(
            "# Tạo bộ hình cuối cho báo cáo luận văn\n\n"
            "Notebook này chỉ đọc dữ liệu thực trong `reports/final`: metric locked test, "
            "Optuna best CV, prediction, permutation importance, learning path và đánh giá MLP khuyến nghị. "
            "Không sinh số ngẫu nhiên và không dùng số liệu viết tay.\n"
        ),
        code(
            "from pathlib import Path\n"
            "import json, sys\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT))\n"
            "REPORTS = ROOT / 'reports' / 'final'\n"
            "FIGURES = REPORTS / 'figures' / 'current'\n"
            "DATASETS = ('student-mat', 'student-por', 'xapi')\n"
            "print('Project root:', ROOT)\n"
        ),
        markdown("## 1. Kiểm tra nguồn số liệu\n"),
        code(
            "summary = []\n"
            "for ds in DATASETS:\n"
            "    metrics = json.loads((REPORTS/'metrics'/f'{ds}_3class_locked_test_metrics.json').read_text(encoding='utf-8'))\n"
            "    cv = json.loads((REPORTS/'metrics'/f'{ds}_3class_optuna_cv.json').read_text(encoding='utf-8'))\n"
            "    rec = json.loads((REPORTS/'recommendations'/f\"{ds.replace('-', '_')}_evaluation.json\").read_text(encoding='utf-8'))\n"
            "    pred = pd.read_csv(REPORTS/'predictions'/f'{ds}_3class_predictions.csv')\n"
            "    summary.append({'dataset': ds, 'test_rows': len(pred), 'accuracy': metrics['Accuracy'], 'f1_macro': metrics['F1-Macro'], 'optuna_best_cv_f1': cv['f1_macro_best'], 'rec_ndcg_at_3': rec['ranking']['ndcg_at_3']})\n"
            "display(pd.DataFrame(summary))\n"
        ),
        markdown(
            "## 2. Sinh toàn bộ hình\n\n"
            "Mã vẽ được lưu tại `scripts/generate_final_figures.py` để có thể chạy từ notebook hoặc dòng lệnh. "
            "Các hình dùng phong cách đơn giản và được xuất trực tiếp từ artifact cuối.\n"
        ),
        code(
            "from scripts.generate_final_figures import main as generate_all_figures\n"
            "generate_all_figures()\n"
            "print('Đã tạo hình tại:', FIGURES)\n"
        ),
        markdown("## 3. Danh mục và xem trước hình\n"),
        code(
            "figure_files = sorted(FIGURES.glob('*.png'))\n"
            "manifest = pd.DataFrame({'file': [p.name for p in figure_files], 'bytes': [p.stat().st_size for p in figure_files]})\n"
            "display(manifest)\n"
            "for path in figure_files:\n"
            "    print(path.name)\n"
            "    display(Image(filename=str(path), width=760))\n"
        ),
        markdown(
            "## 4. Nguyên tắc diễn giải\n\n"
            "- Metric phân loại lấy từ locked test sau khi resampling hỗn hợp được xử lý bằng SMOTENC.\n"
            "- Cột CV là Optuna best CV của đúng phiên bản final model v1; locked test được báo cáo riêng.\n"
            "- Precision@K, Recall@K và NDCG@K của khuyến nghị đo độ trung thành với bộ tiêu chí weak supervision.\n"
            "- LLM-Judge chưa thực hiện vì chưa có bộ chấm độc lập; không tự tạo điểm thay thế.\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (kltn)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
