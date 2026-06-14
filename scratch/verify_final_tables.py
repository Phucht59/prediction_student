import docx
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Huflit/kltn")
REPORTS_DIR = ROOT / "reports" / "final" / "recommendations"
DOCX_PATH = ROOT / "reports" / "final" / "LUAN_VAN_HOAN_CHINH_FINAL.docx"

def format_decimal(val, digits=3):
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}".replace(".", ",")

def main():
    print("=== Loading Source JSON Files ===")
    datasets = ["student-mat", "student-por", "xapi"]
    json_data = {}
    for ds in datasets:
        filename = f"{ds.replace('-', '_')}_evaluation.json"
        path = REPORTS_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            json_data[ds] = json.load(f)
            
    print("\n=== Opening LUAN_VAN_HOAN_CHINH_FINAL.docx ===")
    doc = docx.Document(DOCX_PATH)
    
    # We know from the inspection that Table 12 contains the recommendation metrics
    t = doc.tables[12]
    print(f"Table 12 style: {t.style.name}")
    
    headers = [cell.text.strip() for cell in t.rows[0].cells]
    print(f"Table 12 headers: {headers}")
    expected_headers = ["Dataset", "F1 đa nhãn", "P@3", "R@3", "NDCG@3", "LLM-Judge"]
    assert headers == expected_headers, f"Headers mismatch: {headers} vs {expected_headers}"
    
    dataset_keys = ["student-mat", "student-por", "xapi"]
    dataset_row_labels = ["Mat", "Por", "xAPI"]
    
    for row_idx, (ds_key, ds_label) in enumerate(zip(dataset_keys, dataset_row_labels), start=1):
        cells = t.rows[row_idx].cells
        cell_texts = [c.text.strip() for c in cells]
        print(f"Row {row_idx}: {cell_texts}")
        
        # Load JSON metrics
        f1_macro = json_data[ds_key]["multilabel"]["f1_macro"]
        p_at_3 = json_data[ds_key]["ranking"]["precision_at_3"]
        r_at_3 = json_data[ds_key]["ranking"]["recall_at_3"]
        ndcg_at_3 = json_data[ds_key]["ranking"]["ndcg_at_3"]
        llm_status = json_data[ds_key]["llm_judge"]["status"]
        
        expected_f1 = format_decimal(f1_macro)
        expected_p3 = format_decimal(p_at_3)
        expected_r3 = format_decimal(r_at_3)
        expected_n3 = format_decimal(ndcg_at_3)
        expected_llm = "Chưa chạy" if llm_status == "not_run" else llm_status
        
        expected_row = [ds_label, expected_f1, expected_p3, expected_r3, expected_n3, expected_llm]
        print(f"  Expected: {expected_row}")
        
        for col_idx in range(6):
            assert cell_texts[col_idx] == expected_row[col_idx], f"Mismatch at row {row_idx} col {col_idx}. Expected: '{expected_row[col_idx]}', Got: '{cell_texts[col_idx]}'"
            
    print("\nTable 12 verified successfully!")

if __name__ == "__main__":
    main()
