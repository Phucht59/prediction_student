import os
import json
import sys
from pathlib import Path
from docx import Document

# Set standard output to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def verify():
    docx_path = Path("c:/Huflit/kltn/Bao_cao_cuoi_cung.docx")
    if not docx_path.exists():
        print(f"Error: {docx_path} does not exist.")
        return
        
    print(f"File size of {docx_path}: {docx_path.stat().st_size} bytes")
    
    # Load JSON files
    recommendations_dir = Path("c:/Huflit/kltn/reports/final/recommendations")
    eval_files = {
        "student-mat": recommendations_dir / "student_mat_evaluation.json",
        "student-por": recommendations_dir / "student_por_evaluation.json",
        "xapi": recommendations_dir / "xapi_evaluation.json"
    }
    
    eval_data = {}
    for dataset_name, filepath in eval_files.items():
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                eval_data[dataset_name] = json.load(f)
        else:
            print(f"Warning: {filepath} does not exist.")
            
    doc = Document(docx_path)
    
    # Let's inspect the tables
    tables = doc.tables
    print(f"Number of tables found: {len(tables)}")
    
    for idx, table in enumerate(tables):
        print(f"\nTable {idx + 1}:")
        for r_idx, row in enumerate(table.rows):
            cells_text = [cell.text.strip() for cell in row.cells]
            print(f"  Row {r_idx}: {cells_text}")
            
    print(f"\nNumber of inline shapes (images): {len(doc.inline_shapes)}")
    for s_idx, shape in enumerate(doc.inline_shapes):
        print(f"  Shape {s_idx + 1}: Type: {shape.type}, Width: {shape.width}, Height: {shape.height}")
            
    print("\nVerification of table data:")
    # Table 1 should contain ranking metrics
    # Row 0: Header
    # Row 1: Student-Mat, 1, precision_at_1, recall_at_1, ndcg_at_1
    # ...
    
if __name__ == "__main__":
    verify()
