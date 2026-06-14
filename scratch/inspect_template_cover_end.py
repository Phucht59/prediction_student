import docx
import sys

sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document("scratch/HUFLIT_Mau_Hinh_Thuc_KLTN.docx")
print("=== Analyzing paragraphs around the bottom of the cover in template ===")
for idx, p in enumerate(doc.paragraphs[30:70]):
    text = p.text.strip().replace('\n', ' ')
    if not text:
        continue
    if "hồ chí minh" in text.lower() or "năm" in text.lower():
        print(f"P {idx+30} (align={p.alignment}): '{text}'")
        for r_idx, r in enumerate(p.runs):
            print(f"  Run {r_idx}: font={r.font.name}, size={r.font.size.pt if r.font.size else 'None'}, bold={r.bold}, text='{r.text.strip().replace(chr(10), ' ')[:30]}...'")
