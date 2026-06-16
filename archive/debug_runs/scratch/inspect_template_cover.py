import docx
import sys

sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document("scratch/HUFLIT_Mau_Hinh_Thuc_KLTN.docx")
print("=== Analyzing first 50 paragraphs of template ===")
for idx, p in enumerate(doc.paragraphs[:50]):
    text = p.text.strip().replace('\n', ' ')
    if not text:
        continue
    # Let's print formatting details
    p_format = p.paragraph_format
    align = p.alignment
    left_indent = p_format.left_indent.inches if p_format.left_indent else 'None'
    print(f"P {idx} (align={align}, indent={left_indent}): '{text[:60]}...'")
    for r_idx, r in enumerate(p.runs):
        print(f"  Run {r_idx}: font={r.font.name}, size={r.font.size.pt if r.font.size else 'None'}, bold={r.bold}, text='{r.text.strip().replace(chr(10), ' ')[:30]}...'")
