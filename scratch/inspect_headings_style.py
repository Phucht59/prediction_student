import docx
import sys

sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document("Bao_cao_cuoi_cung.docx")
print("=== Auditing Paragraphs in Chapter 3 and 4 ===")
for idx, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if not text:
        continue
    # We look for lines starting with digit + dot + digit (like 3.1, 4.2)
    words = text.split()
    if len(words) > 0 and (words[0][0].isdigit() if words[0] else False) and '.' in words[0]:
        print(f"P {idx} (style={p.style.name}, align={p.alignment}): '{text[:60]}...'")
        for r_idx, r in enumerate(p.runs):
            print(f"  Run {r_idx}: font={r.font.name}, size={r.font.size.pt if r.font.size else 'None'}, bold={r.bold}, text='{r.text.strip()[:30]}...'")
