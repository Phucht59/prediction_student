import docx
import sys
from docx.shared import Pt, Cm, Inches

sys.stdout.reconfigure(encoding='utf-8')

def inspect_doc(path):
    print(f"\n=================== INSPECTING: {path} ===================")
    doc = docx.Document(path)
    
    # 1. Sections and Margins
    for i, sec in enumerate(doc.sections):
        print(f"Section {i}:")
        print(f"  Page Width: {sec.page_width.cm if sec.page_width else 'N/A'} cm")
        print(f"  Page Height: {sec.page_height.cm if sec.page_height else 'N/A'} cm")
        print(f"  Margins: Top={sec.top_margin.cm if sec.top_margin else 'N/A'} cm, "
              f"Bottom={sec.bottom_margin.cm if sec.bottom_margin else 'N/A'} cm, "
              f"Left={sec.left_margin.cm if sec.left_margin else 'N/A'} cm, "
              f"Right={sec.right_margin.cm if sec.right_margin else 'N/A'} cm")

    # 2. Normal Style
    if 'Normal' in doc.styles:
        normal = doc.styles['Normal']
        print("Normal Style:")
        print(f"  Font Name: {normal.font.name}")
        print(f"  Font Size: {normal.font.size.pt if normal.font.size else 'N/A'} pt")
        print(f"  Line Spacing: {normal.paragraph_format.line_spacing if normal.paragraph_format else 'N/A'}")
        print(f"  Space After: {normal.paragraph_format.space_after.pt if normal.paragraph_format.space_after else 'N/A'} pt")
        print(f"  Space Before: {normal.paragraph_format.space_before.pt if normal.paragraph_format.space_before else 'N/A'} pt")
    else:
        print("Normal style not found")

    # 3. Analyze cover pages (first 20 paragraphs)
    print("\nCover/First 20 paragraphs:")
    for idx, p in enumerate(doc.paragraphs[:20]):
        text = p.text.strip().replace('\n', ' ')
        if not text:
            continue
        print(f"  Paragraph {idx} (align={p.alignment}): text='{text[:60]}...'")
        for r_idx, r in enumerate(p.runs):
            font_name = r.font.name
            font_size = r.font.size.pt if r.font.size else 'N/A'
            bold = r.bold
            print(f"    Run {r_idx}: text='{r.text.strip().replace(chr(10), ' ')[:30]}...', font={font_name}, size={font_size}, bold={bold}")

inspect_doc("scratch/HUFLIT_Mau_Hinh_Thuc_KLTN.docx")
inspect_doc("Bao_cao_cuoi_cung.docx")
