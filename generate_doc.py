import os
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE

def add_cover(doc, is_sub_cover=False):
    # Cover text
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run("TRƯỜNG ĐẠI HỌC NGOẠI NGỮ-TIN HỌC TP. HỒ CHÍ MINH\nKHOA CNTT\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True
    
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run("_______________________________\n\n\n\n")
    run.font.name = 'Times New Roman'
    
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run("KHÓA LUẬN TỐT NGHIỆP\n\n\n\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(18)
    run.bold = True
    
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run("DỰ ĐOÁN KẾT QUẢ HỌC TẬP CỦA SINH VIÊN BẰNG CÁC MÔ HÌNH HỌC MÁY\n\n\n\n\n\n\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(24)
    run.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    p.paragraph_format.left_indent = Inches(2.5)
    
    run = p.add_run("GIẢNG VIÊN HƯỚNG DẪN: TS. Nguyễn Văn A\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True
    
    run = p.add_run("SINH VIÊN THỰC HIỆN: Nguyễn Thái H – 20DH11xxxx\n")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True

    for _ in range(7):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run("TP. HỒ CHÍ MINH – THÁNG 06 NĂM 2026")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True
    
    doc.add_page_break()

def create_report():
    doc = Document()
    
    # Page setup (A4, Margins: Top 3cm, Bottom 3cm, Left 3.5cm, Right 2cm)
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.left_margin = Cm(3.5)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(3.0)

    # Set normal style font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)

    # Cover Pages
    add_cover(doc, is_sub_cover=False)
    add_cover(doc, is_sub_cover=True)
    
    # Helper to add chapter and filler
    def add_chapter(title, content, page_breaks=1):
        p = doc.add_paragraph()
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(15)
        
        doc.add_paragraph(content)
        for _ in range(page_breaks):
            doc.add_page_break()

    add_chapter("LỜI CẢM ƠN", "[Nội dung lời cảm ơn...]", 1)
    add_chapter("LỜI CAM ĐOAN", "[Nội dung lời cam đoan...]", 1)
    add_chapter("MỤC LỤC", "[Nội dung mục lục...]", 1)
    add_chapter("DANH MỤC CÁC KÝ HIỆU, CHỮ VIẾT TẮT VÀ THUẬT NGỮ", "[Nội dung...]", 1)
    add_chapter("DANH MỤC CÁC BẢNG", "[Danh mục bảng...]", 1)
    add_chapter("DANH MỤC CÁC HÌNH VẼ, ĐỒ THỊ", "[Danh mục hình vẽ...]", 1)

    # Chapter 1
    add_chapter("CHƯƠNG 1. MỞ ĐẦU", 
        "1.1. Lý do chọn đề tài\n"
        "[Phần này trình bày lý do chọn đề tài dự đoán kết quả học tập...]\n\n"
        "1.2. Mục tiêu nghiên cứu\n"
        "[Mục tiêu nghiên cứu...]\n\n"
        "1.3. Đối tượng và phạm vi nghiên cứu\n"
        "[Đối tượng và phạm vi...]\n\n"
        "1.4. Ý nghĩa khoa học và thực tiễn\n"
        "[Ý nghĩa...]\n\n", 2)
        
    # Chapter 2
    add_chapter("CHƯƠNG 2. TỔNG QUAN", 
        "2.1. Tổng quan về các phương pháp học máy trong giáo dục\n"
        "[Nội dung tổng quan...]\n\n"
        "2.2. Các nghiên cứu liên quan\n"
        "[Các nghiên cứu trước đây...]\n\n", 2)
        
    # Chapter 3 - MÔ HÌNH
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run("CHƯƠNG 3. MÔ HÌNH DỰ ĐOÁN VÀ KHUYẾN NGHỊ LỘ TRÌNH HỌC TẬP")
    run.bold = True
    run.font.size = Pt(15)
    
    doc.add_paragraph("3.1. Kiến trúc tổng thể của hệ thống")
    doc.add_paragraph("Kiến trúc hệ thống dự đoán kết quả học tập của sinh viên được thiết kế với nhiều thành phần tích hợp chặt chẽ. Đầu tiên, hệ thống thu thập dữ liệu từ nhiều nguồn khác nhau, bao gồm thông tin nhân khẩu học (demographics), điểm số (grades) và các hoạt động của sinh viên trong quá trình học tập (activity). Dưới đây là hình ảnh tổng quan về kiến trúc của hệ thống:")
    
    # Add Image
    try:
        img_path = r"C:\Users\THPhu\.gemini\antigravity\brain\45358a05-baaa-4e75-92f1-9c5c6264125e\student_prediction_model_1781372236144.png"
        doc.add_picture(img_path, width=Inches(5.5))
        p = doc.add_paragraph()
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        p.add_run("Hình 3.1: Kiến trúc mô hình dự đoán kết quả học tập").italic = True
    except Exception as e:
        doc.add_paragraph(f"[Lỗi chèn hình ảnh: {e}]")

    doc.add_paragraph("\n3.2. Tiền xử lý dữ liệu (Data Preprocessing)")
    doc.add_paragraph("Dữ liệu thu thập thường chưa chuẩn hóa, có thể chứa các giá trị thiếu hoặc bị nhiễu. Bước tiền xử lý dữ liệu sẽ giúp làm sạch và chuẩn bị dữ liệu tốt nhất cho việc huấn luyện mô hình. Các phương pháp như Imputation để điền khuyết, Normalization/Standardization để chuẩn hóa thang đo được áp dụng.")

    doc.add_paragraph("\n3.3. Trích xuất đặc trưng (Feature Engineering)")
    doc.add_paragraph("Việc xây dựng các đặc trưng mới từ dữ liệu thô đóng vai trò quan trọng trong việc cải thiện hiệu suất của mô hình học máy. Ví dụ, tỷ lệ vắng mặt hoặc xu hướng thay đổi điểm số qua các kỳ học.")

    doc.add_paragraph("\n3.4. Mô hình Học máy (Machine Learning Model)")
    doc.add_paragraph("Nhiều mô hình học máy đã được thử nghiệm và đánh giá. Trong đó, các thuật toán như Random Forest, Gradient Boosting, hoặc Neural Networks được đề xuất nhờ khả năng xử lý tốt các tập dữ liệu đa dạng. Chúng tôi thiết lập việc tinh chỉnh siêu tham số (Hyperparameter Tuning) để tối ưu độ chính xác của các mô hình này.")

    doc.add_paragraph("\n3.5. Mô hình khuyến nghị lộ trình học tập PyTorch MLP")
    doc.add_paragraph("Hệ thống đề xuất lộ trình học tập cá nhân hóa sử dụng một mô hình Mạng nơ-ron truyền thẳng (Multi-Layer Perceptron - MLP) được xây dựng trên nền tảng PyTorch. Mô hình này nhận đầu vào là các đặc trưng bối cảnh của sinh viên và dự đoán đồng thời sáu yếu tố rủi ro học tập thông qua bài toán phân loại đa nhãn (Multi-label Classification).")
    
    doc.add_paragraph("Kiến trúc của mô hình MLP bao gồm:\n"
                      "- Tầng đầu vào (Input Layer): Nhận véc-tơ đặc trưng gồm 8 chiều (đối với tập dữ liệu Student-Mat và Student-Por) hoặc 7 chiều (đối với tập dữ liệu xAPI).\n"
                      "- Tầng ẩn thứ nhất: Tầng tuyến tính (Linear layer) chuyển đổi từ số đặc trưng đầu vào thành 64 nút ẩn, sử dụng hàm kích hoạt ReLU và kỹ thuật Dropout với tỷ lệ 10% nhằm giảm hiện tượng quá khớp (overfitting).\n"
                      "- Tầng ẩn thứ hai: Tầng tuyến tính chuyển đổi từ 64 nút ẩn sang 32 nút ẩn, sử dụng hàm kích hoạt ReLU.\n"
                      "- Tầng đầu ra (Output Layer): Tầng tuyến tính chuyển đổi từ 32 nút ẩn thành 6 logit tương ứng với 6 yếu tố rủi ro học tập cần dự báo.")
                      
    doc.add_paragraph("Quy trình huấn luyện và tối ưu hóa:\n"
                      "- Hàm mất mát: Sử dụng hàm BCEWithLogitsLoss (Binary Cross Entropy with Logits Loss) kết hợp với trọng số dương (positive weight) được tính toán động dựa trên phân phối nhãn trong tập huấn luyện để giải quyết mất cân bằng lớp.\n"
                      "- Bộ tối ưu hóa: Thuật toán Adam với tỷ lệ học tập (learning rate) là 0,003 và hệ số phạt trọng số (weight decay) là 1e-4.\n"
                      "- Chiến lược huấn luyện: Dữ liệu được chia theo tỷ lệ 80% huấn luyện và 20% đánh giá (validation). Mô hình dừng sớm (early stopping) nếu tổn thất trên tập đánh giá không cải thiện sau 60 epoch liên tiếp.")
                      
    doc.add_paragraph("Xếp hạng rủi ro và xây dựng lộ trình:\n"
                      "Sau khi có các xác suất rủi ro đầu ra từ mô hình MLP (thông qua hàm Sigmoid), hệ thống thực hiện xếp hạng rủi ro và sinh lộ trình can thiệp học tập theo từng giai đoạn tuần tự. Các yếu tố rủi ro có xác suất lớn hơn hoặc bằng 0,5 sẽ được kích hoạt. Trường hợp không có rủi ro nào đạt ngưỡng 0,5 nhưng sinh viên được dự đoán ở nhóm học lực thấp (Low) hoặc trung bình (Medium), yếu tố rủi ro có xác suất cao nhất sẽ được chọn. Từ danh sách rủi ro đã được MLP xếp hạng theo độ ưu tiên, hệ thống sinh ra một lộ trình hành động có cấu trúc kéo dài trong vòng 4 tuần để hỗ trợ kịp thời cho sinh viên.")

    doc.add_page_break()

    # Chapter 4
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run("CHƯƠNG 4. KẾT QUẢ VÀ THẢO LUẬN")
    run.bold = True
    run.font.size = Pt(15)
    
    doc.add_paragraph("4.1. Môi trường thực nghiệm\n[Trình bày về cấu hình phần cứng, phần mềm...]\n")
    doc.add_paragraph("4.2. Bộ dữ liệu thử nghiệm\n[Mô tả về bộ dữ liệu...]\n")
    doc.add_paragraph("4.3. Kết quả đánh giá mô hình học lực\n[Các bảng và biểu đồ kết quả dự đoán học lực...]\n")
    
    doc.add_paragraph("4.4. Kết quả đánh giá mô hình khuyến nghị lộ trình học tập")
    doc.add_paragraph("Hiệu năng của mô hình khuyến nghị lộ trình học tập dựa trên MLP được đánh giá thông qua các độ đo Precision@K, Recall@K và NDCG@K trên tập kiểm thử độc lập (locked test set). Các kết quả này phản ánh mức độ khớp (fidelity) của mô hình so với bộ tiêu chí chuyên môn được sử dụng làm giám sát yếu (weak supervision). Ngoài ra, bảng đánh giá cũng ghi nhận trạng thái và phản hồi từ LLM-Judge đối với tính hợp lệ của lộ trình khuyến nghị.")
    
    # Load evaluation JSON files
    recommendations_dir = Path("C:/Huflit/kltn/reports/final/recommendations")
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
                
    def format_decimal(val, digits=4):
        if val is None:
            return "N/A"
        return f"{val:.{digits}f}".replace(".", ",")
        
    # Add table 4.1: Ranking metrics
    p_title = doc.add_paragraph()
    p_title.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    run_title = p_title.add_run("Bảng 4.1: Kết quả đánh giá độ trung thành (Fidelity) của mô hình khuyến nghị")
    run_title.bold = True
    
    headers = ["Bộ dữ liệu (Dataset)", "K", "Độ chính xác (Precision@K)", "Độ phủ (Recall@K)", "Điểm NDCG (NDCG@K)"]
    table = doc.add_table(rows=1, cols=5)
    table.style = 'Table Grid'
    
    # Header formatting
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        for paragraph in hdr_cells[i].paragraphs:
            paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            for run in paragraph.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(11)
                run.bold = True
                
    datasets_display = {
        "student-mat": "Student-Mat",
        "student-por": "Student-Por",
        "xapi": "xAPI"
    }
    
    for dataset_key in ["student-mat", "student-por", "xapi"]:
        data = eval_data.get(dataset_key)
        if not data:
            continue
        display_name = datasets_display[dataset_key]
        ranking = data.get("ranking", {})
        
        for idx, k in enumerate([1, 3, 5]):
            row_cells = table.add_row().cells
            row_cells[0].text = display_name if idx == 0 else ""
            row_cells[1].text = str(k)
            
            p_val = ranking.get(f"precision_at_{k}")
            r_val = ranking.get(f"recall_at_{k}")
            n_val = ranking.get(f"ndcg_at_{k}")
            
            row_cells[2].text = format_decimal(p_val)
            row_cells[3].text = format_decimal(r_val)
            row_cells[4].text = format_decimal(n_val)
            
            for col_idx, cell in enumerate(row_cells):
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if col_idx == 0 else WD_PARAGRAPH_ALIGNMENT.CENTER
                    for run in paragraph.runs:
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(11)
                        
    doc.add_paragraph() # Spacing
    
    # Add table 4.2: LLM Judge metrics
    p_title2 = doc.add_paragraph()
    p_title2.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    run_title2 = p_title2.add_run("Bảng 4.2: Kết quả đánh giá bằng LLM-Judge đối với mô hình khuyến nghị")
    run_title2.bold = True
    
    headers2 = ["Bộ dữ liệu (Dataset)", "Trạng thái đánh giá", "Điểm số LLM", "Lý do / Mô tả chi tiết"]
    table2 = doc.add_table(rows=1, cols=4)
    table2.style = 'Table Grid'
    
    hdr_cells2 = table2.rows[0].cells
    for i, header in enumerate(headers2):
        hdr_cells2[i].text = header
        for paragraph in hdr_cells2[i].paragraphs:
            paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            for run in paragraph.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(11)
                run.bold = True
                
    for dataset_key in ["student-mat", "student-por", "xapi"]:
        data = eval_data.get(dataset_key)
        if not data:
            continue
        display_name = datasets_display[dataset_key]
        judge = data.get("llm_judge", {})
        
        row_cells = table2.add_row().cells
        row_cells[0].text = display_name
        
        status = judge.get("status", "N/A")
        status_vi = "Chưa thực hiện" if status == "not_run" else status
        row_cells[1].text = status_vi
        
        score = judge.get("score")
        row_cells[2].text = format_decimal(score) if score is not None else "N/A"
        
        reason = judge.get("reason", "")
        reason_vi = "Không có dữ liệu đánh giá từ LLM bên ngoài hoặc tập gán nhãn thủ công." if "No external LLM annotations" in reason else reason
        row_cells[3].text = reason_vi
        
        for col_idx, cell in enumerate(row_cells):
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if col_idx in [0, 3] else WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(11)

    doc.add_paragraph() # Spacing
    doc.add_paragraph("4.5. Thảo luận\n[Phân tích và thảo luận kết quả nghiên cứu...]\n")
    doc.add_page_break()

    # Chapter 5
    add_chapter("CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", 
        "5.1. Kết luận\n"
        "[Tóm tắt những kết quả đạt được...]\n\n"
        "5.2. Hướng phát triển\n"
        "[Các định hướng mở rộng đề tài trong tương lai...]\n\n", 1)

    # References
    add_chapter("TÀI LIỆU THAM KHẢO", 
        "[1] Nguyễn Văn A, Tên sách, Nhà xuất bản, Năm.\n"
        "[2] Tên Tác Giả, \"Tên bài báo\", Tên Tạp chí, Số, Trang, Năm.\n", 1)

    # Appendix
    add_chapter("PHỤ LỤC", "[Các bảng phụ lục, mã nguồn...]", 1)

    doc.save(r"C:\Huflit\kltn\Bao_cao_cuoi_cung.docx")
    print("Done")

if __name__ == '__main__':
    create_report()
