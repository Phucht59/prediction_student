def generate_recommendations(student_id, predicted_class, confidence, feature_dict, spec):
    recs = []
    reasons = []
    
    if spec.kind == 'student':
        # grade_delta
        if feature_dict.get('grade_delta', 0) < 0:
            reasons.append("Điểm G2 giảm so với G1")
            recs.append("Cần ôn tập lại các nội dung trong giai đoạn gần nhất, kiểm tra nguyên nhân sa sút.")
            
        # absences
        if feature_dict.get('absences', 0) >= 10:
            reasons.append("Số buổi vắng mặt cao")
            recs.append("Cần cải thiện chuyên cần, theo dõi lịch học và các deadline bài tập.")
            
        # failures
        if feature_dict.get('failures', 0) > 0:
            reasons.append("Đã từng rớt môn trước đây")
            recs.append("Nên bổ sung kế hoạch học lại kiến thức nền tảng bị hổng.")
            
        # studytime
        if feature_dict.get('studytime', 1) <= 1:
            reasons.append("Thời gian tự học thấp")
            recs.append("Cần tăng thêm thời lượng tự học cá nhân hàng tuần.")
            
        # alcohol/social
        if feature_dict.get('social_alcohol_risk', 0) > 6:
            reasons.append("Có dấu hiệu ảnh hưởng từ sinh hoạt cá nhân/hoạt động ngoài giờ")
            recs.append("Nên điều chỉnh lịch sinh hoạt cân bằng hơn, giảm thiểu các yếu tố gây xao nhãng.")
            
    elif spec.kind == 'xapi':
        # VisitedResources
        if feature_dict.get('VisitedResources', 0) < 30:
            reasons.append("Lượt xem tài nguyên học tập thấp")
            recs.append("Cần tăng cường truy cập và đọc tài liệu môn học thường xuyên hơn.")
            
        # raisedhands
        if feature_dict.get('raisedhands', 0) < 20:
            reasons.append("Ít giơ tay xây dựng bài")
            recs.append("Nên mạnh dạn tương tác và tham gia thảo luận trong lớp để hiểu bài sâu hơn.")
            
        # Discussion
        if feature_dict.get('Discussion', 0) < 20:
            reasons.append("Chưa tích cực tham gia các diễn đàn/nhóm thảo luận")
            recs.append("Hãy tham gia nhóm học tập và thảo luận để giải quyết các thắc mắc.")
            
        # AnnouncementsView
        if feature_dict.get('AnnouncementsView', 0) < 15:
            reasons.append("Ít theo dõi thông báo môn học")
            recs.append("Cần chủ động theo dõi các thông báo của giảng viên để không lỡ sự kiện quan trọng.")
            
        # absence_binary
        if feature_dict.get('absence_binary', 0) == 1:
            reasons.append("Số ngày vắng mặt nhiều (Trên 7 ngày)")
            recs.append("Cải thiện tính chuyên cần là ưu tiên hàng đầu lúc này.")
            
        # ParentAnsweringSurvey
        if feature_dict.get('ParentAnsweringSurvey', 'Yes') == 'No':
            reasons.append("Chưa có sự phối hợp tốt từ phía gia đình")
            recs.append("Nhà trường cần tăng cường liên lạc và phối hợp với phụ huynh.")
            
    # Risk Level
    risk_level = "Thấp"
    if spec.kind == 'student':
        if predicted_class in [0, 1]: # "0-4", "5-8"
            risk_level = "Cao"
        elif predicted_class == 2: # "9-12"
            risk_level = "Trung bình"
    else: # xAPI
        if predicted_class == 0: # Low
            risk_level = "Cao"
        elif predicted_class == 1: # Middle
            risk_level = "Trung bình"
            
    if confidence < 0.5:
        recs.append("Mô hình chưa hoàn toàn chắc chắn về dự đoán, cần cố vấn học tập xem xét hồ sơ thực tế.")
        
    return {
        "student_id": student_id,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "risk_level": risk_level,
        "main_reasons": reasons,
        "recommendations": recs
    }
