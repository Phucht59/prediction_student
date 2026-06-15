def generate_friendly_explanation(rec: dict, diagnosed_risks: dict, dataset_kind: str) -> str:
    """
    Sinh diễn giải thân thiện giải thích lý do lựa chọn can thiệp cho học sinh.
    """
    name = rec.get("intervention_name", "Biện pháp hỗ trợ")
    risk_match = rec.get("risk_match", 0.0)
    perf_need = rec.get("performance_need", 0.0)
    diff_fit = rec.get("difficulty_fit", 0.0)
    time_fit = rec.get("time_fit", 0.0)
    
    reasons = []
    # Check if there is a risk matched
    if risk_match > 0.3:
        # Find which risk had high probability
        highest_risk = None
        highest_prob = -1.0
        for r, prob in diagnosed_risks.items():
            if prob > highest_prob:
                highest_prob = prob
                highest_risk = r
        if highest_risk:
            # Map risk code to Vietnamese name
            risk_names = {
                "R1_LOW_PRIOR_PERFORMANCE": "học lực đầu vào chưa tốt",
                "R2_DECLINING_TREND": "xu hướng điểm số đang giảm",
                "R3_ATTENDANCE_RISK": "rủi ro nghỉ học nhiều",
                "R4_LOW_ENGAGEMENT": "mức độ tương tác lớp học thấp",
                "R5_INSUFFICIENT_STUDY_TIME": "thời gian tự học chưa đủ",
                "R6_HIGH_FAILURE_PROBABILITY": "rủi ro không đạt kết quả môn học",
            }
            friendly_risk = risk_names.get(highest_risk, "rủi ro học tập được phát hiện")
            reasons.append(f"hỗ trợ khắc phục tình trạng {friendly_risk}")
            
    if perf_need > 0.5:
        reasons.append("đáp ứng nhu cầu cải thiện kết quả học tập hiện tại")
    if diff_fit > 0.7:
        reasons.append("vừa sức với năng lực hiện tại của bạn")
    if time_fit > 0.7:
        reasons.append("phù hợp với thời gian học tập hàng tuần của bạn")
        
    if not reasons:
        reasons.append("giúp nâng cao hiệu quả và duy trì phong độ học tập tốt")
        
    explanation = f"Đề xuất '{name}' được lựa chọn vì nó " + ", ".join(reasons) + "."
    return explanation
