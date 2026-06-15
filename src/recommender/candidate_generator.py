import pandas as pd
from typing import Any

class CandidateGenerator:
    def __init__(self, catalog_df: pd.DataFrame):
        self.catalog_df = catalog_df

    def generate_candidates(
        self,
        diagnosed_risks: dict[str, float],
        predicted_class: int
    ) -> pd.DataFrame:
        """
        Lọc các catalog items phù hợp dựa trên rủi ro chẩn đoán (>= 0.3)
        và lớp học lực dự đoán trước khi chuyển qua Scorer.
        """
        # Xác định rủi ro tích cực (prob >= 0.3)
        active_risks = [risk for risk, prob in diagnosed_risks.items() if prob >= 0.3]
        
        candidates = []
        for _, row in self.catalog_df.iterrows():
            target_risks_str = str(row.get("target_risks", ""))
            if pd.isna(row.get("target_risks")):
                target_risks_str = ""
            target_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]
            
            # Nếu không có target_risks, đây là can thiệp nâng cao hoặc chung.
            # Chỉ giới thiệu nếu học sinh học lực tốt (predicted_class == 2) hoặc không có rủi ro nghiêm trọng nào.
            if not target_risks:
                if predicted_class == 2 or not active_risks:
                    candidates.append(row)
                continue
                
            # Nếu can thiệp nhắm vào bất kỳ rủi ro tích cực nào của học sinh
            if any(r in active_risks for r in target_risks):
                candidates.append(row)
                continue
                
            # Nếu học sinh có học lực yếu (predicted_class == 0), giữ các can thiệp cốt lõi (như R1, R6)
            if predicted_class == 0 and any(r in ["R1_LOW_PRIOR_PERFORMANCE", "R6_HIGH_FAILURE_PROBABILITY"] for r in target_risks):
                candidates.append(row)
                continue
                
        # Fallback: Đảm bảo luôn có ít nhất 3 ứng viên để tránh thiếu hụt can thiệp.
        if len(candidates) < 3:
            return self.catalog_df.copy()
            
        return pd.DataFrame(candidates)
