# Handoff Report - Recommendation Model & Document Generation Plan

## 1. Observation
- **Original Document Generation Script (`generate_doc.py`)**: Located at `c:\Huflit\kltn\generate_doc.py`.
  - Line 181: `doc.save(r"C:\Huflit\kltn\Bao_cao_tien_do.docx")` saves the report as `Bao_cao_tien_do.docx`.
  - Lines 149-150: The recommendation section contains only a brief placeholder paragraph with no technical details of the PyTorch MLP model:
    ```python
    doc.add_paragraph("\n3.5. Đầu ra và Khuyến nghị (Output & Recommendations)")
    doc.add_paragraph("Sau khi huấn luyện, mô hình có khả năng dự đoán điểm số và nguy cơ rớt môn của sinh viên. Từ đó, hệ thống đưa ra các cảnh báo sớm và đề xuất các phương án hỗ trợ học tập phù hợp để cải thiện kết quả.")
    ```
  - Lines 155-164: Chapter 4 is generated as a single text block using `add_chapter` without dynamic table injection or detailed evaluation metrics:
    ```python
    add_chapter("CHƯƠNG 4. KẾT QUẢ VÀ THẢO LUẬN", 
        "4.1. Môi trường thực nghiệm\n" ... "4.3. Kết quả đánh giá mô hình\n" ... )
    ```
- **Recommendation Model Source Code (`src/recommendation.py`)**: Located at `c:\Huflit\kltn\src\recommendation.py`.
  - Architecture defined in class `RecommendationMLP` (lines 138-148):
    ```python
    class RecommendationMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim),
            )
    ```
  - Input dimension depends on the dataset: 8 features for `student` datasets (absences, studytime, failures, G1, G2, Dalc, Walc, goout) and 7 features for `xapi` (raisedhands, VisITedResources, AnnouncementsView, Discussion, StudentAbsenceDays, ParentAnsweringSurvey, ParentschoolSatisfaction) (lines 65-96).
  - Target dimension is 6 (representing 6 risk factors) (lines 99-135).
  - Loss function is `BCEWithLogitsLoss` using a calculated class positive weight `pos_weight` clamped between 0.5 and 10.0 (lines 187-193).
  - Training config: Adam optimizer (learning rate 0.003, weight decay 1e-4), 80/20 train/validation split, early stopping patience of 60 epochs (lines 191-215).
  - Learning path assembly: Done via `MLPLearningPathEngine` which maps the MLP-ranked risk probabilities (Sigmoid-activated logits) exceeding a threshold of 0.5 (or fallback to the maximum risk if none exceed 0.5 for Low/Medium classes) to 4-week staged interventions (lines 356-385).
- **Recommendation Evaluation JSON Files**:
  - Located under `c:\Huflit\kltn\reports\final\recommendations`.
  - Contains multilabel metrics (e.g. macro precision, recall, F1, hamming loss), ranking metrics (Precision@K, Recall@K, NDCG@K for K=1,3,5), structural quality metrics, LLM-Judge scores/status/reasons, and model training metadata.
  - Verbatim values observed in `student_mat_evaluation.json`:
    - Precision@1: `0.8607594936708861`, Recall@1: `0.6036764705882354`, NDCG@1: `1.0`
    - Precision@3: `0.561181434599156`, Recall@3: `0.9397058823529412`, NDCG@3: `0.9875956891190583`
    - Precision@5: `0.37974683544303794`, Recall@5: `1.0`, NDCG@5: `0.994036819313503`
    - LLM-Judge status: `not_run`, score: `null`, reason: `No external LLM annotations or validated human rating set was supplied.`

## 2. Logic Chain
1. The thesis report must accurately reflect the final technical implementation details of the recommendation system. Therefore, the brief placeholder in `3.5` of `generate_doc.py` needs to be replaced with a detailed section explaining the PyTorch MLP model architecture, its multi-label prediction task, the specific 7/8 input dimensions, the layer configuration (Linear-64-ReLU-Dropout(10%)-Linear-32-ReLU-Linear-6), the `BCEWithLogitsLoss` loss function with dynamic class weighting, and the MLP-ranked 4-week intervention mapping logic.
2. The report must present the verified evaluation results. The metrics (Precision@K, Recall@K, NDCG@K for K=1,3,5) and the LLM-Judge evaluation metrics (status, score, reason) should be parsed dynamically from the evaluation JSON files.
3. Using `python-docx`'s table generation APIs, we can programmatically build:
   - **Bảng 4.1**: Displaying Precision@K, Recall@K, and NDCG@K for K=1,3,5 across the three datasets (`student-mat`, `student-por`, `xapi`).
   - **Bảng 4.2**: Displaying the LLM-Judge evaluation status, score, and reasons across the three datasets.
4. To meet the user's constraints:
   - The output save path must be changed from `Bao_cao_tien_do.docx` to `Bao_cao_cuoi_cung.docx`.
   - The word "Rule-based" must be avoided in the text (referring to the system as "PyTorch MLP recommendation model" or "MLP-based risk-ranking and learning path recommendation").
   - There must be no mention of resampling algorithm fixes (like SMOTE or ADASYN) in this recommendation module report section.
5. A proposed script (`proposed_generate_doc.py`) was written in the agent folder to implement this plan, and it was successfully executed to verify that the tables are constructed correctly and the file saves to `C:\Huflit\kltn\Bao_cao_cuoi_cung.docx`.

## 3. Caveats
- The LLM-Judge metrics in the JSON files currently have a status of `"not_run"` and score of `null` with a reason indicating no annotations were supplied. The proposed code structure formats these values correctly but they will display as "Chưa thực hiện" (Not run) and "N/A" with the reason in Vietnamese.
- We assume the evaluation JSON files are statically stored at `C:\Huflit\kltn\reports\final\recommendations` and will not change paths.

## 4. Conclusion
The document generator in `generate_doc.py` can be updated with the code structure provided in `proposed_generate_doc.py` to:
1. Save the thesis report as `Bao_cao_cuoi_cung.docx`.
2. Add a comprehensive mathematical and architectural description of the PyTorch MLP recommendation engine.
3. Automatically load and render evaluation metrics (Precision@K, Recall@K, NDCG@K) and LLM-Judge evaluations into formatted Word tables.
4. Fully conform to the vocabulary guidelines (no "Rule-based" or "SMOTE/ADASYN" mentions).

## 5. Verification Method
- Execute the proposed script to ensure no exceptions are raised:
  ```powershell
  python c:\Huflit\kltn\.agents\teamwork_preview_explorer_report_update_1\proposed_generate_doc.py
  ```
- Inspect that the file `C:\Huflit\kltn\Bao_cao_cuoi_cung.docx` is created successfully.
- Open the document to verify that the table structures (Bảng 4.1 and Bảng 4.2) and the PyTorch MLP description match the design perfectly.

---

### Code Diff & Proposed Implementation
The implementation details have been written to `c:\Huflit\kltn\.agents\teamwork_preview_explorer_report_update_1\proposed_generate_doc.py`.

The key structural changes made to `generate_doc.py` are:
1. **JSON Loading & Helper Function**:
   ```python
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
   ```
2. **Replacing Section 3.5 Description**:
   Replaces the generic text in 3.5 with the precise description of the PyTorch MLP's structure: input dimensions, hidden layer layout, BCEWithLogitsLoss with positive weights, Adam optimizer, early stopping, and MLP-ranked learning path generation.
3. **Restructuring Chapter 4**:
   Replaces the monolithic `add_chapter("CHƯƠNG 4. KẾT QUẢ VÀ THẢO LUẬN", ...)` with step-by-step additions of sections 4.1 to 4.5, injecting Table 4.1 (Fidelity metrics Precision/Recall/NDCG@K) and Table 4.2 (LLM-Judge evaluation status/scores/reasons) dynamically formatted with Times New Roman and correct cell alignments.
4. **Modifying Save Location**:
   ```python
   doc.save(r"C:\Huflit\kltn\Bao_cao_cuoi_cung.docx")
   ```
