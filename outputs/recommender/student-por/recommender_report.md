# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: student-por

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9091
* **Macro F1**: 0.9443
* **Micro Precision**: 0.8969
* **Macro Precision**: 0.9257
* **Micro Recall**: 0.9216
* **Macro Recall**: 0.9667
* **Hamming Loss**: 0.0603

### Ranking Metrics (at K=3)
* **Precision@3**: 0.8103
* **Recall@3**: 0.5251
* **NDCG@3**: 0.8341
* **Catalog Coverage@3**: 1.0000

### Path Quality Metrics
* **Risk Coverage Rate**: 0.9554
* **Workload Balance (std hours/week)**: 1.6752
* **Difficulty Progression Rate**: 0.6128
* **Prerequisite Violation Rate**: 0.0179

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 6)
* **Student Context**: Absences: 2, Study Time: 4/4, Failures: 1, G1: 10, G2: 8
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.73, Medium: 0.27, High: 0.00]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 1.0000 probability
  - `R2_DECLINING_TREND`: 1.0000 probability
  - `R3_ATTENDANCE_RISK`: 0.0000 probability
  - `R4_LOW_ENGAGEMENT`: 0.9993 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 1.0000 probability

* **Top 3 Recommended Interventions**:
  1. **LMS Interactive Quizzing** (Score: 0.9283)
     * Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
     * *Score Breakdown*: Đề xuất 'LMS Interactive Quizzing' được lựa chọn vì nó hỗ trợ khắc phục tình trạng học lực đầu vào chưa tốt, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.72, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.85)
  2. **LMS Navigation Tutorial** (Score: 0.9233)
     * Interactive walkthrough of the digital learning system to track course updates and resources.
     * *Score Breakdown*: Đề xuất 'LMS Navigation Tutorial' được lựa chọn vì nó hỗ trợ khắc phục tình trạng học lực đầu vào chưa tốt, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.72, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.80)
  3. **Parent-Teacher Engagement Sync** (Score: 0.9183)
     * Establishing weekly progress reporting channels between school and family to reinforce oversight.
     * *Score Breakdown*: Đề xuất 'Parent-Teacher Engagement Sync' được lựa chọn vì nó hỗ trợ khắc phục tình trạng học lực đầu vào chưa tốt, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.72, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.75)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - LMS Navigation Tutorial: Interactive walkthrough of the digital learning system to track course updates and resources.
      - Parent-Teacher Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (lms_onboarding, parent_sync) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Remediate core knowledge gaps and practice key concepts to catch up.
    * *Recommended Actions*:
      - Peer-Led Study Tutoring: Collaborative learning sessions with top-performing peers to target specific concept gaps.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (peer_tutoring) to reinforce basic subject mastery.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - LMS Interactive Quizzing: Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
      - Biweekly Academic Coaching: Biweekly academic coaching for students showing declining performance trends.
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (interactive_quiz, academic_coaching, study_group) to sustain motivation and learning speed.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Self-evaluate progress and adjust study goals for the coming cycle.
    * *Recommended Actions*:
      - Review weekly study metrics and grade logs. Plan study targets for the next month.
    * *Expected Outcome*: Clear understanding of progress and updated self-study goals.
    * *Educational Rationale*: Cycle wrap-up: reflection on achievements and setting goals for the next month.

---

### Case Study: Moderate Risk (Average) Student (Test Index 2)
* **Student Context**: Absences: 2, Study Time: 3/4, Failures: 0, G1: 11, G2: 11
* **Predicted Academic Performance**: Class 1 (Medium) - Probabilities: [Low: 0.16, Medium: 0.81, High: 0.03]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 0.0002 probability
  - `R2_DECLINING_TREND`: 0.0011 probability
  - `R3_ATTENDANCE_RISK`: 0.0000 probability
  - `R4_LOW_ENGAGEMENT`: 0.7057 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Facilitated Study Group** (Score: 0.8001)
     * Weekly group discussions focusing on course concepts and collaborative exercises.
     * *Score Breakdown*: Đề xuất 'Facilitated Study Group' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.71, Perf Need: 0.57, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.75)
  2. **LMS Interactive Quizzing** (Score: 0.7298)
     * Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
     * *Score Breakdown*: Đề xuất 'LMS Interactive Quizzing' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.71, Perf Need: 0.54, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.85)
  3. **LMS Navigation Tutorial** (Score: 0.7248)
     * Interactive walkthrough of the digital learning system to track course updates and resources.
     * *Score Breakdown*: Đề xuất 'LMS Navigation Tutorial' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.71, Perf Need: 0.54, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.80)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - LMS Navigation Tutorial: Interactive walkthrough of the digital learning system to track course updates and resources.
      - Parent-Teacher Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (lms_onboarding, parent_sync) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Standard practice and concept review.
    * *Recommended Actions*:
      - Complete standard homework assignments. Optional: review previous exam questions.
    * *Expected Outcome*: Consistent completion of weekly homework.
    * *Educational Rationale*: No significant knowledge gaps diagnosed. Continuing with standard practice.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
      - LMS Interactive Quizzing: Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (study_group, interactive_quiz) to sustain motivation and learning speed.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Self-evaluate progress and adjust study goals for the coming cycle.
    * *Recommended Actions*:
      - Review weekly study metrics and grade logs. Plan study targets for the next month.
    * *Expected Outcome*: Clear understanding of progress and updated self-study goals.
    * *Educational Rationale*: Cycle wrap-up: reflection on achievements and setting goals for the next month.

---

### Case Study: Stable (High Performer) Student (Test Index 0)
* **Student Context**: Absences: 4, Study Time: 3/4, Failures: 0, G1: 13, G2: 14
* **Predicted Academic Performance**: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.27, High: 0.73]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 0.0000 probability
  - `R2_DECLINING_TREND`: 0.0000 probability
  - `R3_ATTENDANCE_RISK`: 0.0001 probability
  - `R4_LOW_ENGAGEMENT`: 0.5370 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Advanced Subject Seminar** (Score: 0.9053)
     * Enrichment seminar focusing on applications and advanced extensions of the course materials.
     * *Score Breakdown*: Đề xuất 'Advanced Subject Seminar' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.73, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.60)
  2. **Facilitated Study Group** (Score: 0.5885)
     * Weekly group discussions focusing on course concepts and collaborative exercises.
     * *Score Breakdown*: Đề xuất 'Facilitated Study Group' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.54, Perf Need: 0.14, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.75)
  3. **LMS Interactive Quizzing** (Score: 0.5525)
     * Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
     * *Score Breakdown*: Đề xuất 'LMS Interactive Quizzing' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.54, Perf Need: 0.28, Diff Fit: 0.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.85)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - LMS Navigation Tutorial: Interactive walkthrough of the digital learning system to track course updates and resources.
      - Parent-Teacher Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (lms_onboarding, parent_sync) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Standard practice and concept review.
    * *Recommended Actions*:
      - Complete standard homework assignments. Optional: review previous exam questions.
    * *Expected Outcome*: Consistent completion of weekly homework.
    * *Educational Rationale*: No significant knowledge gaps diagnosed. Continuing with standard practice.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
      - LMS Interactive Quizzing: Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (study_group, interactive_quiz) to sustain motivation and learning speed.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Pursue advanced topics to challenge capacity and expand skills.
    * *Recommended Actions*:
      - Advanced Subject Seminar: Enrichment seminar focusing on applications and advanced extensions of the course materials.
    * *Expected Outcome*: Completion of an enrichment topic or advanced challenge.
    * *Educational Rationale*: High performance indicates capability to handle advanced challenges.

---

