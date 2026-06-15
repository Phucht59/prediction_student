# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: student-mat

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9593
* **Macro F1**: 0.9613
* **Micro Precision**: 0.9516
* **Macro Precision**: 0.9516
* **Micro Recall**: 0.9672
* **Macro Recall**: 0.9719
* **Hamming Loss**: 0.0422

### Ranking Metrics (at K=3)
* **Precision@3**: 0.8650
* **Recall@3**: 0.4835
* **NDCG@3**: 0.8830
* **Catalog Coverage@3**: 1.0000

### Path Quality Metrics
* **Risk Coverage Rate**: 0.8707
* **Workload Balance (std hours/week)**: 1.8242
* **Difficulty Progression Rate**: 0.4852
* **Prerequisite Violation Rate**: 0.0401

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 3)
* **Student Context**: Absences: 4, Study Time: 2/4, Failures: 0, G1: 8, G2: 7
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.92, Medium: 0.08, High: 0.00]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 0.9961 probability
  - `R2_DECLINING_TREND`: 0.9979 probability
  - `R3_ATTENDANCE_RISK`: 0.0071 probability
  - `R4_LOW_ENGAGEMENT`: 0.9616 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0006 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.9995 probability

* **Top 3 Recommended Interventions**:
  1. **LMS Interactive Quizzing** (Score: 0.9286)
     * Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
     * *Score Breakdown*: Đề xuất 'LMS Interactive Quizzing' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.96, Perf Need: 0.78, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.85)
  2. **Biweekly Academic Coaching** (Score: 0.9244)
     * Biweekly academic coaching for students showing declining performance trends.
     * *Score Breakdown*: Đề xuất 'Biweekly Academic Coaching' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.78, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.70)
  3. **LMS Navigation Tutorial** (Score: 0.9236)
     * Interactive walkthrough of the digital learning system to track course updates and resources.
     * *Score Breakdown*: Đề xuất 'LMS Navigation Tutorial' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.96, Perf Need: 0.78, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.80)

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

### Case Study: Moderate Risk (Average) Student (Test Index 0)
* **Student Context**: Absences: 0, Study Time: 1/4, Failures: 0, G1: 11, G2: 12
* **Predicted Academic Performance**: Class 1 (Medium) - Probabilities: [Low: 0.03, Medium: 0.95, High: 0.02]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 0.0004 probability
  - `R2_DECLINING_TREND`: 0.0004 probability
  - `R3_ATTENDANCE_RISK`: 0.0008 probability
  - `R4_LOW_ENGAGEMENT`: 0.6836 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.9993 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.9991 probability

* **Top 3 Recommended Interventions**:
  1. **Facilitated Study Group** (Score: 0.8757)
     * Weekly group discussions focusing on course concepts and collaborative exercises.
     * *Score Breakdown*: Đề xuất 'Facilitated Study Group' được lựa chọn vì nó hỗ trợ khắc phục tình trạng thời gian tự học chưa đủ, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.50, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.75)
  2. **Time Management Workshop** (Score: 0.7954)
     * Guided sessions on creating weekly study schedules prioritizing tasks and minimizing distractions.
     * *Score Breakdown*: Đề xuất 'Time Management Workshop' được lựa chọn vì nó hỗ trợ khắc phục tình trạng thời gian tự học chưa đủ, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.50, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.70)
  3. **Remedial Topic Bootcamps** (Score: 0.7607)
     * Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
     * *Score Breakdown*: Đề xuất 'Remedial Topic Bootcamps' được lựa chọn vì nó hỗ trợ khắc phục tình trạng thời gian tự học chưa đủ, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại. (Risk Match: 1.00, Perf Need: 0.50, Diff Fit: 0.50, Time Fit: 0.60, Prereq Fit: 1.00, Effect: 0.95)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Time Management Workshop: Guided sessions on creating weekly study schedules prioritizing tasks and minimizing distractions.
      - LMS Navigation Tutorial: Interactive walkthrough of the digital learning system to track course updates and resources.
      - Parent-Teacher Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (time_planning, lms_onboarding, parent_sync) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Remediate core knowledge gaps and practice key concepts to catch up.
    * *Recommended Actions*:
      - Remedial Topic Bootcamps: Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (remedial_class) to reinforce basic subject mastery.
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

### Case Study: Stable (High Performer) Student (Test Index 1)
* **Student Context**: Absences: 4, Study Time: 2/4, Failures: 0, G1: 15, G2: 14
* **Predicted Academic Performance**: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.33, High: 0.67]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 0.0000 probability
  - `R2_DECLINING_TREND`: 0.9984 probability
  - `R3_ATTENDANCE_RISK`: 0.0007 probability
  - `R4_LOW_ENGAGEMENT`: 0.9281 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0002 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 1.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Advanced Subject Seminar** (Score: 0.8931)
     * Enrichment seminar focusing on applications and advanced extensions of the course materials.
     * *Score Breakdown*: Đề xuất 'Advanced Subject Seminar' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.67, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.60)
  2. **Remedial Topic Bootcamps** (Score: 0.8286)
     * Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
     * *Score Breakdown*: Đề xuất 'Remedial Topic Bootcamps' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.17, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.95)
  3. **Targeted Practice Exercises** (Score: 0.7481)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Đề xuất 'Targeted Practice Exercises' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.17, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.90)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Maintain current academic stability and check-in with advisor.
    * *Recommended Actions*:
      - Maintain current attendance and standard study schedules. No urgent stabilization required.
    * *Expected Outcome*: Consistent attendance and stable scheduling maintained.
    * *Educational Rationale*: No critical stabilization issues detected based on current student profile.
  * **Week 2 - Theme: Practice**
    * *Objective*: Remediate core knowledge gaps and practice key concepts to catch up.
    * *Recommended Actions*:
      - Remedial Topic Bootcamps: Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
      - Targeted Practice Exercises: Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (remedial_class, extra_exercises) to reinforce basic subject mastery.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
      - Biweekly Academic Coaching: Biweekly academic coaching for students showing declining performance trends.
      - LMS Interactive Quizzing: Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (study_group, academic_coaching, interactive_quiz) to sustain motivation and learning speed.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Pursue advanced topics to challenge capacity and expand skills.
    * *Recommended Actions*:
      - Advanced Subject Seminar: Enrichment seminar focusing on applications and advanced extensions of the course materials.
    * *Expected Outcome*: Completion of an enrichment topic or advanced challenge.
    * *Educational Rationale*: High performance indicates capability to handle advanced challenges.

---

