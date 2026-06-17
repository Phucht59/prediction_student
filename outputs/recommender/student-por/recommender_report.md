# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: student-por

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9094
* **Macro F1**: 0.9359
* **Micro Precision**: 0.9011
* **Macro Precision**: 0.9219
* **Micro Recall**: 0.9179
* **Macro Recall**: 0.9520
* **Hamming Loss**: 0.0628

### Ranking Metrics (at K=3)
* **Precision@3**: 0.6641
* **Recall@3**: 0.3185
* **NDCG@3**: 0.7455
* **Catalog Coverage@3**: 0.5500

### Path Quality Metrics
* **Risk Coverage Rate**: 0.9508
* **Workload Balance (std hours/week)**: 1.3137
* **Difficulty Progression Rate**: 0.6000
* **Prerequisite Violation Rate**: 0.0449

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 6)
* **Student Context**: Absences: 2, Study Time: 4/4, Failures: 1, G1: 10, G2: 8
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.73, Medium: 0.27, High: 0.00]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 1.0000 probability
  - `R2_DECLINING_TREND`: 1.0000 probability
  - `R3_ATTENDANCE_RISK`: 0.0000 probability
  - `R4_LOW_ENGAGEMENT`: 0.9988 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 1.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Peer-Led Study Tutoring** (Score: 1.0000)
     * Collaborative learning sessions with top-performing peers to target specific concept gaps.
     * *Score Breakdown*: Đề xuất 'Peer-Led Study Tutoring' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 1.00, Diff Fit: 0.88, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.80, pLow: 0.73, MaxRisk: 1.00)
  2. **Targeted Practice Exercises** (Score: 1.0000)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Đề xuất 'Targeted Practice Exercises' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 1.00, Diff Fit: 0.88, Time Fit: 1.00, Prereq Fit: 0.85, Effect: 0.90, pLow: 0.73, MaxRisk: 1.00)
  3. **Biweekly Academic Coaching** (Score: 1.0000)
     * Biweekly academic coaching for students showing declining performance trends.
     * *Score Breakdown*: Đề xuất 'Biweekly Academic Coaching' được lựa chọn vì nó hỗ trợ khắc phục tình trạng rủi ro không đạt kết quả môn học, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.56, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.70, pLow: 0.73, MaxRisk: 1.00)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Stabilize attendance, support contact and basic study routine.
    * *Recommended Actions*:
      - Parent-School Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
      - Family Progress Contract: A simple family-school agreement with one weekly learning target and one progress check-in.
    * *Expected Outcome*: Immediate barriers are identified and the student has a concrete weekly plan.
    * *Educational Rationale*: Selected top-scoring stabilize interventions linked to diagnosed risks: R6_HIGH_FAILURE_PROBABILITY, R1_LOW_PRIOR_PERFORMANCE, R2_DECLINING_TREND.
  * **Week 2 - Theme: Practice**
    * *Objective*: Close the highest-priority knowledge or engagement gap.
    * *Recommended Actions*:
      - Peer-Led Study Tutoring: Collaborative learning sessions with top-performing peers to target specific concept gaps.
      - Targeted Practice Exercises: Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
    * *Expected Outcome*: The student completes measurable practice tasks and receives feedback.
    * *Educational Rationale*: Selected top-scoring practice interventions linked to diagnosed risks: R6_HIGH_FAILURE_PROBABILITY, R1_LOW_PRIOR_PERFORMANCE, R2_DECLINING_TREND.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Reinforce learning through interaction, resources and repetition.
    * *Recommended Actions*:
      - Biweekly Academic Coaching: Biweekly academic coaching for students showing declining performance trends.
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
    * *Expected Outcome*: Engagement indicators improve and weak topics are revisited.
    * *Educational Rationale*: Selected top-scoring reinforce interventions linked to diagnosed risks: R6_HIGH_FAILURE_PROBABILITY, R1_LOW_PRIOR_PERFORMANCE, R2_DECLINING_TREND.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Evaluate progress and decide whether to continue, reduce or escalate support.
    * *Recommended Actions*:
      - Review attendance, LMS/resource usage and practice completion; compare with Week 1 baseline.
      - If risk indicators remain high, continue the strongest Week 2 intervention for another cycle.
    * *Expected Outcome*: A clear next-cycle decision based on measured progress.
    * *Educational Rationale*: The final week evaluates whether the intervention reduced the predicted risk signals.

---

### Case Study: Moderate Risk (Average) Student (Test Index 2)
* **Student Context**: Absences: 2, Study Time: 3/4, Failures: 0, G1: 11, G2: 11
* **Predicted Academic Performance**: Class 1 (Medium) - Probabilities: [Low: 0.16, Medium: 0.81, High: 0.03]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 0.0002 probability
  - `R2_DECLINING_TREND`: 0.0032 probability
  - `R3_ATTENDANCE_RISK`: 0.0001 probability
  - `R4_LOW_ENGAGEMENT`: 0.6524 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0009 probability

* **Top 3 Recommended Interventions**:
  1. **Facilitated Study Group** (Score: 0.8030)
     * Weekly group discussions focusing on course concepts and collaborative exercises.
     * *Score Breakdown*: Đề xuất 'Facilitated Study Group' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.70, Perf Need: 0.73, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.75, pLow: 0.16, MaxRisk: 0.65)
  2. **Daily Attendance Monitoring** (Score: 0.5301)
     * Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
     * *Score Breakdown*: Đề xuất 'Daily Attendance Monitoring' được lựa chọn vì nó đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.73, Diff Fit: 0.75, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.90, pLow: 0.16, MaxRisk: 0.65)
  3. **Academic Counselor Consultation** (Score: 0.5251)
     * One-on-one sessions to address personal wellbeing academic anxiety and home/school support alignment.
     * *Score Breakdown*: Đề xuất 'Academic Counselor Consultation' được lựa chọn vì nó đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.73, Diff Fit: 0.75, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.85, pLow: 0.16, MaxRisk: 0.65)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Stabilize attendance, support contact and basic study routine.
    * *Recommended Actions*:
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
      - Academic Counselor Consultation: One-on-one sessions to address personal wellbeing academic anxiety and home/school support alignment.
    * *Expected Outcome*: Immediate barriers are identified and the student has a concrete weekly plan.
    * *Educational Rationale*: Selected top-scoring stabilize interventions linked to diagnosed risks: R4_LOW_ENGAGEMENT.
  * **Week 2 - Theme: Practice**
    * *Objective*: Close the highest-priority knowledge or engagement gap.
    * *Recommended Actions*:
      - Complete standard homework plus one targeted review exercise set.
    * *Expected Outcome*: The student completes measurable practice tasks and receives feedback.
    * *Educational Rationale*: No catalog item was required for Practice; fallback action keeps the 4-week path complete.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Reinforce learning through interaction, resources and repetition.
    * *Recommended Actions*:
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
    * *Expected Outcome*: Engagement indicators improve and weak topics are revisited.
    * *Educational Rationale*: Selected top-scoring reinforce interventions linked to diagnosed risks: R4_LOW_ENGAGEMENT.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Evaluate progress and decide whether to continue, reduce or escalate support.
    * *Recommended Actions*:
      - Review attendance, LMS/resource usage and practice completion; compare with Week 1 baseline.
      - If risk indicators remain high, continue the strongest Week 2 intervention for another cycle.
    * *Expected Outcome*: A clear next-cycle decision based on measured progress.
    * *Educational Rationale*: The final week evaluates whether the intervention reduced the predicted risk signals.

---

### Case Study: Stable (High Performer) Student (Test Index 0)
* **Student Context**: Absences: 4, Study Time: 3/4, Failures: 0, G1: 13, G2: 14
* **Predicted Academic Performance**: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.27, High: 0.73]
* **Diagnosed Academic Risks**:
  - `R1_LOW_PRIOR_PERFORMANCE`: 0.0000 probability
  - `R2_DECLINING_TREND`: 0.0000 probability
  - `R3_ATTENDANCE_RISK`: 0.0001 probability
  - `R4_LOW_ENGAGEMENT`: 0.4862 probability
  - `R5_INSUFFICIENT_STUDY_TIME`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Advanced Subject Seminar** (Score: 0.6653)
     * Enrichment seminar focusing on applications and advanced extensions of the course materials.
     * *Score Breakdown*: Đề xuất 'Advanced Subject Seminar' được lựa chọn vì nó đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.20, Perf Need: 0.73, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.60, pLow: 0.00, MaxRisk: 0.49)
  2. **Standard Practice Plan** (Score: 0.5416)
     * A light weekly practice plan for students with no active risk but continued need for routine reinforcement.
     * *Score Breakdown*: Đề xuất 'Standard Practice Plan' được lựa chọn vì nó đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.20, Perf Need: 0.52, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.68, pLow: 0.00, MaxRisk: 0.49)
  3. **Weekly Progress Review** (Score: 0.5386)
     * A short weekly review of attendance practice completion resource usage and next-week priorities.
     * *Score Breakdown*: Đề xuất 'Weekly Progress Review' được lựa chọn vì nó đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.20, Perf Need: 0.52, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.65, pLow: 0.00, MaxRisk: 0.49)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Stabilize attendance, support contact and basic study routine.
    * *Recommended Actions*:
      - Set a minimum weekly study schedule and complete one advisor or school check-in.
    * *Expected Outcome*: Immediate barriers are identified and the student has a concrete weekly plan.
    * *Educational Rationale*: No catalog item was required for Stabilize; fallback action keeps the 4-week path complete.
  * **Week 2 - Theme: Practice**
    * *Objective*: Close the highest-priority knowledge or engagement gap.
    * *Recommended Actions*:
      - Standard Practice Plan: A light weekly practice plan for students with no active risk but continued need for routine reinforcement.
    * *Expected Outcome*: The student completes measurable practice tasks and receives feedback.
    * *Educational Rationale*: Selected top-scoring practice interventions linked to diagnosed risks: R4_LOW_ENGAGEMENT.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Reinforce learning through interaction, resources and repetition.
    * *Recommended Actions*:
      - Join class discussion or LMS activity at least twice during the week.
    * *Expected Outcome*: Engagement indicators improve and weak topics are revisited.
    * *Educational Rationale*: No catalog item was required for Reinforce; fallback action keeps the 4-week path complete.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Evaluate progress and decide whether to continue, reduce or escalate support.
    * *Recommended Actions*:
      - Review attendance, LMS/resource usage and practice completion; compare with Week 1 baseline.
      - If risk indicators remain high, continue the strongest Week 2 intervention for another cycle.
    * *Expected Outcome*: A clear next-cycle decision based on measured progress.
    * *Educational Rationale*: The final week evaluates whether the intervention reduced the predicted risk signals.

---

