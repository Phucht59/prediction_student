# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: xapi

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9813
* **Macro F1**: 0.9831
* **Micro Precision**: 0.9776
* **Macro Precision**: 0.9794
* **Micro Recall**: 0.9850
* **Macro Recall**: 0.9869
* **Hamming Loss**: 0.0174

### Ranking Metrics (at K=3)
* **Precision@3**: 0.7014
* **Recall@3**: 0.4719
* **NDCG@3**: 0.8407
* **Catalog Coverage@3**: 0.6875

### Path Quality Metrics
* **Risk Coverage Rate**: 0.9306
* **Workload Balance (std hours/week)**: 1.3152
* **Difficulty Progression Rate**: 0.7396
* **Prerequisite Violation Rate**: 0.0021

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 1)
* **Student Context**: Raised Hands: 17, Visited Resources: 61, Discussion: 14, Absences: Under-7
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.59, Medium: 0.40, High: 0.00]
* **Diagnosed Academic Risks**:
  - `R3_ATTENDANCE_RISK`: 0.0003 probability
  - `R4_LOW_ENGAGEMENT`: 0.9999 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.9993 probability

* **Top 3 Recommended Interventions**:
  1. **Daily LMS Resource Checklist** (Score: 0.9682)
     * A short daily checklist that requires opening key resources announcements and practice links before class.
     * *Score Breakdown*: Đề xuất 'Daily LMS Resource Checklist' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.92, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.88, pLow: 0.59, MaxRisk: 1.00)
  2. **Parent-Teacher Engagement Sync** (Score: 0.9673)
     * Establishing weekly progress reporting channels between school and family to reinforce oversight.
     * *Score Breakdown*: Đề xuất 'Parent-Teacher Engagement Sync' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.96, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.78, pLow: 0.59, MaxRisk: 1.00)
  3. **Family Progress Contract** (Score: 0.9673)
     * A simple family-school agreement with one weekly learning target and one progress check-in.
     * *Score Breakdown*: Đề xuất 'Family Progress Contract' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.96, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.78, pLow: 0.59, MaxRisk: 1.00)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Stabilize attendance, support contact and basic study routine.
    * *Recommended Actions*:
      - Daily LMS Resource Checklist: A short daily checklist that requires opening key resources announcements and practice links before class.
      - Parent-Teacher Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
    * *Expected Outcome*: Immediate barriers are identified and the student has a concrete weekly plan.
    * *Educational Rationale*: Selected top-scoring stabilize interventions linked to diagnosed risks: R4_LOW_ENGAGEMENT, R6_HIGH_FAILURE_PROBABILITY.
  * **Week 2 - Theme: Practice**
    * *Objective*: Close the highest-priority knowledge or engagement gap.
    * *Recommended Actions*:
      - Absence Recovery Pack: Short catch-up package for missed lessons with teacher verification after each completed unit.
    * *Expected Outcome*: The student completes measurable practice tasks and receives feedback.
    * *Educational Rationale*: Selected top-scoring practice interventions linked to diagnosed risks: R4_LOW_ENGAGEMENT, R6_HIGH_FAILURE_PROBABILITY.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Reinforce learning through interaction, resources and repetition.
    * *Recommended Actions*:
      - LMS Interactive Quizzing: Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
      - Guided Discussion Prompts: Teacher-provided prompts that require short answers or peer replies to increase class interaction.
    * *Expected Outcome*: Engagement indicators improve and weak topics are revisited.
    * *Educational Rationale*: Selected top-scoring reinforce interventions linked to diagnosed risks: R4_LOW_ENGAGEMENT, R6_HIGH_FAILURE_PROBABILITY.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Evaluate progress and decide whether to continue, reduce or escalate support.
    * *Recommended Actions*:
      - Review attendance, LMS/resource usage and practice completion; compare with Week 1 baseline.
      - If risk indicators remain high, continue the strongest Week 2 intervention for another cycle.
    * *Expected Outcome*: A clear next-cycle decision based on measured progress.
    * *Educational Rationale*: The final week evaluates whether the intervention reduced the predicted risk signals.

---

### Case Study: Moderate Risk (Average) Student (Test Index 0)
* **Student Context**: Raised Hands: 72, Visited Resources: 80, Discussion: 66, Absences: Under-7
* **Predicted Academic Performance**: Class 1 (Medium) - Probabilities: [Low: 0.00, Medium: 0.69, High: 0.30]
* **Diagnosed Academic Risks**:
  - `R3_ATTENDANCE_RISK`: 0.0002 probability
  - `R4_LOW_ENGAGEMENT`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Daily Attendance Monitoring** (Score: 0.5216)
     * Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
     * *Score Breakdown*: Đề xuất 'Daily Attendance Monitoring' được lựa chọn vì nó vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.38, Diff Fit: 0.75, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.90, pLow: 0.00, MaxRisk: 0.00)
  2. **Daily LMS Resource Checklist** (Score: 0.5196)
     * A short daily checklist that requires opening key resources announcements and practice links before class.
     * *Score Breakdown*: Đề xuất 'Daily LMS Resource Checklist' được lựa chọn vì nó vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.38, Diff Fit: 0.75, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.88, pLow: 0.00, MaxRisk: 0.00)
  3. **Academic Counselor Consultation** (Score: 0.5166)
     * One-on-one sessions to address personal wellbeing academic anxiety and home/school support alignment.
     * *Score Breakdown*: Đề xuất 'Academic Counselor Consultation' được lựa chọn vì nó vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.38, Diff Fit: 0.75, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.85, pLow: 0.00, MaxRisk: 0.00)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Stabilize attendance, support contact and basic study routine.
    * *Recommended Actions*:
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
      - Daily LMS Resource Checklist: A short daily checklist that requires opening key resources announcements and practice links before class.
    * *Expected Outcome*: Immediate barriers are identified and the student has a concrete weekly plan.
    * *Educational Rationale*: Selected top-scoring stabilize interventions linked to diagnosed risks: general support.
  * **Week 2 - Theme: Practice**
    * *Objective*: Close the highest-priority knowledge or engagement gap.
    * *Recommended Actions*:
      - Complete standard homework plus one targeted review exercise set.
    * *Expected Outcome*: The student completes measurable practice tasks and receives feedback.
    * *Educational Rationale*: No catalog item was required for Practice; fallback action keeps the 4-week path complete.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Reinforce learning through interaction, resources and repetition.
    * *Recommended Actions*:
      - LMS Interactive Quizzing: Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
    * *Expected Outcome*: Engagement indicators improve and weak topics are revisited.
    * *Educational Rationale*: Selected top-scoring reinforce interventions linked to diagnosed risks: general support.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Evaluate progress and decide whether to continue, reduce or escalate support.
    * *Recommended Actions*:
      - Review attendance, LMS/resource usage and practice completion; compare with Week 1 baseline.
      - If risk indicators remain high, continue the strongest Week 2 intervention for another cycle.
    * *Expected Outcome*: A clear next-cycle decision based on measured progress.
    * *Educational Rationale*: The final week evaluates whether the intervention reduced the predicted risk signals.

---

### Case Study: Stable (High Performer) Student (Test Index 4)
* **Student Context**: Raised Hands: 70, Visited Resources: 80, Discussion: 70, Absences: Under-7
* **Predicted Academic Performance**: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.10, High: 0.90]
* **Diagnosed Academic Risks**:
  - `R3_ATTENDANCE_RISK`: 0.0000 probability
  - `R4_LOW_ENGAGEMENT`: 0.0001 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Advanced Subject Seminar** (Score: 0.9154)
     * Enrichment seminar focusing on applications and advanced extensions of the course materials.
     * *Score Breakdown*: Đề xuất 'Advanced Subject Seminar' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.90, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.60, pLow: 0.00, MaxRisk: 0.00)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Stabilize attendance, support contact and basic study routine.
    * *Recommended Actions*:
      - Set a minimum weekly study schedule and complete one advisor/teacher check-in.
    * *Expected Outcome*: Immediate barriers are identified and the student has a concrete weekly plan.
    * *Educational Rationale*: No catalog item was required for Stabilize; fallback action keeps the 4-week path complete.
  * **Week 2 - Theme: Practice**
    * *Objective*: Close the highest-priority knowledge or engagement gap.
    * *Recommended Actions*:
      - Complete standard homework plus one targeted review exercise set.
    * *Expected Outcome*: The student completes measurable practice tasks and receives feedback.
    * *Educational Rationale*: No catalog item was required for Practice; fallback action keeps the 4-week path complete.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Reinforce learning through interaction, resources and repetition.
    * *Recommended Actions*:
      - Join class discussion or LMS activity at least twice during the week.
    * *Expected Outcome*: Engagement indicators improve and weak topics are revisited.
    * *Educational Rationale*: No catalog item was required for Reinforce; fallback action keeps the 4-week path complete.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Evaluate progress and decide whether to continue, reduce or escalate support.
    * *Recommended Actions*:
      - Advanced Subject Seminar: Enrichment seminar focusing on applications and advanced extensions of the course materials.
    * *Expected Outcome*: A clear next-cycle decision based on measured progress.
    * *Educational Rationale*: Stable prediction and low diagnosed risk allow enrichment-oriented follow-up.

---

