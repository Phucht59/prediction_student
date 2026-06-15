# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: xapi

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9765
* **Macro F1**: 0.6543
* **Micro Precision**: 0.9630
* **Macro Precision**: 0.6473
* **Micro Recall**: 0.9905
* **Macro Recall**: 0.6616
* **Hamming Loss**: 0.0174

### Ranking Metrics (at K=3)
* **Precision@3**: 0.7708
* **Recall@3**: 0.6719
* **NDCG@3**: 0.9081
* **Catalog Coverage@3**: 0.8333

### Path Quality Metrics
* **Risk Coverage Rate**: 1.0000
* **Workload Balance (std hours/week)**: 2.0100
* **Difficulty Progression Rate**: 0.6528
* **Prerequisite Violation Rate**: 0.0052

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 1)
* **Student Context**: Raised Hands: 17, Visited Resources: 61, Discussion: 14, Absences: Under-7
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.59, Medium: 0.40, High: 0.00]
* **Diagnosed Academic Risks**:
  - `R3_ATTENDANCE_RISK`: 0.0008 probability
  - `R4_LOW_ENGAGEMENT`: 1.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **LMS Interactive Quizzing** (Score: 0.9205)
     * Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
     * *Score Breakdown*: Đề xuất 'LMS Interactive Quizzing' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.68, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.85)
  2. **LMS Navigation Tutorial** (Score: 0.9155)
     * Interactive walkthrough of the digital learning system to track course updates and resources.
     * *Score Breakdown*: Đề xuất 'LMS Navigation Tutorial' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.68, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.80)
  3. **Parent-Teacher Engagement Sync** (Score: 0.9105)
     * Establishing weekly progress reporting channels between school and family to reinforce oversight.
     * *Score Breakdown*: Đề xuất 'Parent-Teacher Engagement Sync' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.68, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.75)

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
      - Targeted Practice Exercises: Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (peer_tutoring, extra_exercises) to reinforce basic subject mastery.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - LMS Interactive Quizzing: Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (interactive_quiz, study_group) to sustain motivation and learning speed.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Self-evaluate progress and adjust study goals for the coming cycle.
    * *Recommended Actions*:
      - Review weekly study metrics and grade logs. Plan study targets for the next month.
    * *Expected Outcome*: Clear understanding of progress and updated self-study goals.
    * *Educational Rationale*: Cycle wrap-up: reflection on achievements and setting goals for the next month.

---

### Case Study: Moderate Risk (Average) Student (Test Index 0)
* **Student Context**: Raised Hands: 72, Visited Resources: 80, Discussion: 66, Absences: Under-7
* **Predicted Academic Performance**: Class 1 (Medium) - Probabilities: [Low: 0.00, Medium: 0.69, High: 0.30]
* **Diagnosed Academic Risks**:
  - `R3_ATTENDANCE_RISK`: 0.0001 probability
  - `R4_LOW_ENGAGEMENT`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Targeted Practice Exercises** (Score: 0.5597)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Đề xuất 'Targeted Practice Exercises' được lựa chọn vì nó vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.35, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.90)
  2. **Peer-Led Study Tutoring** (Score: 0.5497)
     * Collaborative learning sessions with top-performing peers to target specific concept gaps.
     * *Score Breakdown*: Đề xuất 'Peer-Led Study Tutoring' được lựa chọn vì nó vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.35, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.80)
  3. **Facilitated Study Group** (Score: 0.5447)
     * Weekly group discussions focusing on course concepts and collaborative exercises.
     * *Score Breakdown*: Đề xuất 'Facilitated Study Group' được lựa chọn vì nó vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.35, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.75)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
      - Academic Counselor Consultation: One-on-one sessions to address personal wellbeing academic anxiety and home/school support alignment.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (attendance_monitoring, counselor_meeting) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Remediate core knowledge gaps and practice key concepts to catch up.
    * *Recommended Actions*:
      - Targeted Practice Exercises: Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
      - Peer-Led Study Tutoring: Collaborative learning sessions with top-performing peers to target specific concept gaps.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (extra_exercises, peer_tutoring) to reinforce basic subject mastery.
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

### Case Study: Stable (High Performer) Student (Test Index 4)
* **Student Context**: Raised Hands: 70, Visited Resources: 80, Discussion: 70, Absences: Under-7
* **Predicted Academic Performance**: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.10, High: 0.90]
* **Diagnosed Academic Risks**:
  - `R3_ATTENDANCE_RISK`: 0.0000 probability
  - `R4_LOW_ENGAGEMENT`: 0.0000 probability
  - `R6_HIGH_FAILURE_PROBABILITY`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Advanced Subject Seminar** (Score: 0.9390)
     * Enrichment seminar focusing on applications and advanced extensions of the course materials.
     * *Score Breakdown*: Đề xuất 'Advanced Subject Seminar' được lựa chọn vì nó hỗ trợ khắc phục tình trạng mức độ tương tác lớp học thấp, đáp ứng nhu cầu cải thiện kết quả học tập hiện tại, vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 1.00, Perf Need: 0.90, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.60)
  2. **Remedial Topic Bootcamps** (Score: 0.5055)
     * Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
     * *Score Breakdown*: Đề xuất 'Remedial Topic Bootcamps' được lựa chọn vì nó vừa sức với năng lực hiện tại của bạn, phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.05, Diff Fit: 1.00, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.95)
  3. **Targeted Practice Exercises** (Score: 0.4255)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Đề xuất 'Targeted Practice Exercises' được lựa chọn vì nó phù hợp với thời gian học tập hàng tuần của bạn. (Risk Match: 0.00, Perf Need: 0.05, Diff Fit: 0.50, Time Fit: 1.00, Prereq Fit: 1.00, Effect: 0.90)

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (attendance_monitoring) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Remediate core knowledge gaps and practice key concepts to catch up.
    * *Recommended Actions*:
      - Remedial Topic Bootcamps: Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
      - Targeted Practice Exercises: Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
      - Peer-Led Study Tutoring: Collaborative learning sessions with top-performing peers to target specific concept gaps.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (remedial_class, extra_exercises, peer_tutoring) to reinforce basic subject mastery.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (study_group) to sustain motivation and learning speed.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Pursue advanced topics to challenge capacity and expand skills.
    * *Recommended Actions*:
      - Advanced Subject Seminar: Enrichment seminar focusing on applications and advanced extensions of the course materials.
    * *Expected Outcome*: Completion of an enrichment topic or advanced challenge.
    * *Educational Rationale*: High performance indicates capability to handle advanced challenges.

---

