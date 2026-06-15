# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: student-por

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9756
* **Macro F1**: 0.9719
* **Micro Precision**: 0.9615
* **Macro Precision**: 0.9647
* **Micro Recall**: 0.9901
* **Macro Recall**: 0.9825
* **Hamming Loss**: 0.0128

### Ranking Metrics (at K=3)
* **Precision@3**: 0.7026
* **Recall@3**: 0.6670
* **NDCG@3**: 0.8462
* **Catalog Coverage@3**: 0.7500

### Path Quality Metrics
* **Risk Coverage Rate**: 1.0000
* **Workload Balance (std hours/week)**: 2.7959
* **Difficulty Progression Rate**: 0.5000
* **Prerequisite Violation Rate**: 0.0487

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 6)
* **Student Context**: Absences: 2, Study Time: 4/4, Failures: 1, G1: 10, G2: 8
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.73, Medium: 0.27, High: 0.00]
* **Diagnosed Academic Risks**:
  - `attendance`: 0.0000 probability
  - `failure_history`: 0.9997 probability
  - `grade_gap`: 1.0000 probability
  - `study_time`: 0.0000 probability
  - `wellbeing`: 1.0000 probability
  - `time_management`: 1.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Academic Counselor Consultation** (Score: 0.9285)
     * One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.72, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.85
  2. **Stress & Lifestyle Management Workshop** (Score: 0.9135)
     * Group counseling sessions regarding balanced lifestyle, reducing harmful habits, and exam stress relief.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.72, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.70
  3. **Time Management Workshop** (Score: 0.9135)
     * Guided sessions on creating weekly study schedules, prioritizing tasks, and minimizing distractions.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.72, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.70

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Academic Counselor Consultation: One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
      - Time Management Workshop: Guided sessions on creating weekly study schedules, prioritizing tasks, and minimizing distractions.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (counselor_meeting, time_planning) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Remediate core knowledge gaps and practice key concepts to catch up.
    * *Recommended Actions*:
      - Peer-Led Study Tutoring: Collaborative learning sessions with top-performing peers to target specific concept gaps.
      - Targeted Practice Exercises: Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
      - Remedial Topic Bootcamps: Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (peer_tutoring, extra_exercises, remedial_class) to reinforce basic subject mastery.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - Stress & Lifestyle Management Workshop: Group counseling sessions regarding balanced lifestyle, reducing harmful habits, and exam stress relief.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (stress_management) to sustain motivation and learning speed.
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
  - `attendance`: 0.0000 probability
  - `failure_history`: 0.0001 probability
  - `grade_gap`: 0.0159 probability
  - `study_time`: 0.0000 probability
  - `wellbeing`: 0.0008 probability
  - `time_management`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Targeted Practice Exercises** (Score: 0.6082)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Risk Match (30%): 0.02, Performance Need (20%): 0.57, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.90
  2. **Peer-Led Study Tutoring** (Score: 0.5982)
     * Collaborative learning sessions with top-performing peers to target specific concept gaps.
     * *Score Breakdown*: Risk Match (30%): 0.02, Performance Need (20%): 0.57, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.80
  3. **Facilitated Study Group** (Score: 0.5884)
     * Weekly group discussions focusing on course concepts and collaborative exercises.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.57, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.75

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
      - Academic Counselor Consultation: One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (attendance_monitoring, counselor_meeting) to build a stable learning foundation.
  * **Week 2 - Theme: Practice**
    * *Objective*: Remediate core knowledge gaps and practice key concepts to catch up.
    * *Recommended Actions*:
      - Targeted Practice Exercises: Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
      - Peer-Led Study Tutoring: Collaborative learning sessions with top-performing peers to target specific concept gaps.
      - Remedial Topic Bootcamps: Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (extra_exercises, peer_tutoring, remedial_class) to reinforce basic subject mastery.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (study_group) to sustain motivation and learning speed.
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
  - `attendance`: 0.0000 probability
  - `failure_history`: 0.0000 probability
  - `grade_gap`: 0.0014 probability
  - `study_time`: 0.0000 probability
  - `wellbeing`: 0.0000 probability
  - `time_management`: 0.0003 probability

* **Top 3 Recommended Interventions**:
  1. **Advanced Subject Seminar** (Score: 0.6053)
     * Enrichment seminar focusing on applications and advanced extensions of the course materials.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.73, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.60
  2. **Remedial Topic Bootcamps** (Score: 0.5228)
     * Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.14, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.95
  3. **Targeted Practice Exercises** (Score: 0.4428)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.14, Difficulty Fit (15%): 0.50, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.90

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

