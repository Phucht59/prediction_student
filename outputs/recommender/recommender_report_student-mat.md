# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: student-mat

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9559
* **Macro F1**: 0.9552
* **Micro Precision**: 0.9724
* **Macro Precision**: 0.9725
* **Micro Recall**: 0.9400
* **Macro Recall**: 0.9432
* **Hamming Loss**: 0.0274

### Ranking Metrics (at K=3)
* **Precision@3**: 0.7975
* **Recall@3**: 0.6623
* **NDCG@3**: 0.8751
* **Catalog Coverage@3**: 0.8333

### Path Quality Metrics
* **Risk Coverage Rate**: 0.9937
* **Workload Balance (std hours/week)**: 2.8092
* **Difficulty Progression Rate**: 0.5485
* **Prerequisite Violation Rate**: 0.0992

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 3)
* **Student Context**: Absences: 4, Study Time: 2/4, Failures: 0, G1: 8, G2: 7
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.92, Medium: 0.08, High: 0.00]
* **Diagnosed Academic Risks**:
  - `attendance`: 0.0007 probability
  - `failure_history`: 0.0007 probability
  - `grade_gap`: 0.9999 probability
  - `study_time`: 0.0010 probability
  - `wellbeing`: 0.0000 probability
  - `time_management`: 0.9997 probability

* **Top 3 Recommended Interventions**:
  1. **Time Management Workshop** (Score: 0.9250)
     * Guided sessions on creating weekly study schedules, prioritizing tasks, and minimizing distractions.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.78, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.70
  2. **Stress & Lifestyle Management Workshop** (Score: 0.9250)
     * Group counseling sessions regarding balanced lifestyle, reducing harmful habits, and exam stress relief.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.78, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.70
  3. **Peer-Led Study Tutoring** (Score: 0.8967)
     * Collaborative learning sessions with top-performing peers to target specific concept gaps.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.96, Difficulty Fit (15%): 0.50, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.80

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Time Management Workshop: Guided sessions on creating weekly study schedules, prioritizing tasks, and minimizing distractions.
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (time_planning, attendance_monitoring) to build a stable learning foundation.
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

### Case Study: Moderate Risk (Average) Student (Test Index 0)
* **Student Context**: Absences: 0, Study Time: 1/4, Failures: 0, G1: 11, G2: 12
* **Predicted Academic Performance**: Class 1 (Medium) - Probabilities: [Low: 0.03, Medium: 0.95, High: 0.02]
* **Diagnosed Academic Risks**:
  - `attendance`: 0.0000 probability
  - `failure_history`: 0.0005 probability
  - `grade_gap`: 0.0657 probability
  - `study_time`: 0.9997 probability
  - `wellbeing`: 0.9999 probability
  - `time_management`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Facilitated Study Group** (Score: 0.8758)
     * Weekly group discussions focusing on course concepts and collaborative exercises.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.50, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.75
  2. **Academic Counselor Consultation** (Score: 0.8105)
     * One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.50, Difficulty Fit (15%): 0.50, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.85
  3. **Stress & Lifestyle Management Workshop** (Score: 0.7955)
     * Group counseling sessions regarding balanced lifestyle, reducing harmful habits, and exam stress relief.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.50, Difficulty Fit (15%): 0.50, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.70

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
    * *Expected Outcome*: Completion of initial practice exercises and reduction in concept gaps.
    * *Educational Rationale*: Prioritizes targeted tasks (peer_tutoring, extra_exercises) to reinforce basic subject mastery.
  * **Week 3 - Theme: Reinforce**
    * *Objective*: Engage in collaborative study and leverage interactive resources to deepen understanding.
    * *Recommended Actions*:
      - Facilitated Study Group: Weekly group discussions focusing on course concepts and collaborative exercises.
      - Stress & Lifestyle Management Workshop: Group counseling sessions regarding balanced lifestyle, reducing harmful habits, and exam stress relief.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (study_group, stress_management) to sustain motivation and learning speed.
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
  - `attendance`: 0.0002 probability
  - `failure_history`: 0.0012 probability
  - `grade_gap`: 0.7218 probability
  - `study_time`: 0.0014 probability
  - `wellbeing`: 0.0984 probability
  - `time_management`: 0.9993 probability

* **Top 3 Recommended Interventions**:
  1. **Remedial Topic Bootcamps** (Score: 0.7451)
     * Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
     * *Score Breakdown*: Risk Match (30%): 0.72, Performance Need (20%): 0.17, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.95
  2. **Time Management Workshop** (Score: 0.6799)
     * Guided sessions on creating weekly study schedules, prioritizing tasks, and minimizing distractions.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.30, Difficulty Fit (15%): 0.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.70
  3. **Stress & Lifestyle Management Workshop** (Score: 0.6799)
     * Group counseling sessions regarding balanced lifestyle, reducing harmful habits, and exam stress relief.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.30, Difficulty Fit (15%): 0.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.70

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Time Management Workshop: Guided sessions on creating weekly study schedules, prioritizing tasks, and minimizing distractions.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (time_planning) to build a stable learning foundation.
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
      - Stress & Lifestyle Management Workshop: Group counseling sessions regarding balanced lifestyle, reducing harmful habits, and exam stress relief.
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (stress_management) to sustain motivation and learning speed.
  * **Week 4 - Theme: Evaluate & Adjust**
    * *Objective*: Pursue advanced topics to challenge capacity and expand skills.
    * *Recommended Actions*:
      - Advanced Subject Seminar: Enrichment seminar focusing on applications and advanced extensions of the course materials.
    * *Expected Outcome*: Completion of an enrichment topic or advanced challenge.
    * *Educational Rationale*: High performance indicates capability to handle advanced challenges.

---

