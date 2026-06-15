# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: xapi

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: 0.9885
* **Macro F1**: 0.9906
* **Micro Precision**: 0.9848
* **Macro Precision**: 0.9872
* **Micro Recall**: 0.9923
* **Macro Recall**: 0.9943
* **Hamming Loss**: 0.0104

### Ranking Metrics (at K=3)
* **Precision@3**: 0.8472
* **Recall@3**: 0.6766
* **NDCG@3**: 0.9864
* **Catalog Coverage@3**: 0.8333

### Path Quality Metrics
* **Risk Coverage Rate**: 1.0000
* **Workload Balance (std hours/week)**: 2.0276
* **Difficulty Progression Rate**: 0.5208
* **Prerequisite Violation Rate**: 0.0000

---

## 2. Student Case Studies

### Case Study: High Risk (Struggling) Student (Test Index 1)
* **Student Context**: Raised Hands: 17, Visited Resources: 61, Discussion: 14, Absences: Under-7
* **Predicted Academic Performance**: Class 0 (Low) - Probabilities: [Low: 0.59, Medium: 0.40, High: 0.00]
* **Diagnosed Academic Risks**:
  - `attendance`: 0.0001 probability
  - `resource_usage`: 0.0003 probability
  - `class_engagement`: 0.9995 probability
  - `course_updates`: 0.0003 probability
  - `parent_support`: 0.9994 probability
  - `school_support`: 1.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Academic Counselor Consultation** (Score: 0.9205)
     * One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.68, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.85
  2. **LMS Interactive Quizzing** (Score: 0.9204)
     * Gamified weekly self-assessment quizzes on LMS to boost digital resource engagement.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.68, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.85
  3. **Parent-Teacher Engagement Sync** (Score: 0.9105)
     * Establishing weekly progress reporting channels between school and family to reinforce oversight.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.68, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.75

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Academic Counselor Consultation: One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
      - Parent-Teacher Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (counselor_meeting, parent_sync, attendance_monitoring) to build a stable learning foundation.
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
  - `attendance`: 0.0001 probability
  - `resource_usage`: 0.0001 probability
  - `class_engagement`: 0.0003 probability
  - `course_updates`: 0.0000 probability
  - `parent_support`: 0.9999 probability
  - `school_support`: 1.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Academic Counselor Consultation** (Score: 0.7918)
     * One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.41, Difficulty Fit (15%): 0.50, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.85
  2. **Parent-Teacher Engagement Sync** (Score: 0.7818)
     * Establishing weekly progress reporting channels between school and family to reinforce oversight.
     * *Score Breakdown*: Risk Match (30%): 1.00, Performance Need (20%): 0.41, Difficulty Fit (15%): 0.50, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.75
  3. **Targeted Practice Exercises** (Score: 0.5597)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.35, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.90

* **Generated 4-Week Learning Path**:
  * **Week 1 - Theme: Stabilize**
    * *Objective*: Establish basic academic stability and resolve immediate attendance or support barriers.
    * *Recommended Actions*:
      - Academic Counselor Consultation: One-on-one sessions to address personal wellbeing, academic anxiety, and home/school support alignment.
      - Parent-Teacher Engagement Sync: Establishing weekly progress reporting channels between school and family to reinforce oversight.
      - Daily Attendance Monitoring: Sign-in sheets and weekly academic advisor check-ins to rebuild class attendance consistency.
    * *Expected Outcome*: Regular class attendance established and a structured weekly study schedule created.
    * *Educational Rationale*: Addressed high priority risks (counselor_meeting, parent_sync, attendance_monitoring) to build a stable learning foundation.
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
    * *Expected Outcome*: Active participation in peer groups and increased digital platform engagement.
    * *Educational Rationale*: Uses interactive activities (study_group) to sustain motivation and learning speed.
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
  - `attendance`: 0.0000 probability
  - `resource_usage`: 0.0000 probability
  - `class_engagement`: 0.0019 probability
  - `course_updates`: 0.0000 probability
  - `parent_support`: 0.0000 probability
  - `school_support`: 0.0000 probability

* **Top 3 Recommended Interventions**:
  1. **Advanced Subject Seminar** (Score: 0.6390)
     * Enrichment seminar focusing on applications and advanced extensions of the course materials.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.90, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.60
  2. **Remedial Topic Bootcamps** (Score: 0.5055)
     * Instructor-led intensive review sessions on fundamental subjects to fix cumulative failure histories.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.05, Difficulty Fit (15%): 1.00, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.95
  3. **Targeted Practice Exercises** (Score: 0.4255)
     * Curated weekly practice worksheets focusing on areas with negative grade gaps or failing histories.
     * *Score Breakdown*: Risk Match (30%): 0.00, Performance Need (20%): 0.05, Difficulty Fit (15%): 0.50, Time Fit (15%): 1.00, Prereq Fit (10%): 1.00, Expected Effect (10%): 0.90

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

