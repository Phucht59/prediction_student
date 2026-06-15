# Weak-Labeling Rules Explanation

This document outlines the domain-driven weak-supervision rules mapping students to six critical academic risks (R1 to R6).

---

## 1. Student Dataset (student-mat / student-por)
Applicable to student academic records (containing absences, study time, fail history, and grades).

### **R1: Attendance Risk (attendance)**
* **Criteria**: `absences >= 10` OR `(absences / studytime) >= 5` (where `studytime` defaults to 1.0, minimum denominator is capped at 0.5).
* **Explanation**: High absenteeism directly correlates with academic decline. If a student's absences exceed 10 sessions or the ratio of absences to study time is excessively high, they are flagged at risk.

### **R2: Failure History Risk (failure_history)**
* **Criteria**: `failures > 0`
* **Explanation**: Any past failures indicate cumulative knowledge gaps that could hinder progress in subsequent terms.

### **R3: Grade Gap Risk (grade_gap)**
* **Criteria**: `G2 < 10` OR `(G1 > 0 and G2 < G1)`
* **Explanation**: A G2 score below passing (<10) or a downward trend between G1 and G2 shows the student is currently struggling or declining academically.

### **R4: Study Time Risk (study_time)**
* **Criteria**: `studytime <= 1`
* **Explanation**: Low self-reported study time (1 hour or less per week) indicates insufficient effort to master the coursework.

### **R5: Wellbeing Risk (wellbeing)**
* **Criteria**: `Dalc + Walc >= 6`
* **Explanation**: High alcohol consumption (daily and weekend scores summed to 6 or more) indicates lifestyle choices affecting academic concentration.

### **R6: Time Management Risk (time_management)**
* **Criteria**: `goout >= 4`
* **Explanation**: High frequency of going out with friends (score 4 or 5 out of 5) suggests poor time management and potential neglect of study schedules.

---

## 2. xAPI Dataset (xapi-eg)
Applicable to student activity/engagement records on the Learning Management System (LMS).

### **R1: Attendance Risk (attendance)**
* **Criteria**: `StudentAbsenceDays == "Above-7"`
* **Explanation**: Students missing more than 7 days of school are at risk of falling behind due to lack of class presence.

### **R2: Resource Usage Risk (resource_usage)**
* **Criteria**: `VisITedResources < 40`
* **Explanation**: Low interactions with course materials (less than 40 clicks/views) indicate poor preparation and low learning engagement.

### **R3: Class Engagement Risk (class_engagement)**
* **Criteria**: `raisedhands < 30` OR `Discussion < 30`
* **Explanation**: Minimal classroom participation (raising hands or participating in discussions less than 30 times) indicates passive learning and potential disinterest.

### **R4: Course Updates Risk (course_updates)**
* **Criteria**: `AnnouncementsView < 30`
* **Explanation**: Checking announcements less than 30 times suggests the student is out-of-sync with important updates, deadlines, and schedule changes.

### **R5: Parent Support Risk (parent_support)**
* **Criteria**: `ParentAnsweringSurvey == "No"`
* **Explanation**: If parents do not answer school surveys, it implies lower parental involvement, which correlates with reduced academic oversight at home.

### **R6: School Support Risk (school_support)**
* **Criteria**: `ParentschoolSatisfaction == "Bad"`
* **Explanation**: Poor parent satisfaction with the school is often linked to student alienation or friction between the school and the family.
