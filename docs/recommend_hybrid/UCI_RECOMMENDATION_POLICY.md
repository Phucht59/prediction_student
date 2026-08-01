# UCI recommendation policy

## Scope and routing

The UCI branch supports `student_mat` and `student_por` through common code and separate configs. S0 is valid only when G1/G2 are known not yet available; S1 requires G1 without G2; S2 requires both. A percentage request without known assessment availability returns `INSUFFICIENT_STAGE_EVIDENCE`. G2 without G1 also abstains. G3, final outcome, test and outer labels are rejected.

## Evidence

Eligible academic evidence includes stage-valid G1/G2, grade decline/improvement, absences, study time, prior failures and whether another assessment is ahead. Missing G1 cannot generate a G1 reason; missing G2 cannot generate a G1→G2 trend. All values retain stage/cutoff/source lineage.

## Actions

The branch declares `STUDY_SCHEDULE`, `ATTENDANCE_IMPROVEMENT`, `TARGETED_REVISION`, `PRACTICE_EXERCISES`, `ASSESSMENT_PREPARATION`, `INSTRUCTOR_CONTACT`, `ADVISOR_SUPPORT`, `PROGRESS_MONITORING`, and `LEARNING_CONSOLIDATION`. VLE-only actions are prohibited.

Worsening absence or study-time evidence cannot reduce its related action priority. Low grade plus a future assessment may support preparation. Grade improvement cannot increase escalation. MAT and POR distributions/thresholds are never mixed.
