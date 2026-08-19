"""Deterministic personalized plans. No LLM. No causal claim."""

from __future__ import annotations

from .contracts import ActionScore, CanonicalAction, RecommendationFeatures, StructuredLearningPlan


def _evidence_lines(features: RecommendationFeatures) -> tuple[str, ...]:
    lines = [f"stage={features.stage.value}", f"course_progress={features.course_progress:.2f}"]
    if features.inactivity_streak is not None:
        lines.append(f"inactivity_streak={features.inactivity_streak}")
    if features.active_day_rate is not None:
        lines.append(f"active_day_rate={features.active_day_rate:.2f}")
    if features.regularity_score is not None:
        lines.append(f"regularity_score={features.regularity_score:.2f}")
    if features.completion_rate is not None:
        lines.append(f"completion_rate={features.completion_rate:.2f}")
    if features.time_to_deadline_days is not None:
        lines.append(f"time_to_deadline_days={features.time_to_deadline_days}")
    if features.due_soon_count:
        lines.append(f"due_soon_count={features.due_soon_count}")
    return tuple(lines)


def build_personalized_plan(top: ActionScore, features: RecommendationFeatures) -> StructuredLearningPlan:
    action = top.action
    evidence = _evidence_lines(features)
    duration = 7
    frequency = "2-3 sessions this week"
    reeval = 7
    if action is CanonicalAction.ASSESSMENT_COMPLETION:
        deadline = features.time_to_deadline_days
        duration = 3 if deadline is not None and deadline <= 7 else 7
        target = (
            f"Submit the open assessment at least 24 hours before the due date ({deadline} days remaining)."
            if deadline is not None
            else "Submit the open assessment at least 24 hours before the due date."
        )
        reason = "There is a missing or soon-due assessment before the next information state."
        what = "Review the brief, finish remaining items, and submit through the VLE."
        safety = "Contact the tutor if a technical submission issue appears."
    elif action is CanonicalAction.RECOVER_ENGAGEMENT:
        streak = features.inactivity_streak or 0
        duration = 5 if streak >= 7 else 7
        frequency = "15-20 minutes daily"
        target = "Record at least 4 distinct VLE activity days in the next 7 days."
        reason = f"Recent engagement is low (inactivity streak {streak} days)."
        what = "Log into the VLE, open the current week's materials, and complete one short activity."
        safety = "If absence is due to personal circumstances, notify an advisor instead of forcing daily load."
    elif action is CanonicalAction.STUDY_REGULARITY:
        duration = 14
        frequency = "3 sessions per week (25-30 minutes)"
        reeval = 14
        target = "Keep the longest gap between study days under 3 days."
        reason = "Study cadence is irregular relative to the pre-cutoff VLE pattern."
        what = "Split work into short fixed sessions instead of one long block."
        safety = "Adjust session length for health and existing workload."
    elif action is CanonicalAction.TARGETED_CONTENT_REVIEW:
        coverage = features.content_coverage
        target = (
            f"Raise content coverage from {coverage:.0%} to at least 80% of current-module materials."
            if coverage is not None
            else "Reach at least 80% coverage of current-module materials."
        )
        reason = "Pre-cutoff content coverage is below the review threshold."
        what = "Re-read the unread content items and write a short summary of each."
        safety = "Prefer understanding over skimming."
    else:
        quiz = features.quiz_activity
        target = "Score at least 70% on a self-check quiz this week."
        reason = (
            f"Retrieval practice is available (recent quiz activity {quiz:.2f})."
            if quiz is not None
            else "A quiz activity is available for retrieval practice."
        )
        what = "Complete a practice quiz and review explanations for incorrect items."
        safety = "Do not treat practice scores as the official grade."
    return StructuredLearningPlan(
        action=action.value,
        reason=reason,
        observed_evidence=evidence,
        what_to_do=what,
        suggested_duration_days=duration,
        suggested_frequency=frequency,
        measurable_target=target,
        reevaluation_time_days=reeval,
        safety_note=safety,
    )
