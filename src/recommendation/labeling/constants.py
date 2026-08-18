"""Frozen, provider-neutral Phase 5 labeling contract."""

PROMPT_VERSION = "recommendation_label_v1"
PROMPT_VERSION_B = "recommendation_label_v1b"
A4_REPLACEMENT_PROMPT_VERSION = "recommendation_a4_replacement_v1"
A4_PROGRESS_GEMMA_PROMPT_VERSION = "recommendation_progress_monitoring_gemma_v1"
A4_PROGRESS_GEMINI31_PROMPT_VERSION = "recommendation_progress_monitoring_gemini31_v1"
A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION = "recommendation_academic_help_seeking_gemma_v1"
SCHEMA_VERSION = "recommendation.labeling.v1"
ACTION_IDS = ("A1", "A2", "A3", "A4", "A5")
FEASIBILITY_STATUSES = ("FEASIBLE", "INFEASIBLE", "UNKNOWN")
LABEL_VALUES = (0, 1, 2, 3, "ABSTAIN")
CONFIDENCE_VALUES = ("LOW", "MEDIUM", "HIGH")
ABSTAIN_REASONS = ("INFEASIBLE", "INSUFFICIENT_INFORMATION")

ACTION_DEFINITIONS = {
    "A1": "Assessment Recovery: prioritize completing or recovering missing or incomplete assessments.",
    "A2": "Re-engagement: encourage returning to interaction with the learning environment when engagement has reduced or stopped.",
    "A3": "Study Planning: improve study rhythm and organize a more regular learning plan.",
    "A4": "Content Review: review learning content that needs reinforcement.",
    "A5": "Retrieval Practice: practice recalling knowledge through quizzes, self-tests, or retrieval activities.",
}

RUBRIC = {
    "0": "NOT_RELEVANT",
    "1": "SLIGHTLY_RELEVANT",
    "2": "RELEVANT",
    "3": "HIGHLY_RELEVANT / PRIORITY",
    "ABSTAIN": "information is insufficient to assess relevance",
}
