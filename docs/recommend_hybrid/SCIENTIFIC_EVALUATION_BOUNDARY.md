# Scientific evaluation boundary

The final authority is the Hybrid CNN-BiLSTM Evidence-Based Learning Support
Recommender. It uses frozen Hybrid CNN-BiLSTM prediction outputs together with a
deterministic, evidence-based policy and a safety constraint solver.

Technical evaluation establishes routing, temporal safety, evidence/source lineage,
constraint compliance, serialization, and deterministic replay. It does not estimate
ranking accuracy, educational effectiveness, causal impact, expert agreement, or user
acceptance because independent relevance labels and deployment observations are not
available.

The silver-label neural-ranker investigation is a `NON_RELEASE_RESEARCH_DIAGNOSTIC`.
It is excluded because shortcut-confounded targets do not support a student-specific
ranking claim. No neural-ranker checkpoint or prediction is part of the final registry.
