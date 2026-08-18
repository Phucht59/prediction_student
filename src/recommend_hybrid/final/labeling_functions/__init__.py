"""Deterministic, abstaining weak-label interfaces; no fabricated LLM labels."""
from .core import ABSTAIN, LabelVote, literature_vote, behavior_vote, feasibility_vote

__all__ = ["ABSTAIN", "LabelVote", "literature_vote", "behavior_vote", "feasibility_vote"]
