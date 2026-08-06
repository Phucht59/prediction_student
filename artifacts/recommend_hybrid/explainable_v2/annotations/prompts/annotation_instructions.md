# LLM Annotation Instructions for Student Action Ranking

## Relevance Scale
- **0**: Unsuitable or potential harm (e.g. recommending quiz practice when no quizzes exist).
- **1**: Weakly relevant (generic advice, low specificity).
- **2**: Relevant (direct alignment with observed student behavioral gaps).
- **3**: Highly relevant (urgent action matching specific missing assessment or inactivity streak).

## Rules
- Do NOT assume future student outcome after cutoff.
- Base evaluation strictly on provided pre-cutoff evidence.
- Abstain if evidence is ambiguous or missing.
