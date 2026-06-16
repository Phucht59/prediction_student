# BRIEFING — 2026-06-15T15:05:22+07:00

## Mission
Improve the student performance prediction model (CNN-BiLSTM + Context MLP) on three datasets (student-mat, student-por, xapi) to optimize Macro-F1 and Recall for the Low performance group, strictly following the thesis outline and avoiding data leakage.

## 🔒 My Identity
- Archetype: teamwork_preview_orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_v27
- Original parent: main agent
- Original parent conversation ID: c34ee600-689e-46c3-a9a6-c6bf5e7a969c

## 🔒 My Workflow
- **Pattern**: Project Pattern
- **Scope document**: c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_v27\PROJECT_V27.md
1. **Decompose**: Decompose the requirements into structured milestones, including exploration, resampling audit/fix, architecture implementation, hyperparameter tuning, ensembling, ablation study, and final evaluation/reporting.
2. **Dispatch & Execute**:
   - **Delegate (sub-orchestrator)**: If tasks are too large, spawn a sub-orchestrator.
   - **Direct (iteration loop)**: For specific milestones, run the Explorer -> Worker -> Reviewer -> Challenger -> Auditor cycle.
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (sub-orchestrators only, last resort)
4. **Succession**: Self-succeed at 16 spawns. Write handoff.md, spawn successor.
- **Work items**:
  - M1: Exploration and Pipeline Audit [pending]
  - M2: Resampling Fix and Loss/Architecture Implementation [pending]
  - M3: Optuna Hyperparameter & Threshold Tuning [pending]
  - M4: Seed Ensembling & Ablation Study [pending]
  - M5: Evaluation & Final Reporting [pending]
- **Current phase**: 1
- **Current focus**: Exploration and Pipeline Audit

## 🔒 Key Constraints
- Avoid data leakage: locked test must be isolated completely.
- Keep CNN-BiLSTM + Context MLP architecture (no traditional ML algorithms as main models).
- Do not fabricate metrics. If V27 is not better, keep baseline and report V27 as experimental.
- Forensic Auditor verdict must be CLEAN for any iteration gate.
- Do not reuse subagents after handoff.

## Current Parent
- Conversation ID: c34ee600-689e-46c3-a9a6-c6bf5e7a969c
- Updated: not yet

## Key Decisions Made
- [TBD]

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_1 | teamwork_preview_explorer | Milestone 1: Exploration and Pipeline Audit | completed | 0db65ef1-b3b0-45b8-a2e5-4e10daefb216 |
| worker_1 | teamwork_preview_worker | Milestone 2: Resampling Fix and Loss/Architecture Implementation | completed | 369625da-5db3-49c8-9991-d298107f902b |
| worker_2 | teamwork_preview_worker | Milestone 3: Optuna Hyperparameter & Threshold Tuning | completed | a86adcea-657d-4c1f-a4b3-45fb1823ad3f |
| worker_3 | teamwork_preview_worker | Milestone 4: Seed Ensembling & Ablation Study | completed | e78b0451-ad8a-4879-ad48-e72ba8c33b5c |
| auditor_1 | teamwork_preview_auditor | Forensic Integrity Audit | completed | a5b936f3-ac62-4b93-b729-3e61139b2858 |
| worker_4 | teamwork_preview_worker | Milestone 5: Final Report Compilation | completed | bbd624bd-7336-4e3f-b9ac-b65f5498994f |

## Succession Status
- Spawn count: 6 / 16
- Pending subagents: none
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: task-27
- Safety timer: none
- On succession: kill all timers before spawning successor
- On context truncation: run `manage_task(Action="list")` — re-create if missing

## Artifact Index
- c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_v27\BRIEFING.md — Persistent memory
- c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_v27\progress.md — Liveness and checkpoint tracking
- c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_v27\PROJECT_V27.md — Global index for this project
