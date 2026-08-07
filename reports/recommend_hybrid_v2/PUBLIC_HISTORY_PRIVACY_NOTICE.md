# Public History Privacy Notice — Recommendation V2

## Overview
This document records an adversarial scientific audit finding regarding historical Git commits in the `Module_recomend` branch.

## Audit Findings
1. **Historical Unblinded Artifacts**:
   - In commit `884bf01ec1da8c2fb4472d1cbac997ede09edf36`, `private_case_mapping.json` was committed to the Git tree under `artifacts/recommend_hybrid/explainable_v2/annotations/private/`.
   - This file contained private mapping linkages between blinded `case_id` hashes and raw OULAD identifiers (`source_query_id`, `source_student_group_id`, `outer_fold`).

2. **Remediation Action**:
   - `private_case_mapping.json` has been un-tracked from the active Git index (`git rm --cached`).
   - `.gitignore` has been updated with explicit rules (`artifacts/recommend_hybrid/explainable_v2/annotations/private/*`).
   - The file now exists ONLY as an un-tracked local runtime file generated dynamically on exporter execution.

3. **Git History Notice**:
   - Git history has NOT been force-rewritten in this commit to preserve audit provenance.
   - The repository MUST NOT be made public or merged until an official decision is made regarding Git history squashing/cleaning.
