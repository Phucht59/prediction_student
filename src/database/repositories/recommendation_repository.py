from __future__ import annotations

from psycopg2.extras import Json

from .base import Repository


class RecommendationRepository(Repository):
    def register_policy(self, name: str, version: str, rules: dict, sha256: str, status: str = "draft") -> int:
        return int(
            self.scalar(
                """INSERT INTO recommendation.policy(policy_name,version_label,rules,policy_sha256,status)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (policy_name,version_label) DO UPDATE SET status=EXCLUDED.status
                   RETURNING policy_id""",
                (name, version, Json(rules), sha256, status),
            )
        )

    def create_plan(self, case_id: int, revision_no: int, goal: str, rationale: str, supersedes: int | None = None) -> int:
        return int(
            self.scalar(
                """INSERT INTO recommendation.plan(case_id,revision_no,goal,rationale,status,supersedes_plan_id)
                   VALUES (%s,%s,%s,%s,'draft',%s) RETURNING plan_id""",
                (case_id, revision_no, goal, rationale, supersedes),
            )
        )


__all__ = ["RecommendationRepository"]

