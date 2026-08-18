"""Fail-closed structural checks for the released thesis repository."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("README.md", "docs/project_map/PROJECT_CODE_MAP.md", "docs/project_map/CONFIG_AUTHORITY.md", "docs/project_map/ARTIFACT_AUTHORITY.md", "configs/final/final_model_authority.yaml", "configs/final/recommendation.yaml", "artifacts/final/recommendation/final_recommendation_registry.json", "scripts/recommend_hybrid/validate_final_evidence_recommender.py", "archive/non_release_research/neural_ranker_diagnostics/README.md", "artifacts/final_release/project_structure_manifest.json")
BAD = re.compile(r"^(?:temp|final_final|new2)(?:[._-]|$)", re.I)

def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    excluded = {".git", ".venv-oulad-v2", "test_lab", "backups", "archive", "node_modules"}
    bad = [p.name for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & excluded) and BAD.search(p.name)]
    registry = json.loads((ROOT / "artifacts/final/recommendation/final_recommendation_registry.json").read_text())
    if registry.get("neural_ranker_artifacts_excluded") is not True: missing.append("final registry neural-ranker exclusion")
    if "NON_RELEASE_RESEARCH_DIAGNOSTIC" not in (ROOT / "archive/non_release_research/neural_ranker_diagnostics/README.md").read_text(): missing.append("diagnostic declaration")
    if missing or bad:
        print("PROJECT_STRUCTURE_CLEAN_FAILED"); print("missing=" + ", ".join(missing) + " bad=" + ", ".join(bad)); return 1
    print("PROJECT_STRUCTURE_CLEAN_PASS"); return 0

if __name__ == "__main__": raise SystemExit(main())
