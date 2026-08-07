import json
from pathlib import Path

from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import (
    REQUIRED_SCHEMA_FIELDS,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts/response_schema.json"
)


def test_response_schema_matches_importer_required_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = set(schema["required"])
    assert REQUIRED_SCHEMA_FIELDS <= required
    assert "query_id" not in required


def test_response_schema_requires_nonnullable_provider_ids_and_hashes():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    properties = schema["properties"]

    assert properties["response_id"]["type"] == "string"
    assert properties["response_record_index"]["type"] == "integer"

    for field in (
        "prompt_sha256",
        "request_batch_sha256",
        "raw_request_sha256",
        "raw_response_sha256",
        "response_record_sha256",
    ):
        assert properties[field]["type"] == "string"
        assert properties[field]["pattern"] == "^[a-f0-9]{64}$"


def test_external_schema_does_not_accept_pseudo_or_human_reviewer_type():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["reviewer_type"]["enum"] == [
        "REAL_EXTERNAL_LLM_REVIEW"
    ]
