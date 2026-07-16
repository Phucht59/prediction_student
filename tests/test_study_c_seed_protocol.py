import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_declared_seed_stability_protocol_has_no_best_seed_selection():
    protocol = json.loads((ROOT / "configs" / "extension_protocol_v1.yaml").read_text(encoding="utf-8"))
    seeds = protocol["study_c"]["seeds"]
    assert seeds == [42, 2026, 3407]
    assert len(set(seeds)) == 3
    assert {"C-H1", "C-H2"}.issubset(protocol["study_c"]["models"])
