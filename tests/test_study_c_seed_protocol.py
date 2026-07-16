from scripts.run_study_c_seed_stability import NEURAL, SEEDS


def test_declared_seed_stability_protocol_has_no_best_seed_selection():
    assert SEEDS == [42, 2026, 3407]
    assert len(set(SEEDS)) == 3
    assert NEURAL == ["C-H1", "C-H2"]
