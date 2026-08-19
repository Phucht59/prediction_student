from experiments.hybrid_vnext.baselines import ACTIVE_PHASE4, make_model
from experiments.hybrid_vnext.model import VNextHybrid, assert_c0_topology, availability_unit_cases, make_c0_config
from experiments.hybrid_vnext.phase4_common import ACTIVE_FAMILIES, OULAD_STATES, SHARED_STRUCTURAL, UCI_STATES


def test_active_roster_has_svm_no_xgb():
    assert ACTIVE_FAMILIES == ("LR", "DT", "RF", "SVM", "MLP")
    assert ACTIVE_PHASE4 == ACTIVE_FAMILIES
    assert "XGB" not in ACTIVE_FAMILIES


def test_make_model_rejects_xgb():
    try:
        make_model("XGB", 42, None)
    except ValueError as exc:
        assert "XGB" in str(exc)
    else:
        raise AssertionError("XGB must not be constructible as an active baseline")


def test_one_model_states():
    assert UCI_STATES == ("S0", "S1", "S2")
    assert OULAD_STATES == ("20pct", "35pct", "50pct", "75pct", "100pct")


def test_shared_structural_and_availability():
    cfg = make_c0_config(8, 4, 5, 12, **SHARED_STRUCTURAL)
    assert_c0_topology(cfg)
    assert cfg.d_fuse == SHARED_STRUCTURAL["d_fuse"]
    cases = availability_unit_cases(VNextHybrid(cfg))
    assert all(c["pass"] for c in cases)
    s0 = next(c for c in cases if c["temporal"] == 0)
    assert s0["cnn_mass"] == 0 and s0["lstm_mass"] == 0
    temp_no_agg = next(c for c in cases if c["temporal"] == 1 and c["aggregate"] == 0)
    assert temp_no_agg["lstm_mass"] > 0 or temp_no_agg["cnn_mass"] > 0
