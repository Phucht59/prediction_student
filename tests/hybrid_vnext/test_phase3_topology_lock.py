from experiments.hybrid_vnext.model import assert_c0_topology, make_c0_config, availability_unit_cases, VNextHybrid
from experiments.hybrid_vnext.phase3_common import TOPOLOGY_SPEC, topology_hash


def test_c0_topology_identity():
    cfg = make_c0_config(8, 4, 5, 12, d_fuse=96, cnn_channels=96, bilstm_hidden=96)
    assert_c0_topology(cfg)
    assert cfg.temporal_path == "parallel"
    assert cfg.fusion == "softmax_3way"
    assert TOPOLOGY_SPEC["public_model_class"] == "Hybrid"
    assert topology_hash()


def test_availability_s0_and_aggregate_independence():
    model = VNextHybrid(make_c0_config(8, 4, 5, 12))
    results = availability_unit_cases(model)
    assert all(item["pass"] for item in results)
    s0 = next(item for item in results if item["temporal"] == 0 and item["aggregate"] == 1)
    assert s0["cnn_mass"] == 0 and s0["lstm_mass"] == 0
    both_temp = next(item for item in results if item["temporal"] == 1 and item["aggregate"] == 0)
    assert both_temp["lstm_mass"] > 0 or both_temp["cnn_mass"] > 0
