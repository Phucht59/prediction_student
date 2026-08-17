"""Frozen Phase 6E shared-head factorial definitions."""
from __future__ import annotations
from src.hybrid.models import SharedHeadConfig,SharedHeadHybrid

CAPACITIES={
 "C1":{"d_model":64,"cnn_channels":96,"cnn_blocks":2,"bilstm_hidden":96,"bilstm_layers":1,"context_hidden":64,"head_hidden":128},
 "C2":{"d_model":96,"cnn_channels":128,"cnn_blocks":2,"bilstm_hidden":128,"bilstm_layers":1,"context_hidden":96,"head_hidden":128},
 "C3":{"d_model":160,"cnn_channels":128,"cnn_blocks":3,"bilstm_hidden":160,"bilstm_layers":1,"context_hidden":128,"head_hidden":128},
}
LOSSES={"L1":{"class_weight_mode":"sqrt","lambda_rank":.25},"L2":{"class_weight_mode":"full","lambda_rank":0.}}
CANDIDATES={f"E{index}":{"capacity":capacity,"loss":loss} for index,(capacity,loss) in enumerate((("C1","L1"),("C1","L2"),("C2","L1"),("C2","L2"),("C3","L1"),("C3","L2")),1)}

def candidate_params(base,candidate):
    spec=CANDIDATES[candidate];params=dict(base);params.update(CAPACITIES[spec["capacity"]]);params.update(LOSSES[spec["loss"]]);params["shared_head_hidden"]=params.pop("head_hidden");params["ema_decay"]=None;params["uci_wide_context"]=False
    return params
def shared_config(domain,temporal_dim,context_dim,params):
    if domain!="uci":raise ValueError("Phase6E is UCI-only")
    return SharedHeadConfig(temporal_dim,context_dim,params["d_model"],params["cnn_channels"],params["cnn_blocks"],params["bilstm_hidden"],params["bilstm_layers"],params["context_hidden"],params["shared_head_hidden"],params["dropout"])
def parameter_count(temporal_dim,context_dim,params):return sum(value.numel() for value in SharedHeadHybrid(shared_config("uci",temporal_dim,context_dim,params)).parameters())
