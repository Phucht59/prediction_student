from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationResult:
    status: str
    risk_delta: float | None
    causal_claim_allowed: bool = False
    runtime_authorized: bool = False

def validate_empirical_support(support_rate: float) -> None:
    if not 0.0 <= support_rate <= 1.0: raise ValueError("support rate must be in [0,1]")
    if support_rate == 0.0: raise RuntimeError("simulation scenario has no empirical support")
