"""student-mat V5 study."""

from src.studies.v5.common.uci_runner import run_uci_study


def run(*, device: str = "cuda", force: bool = False):
    return run_uci_study("student-mat", device=device, force=force)


__all__ = ["run"]

