"""Dataset contracts and preprocessing safety checks."""

from .contracts import OULADInputContract, UCIInputContract, assert_train_only_fit

__all__ = ["OULADInputContract", "UCIInputContract", "assert_train_only_fit"]
