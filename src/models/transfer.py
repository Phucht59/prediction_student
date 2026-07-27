"""Transfer-learning metadata for CNN-BiLSTM MAT."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferStrategy:
    shared_trunk: bool = True
    subject_specific_heads: bool = True
    selection_scope: str = "inner_validation_only"
