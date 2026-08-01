# Phase 2 labeling-function audit

Seven deterministic labeling functions implement seven independent families: action applicability, evidence availability, published evidence, prediction risk/uncertainty, human-review safety, UCI state-action fit, and OULAD state-action fit. Their declarations, permitted fields, output domain, version, and source lineage are in `artifacts/recommend_hybrid/scientific_labeling/lf_registry.yaml`.

All functions receive only the fields declared in their contract and return `-1`, `0`, `1`, or `2`. G3, final outcomes, target labels, post-cutoff evidence, and sensitive attributes are absent from candidate records. The Snorkel Label Model consumes only train-split vote matrices.
