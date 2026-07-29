# OULAD Multi-stage Cutoff Audit

PASS. M1 uses the legacy `floor(length * 0.50)` F2 definition. E1/E2/L1 use 20%/35%/75%; L1 retains a 14-day outcome guard. Events use `date < cutoff_day`. Assessment scores are excluded because raw OULAD supplies no score-release timestamp.
