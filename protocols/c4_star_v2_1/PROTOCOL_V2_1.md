# PROTOCOL_V2_1 — C4-STAR

Parent: `hybrid_superiority_v2.0` hash `eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2`.

v2.1 hash is produced by `experiments.c4_star.protocol.protocol_hash()` and frozen in `protocol_v2_1.json` before HPO.

## Amendment

Candidate selection is **joint-domain**. UCI tests short-prefix (T≤2). OULAD tests long-sequence capacity (T up to 39). Ranking C0–C3 on UCI alone is not allowed.

Outer splits are **not** regenerated.

## Integrity

- Primary metric: AP (`sklearn.metrics.average_precision_score`)
- G3 never a feature; G1/G2 Hybrid temporal-only
- OULAD events strictly before cutoff
- FIT-only scaling / teacher OOF
- SPEED_FINISH is not confirmatory
- Outer test closed until development gate
- DT timeout 90s
- GPU software thermal cap 80°C
- Gemini not used in prediction HPO
