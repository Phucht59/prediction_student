# One Hybrid, one parameter spec

Scientific rule for this thesis: **do not multiply models**.

1. **One architecture** — Hybrid CNN ∥ BiLSTM, `architecture_id=C0`, same fusion and widths on UCI and OULAD.
2. **One training family** — `L1_control`.
3. **One frozen numeric spec** — `artifacts/prediction/final/TRAINING_CONFIG.json`. UCI/OULAD lr–batch–dropout differ only by dataset scale, already locked in Phase 4. That is not a search.
4. **One fitted instance per dataset** — UCI weights and OULAD weights share the class; they are not different topologies.
5. **Information levels are views**, not models (S0/S1/S2 and 20/35/50/75/100).
6. **CV folds and seeds evaluate the spec**; they do not create candidates to cherry-pick.
7. **S0 and OULAD 20% being slightly worse is accepted.** Do not invent a second Hybrid to win those two cells.

Anything that would produce “Hybrid-A / Hybrid-B / Hybrid-early / Hybrid-S0” is out of protocol.
