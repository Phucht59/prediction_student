# Cutoff routing policy

## UCI

UCI routing uses assessment availability rather than course percentage: S0 has no G1, S1 has G1 only, S2 has G1/G2. If availability is unknown, routing abstains instead of inferring a stage. UCI does not report prediction age because assessment-stage ordinals and client percentages are not commensurate.

## OULAD

Validated anchors are 20, 35, 50, 75 and 100 percent. For a request `r`, the router selects `max(anchor ≤ r)`. Thus 25→20, 34→20, 36→35, 49→35, 63→50 and 76→75. Future anchors are prohibited.

Prediction anchor and observed-evidence cutoff are distinct. For a 25% request, the prediction is validated at 20% while evidence may be observed strictly before 25%. `prediction_age = requested_cutoff - anchor_cutoff` is lineage only.

Requests below 20% abstain; exactly 100% is final evaluation only; negative, non-finite or above-100 requests are rejected.
