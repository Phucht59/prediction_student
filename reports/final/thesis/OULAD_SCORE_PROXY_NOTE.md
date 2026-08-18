# OULAD score-availability proxy caveat

OULAD records an assessment date, a submission date and a score. It does not
record an explicit timestamp for when the marked score was released to the
student or became available to a real-time predictor.

Therefore:

```text
score observed in the database
!=
score provably available at the prediction cutoff
```

Historical H0 used a conservative proxy based on known submission and
assessment dates. This is retained as legacy evidence with a caveat; it is not
described as proven target leakage. The strict H1 endpoint excludes score-
progress values whose release time cannot be defended.
