# Database Final Schema

The locked final design uses four active schemas:

```text
system
catalog
ml
recommendation
```

It has exactly 16 core tables and two views. `public` is removed from the
application search path and contains no application tables after cutover.

| Schema | Base tables |
|---|---:|
| system | 1 |
| catalog | 3 |
| ml | 4 |
| recommendation | 8 |
| **Total** | **13** |

Two immutability triggers protect sealed dataset versions and completed final
runs. Non-primary indexes are capped at 20 and must have a documented query.

The authoritative contract is
[`database/final/FINAL_SCHEMA_CONTRACT.md`](../../database/final/FINAL_SCHEMA_CONTRACT.md).
The complete 29-table mapping is
[`database/final/LEGACY_TO_FINAL_MAPPING.yaml`](../../database/final/LEGACY_TO_FINAL_MAPPING.yaml).
