# PostgreSQL Query Plans

## Evidence lookup

```text
Seq Scan on ml_evidence_bundles  (cost=0.00..1.04 rows=1 width=982) (actual time=0.024..0.024 rows=0 loops=1)
  Filter: ((study_id = 'study_c_oulad'::text) AND (source_commit = 'f7ce7b1c53fc494a9a252014e973d77317f902e6'::bpchar))
  Rows Removed by Filter: 3
  Buffers: shared hit=1
Planning:
  Buffers: shared hit=123
Planning Time: 2.299 ms
Execution Time: 0.045 ms
```

## Prediction reproduction

```text
Hash Join  (cost=1109.15..9785.97 rows=15221 width=68) (actual time=10.139..27.014 rows=15378 loops=1)
  Hash Cond: (p.record_id = sr.record_id)
  Buffers: shared hit=1572
  ->  Bitmap Heap Scan on ml_predictions p  (cost=222.26..8783.01 rows=15221 width=446) (actual time=0.774..3.847 rows=15378 loops=1)
        Recheck Cond: (run_id = '313c40d0-31a9-5f63-97a5-8ad88a7a4210'::uuid)
        Heap Blocks: exact=1026
        Buffers: shared hit=1040
        ->  Bitmap Index Scan on idx_ml_predictions_run  (cost=0.00..218.45 rows=15221 width=0) (actual time=0.589..0.589 rows=15378 loops=1)
              Index Cond: (run_id = '313c40d0-31a9-5f63-97a5-8ad88a7a4210'::uuid)
              Buffers: shared hit=14
  ->  Hash  (cost=689.73..689.73 rows=15773 width=218) (actual time=9.319..9.319 rows=15773 loops=1)
        Buckets: 16384  Batches: 1  Memory Usage: 4005kB
        Buffers: shared hit=532
        ->  Seq Scan on source_records sr  (cost=0.00..689.73 rows=15773 width=218) (actual time=0.018..3.018 rows=15773 loops=1)
              Buffers: shared hit=532
Planning:
  Buffers: shared hit=145 read=1
Planning Time: 4.750 ms
Execution Time: 28.151 ms
```
