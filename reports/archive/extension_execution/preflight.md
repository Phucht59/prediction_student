# Extension execution preflight

- Initial code state: `main` at `a8c36a48e2f2b98b51a0ee65e98e7b541218c38b`; clean working tree.
- Execution branch: `feature/study-b-oulad-extension`.
- Project runtime: Python 3.10.8. The shell-default Python 3.14 is not used because it has no ML dependencies.
- PyTorch: 2.12.0 CPU build. The host has a GTX 1650 4 GB, but CUDA is unavailable in this environment.
- RAM: 16.47 GB; free C: drive space at preflight: 137.28 GB.
- PostgreSQL TCP port 5432 is reachable. Destructive migration behavior will not be tested without a disposable DSN.
- Required local raw data: both UCI files and all seven OULAD tables found.
- Baseline Study A suite: `151 passed, 5 skipped, 0 failed` in 11.91 seconds.

Preflight status: **PASS_WITH_CPU_ONLY_CONSTRAINT**. Trial budgets remain maxima and the wall-clock controller must stop opening long trials before the configured limit.
