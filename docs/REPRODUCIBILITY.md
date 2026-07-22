# Reproducibility

Run from the repository root:

```powershell
python project.py final status
python project.py final report
python project.py final validate
```

`status` reads canonical state. `report` regenerates Markdown from canonical JSON. `validate` rebuilds the expected payload in memory from frozen evidence, checks every metric source/checksum, verifies CSV/report synchronization, model order, per-class identities, Top-k provenance, Future OULAD lock, expert status and the local laboratory ignore policy. None of these commands trains a model.
