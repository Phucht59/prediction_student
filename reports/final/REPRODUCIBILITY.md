# Reproducibility

Run:

```powershell
python project.py final status
python project.py final report
python project.py final validate
pytest
```

The release validates canonical tables, source checksums, 65 final ensemble
checkpoints, recommendation replay and Future OULAD lock without training.
`test_lab/` is ignored and is never required for prediction, validation,
checksum replay or thesis evidence.
