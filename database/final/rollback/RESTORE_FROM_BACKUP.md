# Restore from the pre-cutover backup

Use a newly created disposable database. Never overwrite or drop the active
database.

```powershell
createdb -T template0 student_predict_restore_test
pg_restore --exit-on-error --no-owner --no-acl `
  --dbname student_predict_restore_test `
  backups/student_predict_before_final_database.dump
```

Compare the restored schema checksum to
`artifacts/final/database/backup_manifest.json` before declaring restore PASS.
