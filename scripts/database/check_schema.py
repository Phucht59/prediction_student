import json
from src.database.schema import check_database_schema
report=check_database_schema(); print(json.dumps(report,indent=2)); raise SystemExit(0 if report["ok"] else 1)
