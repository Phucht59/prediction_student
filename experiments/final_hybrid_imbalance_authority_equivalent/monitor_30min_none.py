"""Status-file-only monitor; it never imports model or GPU libraries."""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
status = ROOT / "runtime" / "NONE_STATUS.json"
log = ROOT / "runtime" / "none_progress_30min.log"
while True:
    data = json.loads(status.read_text(encoding="utf-8")) if status.is_file() else {"status": "PENDING", "completed": 0, "failed": 0}
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\nSTATUS={data['status']}\nCOMPLETED={data.get('completed', 0)}/50\nFAILED={data.get('failed', 0)}\nCURRENT={data.get('current_run')}\nLAST_COMPLETED={data.get('last_completed_run')}\n----------------------------------------\n")
    if data["status"] in {"COMPLETE", "FAILED"}: break
    time.sleep(1800)
