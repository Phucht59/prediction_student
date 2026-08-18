import json
import time
from pathlib import Path
root=Path(__file__).resolve().parent; status=root/'runtime'/'OULAD_NONE_STATUS.json'; log=root/'runtime'/'oulad_none_progress_30min.log'
while True:
 data=json.loads(status.read_text()) if status.exists() else {'status':'PENDING'}
 with log.open('a',encoding='utf-8') as f:f.write(f"STATUS={data['status']} COMPLETED={data.get('completed',0)}/15 CURRENT={data.get('current_run')}\n")
 if data['status'] in ('COMPLETE','FAILED'):break
 time.sleep(1800)
