from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
def sha256(path):
 h=hashlib.sha256();
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
 return h.hexdigest()
def write_json(path, value):
 atomic_write_json(path, value)

def atomic_write_json(path, value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 fd, temporary=tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as handle:
   json.dump(value,handle,indent=2,sort_keys=True); handle.flush(); os.fsync(handle.fileno())
  os.replace(temporary,path)
 finally:
  if os.path.exists(temporary): os.unlink(temporary)
