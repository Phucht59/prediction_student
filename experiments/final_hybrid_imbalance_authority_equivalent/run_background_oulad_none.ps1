$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$python=Join-Path $root '..\..\.venv-oulad-v2\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'runtime'),(Join-Path $root 'logs')|Out-Null
$p=Start-Process $python -ArgumentList (Join-Path $root 'supervisor_oulad_none.py') -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root 'logs\oulad_none_stdout.log') -RedirectStandardError (Join-Path $root 'logs\oulad_none_stderr.log') -WindowStyle Hidden -PassThru
$p.Id|Set-Content -NoNewline (Join-Path $root 'runtime\OULAD_NONE_PID.txt')
$m=Start-Process $python -ArgumentList (Join-Path $root 'monitor_oulad_none.py') -WorkingDirectory $root -WindowStyle Hidden -PassThru
$m.Id|Set-Content -NoNewline (Join-Path $root 'runtime\OULAD_NONE_MONITOR_PID.txt')
