$ErrorActionPreference = 'Stop'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if (Test-Path (Join-Path $root '..\..\.venv-oulad-v2\Scripts\python.exe')) { Join-Path $root '..\..\.venv-oulad-v2\Scripts\python.exe' } else { (Get-Command py).Source }
$runtime = Join-Path $root 'runtime'; $logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $runtime, $logs | Out-Null
$supervisor = Start-Process -FilePath $python -ArgumentList @((Join-Path $root 'supervisor_none.py')) -WorkingDirectory $root -RedirectStandardOutput (Join-Path $logs 'none_background_stdout.log') -RedirectStandardError (Join-Path $logs 'none_background_stderr.log') -WindowStyle Hidden -PassThru
$supervisor.Id | Set-Content -NoNewline (Join-Path $runtime 'NONE_TRAINING_PID.txt')
$monitor = Start-Process -FilePath $python -ArgumentList @((Join-Path $root 'monitor_30min_none.py')) -WorkingDirectory $root -WindowStyle Hidden -PassThru
$monitor.Id | Set-Content -NoNewline (Join-Path $runtime 'NONE_MONITOR_PID.txt')
