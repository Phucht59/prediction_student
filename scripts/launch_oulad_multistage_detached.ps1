$ErrorActionPreference = "Stop"
$Repo = "C:\hufit\kltn"
$ScriptPath = Join-Path $Repo "scripts\run_oulad_multistage_detached.ps1"
$Root = Join-Path $Repo "logs\oulad_multistage\detached"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

@(
    "DONE.json",
    "FAILED.json",
    "CANCELLED.json",
    "cancel.request",
    "job.lock",
    "status.json",
    "wrapper.pid",
    "child.pid"
) | ForEach-Object {
    $target = Join-Path $Root $_
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Force
    }
}

$launcherOut = Join-Path $Root "launcher.stdout.log"
$launcherErr = Join-Path $Root "launcher.stderr.log"
$process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "`"$ScriptPath`""
    ) `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $launcherOut `
    -RedirectStandardError $launcherErr `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath (Join-Path $Root "launcher.pid") -Value $process.Id

[void]([System.Threading.ManualResetEvent]::new($false).WaitOne(5000))
$alive = $null -ne (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)
$wrapperPidPath = Join-Path $Root "wrapper.pid"
$statusPath = Join-Path $Root "status.json"
$wrapperPid = if (Test-Path -LiteralPath $wrapperPidPath) {
    (Get-Content -LiteralPath $wrapperPidPath -Raw).Trim()
} else {
    $null
}
$status = if (Test-Path -LiteralPath $statusPath) {
    Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
} else {
    $null
}
[ordered]@{
    launcher_pid = $process.Id
    launcher_alive = $alive
    wrapper_pid = $wrapperPid
    status_state = if ($status) { $status.state } else { $null }
    status_step = if ($status) { $status.step } else { $null }
    status_file = $statusPath
    launcher_stdout = $launcherOut
    launcher_stderr = $launcherErr
} | ConvertTo-Json
