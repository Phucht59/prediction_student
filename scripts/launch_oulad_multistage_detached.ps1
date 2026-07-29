$ErrorActionPreference = "Stop"
$Repo = "C:\hufit\kltn"
$ScriptPath = Join-Path $Repo "scripts\run_oulad_multistage_detached.ps1"
$Root = Join-Path $Repo "logs\oulad_multistage\detached"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

$statusPath = Join-Path $Root "status.json"
$wrapperPidPath = Join-Path $Root "wrapper.pid"
$launcherPidPath = Join-Path $Root "launcher.pid"
$lockPath = Join-Path $Root "job.lock"
$priorStatus = if (Test-Path -LiteralPath $statusPath) {
    Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
} else {
    $null
}
$priorWrapperPid = if (Test-Path -LiteralPath $wrapperPidPath) {
    [int](Get-Content -LiteralPath $wrapperPidPath -Raw).Trim()
} elseif ($priorStatus -and $priorStatus.wrapper_pid) {
    [int]$priorStatus.wrapper_pid
} else {
    0
}
$priorLauncherPid = if (Test-Path -LiteralPath $launcherPidPath) {
    [int](Get-Content -LiteralPath $launcherPidPath -Raw).Trim()
} else {
    0
}
$wrapperAlive = $priorWrapperPid -gt 0 -and
    $null -ne (Get-Process -Id $priorWrapperPid -ErrorAction SilentlyContinue)
if ($wrapperAlive) {
    throw "An active detached OULAD wrapper already exists (PID $priorWrapperPid)"
}

$oldStatusChecksum = if (Test-Path -LiteralPath $statusPath) {
    (Get-FileHash -LiteralPath $statusPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else {
    $null
}
$checkpointRoot = Join-Path $Repo "artifacts\final\unified_stage_aware_oulad\checkpoints"
$checkpointCounts = [ordered]@{}
if (Test-Path -LiteralPath $checkpointRoot) {
    Get-ChildItem -LiteralPath $checkpointRoot -Directory | Sort-Object Name | ForEach-Object {
        $checkpointCounts[$_.Name] = @(
            Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Include *.joblib,*.pt
        ).Count
    }
}
$archiveRoot = Join-Path $Repo "artifacts\history\partial_svm_probability_true_20260729"
$timestamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$recoveryPath = Join-Path $Root "recovery_$timestamp.json"
$staleAction = if (Test-Path -LiteralPath $lockPath) {
    "REMOVED_STALE_LOCK"
} else {
    "NO_LOCK_PRESENT"
}
[ordered]@{
    schema_version = "oulad_detached_recovery_v1"
    recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    prior_state = if ($priorStatus) { $priorStatus.state } else { $null }
    prior_step = if ($priorStatus) { $priorStatus.step } else { $null }
    prior_wrapper_pid = $priorWrapperPid
    prior_launcher_pid = $priorLauncherPid
    wrapper_pid_alive = $wrapperAlive
    stale_lock_action = $staleAction
    old_status_sha256 = $oldStatusChecksum
    archive_state = [ordered]@{
        path = $archiveRoot
        checkpoint_count = @(
            Get-ChildItem -LiteralPath $archiveRoot -Recurse -Filter *.joblib -ErrorAction SilentlyContinue
        ).Count
        readme_present = Test-Path -LiteralPath (Join-Path $archiveRoot "README.txt")
        manifest_present = Test-Path -LiteralPath (Join-Path $archiveRoot "manifest.json")
    }
    checkpoint_counts = $checkpointCounts
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recoveryPath -Encoding UTF8

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
    recovery_record = $recoveryPath
} | ConvertTo-Json
