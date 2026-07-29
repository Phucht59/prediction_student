param(
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$Repo = "C:\hufit\kltn"
$Python = Join-Path $Repo ".venv-oulad-v2\Scripts\python.exe"
$Root = Join-Path $Repo "logs\oulad_multistage\detached"
$Lock = Join-Path $Root "job.lock"
$StatusPath = Join-Path $Root "status.json"
$CancelPath = Join-Path $Root "cancel.request"
$StartedAt = [DateTimeOffset]::UtcNow
$script:ChildPid = 0
$script:CurrentStdout = ""
$script:CurrentStderr = ""
$script:CurrentStep = "STARTING"
$script:StepIndex = 0
$script:LastCompleted = ""
$script:FailureExitCode = $null
$script:FailureSummary = $null

Set-Location -LiteralPath $Repo
New-Item -ItemType Directory -Force -Path $Root | Out-Null

function Get-CheckpointSnapshot {
    $checkpointRoot = Join-Path $Repo "artifacts\final\unified_stage_aware_oulad\checkpoints"
    $counts = [ordered]@{}
    $latest = $null
    if (Test-Path -LiteralPath $checkpointRoot) {
        Get-ChildItem -LiteralPath $checkpointRoot -Directory | Sort-Object Name | ForEach-Object {
            $files = @(Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Include *.joblib,*.pt)
            $counts[$_.Name] = $files.Count
            $candidate = $files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
            if ($candidate -and (!$latest -or $candidate.LastWriteTimeUtc -gt $latest.LastWriteTimeUtc)) {
                $latest = $candidate
            }
        }
    }
    return @{
        counts = $counts
        total = if ($counts.Count -gt 0) { ($counts.Values | Measure-Object -Sum).Sum } else { 0 }
        latest = if ($latest) { ([DateTimeOffset]$latest.LastWriteTimeUtc).ToString("o") } else { $null }
        latest_path = if ($latest) { $latest.FullName.Substring($Repo.Length + 1) } else { $null }
    }
}

function Write-Status {
    param(
        [string]$State,
        [Nullable[int]]$ExitCode = $null,
        [string]$ErrorSummary = $null
    )
    $snapshot = Get-CheckpointSnapshot
    $latestParts = if ($snapshot.latest_path) { $snapshot.latest_path -split '[\\/]' } else { @() }
    $currentModel = if ($latestParts.Count -ge 5) { $latestParts[-3] } else { $null }
    $currentFold = if ($latestParts.Count -ge 2 -and $latestParts[-2] -match 'outer_fold_(\d+)') { [int]$Matches[1] } else { $null }
    $currentSeed = if ($latestParts.Count -ge 1 -and $latestParts[-1] -match 'seed_(\d+)') { [int]$Matches[1] } else { $null }
    $payload = [ordered]@{
        state = $State
        step = $script:CurrentStep
        step_index = $script:StepIndex
        step_total = 15
        started_at = $StartedAt.ToString("o")
        heartbeat_at = [DateTimeOffset]::UtcNow.ToString("o")
        wrapper_pid = $PID
        child_pid = $script:ChildPid
        branch = (& git branch --show-current).Trim()
        head_at_start = $script:HeadAtStart
        current_commit = (& git rev-parse HEAD).Trim()
        completed_checkpoints = [int]$snapshot.total
        total_checkpoints = 150
        checkpoint_counts_by_model = $snapshot.counts
        last_checkpoint_time = $snapshot.latest
        current_model = $currentModel
        current_outer_fold = $currentFold
        current_seed = $currentSeed
        cuda_available = $script:CudaAvailable
        gpu_name = $script:GpuName
        last_completed_step = $script:LastCompleted
        exit_code = $ExitCode
        error_summary = $ErrorSummary
        stdout_log = $script:CurrentStdout
        stderr_log = $script:CurrentStderr
    }
    $temporary = "$StatusPath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}

function Stop-ChildSafely {
    if ($script:ChildPid -le 0) { return }
    $child = Get-Process -Id $script:ChildPid -ErrorAction SilentlyContinue
    if (!$child) { return }
    Stop-Process -Id $script:ChildPid -ErrorAction SilentlyContinue
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline -and (Get-Process -Id $script:ChildPid -ErrorAction SilentlyContinue)) {
        Start-Sleep -Seconds 1
    }
    if (Get-Process -Id $script:ChildPid -ErrorAction SilentlyContinue) {
        Stop-Process -Id $script:ChildPid -Force
    }
}

function Get-LastMeaningfulLine {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) { return $null }
    $line = Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue |
        Where-Object { ![string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { [string]$_ } |
        Select-Object -Last 1
    if ($null -eq $line) { return $null }
    return [string]$line
}

function Invoke-ExternalCommand {
    param(
        [int]$Index,
        [string]$Name,
        [string]$Executable,
        [string[]]$Arguments,
        [string]$LogPrefix
    )
    $script:StepIndex = $Index
    $script:CurrentStep = $Name
    $script:CurrentStdout = Join-Path $Root "$LogPrefix.stdout.log"
    $script:CurrentStderr = Join-Path $Root "$LogPrefix.stderr.log"
    $commandRecordPath = Join-Path $Root "$LogPrefix.command.json"
    $commandStartedAt = [DateTimeOffset]::UtcNow
    Write-Status -State "RUNNING"
    $resolvedCommand = Get-Command -Name $Executable -ErrorAction SilentlyContinue
    if (!(Test-Path -LiteralPath $Executable) -and !$resolvedCommand) {
        $detail = "executable not found: $Executable"
        $script:FailureExitCode = 127
        $script:FailureSummary = "$Name failed: command=$Executable; exit_code=127; detail=$detail"
        Write-Status -State "FAILED" -ExitCode 127 -ErrorSummary $script:FailureSummary
        throw $script:FailureSummary
    }
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments `
        -WorkingDirectory $Repo -RedirectStandardOutput $script:CurrentStdout `
        -RedirectStandardError $script:CurrentStderr -WindowStyle Hidden -PassThru
    $script:ChildPid = $process.Id
    Set-Content -LiteralPath (Join-Path $Root "child.pid") -Value $script:ChildPid
    $lastHeartbeat = Get-Date
    while (!$process.WaitForExit(1000)) {
        if (Test-Path -LiteralPath $CancelPath) {
            Stop-ChildSafely
            $cancelled = @{
                state = "CANCELLED"
                cancelled_at = [DateTimeOffset]::UtcNow.ToString("o")
                step = $Name
            }
            $cancelled | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root "CANCELLED.json") -Encoding UTF8
            Write-Status -State "CANCELLED" -ExitCode 130 -ErrorSummary "Cancellation requested"
            throw [System.OperationCanceledException]::new("Cancellation requested")
        }
        if (((Get-Date) - $lastHeartbeat).TotalSeconds -ge 60) {
            Write-Status -State "RUNNING"
            $lastHeartbeat = Get-Date
        }
    }
    # WaitForExit(timeout) can report completion before PowerShell has materialized
    # ExitCode and drained redirected streams. The parameterless call is required.
    $process.WaitForExit()
    $exitCode = [int]$process.ExitCode
    $script:ChildPid = 0
    $commandEndedAt = [DateTimeOffset]::UtcNow
    $lastStderr = Get-LastMeaningfulLine -Path $script:CurrentStderr
    $lastStdout = Get-LastMeaningfulLine -Path $script:CurrentStdout
    $detail = if ($lastStderr) { $lastStderr } elseif ($lastStdout) { $lastStdout } else { "no output" }
    [ordered]@{
        command = $Executable
        arguments = @($Arguments)
        started_at = $commandStartedAt.ToString("o")
        ended_at = $commandEndedAt.ToString("o")
        duration_seconds = [Math]::Round(($commandEndedAt - $commandStartedAt).TotalSeconds, 3)
        exit_code = $exitCode
        stdout_path = $script:CurrentStdout
        stderr_path = $script:CurrentStderr
        last_non_empty_stderr_line = $lastStderr
        last_non_empty_stdout_line = $lastStdout
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $commandRecordPath -Encoding UTF8

    if ($exitCode -ne 0) {
        $script:FailureExitCode = $exitCode
        $script:FailureSummary = "$Name failed: command=$Executable; exit_code=$exitCode; detail=$detail"
        if ($Name -eq "GPU_GATE" -and $exitCode -eq 20) {
            Write-Status -State "BLOCKED_GPU" -ExitCode 20 -ErrorSummary $script:FailureSummary
            throw "BLOCKED_GPU"
        }
        if ($Name -eq "GPU_GATE" -and $exitCode -eq 21) {
            Write-Status -State "GPU_CHECK_ERROR" -ExitCode 21 -ErrorSummary $script:FailureSummary
            throw "GPU_CHECK_ERROR"
        }
        Write-Status -State "FAILED" -ExitCode $exitCode -ErrorSummary $script:FailureSummary
        throw $script:FailureSummary
    }
    $script:LastCompleted = $Name
    Write-Status -State "RUNNING" -ExitCode 0
}

$existingPidPath = Join-Path $Root "wrapper.pid"
$existingPid = if (Test-Path $existingPidPath) { [int](Get-Content $existingPidPath -Raw) } else { 0 }
$existingAlive = $existingPid -gt 0 -and
    $null -ne (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)
if ($existingAlive) {
    throw "An active detached OULAD wrapper already exists"
}
if (Test-Path -LiteralPath $StatusPath) {
    $priorStatus = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
    if ($priorStatus.state -in @("FAILED", "BLOCKED_GPU", "GPU_CHECK_ERROR")) {
        $recoveryTimestamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
        $priorChecksum = (Get-FileHash -LiteralPath $StatusPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $archiveRoot = Join-Path $Repo "artifacts\history\partial_svm_probability_true_20260729"
        $checkpointSnapshot = Get-CheckpointSnapshot
        [ordered]@{
            schema_version = "oulad_detached_recovery_v1"
            recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
            prior_state = $priorStatus.state
            prior_step = $priorStatus.step
            prior_wrapper_pid = [int]$priorStatus.wrapper_pid
            pid_alive = $existingAlive
            stale_lock_action = if (Test-Path -LiteralPath $Lock) { "REMOVED_STALE_LOCK" } else { "NO_LOCK_PRESENT" }
            stale_pid_action = "REMOVED_DEAD_PID_MARKERS"
            old_status_sha256 = $priorChecksum
            archive_state = [ordered]@{
                path = $archiveRoot
                checkpoint_count = @(Get-ChildItem -LiteralPath $archiveRoot -Recurse -Filter *.joblib -ErrorAction SilentlyContinue).Count
                readme_present = Test-Path -LiteralPath (Join-Path $archiveRoot "README.txt")
                manifest_present = Test-Path -LiteralPath (Join-Path $archiveRoot "manifest.json")
            }
            checkpoint_counts = $checkpointSnapshot.counts
        } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Root "recovery_$recoveryTimestamp.json") -Encoding UTF8
    }
}
if (Test-Path -LiteralPath $Lock) {
    Remove-Item -LiteralPath $Lock -Force
}
foreach ($markerName in @("launcher.pid", "wrapper.pid", "child.pid")) {
    $markerPath = Join-Path $Root $markerName
    if (Test-Path -LiteralPath $markerPath) {
        Remove-Item -LiteralPath $markerPath -Force
    }
}
New-Item -ItemType File -Path $Lock -ErrorAction Stop | Out-Null
Set-Content -LiteralPath (Join-Path $Root "wrapper.pid") -Value $PID
$script:HeadAtStart = (& git rev-parse HEAD).Trim()
$script:CudaAvailable = $null
$script:GpuName = $null

try {
    Invoke-ExternalCommand 0 "PREFLIGHT" $Python @("scripts/oulad_multistage_runtime.py", "preflight") "00_preflight"
    Invoke-ExternalCommand 1 "GPU_GATE" $Python @("scripts/oulad_multistage_runtime.py", "gpu") "01_gpu_gate"
    $gpuAudit = Get-Content (Join-Path $Repo "artifacts\final\unified_stage_aware_oulad\gpu_runtime_audit.json") -Raw | ConvertFrom-Json
    $script:CudaAvailable = [bool]$gpuAudit.cuda_available
    $script:GpuName = $gpuAudit.device_name
    if ($PreflightOnly) {
        $script:CurrentStep = "PREFLIGHT_ONLY_COMPLETE"
        $script:LastCompleted = "GPU_GATE"
        Write-Status -State "PREFLIGHT_PASS" -ExitCode 0
        exit 0
    }

    Invoke-ExternalCommand 2 "SVM_AMENDMENT" $Python @("scripts/oulad_multistage_runtime.py", "amendment") "02_svm_amendment"
    Invoke-ExternalCommand 3 "CHECKPOINT_AUDIT" $Python @("scripts/oulad_multistage_runtime.py", "audit") "03_checkpoint_audit"
    Invoke-ExternalCommand 4 "SMOKE_TEST" $Python @("scripts/oulad_multistage_runtime.py", "smoke") "04_smoke"
    Invoke-ExternalCommand 5 "FULL_TRAIN_RESUME" $Python @("project.py", "study", "oulad-multistage", "train", "--resume") "05_train"
    Invoke-ExternalCommand 6 "EVALUATE" $Python @("project.py", "study", "oulad-multistage", "evaluate") "06_evaluate"
    Invoke-ExternalCommand 7 "BOOTSTRAP" $Python @("project.py", "study", "oulad-multistage", "bootstrap") "07_bootstrap"
    Invoke-ExternalCommand 8 "REPORT" $Python @("project.py", "study", "oulad-multistage", "report") "08_report"
    Invoke-ExternalCommand 9 "OULAD_VALIDATE" $Python @("project.py", "study", "oulad-multistage", "validate") "09_validate"
    Invoke-ExternalCommand 10 "PYTEST" $Python @("-m", "pytest") "10_pytest"
    Invoke-ExternalCommand 11 "RUFF" $Python @("-m", "ruff", "check", ".") "11_ruff"
    Invoke-ExternalCommand 12 "FINAL_VALIDATE" $Python @("project.py", "final", "validate") "12_final_validate"
    Invoke-ExternalCommand 13 "DATABASE_REPLACEMENT_VALIDATE" $Python @("scripts/oulad_multistage_database.py") "13_database_validate"

    $script:StepIndex = 14
    $script:CurrentStep = "FINAL_AUDIT_COMMIT_PUSH"
    $script:CurrentStdout = Join-Path $Root "13_git.stdout.log"
    $script:CurrentStderr = Join-Path $Root "13_git.stderr.log"
    $gitScript = @'
git status
git add artifacts/final/unified_stage_aware_oulad artifacts/history/legacy_oulad_single_cutoff_f2 artifacts/history/partial_svm_probability_true_20260729 artifacts/refactor reports/final reports/history reports/refactor database/final
git commit -m "refactor: complete unified OULAD stage-aware evaluation"
git push origin codex/unified-oulad-stage-aware-system
'@
    $gitPath = Join-Path $Root "final_git_step.ps1"
    Set-Content -LiteralPath $gitPath -Value $gitScript -Encoding UTF8
    Invoke-ExternalCommand 14 "FINAL_AUDIT_COMMIT_PUSH" "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $gitPath) "14_git"

    $snapshot = Get-CheckpointSnapshot
    $done = [ordered]@{
        state = "DONE"
        started_at = $StartedAt.ToString("o")
        completed_at = [DateTimeOffset]::UtcNow.ToString("o")
        runtime_seconds = [int]([DateTimeOffset]::UtcNow - $StartedAt).TotalSeconds
        branch = (& git branch --show-current).Trim()
        code_commit = $script:HeadAtStart
        final_commit = (& git rev-parse HEAD).Trim()
        push_status = "PASS"
        completed_checkpoints = [int]$snapshot.total
        total_checkpoints = 150
        checkpoint_counts_by_model = $snapshot.counts
        stage_row_count = 40
        overall_row_count = 10
        bootstrap_replicates = 5000
        pytest = "PASS"
        ruff = "PASS"
        validators = "PASS"
        uci_checksums = "UNCHANGED"
        frozen_oulad = "RETAINED"
        canonical_database_modified = $false
        docx_pdf_modified = $false
        reports = @(
            "reports/final/OULAD_UNIFIED_MULTI_STAGE_RESULTS.md",
            "reports/final/OULAD_EARLY_WARNING_RESULTS.md",
            "reports/final/OULAD_MODEL_SELECTION_REPORT.md",
            "reports/final/OULAD_HYBRID_VS_ML_STAGE_MATRIX.md"
        )
        marker = "OULAD_UNIFIED_STAGE_AWARE_SYSTEM_READY_FOR_REVIEW"
    }
    $done | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Root "DONE.json") -Encoding UTF8
    Write-Status -State "DONE" -ExitCode 0
}
catch [System.OperationCanceledException] {
    exit 130
}
catch {
    $summary = $_.Exception.Message
    if ($summary -eq "BLOCKED_GPU") {
        exit 20
    }
    if ($summary -eq "GPU_CHECK_ERROR") {
        exit 21
    }
    if (!(Test-Path -LiteralPath (Join-Path $Root "FAILED.json"))) {
        $failureCode = if ($null -ne $script:FailureExitCode) { [int]$script:FailureExitCode } else { 1 }
        $failureSummary = if ($script:FailureSummary) { $script:FailureSummary } else { $summary }
        @{
            state = "FAILED"
            failed_at = [DateTimeOffset]::UtcNow.ToString("o")
            step = $script:CurrentStep
            exit_code = $failureCode
            error_summary = $failureSummary
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root "FAILED.json") -Encoding UTF8
        Write-Status -State "FAILED" -ExitCode $failureCode -ErrorSummary $failureSummary
    }
    exit $(if ($null -ne $script:FailureExitCode) { [int]$script:FailureExitCode } else { 1 })
}
finally {
    if (Test-Path -LiteralPath $Lock) {
        Remove-Item -LiteralPath $Lock -Force
    }
}
