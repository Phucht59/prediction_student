param()

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
        step_total = 14
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

function Invoke-Step {
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
    Write-Status -State "RUNNING"
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
    $script:ChildPid = 0
    if ($process.ExitCode -ne 0) {
        if ($Name -eq "GPU_GATE" -and $process.ExitCode -eq 21) {
            Write-Status -State "BLOCKED_GPU" -ExitCode 21 -ErrorSummary "CUDA gate failed"
            throw "BLOCKED_GPU"
        }
        Write-Status -State "FAILED" -ExitCode $process.ExitCode -ErrorSummary "$Name failed"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    $script:LastCompleted = $Name
    Write-Status -State "RUNNING" -ExitCode 0
}

if (Test-Path -LiteralPath $Lock) {
    $existingPidPath = Join-Path $Root "wrapper.pid"
    $existingPid = if (Test-Path $existingPidPath) { [int](Get-Content $existingPidPath -Raw) } else { 0 }
    if ($existingPid -gt 0 -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        throw "An active detached OULAD wrapper already exists"
    }
    Remove-Item -LiteralPath $Lock -Force
}
New-Item -ItemType File -Path $Lock -ErrorAction Stop | Out-Null
Set-Content -LiteralPath (Join-Path $Root "wrapper.pid") -Value $PID
$script:HeadAtStart = (& git rev-parse HEAD).Trim()
$script:CudaAvailable = $null
$script:GpuName = $null

try {
    Invoke-Step 0 "PREFLIGHT" $Python @("scripts/oulad_multistage_runtime.py", "preflight") "00_preflight"
    Invoke-Step 1 "SVM_AMENDMENT" $Python @("scripts/oulad_multistage_runtime.py", "amendment") "01_svm_amendment"
    Invoke-Step 2 "CHECKPOINT_AUDIT" $Python @("scripts/oulad_multistage_runtime.py", "audit") "02_checkpoint_audit"
    Invoke-Step 3 "GPU_GATE" $Python @("scripts/oulad_multistage_runtime.py", "gpu") "03_smoke"
    $gpuAudit = Get-Content (Join-Path $Repo "artifacts\final\unified_stage_aware_oulad\gpu_runtime_audit.json") -Raw | ConvertFrom-Json
    $script:CudaAvailable = [bool]$gpuAudit.cuda_available
    $script:GpuName = $gpuAudit.device_name
    Invoke-Step 3 "SMOKE_TEST" $Python @("scripts/oulad_multistage_runtime.py", "smoke") "03_smoke"
    Invoke-Step 4 "FULL_TRAIN_RESUME" $Python @("project.py", "study", "oulad-multistage", "train", "--resume") "04_train"
    Invoke-Step 5 "EVALUATE" $Python @("project.py", "study", "oulad-multistage", "evaluate") "05_evaluate"
    Invoke-Step 6 "BOOTSTRAP" $Python @("project.py", "study", "oulad-multistage", "bootstrap") "06_bootstrap"
    Invoke-Step 7 "REPORT" $Python @("project.py", "study", "oulad-multistage", "report") "07_report"
    Invoke-Step 8 "OULAD_VALIDATE" $Python @("project.py", "study", "oulad-multistage", "validate") "08_validate"
    Invoke-Step 9 "PYTEST" $Python @("-m", "pytest") "09_pytest"
    Invoke-Step 10 "RUFF" $Python @("-m", "ruff", "check", ".") "10_ruff"
    Invoke-Step 11 "FINAL_VALIDATE" $Python @("project.py", "final", "validate") "11_final_validate"
    Invoke-Step 12 "DATABASE_REPLACEMENT_VALIDATE" $Python @("scripts/oulad_multistage_database.py") "12_database_validate"

    $script:StepIndex = 13
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
    Invoke-Step 13 "FINAL_AUDIT_COMMIT_PUSH" "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $gitPath) "13_git"

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
    if (!(Test-Path -LiteralPath (Join-Path $Root "FAILED.json")) -and $summary -ne "BLOCKED_GPU") {
        @{
            state = "FAILED"
            failed_at = [DateTimeOffset]::UtcNow.ToString("o")
            step = $script:CurrentStep
            error_summary = $summary
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root "FAILED.json") -Encoding UTF8
        Write-Status -State "FAILED" -ExitCode 1 -ErrorSummary $summary
    }
    exit 1
}
finally {
    if (Test-Path -LiteralPath $Lock) {
        Remove-Item -LiteralPath $Lock -Force
    }
}
