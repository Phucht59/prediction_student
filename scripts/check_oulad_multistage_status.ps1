$ErrorActionPreference = "Stop"
$Repo = "C:\hufit\kltn"
$Root = Join-Path $Repo "logs\oulad_multistage\detached"
$StatusPath = Join-Path $Root "status.json"

if (!(Test-Path -LiteralPath $StatusPath)) {
    Write-Output "NO_DETACHED_STATUS"
    exit 2
}
$status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
$childAlive = $false
if ($status.child_pid -and [int]$status.child_pid -gt 0) {
    $childAlive = $null -ne (Get-Process -Id ([int]$status.child_pid) -ErrorAction SilentlyContinue)
}
[ordered]@{
    state = $status.state
    step = $status.step
    heartbeat_at = $status.heartbeat_at
    completed_checkpoints = $status.completed_checkpoints
    total_checkpoints = $status.total_checkpoints
    checkpoint_counts_by_model = $status.checkpoint_counts_by_model
    current_model = $status.current_model
    current_outer_fold = $status.current_outer_fold
    current_seed = $status.current_seed
    last_checkpoint_time = $status.last_checkpoint_time
    child_pid = $status.child_pid
    child_alive = $childAlive
    done_marker = Test-Path -LiteralPath (Join-Path $Root "DONE.json")
    failed_marker = Test-Path -LiteralPath (Join-Path $Root "FAILED.json")
} | ConvertTo-Json -Depth 8

if ($status.state -eq "FAILED" -and $status.stderr_log -and (Test-Path -LiteralPath $status.stderr_log)) {
    Write-Output "LAST_STDERR_LINES"
    Get-Content -LiteralPath $status.stderr_log -Tail 30
}
