$ErrorActionPreference = "Stop"
$Root = "C:\hufit\kltn\logs\oulad_multistage\detached"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
Set-Content -LiteralPath (Join-Path $Root "cancel.request") `
    -Value ([DateTimeOffset]::UtcNow.ToString("o")) -Encoding UTF8
Write-Output "CANCEL_REQUESTED"
