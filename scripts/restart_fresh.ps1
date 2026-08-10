# Fresh-start restart: stop both bots (+ dashboard + chart api), archive the
# ledgers and state files from the pre-overhaul era, then bring everything
# back up via start_all.ps1 with clean databases and fresh guard anchors.
# MT5 terminal itself is left running (bots re-attach).
#
# Safe to re-run; archives are timestamped, nothing is deleted.

$dir = "C:\Users\proksa\OneDrive\POTA~1\NEWFOL~1\ruflo\ruflo\METATR~1"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "[1/4] Stopping bot processes..."
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'run_paper|dashboard\.py|chart_api|streamlit'
} | ForEach-Object {
    Write-Host ("  stopping PID {0}" -f $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3

Write-Host "[2/4] Archiving ledgers + state (fresh-start era boundary)..."
$data = Join-Path $dir "data"
$archive = Join-Path $data ("archive_phase5_" + $stamp)
New-Item -ItemType Directory -Path $archive -Force | Out-Null
# WAL sidecars must travel with their database file.
$patterns = @(
    "paper_ledger.sqlite*", "paper_ledger_yolo.sqlite*", "state_portfolio_*.json"
)
foreach ($p in $patterns) {
    Get-ChildItem -Path $data -Filter $p -File -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host ("  archiving {0}" -f $_.Name)
        Move-Item $_.FullName (Join-Path $archive $_.Name) -Force
    }
}

Write-Host "[3/4] Relaunching everything via start_all.ps1..."
& (Join-Path $dir "scripts\start_all.ps1")

Write-Host "[4/4] Done. Fresh ledgers + state will be created on first tick."
Write-Host ("Archive: " + $archive)
