# Reload all paper bots (FTMO, YOLO, FTMO_4DAY) so they pick up a config
# change (e.g. the 2R take-profit) WITHOUT archiving or resetting anything.
# Ledgers, state files and the persisted drawdown anchors are all preserved
# — only the run_paper processes are restarted. MT5, dashboard and chart
# API are left alone (start_all.ps1 re-attaches the bots to them).

$dir = "C:\Users\proksa\OneDrive\POTA~1\NEWFOL~1\ruflo\ruflo\METATR~1"

Write-Host "[1/2] Stopping the run_paper bots (leaving MT5 + dashboard up)..."
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'run_paper'
} | ForEach-Object {
    Write-Host ("  stopping PID {0}" -f $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

# Race-safe: WAIT until both run_paper processes are actually gone before
# relaunching, otherwise start_all.ps1's "is it already running?" check can
# see a dying process, skip the restart, and leave the bots down.
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    $alive = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_paper' }
    if (-not $alive) { break }
    Start-Sleep -Milliseconds 500
}
Start-Sleep -Seconds 2
Write-Host "  both bots confirmed stopped."

Write-Host "[2/2] Relaunching via start_all.ps1 (idempotent — only restarts the stopped bots)..."
& (Join-Path $dir "scripts\start_all.ps1")
Write-Host "Done. Bots reloaded with the current config; ledgers + anchors preserved."
