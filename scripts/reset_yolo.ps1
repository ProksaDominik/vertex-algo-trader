# Reset ONLY the YOLO book to a fresh $200 and restart ONLY the YOLO bot.
# FTMO is never touched - its process keeps running with its profit and open
# positions. (Pure ASCII on purpose so Windows PowerShell parses it cleanly.)

$dir = "C:\Users\proksa\OneDrive\POTA~1\NEWFOL~1\ruflo\ruflo\METATR~1"
$data = Join-Path $dir "data"
$logs = Join-Path $dir "logs"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archive = Join-Path $data ("archive_yoloreset_" + $stamp)

Start-Transcript -Path (Join-Path $logs "reset_yolo_transcript.txt") -Force | Out-Null

Write-Host "[1/4] Stopping ONLY the YOLO bot (FTMO left running)..."
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'portfolio_yolo' } |
    ForEach-Object {
        Write-Host ("  stopping YOLO PID {0}" -f $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if (-not (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'portfolio_yolo' })) { break }
    Start-Sleep -Milliseconds 500
}
Start-Sleep -Seconds 2
Write-Host "  YOLO stopped."

Write-Host "[2/4] Archiving YOLO ledger + state (resets book to 200, clears breaker halt)..."
New-Item -ItemType Directory -Path $archive -Force | Out-Null
foreach ($p in @("paper_ledger_yolo.sqlite*", "state_portfolio_yolo.json*")) {
    Get-ChildItem -Path $data -Filter $p -File -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host ("  archiving {0}" -f $_.Name)
        Move-Item $_.FullName (Join-Path $archive $_.Name) -Force
    }
}

Write-Host "[3/4] Restarting ONLY the YOLO bot via start_all.ps1 (FTMO already running is skipped)..."
& (Join-Path $dir "scripts\start_all.ps1")

Write-Host "[4/4] Done. YOLO fresh at 200, FTMO untouched."
Stop-Transcript | Out-Null
