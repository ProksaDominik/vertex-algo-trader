# Brings up MT5 terminal + both paper-trading bots + the dashboard if any of
# them aren't already running. Idempotent — safe to run repeatedly (e.g. from
# a scheduled task at logon) without spawning duplicates.

# Short (8.3) path — avoids an encoding bug where a spawned powershell.exe
# (e.g. from the Startup-folder .bat) misreads the "č" in the real path,
# corrupting $dir and making every Start-Process below fail with
# DirectoryNotFoundException even though the folder plainly exists.
$dir = "C:\Users\proksa\OneDrive\POTA~1\NEWFOL~1\ruflo\ruflo\METATR~1"
$mt5Path = "C:\Program Files\MetaTrader 5\terminal64.exe"

function Test-Running($pattern) {
    return [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern })
}

if (-not (Get-Process -Name terminal64 -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $mt5Path
    Start-Sleep -Seconds 25  # give MT5 time to load and auto-login before bots try to attach
}

if (-not (Test-Running 'portfolio_ftmo')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\run_paper.py --config config\portfolio_ftmo.yaml' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_ftmo.log" -RedirectStandardError "$dir\logs\stderr_ftmo.log"
}

if (-not (Test-Running 'portfolio_yolo')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\run_paper.py --config config\portfolio_yolo.yaml' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_yolo.log" -RedirectStandardError "$dir\logs\stderr_yolo.log"
}

if (-not (Test-Running 'chart_api')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\chart_api.py' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_chartapi.log" -RedirectStandardError "$dir\logs\stderr_chartapi.log"
}

if (-not (Test-Running 'streamlit')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList '-m streamlit run scripts\dashboard.py --server.headless true' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_dashboard.log" -RedirectStandardError "$dir\logs\stderr_dashboard.log"
}
