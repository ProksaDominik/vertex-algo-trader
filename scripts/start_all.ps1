# Brings up MT5 terminal + both paper-trading bots + the dashboard if any of
# them aren't already running. Idempotent — safe to run repeatedly (e.g. from
# a scheduled task at logon) without spawning duplicates.

# 2026-08-18 -- moved out of OneDrive (was under C:\Users\proksa\OneDrive\
# Počítač\...) to C:\ruflo: the OneDrive sync client was intermittently
# locking the state/log files this project rewrites every 15-30s, causing
# both PermissionError crashes and (very likely) contributing to MT5 IPC
# hangs. This also retires the old short-(8.3)-path workaround -- that was
# only needed to dodge an encoding bug with the "č" in the OneDrive path;
# C:\ruflo has no special characters so the plain path is safe to use directly.
$dir = "C:\Users\proksa\OneDrive\POTA~1\NEWFOL~1\ruflo\ruflo\METATR~1"
$mt5Path = "C:\Program Files\MetaTrader 5\terminal64.exe"

function Test-Running($pattern) {
    return [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern })
}

if (-not (Get-Process -Name terminal64 -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $mt5Path
    Start-Sleep -Seconds 25  # give MT5 time to load and auto-login before bots try to attach
}

# 2026-08-15 -- consolidated down to ONLY the two LucidPro bots (eval trial
# + funded). FTMO was swapped for run_lucidpro_funded.py on 2026-08-14;
# FTMO_4day and YOLO were stopped and removed from this launcher on
# 2026-08-16 so nothing restarts them. Their config files and code are
# untouched -- this is a launcher-scope decision, not a deletion.
if (-not (Test-Running 'run_lucidpro_funded')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\run_lucidpro_funded.py' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_lucidpro_funded.log" -RedirectStandardError "$dir\logs\stderr_lucidpro_funded.log"
}

if (-not (Test-Running 'run_lucidpro_trial')) {
    # Added 2026-08-14 -- this bot was never wired into the launcher, so it
    # silently stopped accumulating eval attempts whenever the machine
    # rebooted or the manual terminal that launched it was closed (dead
    # since 2026-08-12 with zero resolved attempts before this fix).
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\run_lucidpro_trial.py' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_lucidpro.log" -RedirectStandardError "$dir\logs\stderr_lucidpro.log"
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
    Start-Sleep -Seconds 5  # give Streamlit time to bind the port before opening the browser
}

# Open the dashboard in the default browser — MT5 already opens its own
# visible window above, this is the equivalent "visible" step for the web
# interface. Runs every time (harmless if it just opens another tab onto
# the same already-running dashboard).
Start-Process "http://localhost:8501"
