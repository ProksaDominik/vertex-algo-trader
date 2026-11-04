# Silent, frequent version of start_all.ps1's idempotent restart logic --
# built 2026-08-15 after the LucidPro eval trial bot was found to have
# silently died more than once (once for ~2 days, once for ~14 hours) with
# nobody noticing because start_all.ps1 only ever runs at logon or when
# manually invoked. This runs on a short interval instead (see
# register_watchdog.ps1) so a dead bot gets caught in minutes, not days.
#
# Differs from start_all.ps1 in exactly one way: it never opens a browser
# tab. Running that every 15 minutes forever would be a real nuisance --
# the dashboard-open step only makes sense for an interactive/logon launch,
# not a silent background watchdog. Everything else (idempotent
# Test-Running checks, same processes, same log files) is identical on
# purpose -- one restart codepath, not two to keep in sync.

# 2026-08-18 -- moved out of OneDrive to C:\ruflo (sync-client file locks
# were causing PermissionError crashes and likely contributing to MT5 IPC
# hangs). See start_all.ps1 for the full explanation.
$dir = "C:\Users\proksa\OneDrive\POTA~1\NEWFOL~1\ruflo\ruflo\METATR~1"
$mt5Path = "C:\Program Files\MetaTrader 5\terminal64.exe"

function Test-Running($pattern) {
    return [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern })
}

# Two independent checks before concluding "not running": a bare WMI query
# gave a FALSE NEGATIVE once already (2026-08-14, led to a real duplicate
# process being launched by hand) -- an unattended watchdog trusting a
# single flaky check every 15 minutes could do that silently and
# repeatedly, compounding into multiple live copies of the same bot
# trading the same account. Re-check after a short pause; only report
# "confirmed not running" if BOTH checks agree.
function Test-RunningConfirmed($pattern) {
    if (Test-Running $pattern) { return $true }
    Start-Sleep -Seconds 5
    return Test-Running $pattern
}

if (-not (Get-Process -Name terminal64 -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $mt5Path
    Start-Sleep -Seconds 25
}

# 2026-08-16 -- FTMO_4day and YOLO removed (stopped and consolidated down
# to ONLY the two LucidPro bots). Do not re-add without also updating
# start_all.ps1 to match -- one restart codepath, kept in sync on purpose.
if (-not (Test-RunningConfirmed 'run_lucidpro_funded')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\run_lucidpro_funded.py' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_lucidpro_funded.log" -RedirectStandardError "$dir\logs\stderr_lucidpro_funded.log"
}

if (-not (Test-RunningConfirmed 'run_lucidpro_trial')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\run_lucidpro_trial.py' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_lucidpro.log" -RedirectStandardError "$dir\logs\stderr_lucidpro.log"
}

if (-not (Test-RunningConfirmed 'chart_api')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList 'scripts\chart_api.py' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_chartapi.log" -RedirectStandardError "$dir\logs\stderr_chartapi.log"
}

if (-not (Test-RunningConfirmed 'streamlit')) {
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" `
        -ArgumentList '-m streamlit run scripts\dashboard.py --server.headless true' `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput "$dir\logs\stdout_dashboard.log" -RedirectStandardError "$dir\logs\stderr_dashboard.log"
}
