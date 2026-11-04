# Vertex Algo Bot -- application launcher.
#
# Portable by design: every path is derived from this script's own location via
# $PSScriptRoot, so the whole folder can be copied to another PC and run without
# editing anything. That matters because this project has already been broken
# once by absolute paths -- a move out of OneDrive left the Startup shortcut and
# two scheduled tasks pointing at a directory that no longer existed, and they
# failed silently for days.
#
#   .\Vertex.ps1            start everything, open the app
#   .\Vertex.ps1 -Stop      stop everything
#   .\Vertex.ps1 -Restart   stop, then start
#   .\Vertex.ps1 -Status    report what is running
#   .\Vertex.ps1 -NoBots    UI only (no live trading)

param(
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status,
    [switch]$NoBots,
    [switch]$Watchdog,      # quiet re-check from Task Scheduler: start what is missing, open no window
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$Root   = $PSScriptRoot
$Py     = Join-Path $Root ".venv\Scripts\python.exe"
$Logs   = Join-Path $Root "logs"
$MT5    = "C:\Program Files\MetaTrader 5\terminal64.exe"

function Write-Step($msg, $c = "Gray") { Write-Host "  $msg" -ForegroundColor $c }

# VERTEX_BROKER=ninjatrader in .env moves the Lucid trial bot onto NinjaTrader 8
# (Lucid is futures-only; see INSTALL.md). Only that one line is read -- no
# credentials leave the file.
$EnvFile = Join-Path $Root ".env"
$Futures = (Test-Path $EnvFile) -and
    [bool](Select-String -Path $EnvFile -Pattern '^\s*VERTEX_BROKER\s*=\s*ninjatrader\s*$' -Quiet)

# VERTEX_PAPER=1 (2026-10-15): the bots paper-trade on free prices -- NinjaTrader's minute bars when it is open,
# else the internet's (src/metatraderbot/free_prices.py). NinjaTrader is then optional: the watchdog neither
# starts it (its failing Google sign-in looped for hours) nor sends alerts about it.
$Paper = (Test-Path $EnvFile) -and
    [bool](Select-String -Path $EnvFile -Pattern '^\s*VERTEX_PAPER\s*=\s*1\s*$' -Quiet)

# NinjaTrader 8 and the port its VertexBridge add-on listens on (futures mode).
$NinjaExe   = Join-Path $env:ProgramFiles "NinjaTrader 8\bin\NinjaTrader.exe"
$BridgePort = 45973
if (Test-Path $EnvFile) {
    $m = Select-String -Path $EnvFile -Pattern '^\s*NT_BRIDGE_PORT\s*=\s*(\d+)\s*$' | Select-Object -First 1
    if ($m) { $BridgePort = [int]$m.Matches[0].Groups[1].Value }
}

function Get-Procs {
    # run_lab.py was missing from this pattern until 2026-08-28, which is the
    # real reason the strategy lab died unnoticed and stayed dead for two days:
    # invisible here means -Status reported nothing, -Stop never stopped it, and
    # -Restart never brought it back. A component the launcher cannot see is a
    # component nobody is watching.
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'run_lucidpro|run_lucid_futures|run_trading_desk\.py|run_lab\.py|run_prediction_lab\.py|close_legacy_gold\.py|chart_api\.py|vertex_api\.py|streamlit'
    }
}

function Stop-All {
    $p = Get-Procs
    if (-not $p) { Write-Step "nothing running" "DarkGray"; return }
    foreach ($x in $p) { Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    Write-Step "stopped $($p.Count) process(es)" "Yellow"
}

# NOTE: the argument parameter is deliberately NOT called $args -- that is a
# PowerShell automatic variable and cannot be bound as a normal parameter, so
# it silently arrives empty and Start-Process fails on a null ArgumentList.
function Start-One($name, $argList, $out, $err, $notMatching = $null) {
    if (Get-Procs | Where-Object { $_.CommandLine -match [regex]::Escape($name) -and
                                   (-not $notMatching -or $_.CommandLine -notmatch [regex]::Escape($notMatching)) }) {
        Write-Step "$name already running" "DarkGray"; return
    }
    Start-Process -FilePath $Py -ArgumentList $argList -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Logs $out) `
        -RedirectStandardError  (Join-Path $Logs $err)
    Write-Step "started $name" "Green"
}

# True once the VertexBridge add-on accepts connections, i.e. NinjaTrader is up
# and the add-on has loaded. A bare connect is safe: the bridge treats a client
# that sends nothing as finished.
function Test-Bridge {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $ar = $c.BeginConnect('127.0.0.1', $BridgePort, $null, $null)
        return ($ar.AsyncWaitHandle.WaitOne(700) -and $c.Connected)
    } catch { return $false } finally { $c.Close() }
}

Write-Host ""
Write-Host "  VERTEX ALGO BOT" -ForegroundColor White
Write-Host "  $Root" -ForegroundColor DarkGray
Write-Host ""

if (-not (Test-Path $Py)) {
    Write-Host "  Python environment missing." -ForegroundColor Red
    Write-Host "  Run .\Setup.ps1 first to create it." -ForegroundColor Yellow
    exit 1
}
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

if ($Status) {
    # A Windows venv launcher spawns the real interpreter as a CHILD process, so
    # every component legitimately appears twice in a raw process list. Reporting
    # both makes a healthy stack look like it is running duplicate bots -- which
    # is alarming for exactly the wrong reason -- so report one row per component
    # and show the launcher pid.
    $p = Get-Procs
    $seen = @{}
    foreach ($x in $p) {
        $n = if ($x.CommandLine -match 'run_lucidpro_funded|run_lucid_futures_funded') { 'funded bot' }
             elseif ($x.CommandLine -match 'run_lucidpro_trial') { 'trial bot' }
             elseif ($x.CommandLine -match 'run_prediction_lab') { 'prediction lab' }
             elseif ($x.CommandLine -match 'run_lab') { 'strategy lab' }
             elseif ($x.CommandLine -match 'close_legacy_gold') { 'gold closer' }
             elseif ($x.CommandLine -match 'chart_api') { 'chart api' }
             else { 'dashboard' }
        if (-not $seen.ContainsKey($n) -and $x.CommandLine -match '\.venv') {
            $seen[$n] = $x.ProcessId
        }
    }
    foreach ($n in @('funded bot','trial bot','strategy lab','prediction lab','chart api','dashboard')) {
        if ($seen.ContainsKey($n)) { Write-Step ("{0,-12} running  (pid {1})" -f $n, $seen[$n]) "Green" }
        else                       { Write-Step ("{0,-12} stopped" -f $n) "DarkGray" }
    }
    if ($Futures) {
        $nt = Get-Process -Name NinjaTrader -ErrorAction SilentlyContinue
        if ($nt) { Write-Step ("{0,-12} running  (all three books trade Lucid futures here)" -f 'NinjaTrader') "Green" }
        else {
            Write-Step ("{0,-12} NOT RUNNING  <- the bots trade through it" -f 'NinjaTrader') "Yellow"
        }
    } else {
        $mt = Get-Process -Name terminal64 -ErrorAction SilentlyContinue
        if ($mt) { Write-Step ("{0,-12} running" -f 'MetaTrader 5') "Green" }
        else {
            Write-Step ("{0,-12} NOT RUNNING  <- start it yourself before trading" -f 'MetaTrader 5') "Yellow"
        }
    }
    Write-Host ""
    exit 0
}

if ($Stop) { Stop-All; Write-Host ""; exit 0 }
if ($Restart) { Stop-All }

# MetaTrader 5 is deliberately NOT started here -- the operator starts it, so
# the terminal is always launched into a session they are watching, and the app
# never silently attaches to an account nobody meant to trade. The launcher only
# reports the state and carries on: the UI is perfectly usable with MT5 down,
# it just cannot place or read live orders.
$mt5Running = [bool](Get-Process -Name terminal64 -ErrorAction SilentlyContinue)
if ($Futures) {
    # Futures mode uses no MetaTrader at all -- nothing to warn about.
} elseif ($mt5Running) {
    Write-Step "MetaTrader 5 is running" "Green"
} else {
    Write-Host ""
    Write-Host "  ------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "   METATRADER 5 IS NOT RUNNING" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   The app will start, but until MT5 is open and logged in:" -ForegroundColor Gray
    Write-Host "     - no orders can be placed or closed" -ForegroundColor Gray
    Write-Host "     - live positions and prices are unavailable" -ForegroundColor Gray
    Write-Host "     - the bots will retry connecting in the background" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   Start MetaTrader 5, log in, and make sure AlgoTrading is" -ForegroundColor Gray
    Write-Host "   enabled (the toolbar button is green)." -ForegroundColor Gray
    Write-Host "  ------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host ""
}

# Futures mode: one click also starts NinjaTrader (owner's request, 2026-10-01;
# this used to be left to the operator). Connecting it to the Lucid/Sim
# connection is NinjaTrader's own "Connect on startup" setting. The bots start
# once the VertexBridge add-on answers, or after 3 minutes -- each of them keeps
# retrying on its own if NinjaTrader is still not ready.
# NinjaTrader's last start that never got past its login screen (its log holds only the
# "Session Break" line). Starting it again would just try the login again: on 2026-10-09
# NinjaTrader reported "too many incorrect login attempts" after its stored Google sign-in
# had failed for a day. So for 3 hours after such a start the watchdog leaves NinjaTrader
# to the owner's manual sign-in.
function Test-NinjaLoginStuck {
    foreach ($d in @((Join-Path $env:USERPROFILE "Documents\NinjaTrader 8\log"),
                     (Join-Path ([Environment]::GetFolderPath('MyDocuments')) "NinjaTrader 8\log"))) {
        if (-not (Test-Path $d)) { continue }
        $last = Get-ChildItem $d -Filter "log.*.en.txt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $last) { continue }
        $young = $last.CreationTime -gt (Get-Date).AddHours(-3)
        $lines = @(Get-Content $last.FullName -TotalCount 3 -ErrorAction SilentlyContinue).Count
        return ($young -and $lines -le 2)
    }
    return $false
}

if ($Futures -and -not $NoBots -and $Paper) {
    Write-Step "paper trading: prices from NinjaTrader if it is open, else from the internet - NinjaTrader not needed" "Green"
} elseif ($Futures -and -not $NoBots) {
    if (Get-Process -Name NinjaTrader -ErrorAction SilentlyContinue) {
        Write-Step "NinjaTrader is running (Lucid futures mode)" "Green"
    } elseif ($Watchdog -and (Test-NinjaLoginStuck)) {
        Write-Step "NinjaTrader's last start stopped at its login - not starting it again (sign in yourself)" "Yellow"
    } elseif (Test-Path $NinjaExe) {
        Start-Process -FilePath $NinjaExe -WorkingDirectory (Split-Path $NinjaExe)
        Write-Step "started NinjaTrader - waiting for the VertexBridge add-on" "Green"
    } else {
        Write-Step "NinjaTrader 8 not found at $NinjaExe - start it yourself" "Yellow"
    }
    $deadline = (Get-Date).AddMinutes(3)
    while (-not (Test-Bridge) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 3 }
    if (Test-Bridge) { Write-Step "VertexBridge answering on port $BridgePort" "Green" }
    else { Write-Step "VertexBridge not answering yet - the bots keep retrying" "Yellow" }
} elseif ($Futures) {
    if (Get-Process -Name NinjaTrader -ErrorAction SilentlyContinue) {
        Write-Step "NinjaTrader is running (Lucid futures mode)" "Green"
    } else {
        Write-Host "  ------------------------------------------------------------" -ForegroundColor Yellow
        Write-Host "   NINJATRADER IS NOT RUNNING (VERTEX_BROKER=ninjatrader)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "   The bots trade Lucid futures through NinjaTrader 8." -ForegroundColor Gray
        Write-Host "   Start it, connect your Lucid (or Sim) connection, and check" -ForegroundColor Gray
        Write-Host "   the VertexBridge add-on is compiled. The bot keeps retrying." -ForegroundColor Gray
        Write-Host "  ------------------------------------------------------------" -ForegroundColor Yellow
        Write-Host ""
    }
}

if (-not $NoBots) {
    if ($Futures) {
        # Lucid futures mode: the same three books, all on NinjaTrader --
        # funded (its own sim account, NT_FUNDED_ACCOUNT), trial (NT_ACCOUNT)
        # and the lab (bars only, no orders). No MetaTrader component starts.
        Start-One 'run_lucid_futures_funded' 'scripts\run_lucid_futures_funded.py' 'stdout_lucid_futures_funded.log' 'stderr_lucid_futures_funded.log' '--account'
        # Every further funded account (owner, 2026-10-09: "we will be trading multiple accounts") runs its
        # own funded bot: LUCID_FUNDED_EXTRA_ACCOUNTS=<NinjaTrader account>,<NinjaTrader account> in .env.
        $extra = Select-String -Path $EnvFile -Pattern '^\s*LUCID_FUNDED_EXTRA_ACCOUNTS\s*=\s*(.+?)\s*$' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($extra) {
            foreach ($a in ($extra.Matches[0].Groups[1].Value -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
                $safe = $a -replace '[^A-Za-z0-9]+', '_'
                Start-One "--account $a" @('scripts\run_lucid_futures_funded.py', '--account', $a) `
                    "stdout_lucid_futures_funded_$safe.log" "stderr_lucid_futures_funded_$safe.log"
            }
        }
        Start-One 'run_lucidpro_trial'  'scripts\run_lucidpro_trial.py'  'stdout_lucidpro.log'        'stderr_lucidpro.log' '--strategy'
        # A second strategy side by side, on paper only (owner, 2026-10-21: "one tab the current strategy and one
        # tab this new strategy and we will compare them"): a trial bot and a funded account racing the A-D breadth
        # side. Switched on by choice, not by default (the free S&P 500 count did worse than the 200-day rule on
        # history, notes/Research log 2026-10-21): owner 2026-10-21 "keep it for testing for 2 months" -- it runs
        # while today is on or before the date in config\compare_ad_until.txt, or with VERTEX_COMPARE_AD=1 in .env.
        $compareUntil = $null
        $untilFile = Join-Path $Root 'config\compare_ad_until.txt'
        if (Test-Path $untilFile) {
            try {
                $compareUntil = [datetime]::ParseExact(((Get-Content $untilFile -TotalCount 1).Trim() -split '\s+')[0],
                                                       'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture)
            } catch { $compareUntil = $null }
        }
        $compare = ((Test-Path $EnvFile) -and
            [bool](Select-String -Path $EnvFile -Pattern '^\s*VERTEX_COMPARE_AD\s*=\s*1\s*$' -Quiet)) -or
            ($compareUntil -and (Get-Date).Date -le $compareUntil)
        if ($Paper -and $compare) {
            Start-One '--strategy ad' @('scripts\run_lucidpro_trial.py', '--strategy', 'ad') 'stdout_lucidpro_ad.log' 'stderr_lucidpro_ad.log'
            Start-One '--account Sim102-AD' @('scripts\run_lucid_futures_funded.py', '--account', 'Sim102-AD', '--side', 'breadth') `
                'stdout_lucid_futures_funded_Sim102_AD.log' 'stderr_lucid_futures_funded_Sim102_AD.log'
        } elseif ($compareUntil) {
            # The test is over: stop its bots once no race can be open (races run 19:00-16:00 New York = 00:00/01:00
            # to 21:00/22:00 Prague). Their files stay, so the dashboard keeps showing the comparison.
            $now = (Get-Date).TimeOfDay
            if ($now -ge [timespan]'22:05' -and $now -lt [timespan]'23:55') {
                foreach ($x in (Get-Procs | Where-Object { $_.CommandLine -match '--strategy ad|Sim102-AD' })) {
                    Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue
                    Write-Step "side-by-side test over ($($compareUntil.ToString('yyyy-MM-dd'))): stopped $($x.ProcessId)" "Yellow"
                }
            }
        }
        # The trading desk (owner 2026-10-21: Scout, News, Chart and Skeptic agents on their own paper account,
        # trades on big moves, at most 2 a day, AI fees capped at $10 a month) -- paper only.
        if ($Paper) {
            Start-One 'run_trading_desk' 'scripts\run_trading_desk.py' 'stdout_trading_desk.log' 'stderr_trading_desk.log'
        }
        Start-One 'run_lab' @('scripts/run_lab.py','--configs','296','--report-every','20') 'stdout_lab_futures.log' 'stderr_lab_futures.log'
        # Live prediction lab: scores frozen predictors on new days; places no orders.
        Start-One 'run_prediction_lab' @('scripts/run_prediction_lab.py','--loop') 'stdout_prediction_lab.log' 'stderr_prediction_lab.log'
        # Vertex, the voice butler's stats service (read-only, port 8502) for the laptops' Vertex app.
        Start-One 'vertex_api' 'scripts\vertex_api.py' 'stdout_vertex_api.log' 'stderr_vertex_api.log'
    } else {
        Start-One 'run_lucidpro_funded' 'scripts\run_lucidpro_funded.py' 'stdout_lucidpro_funded.log' 'stderr_lucidpro_funded.log'
        Start-One 'run_lucidpro_trial'  'scripts\run_lucidpro_trial.py'  'stdout_lucidpro.log'        'stderr_lucidpro.log'
        # The strategy lab. It was left out of the launcher when first built, so
        # when it stopped on 2026-08-26 nothing restarted it and two days of
        # sample accumulation were lost before anyone noticed. It places NO
        # orders -- it only reads bars -- so starting it here cannot affect the
        # trading books.
        Start-One 'run_lab' @('scripts/run_lab.py','--configs','296','--report-every','20') 'stdout_lab.log' 'stderr_lab.log'
    }
} else { Write-Step "bots skipped (-NoBots)" "Yellow" }

# The chart API serves MetaTrader prices for the MT5 pages' charts; the
# futures pages never use it.
if (-not $Futures) {
    Start-One 'chart_api' 'scripts\chart_api.py' 'stdout_chartapi.log' 'stderr_chartapi.log'
}
Start-One 'streamlit' "-m streamlit run scripts\dashboard.py --server.headless true --server.port $Port" 'stdout_dashboard.log' 'stderr_dashboard.log'

# --- Telegram when NinjaTrader needs a hand (owner away for months on a server, 2026-10-09) --------
function Send-Telegram($text) {
    if (-not (Test-Path $EnvFile)) { return }
    $tok  = Select-String -Path $EnvFile -Pattern '^\s*TELEGRAM_BOT_TOKEN\s*=\s*"?([^"\s]+)' | Select-Object -First 1
    $chat = Select-String -Path $EnvFile -Pattern '^\s*TELEGRAM_CHAT_ID\s*=\s*"?([^"\s]+)' | Select-Object -First 1
    if (-not $tok -or -not $chat) { return }
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-RestMethod -Method Post -TimeoutSec 15 `
            -Uri ("https://api.telegram.org/bot{0}/sendMessage" -f $tok.Matches[0].Groups[1].Value) `
            -Body @{ chat_id = $chat.Matches[0].Groups[1].Value; text = $text } | Out-Null
    } catch { }
}
function Send-AlertOnce($key, $text, $hours = 2) {
    $mark = Join-Path $Root "data\alert_$key.txt"
    if ((Test-Path $mark) -and (Get-Item $mark).LastWriteTime -gt (Get-Date).AddHours(-$hours)) { return }
    Send-Telegram $text
    Set-Content -Path $mark -Value (Get-Date -Format o)
}
function Get-NinjaLoginFailures($minutes = 15) {
    $d = Join-Path $env:USERPROFILE "Documents\NinjaTrader 8\log"
    $last = Get-ChildItem $d -Filter "log.*.en.txt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $last) { return 0 }
    $since = (Get-Date).AddMinutes(-$minutes).ToString("yyyy-MM-dd HH:mm:ss")
    @(Get-Content $last.FullName -Tail 400 -ErrorAction SilentlyContinue | Where-Object {
        $_.Length -ge 19 -and $_.Substring(0, 19) -ge $since -and ($_ -match 'problem authenticating|Logon failed') }).Count
}
function Test-MarketWeek {
    # CME's week: Sunday 18:00 to Friday 17:00 New York, without the daily 17:00-18:00 pause
    $et = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), "Eastern Standard Time")
    $h = $et.Hour; $d = $et.DayOfWeek
    if ($d -eq 'Saturday') { return $false }
    if ($d -eq 'Sunday') { return $h -ge 18 }
    if ($d -eq 'Friday') { return $h -lt 17 }
    return -not ($h -eq 17)
}
if ($Watchdog -and $Futures -and -not $NoBots -and -not $Paper) {
    $fails = Get-NinjaLoginFailures
    if ($fails -ge 3) {
        Send-AlertOnce "nt_login" ("Vertex: NinjaTrader's sign-in failed $fails times in 15 minutes. Close NinjaTrader and " +
            "sign in again (Sign in with Google) before it locks the login - over Remote Desktop on the server.")
    }
    $downMark = Join-Path $Root "data\alert_bridge_down_since.txt"
    if (Test-Bridge) {
        Remove-Item $downMark -ErrorAction SilentlyContinue
    } elseif (Test-MarketWeek) {
        if (-not (Test-Path $downMark)) { Set-Content -Path $downMark -Value (Get-Date -Format o) }
        elseif ((Get-Item $downMark).CreationTime -lt (Get-Date).AddMinutes(-25)) {
            Send-AlertOnce "nt_down" ("Vertex: NinjaTrader has not answered for 30 minutes - the bots cannot trade. " +
                "Open Remote Desktop, start NinjaTrader and sign in (the dashboard's left side shows its state).")
        }
    }
}

# --- the 24/7 computer's results, copied into the OneDrive folder for the other laptop (owner, 2026-10-20:
# "I need to be able to control it also through my laptop ... everything should work on both"). The bots run
# from C:\Vertex, outside OneDrive (OneDrive locks files they write); every watchdog round copies the changed
# data files and the logs (up to 20 MB each) into the OneDrive copy of this folder, so the other laptop -- and
# Claude working there -- sees today's results. Only from a copy outside OneDrive, and only into a folder that
# is a Vertex copy too (has Vertex.ps1 and Setup-Server.ps1). Never the other way: the bots' files stay here.
if ($Watchdog -and $Root -notmatch 'OneDrive' -and $env:OneDrive -and (Test-Path $env:OneDrive)) {
    try {
        $markFile = Join-Path $Root "data\onedrive_copy.txt"
        $dst = if (Test-Path $markFile) { Get-Content $markFile -TotalCount 1 } else { $null }
        if (-not $dst -or -not (Test-Path (Join-Path $dst "Setup-Server.ps1"))) {
            # the owner's folder first (2026-10-21: a search of the whole OneDrive -- tens of thousands of library
            # files -- can outlast the task's 10 minutes, so the copy never started); the search only as a fallback
            $known = Join-Path $env:OneDrive "Počítač\New folder\ruflo\ruflo\metatraderbot"
            if (Test-Path (Join-Path $known "Setup-Server.ps1")) {
                $dst = $known
            } else {
                $hit = Get-ChildItem $env:OneDrive -Recurse -Depth 7 -Filter "Vertex.ps1" -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -notmatch '\\\.venv\\' -and
                                   (Test-Path (Join-Path $_.DirectoryName "Setup-Server.ps1")) } | Select-Object -First 1
                $dst = if ($hit) { $hit.DirectoryName } else { $null }
            }
            if ($dst) { Set-Content -Path $markFile -Value $dst -Encoding UTF8 }
        }
        if ($dst -and ($dst -ne $Root)) {
            robocopy (Join-Path $Root "data") (Join-Path $dst "data") /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP `
                /XF *.sqlite-shm *.sqlite-wal *.tmp onedrive_copy.txt | Out-Null
            robocopy $Logs (Join-Path $dst "logs") *.log /MAX:20971520 /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
        }
    } catch { }
}

if ($Watchdog) { exit 0 }      # the watchdog never opens the app window
Start-Sleep -Seconds 8
Write-Host ""
if (-not $mt5Running -and -not $Futures) {
    Write-Step "reminder: MetaTrader 5 still needs to be started manually" "Yellow"
}

# Open as a DESKTOP APP, not a browser tab.
#
# Chromium's --app= mode gives a standalone window with no address bar, tabs or
# bookmarks, its own taskbar entry and its own icon -- indistinguishable from a
# native app for this purpose, with no extra runtime to install or ship. Edge is
# tried first because it is present on every Windows 11 machine, so this keeps
# working after a transfer; Chrome is the fallback, and a plain browser tab is
# the last resort so the app is never simply unreachable.
$AppUrl = "http://localhost:$Port"
$UserDir = Join-Path $env:LOCALAPPDATA "VertexAlgoBot\browser"
New-Item -ItemType Directory -Force -Path $UserDir | Out-Null

$browsers = @(
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
)
$exe = $browsers | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($exe) {
    # --user-data-dir keeps the app window out of the normal browsing profile,
    # so it gets its own window state and cannot be closed by quitting a browser.
    Start-Process -FilePath $exe -ArgumentList @(
        "--app=$AppUrl",
        "--user-data-dir=`"$UserDir`"",
        "--window-size=1500,950",
        "--no-first-run",
        "--no-default-browser-check"
    )
    Write-Step "opened app window ($(Split-Path $exe -Leaf))" "Green"
} else {
    Write-Step "no Chromium browser found - opening in default browser" "Yellow"
    Start-Process $AppUrl
}
Write-Step $AppUrl "DarkGray"
Write-Host ""
