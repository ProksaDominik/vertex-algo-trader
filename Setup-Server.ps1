# Sets Vertex up on a Windows server (owner, 2026-10-09: "I want it to run on server") -- or on the owner's old
# laptop as the 24/7 server (2026-10-20: "everything should work on both, just the older one will be running 24/7").
#
# Before: unzip Vertex-move.zip to C:\Vertex. Python 3.12 and Tailscale are installed with winget when they are
# missing (sign in to Tailscale when it asks). NinjaTrader 8 only when the bots trade through it -- not in paper
# mode (VERTEX_PAPER=1 in .env). Then, in PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File C:\Vertex\ruflo\ruflo\metatraderbot\Setup-Server.ps1
#
# It: builds the Python environment with the same packages as the old PC, copies the VertexBridge add-on into
# NinjaTrader (not in paper mode), registers the "Vertex Watchdog" task (at sign-in and every 15 minutes it starts
# whatever is not running), keeps the computer awake while plugged in -- a laptop also with its lid closed --, lets
# ONLY your Tailscale devices open the dashboard (8501) and Vertex's stats service (8502), links the Obsidian notes
# to the OneDrive vault when OneDrive is signed in here (so the notes the bots write reach your other laptop),
# puts the Vertex shortcut on the desktop, and prints what is left.

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py   = Join-Path $Root ".venv\Scripts\python.exe"
function Say($m, $c = "Gray") { Write-Host "  $m" -ForegroundColor $c }
$todo = New-Object System.Collections.Generic.List[string]

Write-Host ""
Write-Host "  VERTEX -- SERVER SETUP" -ForegroundColor White
Write-Host "  $Root" -ForegroundColor DarkGray
Write-Host ""
if ($Root -match 'OneDrive') {
    Say "This folder is inside OneDrive. Put it in C:\Vertex instead -- OneDrive locks files the bots write." "Red"
    exit 1
}
$winget = [bool](Get-Command winget -ErrorAction SilentlyContinue)
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

# --- 1. Python 3.12 and the environment, same packages as the old PC -----------------
# exactly 3.12: the package list comes from the old PC's 3.12 (newer Pythons lack some of the packages)
function Find-Py312 {
    foreach ($cand in @(@("py", "-3.12"), @("python"))) {
        try {
            $v = & $cand[0] $cand[1..9] --version 2>&1
            if ("$v" -match "3\.12") { return ,$cand }
        } catch { }
    }
    return $null
}
$base = Find-Py312
if (-not $base -and $winget) {
    Say "installing Python 3.12 (winget, a few minutes)..." "Cyan"
    & winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements | Out-Null
    Refresh-Path
    $base = Find-Py312
}
if (-not $base) {
    Say "Python 3.12 is not installed. Install it from python.org (Python 3.12.x, tick 'Add python.exe to PATH')," "Red"
    Say "then run this script again." "Red"
    exit 1
}
Say "found $(& $base[0] $base[1..9] --version 2>&1)" "Green"
if (-not (Test-Path $Py)) {
    Say "creating the Python environment..." "Cyan"
    & $base[0] $base[1..9] -m venv (Join-Path $Root ".venv")
}
$req = Join-Path $Root "requirements-lock.txt"
if (-not (Test-Path $req)) { $req = Join-Path $Root "requirements.txt" }
Say "installing packages from $(Split-Path $req -Leaf) (5-15 minutes)..." "Cyan"
& $Py -m pip install --upgrade pip --quiet
& $Py -m pip install -r $req --quiet
if ($LASTEXITCODE -ne 0) {
    Say "some packages failed -- trying the short list (requirements.txt)" "Yellow"
    & $Py -m pip install -r (Join-Path $Root "requirements.txt") --quiet
}
Say "Python environment ready" "Green"
foreach ($d in @("logs", "data")) { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) { $todo.Add("The settings file .env is missing -- it should come with the zip.") }

# --- 2. NinjaTrader and the VertexBridge add-on (not in paper mode) -----------------------
$paper = (Test-Path $envFile) -and [bool](Select-String -Path $envFile -Pattern '^\s*VERTEX_PAPER\s*=\s*1\s*$' -Quiet)
$nt = Join-Path $env:ProgramFiles "NinjaTrader 8\bin\NinjaTrader.exe"
if ($paper) {
    Say "paper mode (VERTEX_PAPER=1): free internet prices -- NinjaTrader is not needed" "Green"
} elseif (-not (Test-Path $nt)) {
    $todo.Add("Install NinjaTrader 8 (ninjatrader.com), sign in with Google, start it once, then run this script again.")
} else {
    Say "NinjaTrader 8 found" "Green"
    try {
        & (Join-Path $Root "ninjatrader\Install-VertexBridge.ps1")
        $todo.Add("In NinjaTrader: New > NinjaScript Editor > press F5 once (compiles the VertexBridge add-on).")
    } catch {
        $todo.Add("Start NinjaTrader once (so it creates its Documents folder), then run this script again.")
    }
    $todo.Add("In NinjaTrader: Tools > Options > General: tick 'Connect on startup' for the Simulation connection.")
}

# --- 3. the watchdog: starts the bots, the labs, the dashboard (and NinjaTrader if used) -------------
$vbs = Join-Path $Root "scripts\watchdog_vertex.vbs"
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
$t1 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName "Vertex Watchdog" -Action $action -Trigger @($t1, $t2) -Settings $set `
    -Description "Starts the Vertex bots, labs and dashboard if any of them is not running." -Force | Out-Null
Say "registered the 'Vertex Watchdog' task (at sign-in and every 15 minutes)" "Green"

# --- 4. never sleep while plugged in -- a laptop also with its lid closed; the screen may switch off ----
foreach ($a in @(@("standby-timeout-ac", "0"), @("hibernate-timeout-ac", "0"), @("monitor-timeout-ac", "10"))) {
    & powercfg /change $a[0] $a[1] 2>$null | Out-Null
}
& powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 2>$null | Out-Null    # lid closed: do nothing
& powercfg /setactive SCHEME_CURRENT 2>$null | Out-Null
Say "no sleep while plugged in (the lid may be closed; the screen switches off after 10 minutes)" "Green"

# --- 5. the dashboard and the stats service, for your Tailscale devices only -----------------------
foreach ($fw in @(@("Vertex dashboard (Tailscale devices only)", 8501), @("Vertex voice stats (Tailscale devices only)", 8502))) {
    if (-not (Get-NetFirewallRule -DisplayName $fw[0] -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $fw[0] -Direction Inbound -Protocol TCP -LocalPort $fw[1] `
            -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any | Out-Null
    }
}
Say "ports 8501 (dashboard) and 8502 (Vertex) open to Tailscale addresses only (not to the internet)" "Green"
function Find-Tailscale {
    $c = Get-Command tailscale -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $f = Join-Path $env:ProgramFiles "Tailscale\tailscale.exe"
    if (Test-Path $f) { return $f }
    return $null
}
$ts = Find-Tailscale
if (-not $ts -and $winget) {
    Say "installing Tailscale (winget)..." "Cyan"
    & winget install --id Tailscale.Tailscale -e --silent --accept-package-agreements --accept-source-agreements | Out-Null
    Refresh-Path
    $ts = Find-Tailscale
}
if ($ts) {
    try {
        $me = (& $ts status --json | ConvertFrom-Json).Self
        $name = ($me.DNSName).TrimEnd(".")
        $short = $name.Split(".")[0]
        Say "THIS COMPUTER IN TAILSCALE: $short" "Cyan"
        Say "   dashboard from your other laptop / phone (Tailscale on): http://$($short):8501" "Cyan"
        Say "   or: http://$($me.TailscaleIPs[0]):8501" "Cyan"
        if (-not $me.Online) { $todo.Add("Sign in to Tailscale on this computer (Tailscale icon by the clock > Log in), same account as your laptop.") }
    } catch { $todo.Add("Sign in to Tailscale on this computer (Tailscale icon by the clock > Log in), then run this script again for the address.") }
} else {
    $todo.Add("Install Tailscale (tailscale.com/download), sign in (same account as your laptop), then run this script again.")
}

# --- 6. the Obsidian notes: the OneDrive vault when OneDrive is signed in here ----------------------
# The bots write the Trade ledger, Learning and Prediction lab notes into ruflo\notes. Linked to the vault in
# OneDrive, they reach your other laptop's Obsidian and Vertex's memory note reaches the bots' computer.
$notes = Join-Path (Split-Path (Split-Path $Root -Parent) -Parent) "notes"
$vault = $null
if ($env:OneDrive -and (Test-Path $env:OneDrive)) {
    $hit = Get-ChildItem $env:OneDrive -Recurse -Depth 6 -Filter "Research log.md" -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.DirectoryName "Status.md") } | Select-Object -First 1
    if ($hit) { $vault = $hit.DirectoryName }
}
$isLink = (Test-Path $notes) -and ((Get-Item $notes).Attributes -band [IO.FileAttributes]::ReparsePoint)
if ($vault -and -not $isLink) {
    if (Test-Path $notes) { Rename-Item $notes "notes-from-zip" -Force }
    & cmd /c mklink /J "$notes" "$vault" | Out-Null
    & attrib +P -U /S /D "$vault\*" 2>$null | Out-Null       # OneDrive: always keep these on this computer
    Say "Obsidian notes linked to the OneDrive vault ($vault)" "Green"
} elseif ($isLink) {
    Say "Obsidian notes already linked to the OneDrive vault" "Green"
} else {
    $todo.Add("Optional: sign in to OneDrive here (same account) and run this script again -- then the notes the bots write reach your other laptop's Obsidian.")
}

# --- 7. the desktop shortcut ------------------------------------------------------------------------
try { & (Join-Path $Root "CreateShortcut.ps1") | Out-Null; Say "Vertex shortcut on the desktop" "Green" } catch { }

Write-Host ""
$todo.Add("Optional but recommended: automatic sign-in after Windows restarts it for updates (see the note 'Move to the old laptop').")
Write-Host "  STILL TO DO" -ForegroundColor Yellow
$i = 1
foreach ($x in $todo) { Write-Host "   $i. $x" -ForegroundColor Yellow; $i++ }
Write-Host ""
Write-Host "  Then double-click 'Vertex Algo Bot' on the desktop to start everything." -ForegroundColor Green
Write-Host "  Keep this computer plugged in and signed in (locking it is fine; signing out stops the bots)." -ForegroundColor Cyan
Write-Host ""
