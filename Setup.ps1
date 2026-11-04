# Vertex Algo Bot -- one-time setup for a new machine.
#
# Creates the virtual environment, installs pinned dependencies, and checks the
# things that actually break on a fresh PC. Run once after copying the folder
# across, then use Vertex.ps1 to start the app.
#
#   .\Setup.ps1
#
# Everything resolves from $PSScriptRoot, so the folder can live anywhere.

param([switch]$Force)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py   = Join-Path $Venv "Scripts\python.exe"

function Say($m, $c = "Gray") { Write-Host "  $m" -ForegroundColor $c }

Write-Host ""
Write-Host "  VERTEX ALGO BOT -- SETUP" -ForegroundColor White
Write-Host "  $Root" -ForegroundColor DarkGray
Write-Host ""

# --- 1. Python ---------------------------------------------------------------
$sys = Get-Command python -ErrorAction SilentlyContinue
if (-not $sys) {
    Say "Python not found on PATH." "Red"
    Say "Install Python 3.11 or 3.12 from python.org and tick 'Add to PATH'." "Yellow"
    exit 1
}
$ver = & python --version 2>&1
Say "found $ver" "Green"
# MetaTrader5 has no wheels for 3.13+ yet; a fresh PC with 3.13 fails on install
# with a confusing compiler error rather than a clear "unsupported" message.
if ($ver -match "3\.(1[3-9]|[2-9]\d)") {
    Say "Python 3.13+ detected -- the MetaTrader5 package has no wheel for it." "Red"
    Say "Install Python 3.12 alongside and re-run this script." "Yellow"
    exit 1
}

# --- 2. venv -----------------------------------------------------------------
if ((Test-Path $Venv) -and $Force) {
    Say "removing existing .venv (-Force)" "Yellow"
    Remove-Item -Recurse -Force $Venv
}
if (-not (Test-Path $Py)) {
    Say "creating virtual environment..." "Cyan"
    & python -m venv $Venv
    Say "created .venv" "Green"
} else { Say ".venv already present (use -Force to rebuild)" "DarkGray" }

# --- 3. dependencies ---------------------------------------------------------
Say "installing dependencies (this takes a few minutes)..." "Cyan"
& $Py -m pip install --upgrade pip --quiet
& $Py -m pip install -r (Join-Path $Root "requirements.txt") --quiet
Say "dependencies installed" "Green"

# --- 4. folders --------------------------------------------------------------
foreach ($d in @("logs", "data", "data\archive")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}
Say "data and log folders ready" "Green"

# --- 5. credentials ----------------------------------------------------------
$envF = Join-Path $Root ".env"
$envX = Join-Path $Root ".env.example"
if (-not (Test-Path $envF) -and (Test-Path $envX)) {
    Copy-Item $envX $envF
    Say "created .env from template -- add your MT5 login before trading" "Yellow"
} elseif (Test-Path $envF) { Say ".env present" "Green" }

# --- 6. MetaTrader 5 ---------------------------------------------------------
if (Test-Path "C:\Program Files\MetaTrader 5\terminal64.exe") {
    Say "MetaTrader 5 found" "Green"
} else {
    Say "MetaTrader 5 not installed -- the UI runs, but nothing can trade." "Yellow"
    Say "Install it and log into your account, then start it once manually." "Yellow"
}

# --- 7. import check ---------------------------------------------------------
Say "verifying imports..." "Cyan"
$check = & $Py -c @"
import sys
sys.path.insert(0, r'$Root\src')
mods = ['pandas','numpy','streamlit','plotly','yaml','dotenv']
bad = []
for m in mods:
    try: __import__(m)
    except Exception as e: bad.append(f'{m}: {e}')
try:
    import MetaTrader5  # noqa: F401
except Exception as e:
    bad.append(f'MetaTrader5: {e}')
try:
    from metatraderbot import app_theme, app_stats, app_reset  # noqa: F401
except Exception as e:
    bad.append(f'app modules: {e}')
print('OK' if not bad else 'FAIL:' + ' | '.join(bad))
"@ 2>&1
if ($check -match "^OK") { Say "all imports OK" "Green" }
else { Say $check "Red"; exit 1 }

Write-Host ""
Say "Setup complete." "Green"
Say "Start the app with:  .\Vertex.ps1" "Cyan"
Write-Host ""
