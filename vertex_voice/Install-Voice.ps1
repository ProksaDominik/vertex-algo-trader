# Installs Vertex, the voice butler, on a laptop (owner, 2026-10-09: on both laptops).
# Vertex has no window of its own: it lives by the clock (the V icon), listens for "Hey Vertex", talks, and
# shows everything in the dashboard -- a card on every page and the Vertex page -- and moves that dashboard
# by voice. Install the dashboard app too (viewer\Install-Viewer.ps1) on a laptop that does not run the bots.
#
# Needs: Python 3.12 (python.org, tick "Add python.exe to PATH"), a microphone and speakers, Tailscale signed in
# (same account as the server), and a Claude API key (console.anthropic.com > API keys).
# Copy this "vertex_voice" folder to the laptop, right-click Install-Voice.ps1 > Run with PowerShell.
#
#   Install-Voice.ps1                 install, ask for the server name and the Claude key
#   Install-Voice.ps1 -Startup        also start Vertex at every sign-in
#   Install-Voice.ps1 -Reconfigure    ask for the server name / Claude key again
#   Install-Voice.ps1 -Server NAME    only point Vertex at the 24/7 computer (its Tailscale name), keeping the
#                                     Claude key, Spotify and every other setting
#
# Everything goes to %LOCALAPPDATA%\VertexVoice (a plain-letter path: the speech model cannot open folders
# with letters like "c with a hacek").

param([switch]$Startup, [switch]$Reconfigure, [string]$Server = "")

$ErrorActionPreference = "Stop"
$Src = $PSScriptRoot
$Dst = Join-Path $env:LOCALAPPDATA "VertexVoice"
function Say($m, $c = "Gray") { Write-Host "  $m" -ForegroundColor $c }
Write-Host ""
Write-Host "  VERTEX -- VOICE BUTLER SETUP" -ForegroundColor White
Write-Host ""

# -Server: the bots moved to the 24/7 computer (owner, 2026-10-20) -- Vertex on this laptop talks to it from now
# on. Only VERTEX_SERVER changes (the dashboard address follows it); the key and all other settings stay.
if ($Server) {
    $settings = Join-Path $Dst "settings.env"
    if (-not (Test-Path $settings)) { Say "Vertex is not installed on this laptop yet -- run Install-Voice.ps1 without -Server first." "Red"; exit 1 }
    $name = $Server.Trim() -replace '^https?://', '' -replace '[:/].*$', ''
    $keep = Get-Content $settings -Encoding UTF8 | Where-Object { $_ -notmatch '^\s*(VERTEX_SERVER|VERTEX_DASHBOARD)\s*=' }
    [IO.File]::WriteAllLines($settings, [string[]](@("VERTEX_SERVER=$name") + $keep), (New-Object Text.UTF8Encoding($false)))
    Say "Vertex now talks to $name -- the bots' numbers and the dashboard (http://$($name):8501)." "Green"
    Say "Quit Vertex from the V icon by the clock and double-click 'Vertex Voice' to start it again." "Cyan"
    Write-Host ""
    exit 0
}

# --- 1. Python ---------------------------------------------------------------------------------------
$py = $null
foreach ($cand in @(@("py", "-3.12"), @("py", "-3.11"), @("python"))) {
    try {
        $v = & $cand[0] $cand[1..9] --version 2>&1
        if ($v -match "Python 3\.(1[0-2])") { $py = $cand; break }
    } catch { }
}
if (-not $py) {
    Say "Python 3.10-3.12 not found. Install Python 3.12 from python.org (tick 'Add python.exe to PATH')." "Red"
    exit 1
}
Say "found $(& $py[0] $py[1..9] --version 2>&1)" "Green"

# --- 2. the app ----------------------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $Dst | Out-Null
foreach ($f in @("vertex_voice.py", "pc_control.py", "spotify.py", "requirements.txt", "vertex_v_icon.ico")) {
    Copy-Item (Join-Path $Src $f) (Join-Path $Dst $f) -Force
}
$venvPy = Join-Path $Dst ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Say "creating the Python environment..." "Cyan"
    & $py[0] $py[1..9] -m venv (Join-Path $Dst ".venv")
}
Say "installing packages (a few minutes)..." "Cyan"
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r (Join-Path $Dst "requirements.txt") --quiet
if ($LASTEXITCODE -ne 0) { Say "package install failed -- check the internet connection and run this again." "Red"; exit 1 }
Say "packages installed" "Green"
# An NVIDIA graphics card: the large English speech model runs on it (0.4 s a question, far fewer misheard words)
$gpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Say "NVIDIA card found: installing its libraries for speech recognition (~1 GB)..." "Cyan"
    & $venvPy -m pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10" --quiet
    $gpu = ($LASTEXITCODE -eq 0)
}

# --- 3. the offline wake-word model (Vosk, ~40 MB) -------------------------------------------------
$models = Join-Path $Dst "models"
if (-not (Get-ChildItem $models -Directory -Filter "vosk-model*" -ErrorAction SilentlyContinue)) {
    New-Item -ItemType Directory -Force -Path $models | Out-Null
    $zip = Join-Path $env:TEMP "vosk-model-small-en-us.zip"
    Say "downloading the wake-word model..." "Cyan"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $models -Force
    Remove-Item $zip -Force
}
Say "wake-word model ready" "Green"

# --- 4. settings -----------------------------------------------------------------------------------------
$settings = Join-Path $Dst "settings.env"
if ($Reconfigure -or -not (Test-Path $settings)) {
    $server = Read-Host "  The server's name in Tailscale (e.g. vertex-server)"
    if (-not $server) { $server = "vertex-server" }
    $sec = Read-Host "  Your Claude API key (starts with sk-ant-; it is not shown)" -AsSecureString
    $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
    $address = Read-Host "  How should Vertex address you? (e.g. sir; Enter = no title)"
    @(
        "VERTEX_SERVER=$($server.Trim())",
        "ANTHROPIC_API_KEY=$($key.Trim())",
        "VERTEX_VOICE=en-GB-RyanNeural",
        "VERTEX_WHISPER=base.en",
        "VERTEX_ANNOUNCE=1",
        "VERTEX_ADDRESS=$($address.Trim())"
    ) | Set-Content -Path $settings -Encoding ascii
    Say "settings saved ($settings)" "Green"
}

# --- 5. the speech-to-text model (~150 MB; ~1.5 GB more with an NVIDIA card), so the first question is quick ------
Say "preparing speech recognition (first time downloads ~150 MB$(if ($gpu) { ' + 1.5 GB' }))..." "Cyan"
& $venvPy -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')" 2>$null
if ($gpu) {
    & $venvPy -c "import sys; sys.path.insert(0, r'$Dst'); import vertex_voice as v; v.gpu_libraries(); from faster_whisper import WhisperModel; WhisperModel('distil-large-v3.5', device='cuda', compute_type='float16')" 2>$null
}
Say "speech recognition ready" "Green"

# --- 6. the Vertex icon ------------------------------------------------------------------------------------
$pyw = Join-Path $Dst ".venv\Scripts\pythonw.exe"
$sh = New-Object -ComObject WScript.Shell
$targets = @(Join-Path ([Environment]::GetFolderPath("Desktop")) "Vertex Voice.lnk")
if ($Startup) { $targets += Join-Path ([Environment]::GetFolderPath("Startup")) "Vertex Voice.lnk" }
foreach ($lnkPath in $targets) {
    $lnk = $sh.CreateShortcut($lnkPath)
    $lnk.TargetPath = $pyw
    $lnk.Arguments = "`"$(Join-Path $Dst 'vertex_voice.py')`""
    if ($lnkPath -like "*Startup*") { $lnk.Arguments += " --hidden" }   # at sign-in: straight to the tray
    $lnk.WorkingDirectory = $Dst
    $lnk.IconLocation = Join-Path $Dst "vertex_v_icon.ico"
    $lnk.Description = "Vertex -- voice butler for the trading bots"
    $lnk.Save()
    Say "created $lnkPath" "Green"
}

# --- 7. self-check: the server and Claude ------------------------------------------------------------------
Write-Host ""
Say "checking the server and Claude..." "Cyan"
& $venvPy (Join-Path $Dst "vertex_voice.py") --check
Write-Host ""
Say "Double-click 'Vertex Voice' on the desktop: it opens the dashboard and waits by the clock (V icon)." "Cyan"
Say "Then say: 'Hey Vertex ... how did we do today?' or 'Hey Vertex ... show me the funded accounts'." "Cyan"
Write-Host ""
