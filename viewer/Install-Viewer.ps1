# Vertex VIEWER -- the dashboard as an app on a laptop that does NOT run the bots (owner, 2026-10-09:
# "can it also work as an app not as web app on the old one?"). The bots run on the server; this laptop
# only shows the dashboard, in its own window with the Vertex icon -- no address bar, no tabs.
#
# Needs: Tailscale installed and signed in (same account as the server), and Edge or Chrome.
# Copy this "viewer" folder to the laptop, right-click Install-Viewer.ps1 > Run with PowerShell.
#
#   Install-Viewer.ps1                          asks for the server's Tailscale name
#   Install-Viewer.ps1 -Server vertex-server    desktop shortcut "Vertex"
#   Install-Viewer.ps1 -Server vertex-server -Startup    also opens it at every sign-in
#   Install-Viewer.ps1 -Remove                   removes the shortcuts

param([string]$Server = "", [switch]$Startup, [switch]$Remove, [int]$Port = 8501)

$ErrorActionPreference = "Stop"
$Name = "Vertex"
$Desktop = [Environment]::GetFolderPath("Desktop")
$StartDir = [Environment]::GetFolderPath("Startup")
function Say($m, $c = "Gray") { Write-Host "  $m" -ForegroundColor $c }
Write-Host ""
Write-Host "  VERTEX -- VIEWER APP" -ForegroundColor White
Write-Host ""

if ($Remove) {
    foreach ($p in @((Join-Path $Desktop "$Name.lnk"), (Join-Path $StartDir "$Name.lnk"))) {
        if (Test-Path $p) { Remove-Item $p -Force; Say "removed $p" "Yellow" }
    }
    exit 0
}

if (-not $Server) {
    $Server = Read-Host "  The server's name in Tailscale (e.g. vertex-server) or its 100.x.x.x address"
}
$Server = $Server.Trim() -replace '^https?://', '' -replace '[:/].*$', ''
if (-not $Server) { Say "No server name given." "Red"; exit 1 }
$Url = "http://$($Server):$Port"
# this laptop's name in the link: Vertex (the voice butler) on this laptop moves this window, not the others
$Viewer = ($env:COMPUTERNAME -replace '[^A-Za-z0-9_-]', '')
$AppUrl = "$Url/?viewer=$Viewer"

$browsers = @(
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$exe = $browsers | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { Say "Neither Edge nor Chrome is installed -- install one of them first." "Red"; exit 1 }

# its own browser profile: the app keeps its window size and is not closed by quitting the browser
$profileDir = Join-Path $env:LOCALAPPDATA "VertexViewer\browser"
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
$appArgs = "--app=$AppUrl --user-data-dir=`"$profileDir`" --start-maximized --no-first-run --no-default-browser-check"
$icon = Join-Path $PSScriptRoot "vertex_v_icon.ico"

$sh = New-Object -ComObject WScript.Shell
$targets = @(Join-Path $Desktop "$Name.lnk")
if ($Startup) { $targets += Join-Path $StartDir "$Name.lnk" }
foreach ($lnkPath in $targets) {
    $lnk = $sh.CreateShortcut($lnkPath)
    $lnk.TargetPath = $exe
    $lnk.Arguments = $appArgs
    $lnk.Description = "Vertex dashboard (runs on $Server)"
    if (Test-Path $icon) {
        $keep = Join-Path $env:LOCALAPPDATA "VertexViewer\vertex_v_icon.ico"
        Copy-Item $icon $keep -Force
        $lnk.IconLocation = $keep
    }
    $lnk.Save()
    Say "created $lnkPath" "Green"
}

# can this laptop reach the dashboard right now?
try {
    $r = Invoke-WebRequest -Uri "$Url/_stcore/health" -UseBasicParsing -TimeoutSec 8
    Say "the server answers ($Url) -- opening the app" "Green"
    Start-Process -FilePath $exe -ArgumentList $appArgs
} catch {
    Say "Cannot reach $Url right now. Check: Tailscale is on and signed in here and on the server," "Yellow"
    Say "the server name is right (Tailscale icon > the server's name), and the server is running." "Yellow"
}
Write-Host ""
Say "Double-click 'Vertex' on the desktop any time. It only shows the dashboard -- trading runs on the server." "Cyan"
Write-Host ""
