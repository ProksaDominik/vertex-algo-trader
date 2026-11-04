# Brings code changes made on the owner's other laptop to the 24/7 computer (owner, 2026-10-20: "everything
# should work on both, just the older one will be running 24/7"). Claude works in the OneDrive copy of this
# folder on the other laptop; OneDrive brings the changes here, but the bots run from C:\Vertex, outside OneDrive.
# Run this on the 24/7 computer -- best between 22:05 and 23:55 Prague time, when no race is open:
#
#   right-click Update-From-OneDrive.ps1 > Run with PowerShell
#
# Either copy works: the one in C:\Vertex, or -- since 2026-10-21, so the NEWEST version of this script always runs
# -- the one in the OneDrive folder itself (OneDrive > Pocitac > New folder > ruflo > ruflo > metatraderbot).
#
# Copies the code (src, scripts, vertex_voice, viewer, ninjatrader, tests, config, docs, the .ps1 scripts,
# requirements.txt) from the OneDrive copy, installs any new packages, and restarts everything. Never touches the
# bots' data, logs or this computer's Python environment (.venv). The one thing it may add to the settings (.env):
# the trading desk's Claude key, asked for once if there is none (owner 2026-10-21).

param([switch]$NoRestart, [switch]$Force)

$ErrorActionPreference = "Stop"
$Target = "C:\Vertex\ruflo\ruflo\metatraderbot"
function Say($m, $c = "Gray") { Write-Host "  $m" -ForegroundColor $c }
function Done($code) {
    if ([Environment]::UserInteractive -and -not $Force) { Write-Host ""; Read-Host "  Press Enter to close" | Out-Null }
    exit $code
}
Write-Host ""
Write-Host "  VERTEX -- UPDATE FROM ONEDRIVE" -ForegroundColor White
Write-Host ""

if ($PSScriptRoot -match 'OneDrive') {
    # started from the OneDrive folder: this folder is the source, the 24/7 folder the target
    if (-not (Test-Path (Join-Path $Target "Vertex.ps1"))) {
        Say "This is the OneDrive copy, and $Target was not found -- run it on the 24/7 laptop." "Red"; Done 1
    }
    $src, $Root = $PSScriptRoot, $Target
    # tell the watchdog where the OneDrive copy is, so its data copy back to OneDrive starts at once
    try {
        New-Item -ItemType Directory -Force -Path (Join-Path $Root "data") | Out-Null
        Set-Content -Path (Join-Path $Root "data\onedrive_copy.txt") -Value $src -Encoding UTF8
    } catch { }
} else {
    $Root = $PSScriptRoot
    $mark = Join-Path $Root "data\onedrive_copy.txt"
    $src = if (Test-Path $mark) { Get-Content $mark -TotalCount 1 } else { $null }
    if (-not $src -or -not (Test-Path (Join-Path $src "Setup-Server.ps1"))) {
        $hit = $null
        if ($env:OneDrive) {
            $hit = Get-ChildItem $env:OneDrive -Recurse -Depth 7 -Filter "Vertex.ps1" -ErrorAction SilentlyContinue |
                Where-Object { Test-Path (Join-Path $_.DirectoryName "Setup-Server.ps1") } | Select-Object -First 1
        }
        if (-not $hit) { Say "No Vertex folder found in OneDrive -- is OneDrive signed in here (same account)?" "Red"; Done 1 }
        $src = $hit.DirectoryName
    }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Say "from: $src" "DarkGray"
Say "to:   $Root" "DarkGray"

$now = (Get-Date).TimeOfDay
if (-not $Force -and -not ($now -ge [timespan]'22:05' -and $now -lt [timespan]'23:55')) {
    Say "It is $((Get-Date).ToString('HH:mm')). A race may be open now -- best between 22:05 and 23:55." "Yellow"
    if ((Read-Host "  Type YES to update anyway") -ne "YES") { Say "Nothing changed." "Gray"; Done 0 }
}

$copy = "/E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /XD __pycache__ .pytest_cache".Split(" ")
foreach ($d in @("src", "scripts", "vertex_voice", "viewer", "ninjatrader", "tests", "config", "docs")) {
    if (Test-Path (Join-Path $src $d)) {
        $extra = if ($d -eq "vertex_voice") { @("/XD", "models", ".venv") } else { @() }
        robocopy (Join-Path $src $d) (Join-Path $Root $d) @copy @extra | Out-Null
        if ($LASTEXITCODE -ge 8) { Say "copying $d failed (robocopy $LASTEXITCODE)" "Red"; Done 1 }
    }
}
robocopy $src $Root *.ps1 requirements.txt *.md /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Say "copying the scripts failed (robocopy $LASTEXITCODE)" "Red"; Done 1 }
Say "code copied" "Green"

# The trading desk's agents need a Claude key on this laptop: asked for once, kept in .env (never in OneDrive).
$envFile = Join-Path $Root ".env"
$hasKey = (Test-Path $envFile) -and [bool](Select-String -Path $envFile -Quiet `
    -Pattern '^\s*(DESK_ANTHROPIC_API_KEY|ANTHROPIC_API_KEY)\s*=\s*sk-ant-\S+')
if (-not $hasKey -and [Environment]::UserInteractive -and -not $Force) {
    Write-Host ""
    Say "The trading desk needs a Claude key on this laptop." "Cyan"
    Say "Best: make a new key at console.anthropic.com in a workspace with a `$10 monthly limit, then paste it here." "Gray"
    Say "Press Enter without a key to skip -- the desk then only watches and writes the moves down." "Gray"
    $sec = Read-Host "  Claude key (hidden while you paste)" -AsSecureString
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
    if ($plain.Trim() -match '^sk-ant-\S+$') {
        Add-Content -Path $envFile -Value ("`r`nDESK_ANTHROPIC_API_KEY=" + $plain.Trim()) -Encoding ascii
        Say "key saved for the trading desk" "Green"
    } elseif ($plain.Trim()) {
        Say "That doesn't look like a Claude key (they start with sk-ant-) -- skipped; run this again to add it." "Yellow"
    } else {
        Say "no key -- the desk watches only; run this again any time to add one" "Yellow"
    }
    $plain = $null
}

Say "installing any new packages..." "Cyan"
& $Py -m pip install -r (Join-Path $Root "requirements.txt") --quiet
Say "packages up to date" "Green"

if (-not $NoRestart) {
    Say "restarting the bots, labs and dashboard..." "Cyan"
    & (Join-Path $Root "Vertex.ps1") -Restart
}
Write-Host ""
Say "Done. The other laptop's changes now run here. Tell Claude 'the update is in'." "Green"
Done 0                   # robocopy leaves a non-zero code even when it copied everything
