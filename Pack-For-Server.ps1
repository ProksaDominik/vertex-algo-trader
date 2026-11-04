# Packs Vertex for the move to a server (owner, 2026-10-09: "I want it to run on server").
#
#   .\Pack-For-Server.ps1            pack a copy now (the bots here keep running) -- for a trial setup
#   .\Pack-For-Server.ps1 -Switch    THE MOVE: stop the bots on this PC, switch off its Vertex tasks,
#                                    then pack -- from now on only the server trades
#
# Makes Desktop\Vertex-move.zip with the same folders as here:
#   ruflo\notes                    the Obsidian notes the bots write
#   ruflo\ruflo\metatraderbot      code, settings (.env), data and the bots' memory
# Left out: .venv (rebuilt on the server), scratch, logs, caches.
# The zip holds .env (your settings and keys): keep it private and delete it once the server runs.
#
# Two computers must never run the bots at the same time -- both would trade the same accounts.

param([switch]$Switch)

$ErrorActionPreference = "Stop"
$Bot   = $PSScriptRoot                                  # ...\ruflo\ruflo\metatraderbot
$Vault = Split-Path (Split-Path $Bot -Parent) -Parent   # ...\ruflo  (holds notes\)
$Zip   = Join-Path ([Environment]::GetFolderPath("Desktop")) "Vertex-move.zip"
$Stage = Join-Path $env:TEMP "vertex-move"

function Say($m, $c = "Gray") { Write-Host "  $m" -ForegroundColor $c }
Write-Host ""
Write-Host "  VERTEX -- PACK FOR THE SERVER" -ForegroundColor White
Write-Host ""

if ($Switch) {
    foreach ($t in @("Vertex Watchdog", "Vertex Wake for race")) {
        if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
            Disable-ScheduledTask -TaskName $t | Out-Null
            Say "switched off the task '$t' on this PC" "Yellow"
        }
    }
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'run_lucidpro|run_lucid_futures|run_lab\.py|run_prediction_lab\.py|chart_api\.py|vertex_api\.py|streamlit'
    }
    foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Say "stopped $(@($procs).Count) bot / lab / dashboard process(es) on this PC" "Yellow"
    # the "Vertex Algo Bot" shortcut starts the bots: gone from this PC, so they are never started here by mistake
    try { & (Join-Path $Bot "CreateShortcut.ps1") -Remove | Out-Null; Say "removed the 'Vertex Algo Bot' shortcut on this PC" "Yellow" } catch { }
    Start-Sleep -Seconds 3
}

if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
$dstBot = Join-Path $Stage "ruflo\ruflo\metatraderbot"
$dstNotes = Join-Path $Stage "ruflo\notes"
New-Item -ItemType Directory -Force $dstBot, $dstNotes | Out-Null
# vertex_voice\models: the laptops' speech model -- the server does not listen
robocopy $Bot $dstBot /E /NFL /NDL /NJH /NJS /NP /XD .venv scratch logs __pycache__ .pytest_cache .mypy_cache `
    (Join-Path $Bot "vertex_voice\models") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $dstBot "logs") | Out-Null
# the exact package versions running here, so the server installs the same ones
$py = Join-Path $Bot ".venv\Scripts\python.exe"
if (Test-Path $py) {
    & $py -m pip freeze | Where-Object { $_ -and $_ -notmatch '^(-e |#)' -and $_ -notmatch ' @ file:' } |
        Set-Content -Path (Join-Path $dstBot "requirements-lock.txt") -Encoding ascii
    Say "saved the exact package list (requirements-lock.txt)" "Green"
}
robocopy (Join-Path $Vault "notes") $dstNotes /E /NFL /NDL /NJH /NJS /NP | Out-Null
if (Test-Path (Join-Path $Vault ".obsidian")) {
    robocopy (Join-Path $Vault ".obsidian") (Join-Path $Stage "ruflo\.obsidian") /E /NFL /NDL /NJH /NJS /NP | Out-Null
}

if (Test-Path $Zip) { Remove-Item $Zip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($Stage, $Zip, [System.IO.Compression.CompressionLevel]::Optimal, $false)
Remove-Item $Stage -Recurse -Force
$mb = [math]::Round((Get-Item $Zip).Length / 1MB)
Say "packed: $Zip  ($mb MB)" "Green"
Write-Host ""
if ($Switch) {
    Say "This PC no longer runs the bots. Copy the zip to the server and run Setup-Server.ps1 there." "Cyan"
    Say "Then on this PC: viewer\Install-Viewer.ps1 -Server <its Tailscale name>  (the dashboard as an app)" "Cyan"
    Say "            and: vertex_voice\Install-Voice.ps1 -Server <its Tailscale name>  (Vertex talks to it)" "Cyan"
} else {
    Say "Trial pack only -- the bots here keep running. Use -Switch for the real move." "Cyan"
}
Say "The zip contains your settings (.env): keep it private, delete it when the server runs." "Yellow"
Write-Host ""
exit 0   # robocopy leaves a non-zero code even when it copied everything
