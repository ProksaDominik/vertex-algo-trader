# Wrapper for the scheduled daily review.
#
# Registered as the Windows Scheduled Task "MetatraderbotDailyReview" (see
# scripts/register_daily_review.ps1). Kept as a script rather than inlining the
# command into the task so the arguments -- especially whether --apply is
# passed -- can be changed without re-registering the task.
#
# SAFE BY DEFAULT: runs in report-only mode. Set $ApplyCalibration = $true to
# let it write data/cost_calibration.json from observed fills. Do that only
# after reading a few reports and agreeing with what it wants to change.

param(
    [switch]$Apply
)

$ApplyCalibration = $false   # <- flip to $true to enable auto-calibration

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\daily_review.py"
$LogDir = Join-Path $Root "logs\daily_review"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$stamp = Get-Date -Format "yyyy-MM-dd"
$log = Join-Path $LogDir "review_$stamp.log"

$argList = @("-u", $Script, "--days", "1")
if ($Apply -or $ApplyCalibration) { $argList += "--apply" }

"=== daily review $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') ===" |
    Out-File -FilePath $log -Encoding utf8

& $Python $argList 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

"=== exit code $LASTEXITCODE ===" | Out-File -FilePath $log -Append -Encoding utf8

# Mirror the day into the Obsidian vault (notes/Journal/YYYY-MM-DD.md) so the
# AIs and the human read the same record. Read-only against the ledgers.
& $Python -u (Join-Path $Root "scripts\vault_journal.py") --review-log $log 2>&1 |
    Out-File -FilePath $log -Append -Encoding utf8

# Keep 60 days of reports; the whole point is a trend, not one snapshot.
Get-ChildItem $LogDir -Filter "review_*.log" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 60 |
    Remove-Item -Force -ErrorAction SilentlyContinue
