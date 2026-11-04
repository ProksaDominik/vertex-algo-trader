# Wrapper for the scheduled LucidPro funded-roster re-validation.
#
# Registered as the Windows Scheduled Task "MetatraderbotLucidProFundedReview"
# (see scripts/register_lucidpro_funded_review.ps1). Kept as a script rather
# than inlining the command into the task, same reasoning as
# daily_review_task.ps1 -- arguments can change without re-registering.
#
# REPORT-ONLY: this never edits run_lucidpro_funded.py or any config. It
# writes a dated log to logs/lucidpro_funded_review/ for a human to read.

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\lucidpro_funded_revalidate.py"
$LogDir = Join-Path $Root "logs\lucidpro_funded_review"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$stamp = Get-Date -Format "yyyy-MM-dd"
$runLog = Join-Path $LogDir "run_$stamp.log"

"=== lucidpro funded review $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') ===" |
    Out-File -FilePath $runLog -Encoding utf8

& $Python "-u" $Script 2>&1 | Out-File -FilePath $runLog -Append -Encoding utf8

"=== exit code $LASTEXITCODE ===" | Out-File -FilePath $runLog -Append -Encoding utf8

# The Python script already prunes its own dated review_*.log reports.
# This prunes the separate run_*.log wrapper-invocation logs the same way.
Get-ChildItem $LogDir -Filter "run_*.log" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 24 |
    Remove-Item -Force -ErrorAction SilentlyContinue
