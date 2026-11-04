# Register (or refresh) the "MetatraderbotLucidProFundedReview" scheduled
# task -- mirrors register_daily_review.ps1's pattern exactly, but on a
# MONTHLY cadence instead of daily.
#
# Why monthly, not daily: measured per-leg trade frequency for this roster
# is 0.19-0.87 trades/week (scripts/lucidpro_funded_revalidate.py's
# docstring) -- daily re-checks would mostly find zero new trades. Monthly
# is fast enough to catch a genuinely broken leg or a stale roster without
# pretending there's daily-resolution signal where there isn't any.
#
# PowerShell's basic ScheduledTask cmdlets have no native "1st of the month"
# trigger without CIM complexity -- -Weekly -WeeksInterval 4 approximates
# monthly closely enough for this purpose (drifts slowly against calendar
# months, which doesn't matter here).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\register_lucidpro_funded_review.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_lucidpro_funded_review.ps1 -Remove

param(
    [switch]$Remove,
    [string]$At = "06:00",
    [string]$DayOfWeek = "Sunday"
)

$TaskName = "MetatraderbotLucidProFundedReview"
$Root = Split-Path -Parent $PSScriptRoot
$Wrapper = Join-Path $Root "scripts\lucidpro_funded_review_task.ps1"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "removed existing task '$TaskName'"
}
if ($Remove) { return }

if (-not (Test-Path $Wrapper)) { throw "wrapper not found: $Wrapper" }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Wrapper`"" `
    -WorkingDirectory $Root

$trigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval 4 -DaysOfWeek $DayOfWeek -At $At

# Same interactive-only reasoning as register_daily_review.ps1: avoids
# storing credentials, acceptable for a desktop bot that needs the session
# available anyway (this task also re-downloads price history, which can
# run for several minutes -- give it real headroom).
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 45)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Monthly re-validation of the LucidPro funded-stage roster (run_lucidpro_funded.py). Report-only -- never edits the deployed config." | Out-Null

Write-Output "registered '$TaskName' -- runs every 4 weeks on $DayOfWeek at $At"
Write-Output "  wrapper : $Wrapper"
Write-Output "  logs    : $Root\logs\lucidpro_funded_review\"
Write-Output "  run now : Start-ScheduledTask -TaskName $TaskName"
