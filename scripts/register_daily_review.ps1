# Register (or refresh) the "MetatraderbotDailyReview" scheduled task.
#
# Runs scripts/daily_review_task.ps1 every day at 23:30 local -- after the
# broker's daily bar has rolled, so the day's trades are all closed and
# reconciled before the review reads the ledger.
#
# Report-only by default: the wrapper decides whether to pass --apply, so
# enabling auto-calibration later needs no change here.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily_review.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily_review.ps1 -Remove

param(
    [switch]$Remove,
    [string]$At = "23:30"
)

$TaskName = "MetatraderbotDailyReview"
$Root = Split-Path -Parent $PSScriptRoot
$Wrapper = Join-Path $Root "scripts\daily_review_task.ps1"

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

$trigger = New-ScheduledTaskTrigger -Daily -At $At

# Run whether or not the user is logged on is deliberately NOT used: that
# requires storing credentials. Interactive-only is enough for a desktop bot
# that already needs MT5 running in the same session.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Daily per-trade analysis of the metatraderbot FTMO + YOLO books. Report-only unless the wrapper enables --apply." | Out-Null

Write-Output "registered '$TaskName' -- runs daily at $At"
Write-Output "  wrapper : $Wrapper"
Write-Output "  logs    : $Root\logs\daily_review\"
Write-Output "  run now : Start-ScheduledTask -TaskName $TaskName"
