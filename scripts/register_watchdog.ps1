# Register (or refresh) the "MetatraderbotWatchdog" scheduled task -- runs
# scripts\watchdog.ps1 every 15 minutes so a silently-dead bot process gets
# caught in minutes, not the days it previously took (LucidPro eval trial
# sat dead for ~2 days once, then again for ~14 hours, because
# start_all.ps1 only ever runs at logon or when someone remembers to run
# it by hand).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\register_watchdog.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_watchdog.ps1 -Remove

param(
    [switch]$Remove,
    [int]$IntervalMinutes = 15
)

$TaskName = "MetatraderbotWatchdog"
$Root = Split-Path -Parent $PSScriptRoot
$Wrapper = Join-Path $Root "scripts\watchdog.ps1"

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

# Repeat every $IntervalMinutes for 10 years -- Task Scheduler's XML schema
# rejects [TimeSpan]::MaxValue (out of range), so this is the practical
# stand-in for "indefinitely".
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Restarts any dead metatraderbot paper/eval/funded process every $IntervalMinutes min. Idempotent -- double-checks before launching anything to avoid duplicate processes." | Out-Null

Write-Output "registered '$TaskName' -- runs every $IntervalMinutes minutes"
Write-Output "  wrapper : $Wrapper"
Write-Output "  run now : Start-ScheduledTask -TaskName $TaskName"
