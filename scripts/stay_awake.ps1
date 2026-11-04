# Task Scheduler's "Vertex Wake for race" runs this after waking the PC at 00:55
# (Mon-Fri, Prague): keep Windows from falling back asleep for -Minutes, so
# NinjaTrader reconnects and both bots open the 19:00 New York (01:00 Prague)
# race. Then Windows may sleep again; the bots replay the price history on wake.
param([int]$Minutes = 25)
Add-Type -Namespace Vertex -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]2147483648; $ES_SYSTEM_REQUIRED = [uint32]1
[Vertex.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
Start-Sleep -Seconds ($Minutes * 60)
[Vertex.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
