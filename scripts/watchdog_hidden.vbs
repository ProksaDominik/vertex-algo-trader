' Launches watchdog.ps1 with NO console window.
'
' 2026-08-20 -- MetatraderbotWatchdog runs every 15 minutes as an Interactive
' task, so calling powershell.exe directly popped a console window onto the
' desktop four times an hour. Notes on the approaches that do NOT work:
'
'   * The task's Settings.Hidden property only hides the task in the Task
'     Scheduler UI. It has no effect on the window its action creates.
'   * powershell.exe -WindowStyle Hidden still creates the console host first
'     and hides it a moment later, so it flashes.
'   * Switching the task principal to "run whether user is logged on or not"
'     genuinely has no window, but moves the task to session 0 -- and this
'     watchdog Start-Process-es the MT5 terminal, which needs the interactive
'     desktop session to work at all. That would break the thing it guards.
'
' WScript.Shell.Run with intWindowStyle=0 never creates a visible host in the
' first place, and stays in the user's interactive session. bWaitOnReturn is
' False so this wrapper exits immediately; the scheduled task therefore always
' reports success, and watchdog.ps1's own outcome is visible in the logs it
' writes (logs\stderr_*.log) rather than in the task's LastTaskResult.

Dim shell, ps1, cmd
Set shell = CreateObject("WScript.Shell")

ps1 = "C:\ruflo\ruflo\metatraderbot\scripts\watchdog.ps1"
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & ps1 & """"

shell.Run cmd, 0, False
