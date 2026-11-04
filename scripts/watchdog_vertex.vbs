' Task Scheduler's "Vertex Watchdog": every 15 minutes, start whatever is not
' running (NinjaTrader, bots, labs, dashboard) -- no window, no app pop-up.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
sh.CurrentDirectory = root
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & root & "\Vertex.ps1"" -Watchdog", 0, False
