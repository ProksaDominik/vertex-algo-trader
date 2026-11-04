' Task Scheduler's "Vertex Wake for race" (00:55 Mon-Fri): make sure everything
' runs, then keep the PC awake 25 minutes for the 01:00 race -- no windows.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
sh.CurrentDirectory = root
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & root & "\Vertex.ps1"" -Watchdog", 0, False
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & root & "\scripts\stay_awake.ps1"" -Minutes 25", 0, False
