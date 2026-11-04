# Copies the VertexBridge add-on into NinjaTrader 8's custom-script folder.
#
# Run once after installing NinjaTrader 8 (and again after updating
# VertexBridge.cs), then compile it inside NinjaTrader:
#   New > NinjaScript Editor > press F5
# The NinjaScript Output window (New > NinjaScript Output) then shows
#   VertexBridge 1.0 listening on 127.0.0.1:45973

$ErrorActionPreference = "Stop"
$src  = Join-Path $PSScriptRoot "VertexBridge.cs"
# NinjaTrader uses the Documents folder, wherever Windows has put it
# (OneDrive redirects it on many PCs).
$docs = [Environment]::GetFolderPath("MyDocuments")
$dest = Join-Path $docs "NinjaTrader 8\bin\Custom\AddOns"

if (-not (Test-Path $dest)) {
    Write-Host "  NinjaTrader 8 folder not found:" -ForegroundColor Red
    Write-Host "    $dest" -ForegroundColor Red
    Write-Host "  Install NinjaTrader 8 and start it once, then run this again." -ForegroundColor Yellow
    exit 1
}
Copy-Item $src $dest -Force
Write-Host "  copied VertexBridge.cs -> $dest" -ForegroundColor Green
Write-Host "  Next, in NinjaTrader: New > NinjaScript Editor, then press F5 to compile." -ForegroundColor Cyan
