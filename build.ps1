param(
  [switch]$OneFile = $false,          # --onefile vs --standalone
  [switch]$Release = $true,           # add --lto=yes
  [switch]$Console = $false,          # show console window (default: hidden)
  [switch]$Clean = $false,            # remove previous .build/.dist
  [switch]$AssumeYes = $true,         # auto-yes for tool downloads
  [string]$Entry = "mainwindow.py",   # entry script
  [string]$OutputName = "OpenccPurepyGui.exe", # final exe name
  [string]$Icon = "resource/openccpurepygui.ico"
)

# --- setup & checks ---
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv313u\Scripts\Activate.ps1")) {
  Write-Error "Virtual env .venv313u not found. Create it or adjust the path."
}
. .\.venv313u\Scripts\Activate.ps1

if (-not (Test-Path $Entry)) {
  Write-Error "Entry file '$Entry' not found."
}

if (-not (Test-Path $Icon)) {
  Write-Warning "Icon '$Icon' not found. The build will continue without a custom icon."
}

if ($Clean) {
  Get-ChildItem -Force -Recurse -ErrorAction SilentlyContinue `
    | Where-Object { $_.PSIsContainer -and ($_.Name -like "*.build" -or $_.Name -like "*.dist") } `
    | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# --- common args ---
$common = @(
  "--enable-plugin=pyside6",
  "--include-package=opencc_purepy",
  "--include-data-dir=resource=resource",
  "--include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts",
  "--msvc=latest",
  "--output-filename=$OutputName"
)

if (Test-Path $Icon) { $common += @("--windows-icon-from-ico=$Icon") }

# GUI app by default (no console)
if (-not $Console) { $common += @("--windows-console-mode=disable") }

# Release flags
if ($Release) { $common += @("--lto=yes") }

# CI-friendly (no interactive prompts for dependency tools)
if ($AssumeYes) { $common += @("--assume-yes-for-downloads") }

# --- build ---
Write-Host "Nuitka build starting..."
Write-Host "  OneFile:    $OneFile"
Write-Host "  Release:    $Release"
Write-Host "  Console:    $Console"
Write-Host "  OutputName: $OutputName"
Write-Host "  Entry:      $Entry"
Write-Host ""

if ($OneFile) {
  python -m nuitka --onefile $common $Entry
} else {
  python -m nuitka --standalone $common $Entry
}

Write-Host ""
Write-Host "Build finished."
Write-Host "Tip: your exe will be in '<entry>.dist\' (standalone) or next to the script (onefile)."
