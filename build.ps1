param(
    [switch]$OneFile = $false, # --onefile vs --standalone
    [switch]$Release = $true, # add --lto=yes
    [switch]$Console = $false, # show console window (default: hidden)
    [switch]$Clean = $false, # remove previous .build/.dist
    [switch]$AssumeYes = $true, # auto-yes for tool downloads
    [string]$Entry = "mainwindow.py", # entry script
    [string]$OutputName = "OpenccPurepyGui.exe", # final exe name
    [string]$Icon = "resource/openccpurepygui.ico",
    [string]$PythonExe = "python"              # which Python to use (e.g. 'py -3.13')
)

$ErrorActionPreference = "Stop"

# Basic checks
if (-not (Test-Path $Entry))
{
    Write-Error "Entry file '$Entry' not found."
}

if (-not (Test-Path $Icon))
{
    Write-Warning "Icon '$Icon' not found. The build will continue without a custom icon."
}

# Optional clean
if ($Clean)
{
    Get-ChildItem -Force -Directory | Where-Object {
        $_.Name -like "*.build" -or $_.Name -like "*.dist"
    } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# Common Nuitka args
$common = @(
    "--enable-plugin=pyside6",
    "--include-package=opencc_purepy",
    "--include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts",
    "--msvc=latest",
    "--output-filename=$OutputName"
)

# (Optional) include GUI resources; uncomment if you have a /resource folder to ship
# $common += @("--include-data-dir=resource=resource")

if (Test-Path $Icon)
{
    $common += @("--windows-icon-from-ico=$Icon")
}

# GUI app by default (no console)
if (-not $Console)
{
    $common += @("--windows-console-mode=disable")
}

# Release flags
if ($Release)
{
    $common += @("--lto=yes")
}

# CI-friendly (no interactive prompts for dependency tools)
if ($AssumeYes)
{
    $common += @("--assume-yes-for-downloads")
}

Write-Host "Nuitka build starting..."
Write-Host "  OneFile:     $OneFile"
Write-Host "  Release:     $Release"
Write-Host "  Console:     $Console"
Write-Host "  OutputName:  $OutputName"
Write-Host "  Entry:       $Entry"
Write-Host "  PythonExe:   $PythonExe"
Write-Host ""

# --- build ---
$mode = if ($OneFile)
{
    "--onefile"
}
else
{
    "--standalone"
}

Write-Host "Invoking: $PythonExe -m nuitka $mode $( $common -join ' ' ) $Entry"
& $PythonExe -m nuitka $mode $common $Entry
$code = $LASTEXITCODE

if ($code -ne 0)
{
    Write-Error "`nBuild failed with exit code $code."
    exit $code
}

# Success
$base = [IO.Path]::GetFileNameWithoutExtension($Entry)
$distDir = "$base.dist"
$outHint = if ($OneFile)
{
    (Join-Path (Get-Location) $OutputName)
}
else
{
    (Join-Path $distDir $OutputName)
}

Write-Host "`nBuild finished successfully."
Write-Host "Output: $outHint"
