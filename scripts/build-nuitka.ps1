param(
    [switch]$OneFile = $false, # --onefile vs --standalone
    [switch]$Release = $true, # add --lto=yes
    [switch]$Console = $false, # show console window (default: hidden)
    [switch]$Clean = $false, # remove previous Nuitka build output
    [switch]$AssumeYes = $true, # auto-yes for tool downloads
    [string]$Entry = "mainwindow.py",
    [string]$OutputName = "OpenccPurepyGui.exe",
    [string]$Icon = "resource/openccpurepygui.ico",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Fail([string]$Message)
{
    Write-Error $Message
    exit 1
}

function Confirm-File([string]$Path, [string]$Description)
{
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf))
    {
        Fail "$Description not found: '$Path'"
    }
}

function Read-Version([string]$Path)
{
    Confirm-File $Path "VERSION file"

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8)
    {
        $value = $line.Trim()
        if ($value -and -not $value.StartsWith("#"))
        {
            return $value
        }
    }

    Fail "VERSION file does not contain a valid version: '$Path'"
}

# Project / version
Confirm-File $Entry "Entry file"

$VersionFile = "VERSION"
$Version = Read-Version $VersionFile

if (-not (Test-Path -LiteralPath $Icon -PathType Leaf))
{
    Write-Warning "Icon '$Icon' not found. The build will continue without a custom icon."
}

# Clean previous Nuitka output
if ($Clean)
{
    Get-ChildItem -Force -Directory |
            Where-Object {
                $_.Name -like "*.build" -or
                        $_.Name -like "*.dist" -or
                        $_.Name -like "*.onefile-build"
            } |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# Detect PDFium platform folder (matches pdfium_loader.py convention).
function Get-PdfiumPlatformFolder
{
    # Query the selected Python interpreter, not the PowerShell host.
    $machine = (& $PythonExe -c "import platform; print(platform.machine().lower())").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $machine)
    {
        Fail "Unable to determine Python architecture using '$PythonExe'."
    }

    switch ($machine)
    {
        { $_ -in @("arm64", "aarch64") } { return "win-arm64" }
        { $_ -in @("amd64", "x86_64") } { return "win-x64" }
        { $_ -in @("x86", "i386", "i686") } { return "win-x86" }
        default { Fail "Unsupported Python architecture: '$machine'" }
    }
}

# Common Nuitka arguments
$common = @(
    "--enable-plugin=pyside6",

    "--include-package=opencc_purepy",
    "--include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts",

    # QApplication reads this at runtime.
    "--include-data-files=VERSION=VERSION",

    "--msvc=latest",
    "--output-filename=$OutputName"
)

# Bundle PDFium native library when available.
$pdfiumPlat = Get-PdfiumPlatformFolder
$pdfiumDir = "pdf_module/pdfium/$pdfiumPlat"
$pdfiumDll = Join-Path $pdfiumDir "pdfium.dll"

if (Test-Path -LiteralPath $pdfiumDll -PathType Leaf)
{
    $common += "--include-raw-dir=$pdfiumDir=$pdfiumDir"
    Write-Host "PDFium: bundling natives from '$pdfiumDir'"
}
else
{
    Write-Warning "PDFium: missing '$pdfiumDll' (PDF will be disabled in this build)"
}

# resource_rc.py embeds Qt resources, so the whole resource directory is not
# copied into the runtime distribution. The ICO is still used at build time.
if (Test-Path -LiteralPath $Icon -PathType Leaf)
{
    $common += "--windows-icon-from-ico=$Icon"
}

if (-not $Console)
{
    $common += "--windows-console-mode=disable"
}

if ($Release)
{
    $common += "--lto=yes"
}

if ($AssumeYes)
{
    $common += "--assume-yes-for-downloads"
}

# Build mode
if ($OneFile)
{
    $mode = @(
        "--onefile",
        "--onefile-tempdir-spec={CACHE_DIR}/OpenccPurepyGui/$Version/"
    )
}
else
{
    $mode = @("--standalone")
}

# Summary
Write-Host "Nuitka build starting..."
Write-Host "  Version:     $Version"
Write-Host "  OneFile:     $OneFile"
Write-Host "  Release:     $Release"
Write-Host "  Console:     $Console"
Write-Host "  OutputName:  $OutputName"
Write-Host "  Entry:       $Entry"
Write-Host "  PythonExe:   $PythonExe"
Write-Host ""

# Build
Write-Host "Invoking: $PythonExe -m nuitka $( ($mode + $common) -join ' ' ) $Entry"

& $PythonExe -m nuitka @mode @common $Entry
$code = $LASTEXITCODE

if ($code -ne 0)
{
    Fail "Build failed with exit code $code."
}

# Result
$base = [IO.Path]::GetFileNameWithoutExtension($Entry)
$distDir = "$base.dist"

if ($OneFile)
{
    $outHint = Join-Path (Get-Location) $OutputName
}
else
{
    $outHint = Join-Path $distDir $OutputName
}

Write-Host ""
Write-Host "Build finished successfully."
Write-Host "Version: $Version"
Write-Host "Output:  $outHint"
