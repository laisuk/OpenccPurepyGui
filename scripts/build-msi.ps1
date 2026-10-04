param(
    # WiX source maintained in source control.
    [string]$Wxs = "installer\Product.wxs",

    # Nuitka standalone distribution.
    [string]$DistDir = "mainwindow.dist",

    # VERSION is the single source of truth.
    [string]$VersionFile = "VERSION",

    # Used for output filename.
    [string]$Arch = "win-x64",

    # Generated WiX intermediates.
    [string]$BuildDir = "installer\build",

    # Final MSI output directory.
    [string]$OutputDir = "installer",

    # Optional final MSI filename override.
    [string]$MsiName = "",

    # Open output directory after a successful build.
    [switch]$OpenOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"


# ============================================================
# Helpers
# ============================================================

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host $Title
    Write-Host ("=" * 72)
}

function Write-Info([string]$Message) {
    Write-Host "[*] $Message"
}

function Write-Ok([string]$Message) {
    Write-Host "[+] $Message"
}

function Fail([string]$Message) {
    throw $Message
}

function Require-File(
    [string]$Path,
    [string]$Description
) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "Missing $Description`: $Path"
    }
}

function Require-Directory(
    [string]$Path,
    [string]$Description
) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        Fail "Missing $Description`: $Path"
    }
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Fail (
            "Required WiX 3 tool '$Name' was not found in PATH. " +
            "Ensure WiX Toolset 3 heat/candle/light are available."
        )
    }
}

function Invoke-Checked(
    [string]$Command,
    [string[]]$Arguments
) {
    & $Command @Arguments

    if ($LASTEXITCODE -ne 0) {
        Fail "$Command failed with exit code $LASTEXITCODE."
    }
}

function Read-Version([string]$Path) {
    Require-File $Path "VERSION file"

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $value = $line.Trim()

        if ($value -and -not $value.StartsWith("#")) {
            return $value
        }
    }

    Fail "VERSION file does not contain a valid version: $Path"
}


# ============================================================
# Validate inputs
# ============================================================

$Version = Read-Version $VersionFile

Require-File $Wxs "WiX product source"
Require-Directory $DistDir "Nuitka distribution"

$mainExe = Join-Path $DistDir "OpenccPurepyGui.exe"
Require-File $mainExe "Nuitka application executable"

Require-Command "heat"
Require-Command "candle"
Require-Command "light"


# ============================================================
# Resolve paths
# ============================================================

$Wxs = (Resolve-Path -LiteralPath $Wxs).Path
$DistDir = (Resolve-Path -LiteralPath $DistDir).Path

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

if ([string]::IsNullOrWhiteSpace($MsiName)) {
    $MsiName = "OpenccPurepyGui-$Version-$Arch-setup.msi"
}

$appFilesWxs = Join-Path $BuildDir "AppFiles.wxs"
$objProduct = Join-Path $BuildDir "Product.wixobj"
$objAppFiles = Join-Path $BuildDir "AppFiles.wixobj"

$outMsi = Join-Path $OutputDir $MsiName


# ============================================================
# Summary
# ============================================================

Write-Section "MSI build"

Write-Info "Version : $Version"
Write-Info "WXS     : $Wxs"
Write-Info "DistDir : $DistDir"
Write-Info "BuildDir: $BuildDir"
Write-Info "Arch    : $Arch"
Write-Info "Output  : $outMsi"


# ============================================================
# Clean generated intermediates
# ============================================================

Write-Section "Clean generated WiX files"

$generatedFiles = @(
    $appFilesWxs,
    $objProduct,
    $objAppFiles,
    $outMsi,
    (Join-Path $OutputDir "OpenccPurepyGui-$Version-$Arch-setup.wixpdb")
)

foreach ($file in $generatedFiles) {
    if (Test-Path -LiteralPath $file) {
        Remove-Item -LiteralPath $file -Force
    }
}

Write-Ok "Generated files cleaned"


# ============================================================
# 1) Harvest Nuitka distribution
# ============================================================

Write-Section "Harvest Nuitka distribution (heat)"

Write-Info "Generating: $appFilesWxs"

$heatArgs = @(
    "dir",
    $DistDir,

    "-nologo",

    # Attach harvested files below Product.wxs INSTALLFOLDER.
    "-dr", "INSTALLFOLDER",

    # Product.wxs references this component group.
    "-cg", "AppFiles",

    # Author component GUIDs automatically.
    # Generated AppFiles.wxs therefore remains disposable.
    "-ag",

    # Product.wxs already declares INSTALLFOLDER.
    "-srd",

    # Suppress COM/self-registration harvesting.
    "-sreg",
    "-scom",

    # Generate fragments.
    "-sfrag",

    # Keep source paths relocatable for candle.
    "-var", "var.SourceDir",

    "-out", $appFilesWxs
)

Invoke-Checked "heat" $heatArgs

Require-File $appFilesWxs "generated AppFiles.wxs"

Write-Ok "Harvested Nuitka distribution"


# ============================================================
# 2) Compile Product.wxs
# ============================================================

Write-Section "Compile WiX sources (candle)"

Write-Info "Compiling Product.wxs"

$productArgs = @(
    "-nologo",
    "-arch", "x64",
    "-ext", "WixUIExtension",

    "-dSourceDir=$DistDir",
    "-dAppVersion=$Version",

    "-out", $objProduct,
    $Wxs
)

Invoke-Checked "candle" $productArgs

Require-File $objProduct "Product.wixobj"


# ============================================================
# 3) Compile generated AppFiles.wxs
# ============================================================

Write-Info "Compiling AppFiles.wxs"

$appFilesArgs = @(
    "-nologo",
    "-arch", "x64",

    "-dSourceDir=$DistDir",

    "-out", $objAppFiles,
    $appFilesWxs
)

Invoke-Checked "candle" $appFilesArgs

Require-File $objAppFiles "AppFiles.wixobj"

Write-Ok "Compiled WiX sources"


# ============================================================
# 4) Link MSI
# ============================================================

Write-Section "Link MSI (light)"

Write-Info "Linking: $outMsi"

$lightArgs = @(
    "-nologo",

    "-ext", "WixUIExtension",

    # Preserve current behavior. Remove this later if you want
    # full MSI ICE validation during release builds.
    "-sval",

    "-cultures:en-us",

    "-out", $outMsi,

    $objProduct,
    $objAppFiles
)

Invoke-Checked "light" $lightArgs

Require-File $outMsi "final MSI"


# ============================================================
# Done
# ============================================================

Write-Section "Build complete"

Write-Ok "Version: $Version"
Write-Ok "MSI: $outMsi"

if ($OpenOutput) {
    Start-Process -FilePath $OutputDir
}