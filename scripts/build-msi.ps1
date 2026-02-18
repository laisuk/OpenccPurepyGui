param(
    # WiX source
    [string]$Wxs = "installer\Product.wxs",

    # Nuitka/PySide dist folder
    [string]$DistDir = "mainwindow.dist",

    # MSI metadata (MSI requires x.y.z)
    [string]$Ver = "1.2.0",

    # For filename only
    [string]$Arch = "win-x64",

    # WiX intermediate/output folder
    [string]$WixOut = "installer",

    # Output MSI name (optional override)
    [string]$MsiName = "",

    # Extra: open the output folder when done
    [switch]$OpenOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ----------------------------
# Pretty logging helpers
# ----------------------------
function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host $Title
    Write-Host ("=" * 72)
}

function Write-Info([string]$Msg)  { Write-Host "[*] $Msg" }
function Write-Ok([string]$Msg)    { Write-Host "[+] $Msg" }
function Write-Warn([string]$Msg)  { Write-Warning $Msg }
function Fail([string]$Msg)        { throw $Msg }

function Require-File([string]$Path, [string]$What) {
    if (-not (Test-Path $Path)) { Fail "Missing $($What): $Path" }
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Fail "Required tool '$Name' not found in PATH. Install WiX Toolset and ensure heat/candle/light are available."
    }
}

# ----------------------------
# Resolve inputs
# ----------------------------
Require-File $Wxs "WiX source (.wxs)"
Require-File $DistDir "dist directory"

$DistDir = (Resolve-Path $DistDir).Path
$Wxs     = (Resolve-Path $Wxs).Path

if ([string]::IsNullOrWhiteSpace($MsiName)) {
    $MsiName = "$WixOut\OpenccPurepyGui-$Ver-$Arch-setup.msi"
}

# WiX paths
New-Item -ItemType Directory -Force -Path $WixOut | Out-Null
$WixOut = (Resolve-Path $WixOut).Path

$appFilesWxs = Join-Path $WixOut "AppFiles.wxs"
$objProduct  = Join-Path $WixOut "Product.wixobj"
$objAppFiles = Join-Path $WixOut "AppFiles.wixobj"
$outMsi      = Join-Path (Get-Location) $MsiName

# Tools
Require-Command "heat"
Require-Command "candle"
Require-Command "light"

Write-Section "Inputs"
Write-Info "WXS     : $Wxs"
Write-Info "DistDir : $DistDir"
Write-Info "WixOut  : $WixOut"
Write-Info "Version : $Ver"
Write-Info "Arch    : $Arch"
Write-Info "Output  : $outMsi"

# ----------------------------
# 1) Harvest dist -> AppFiles.wxs
# ----------------------------
Write-Section "Harvest dist folder (heat)"
Write-Info "Generating: $appFilesWxs"

# Notes:
# -dr INSTALLFOLDER : put harvested files under your INSTALLFOLDER directory
# -cg AppFiles      : matches <ComponentGroupRef Id="AppFiles" />
# -gg               : generate stable GUIDs
# -srd              : suppress root directory element (we already have INSTALLFOLDER)
# -sfrag            : output as fragment
# -sreg             : suppress self-reg harvesting (silences HEAT5150 for non-COM DLLs)
# -var var.SourceDir: use $(var.SourceDir) in generated Source paths
& heat dir "$DistDir" `
    -nologo `
    -dr INSTALLFOLDER `
    -cg AppFiles `
    -gg `
    -srd `
    -sfrag `
    -sreg `
    -var var.SourceDir `
    -out "$appFilesWxs"

Require-File $appFilesWxs "heat output (AppFiles.wxs)"
Write-Ok "Harvested files -> AppFiles.wxs"

# ----------------------------
# 2) Compile .wxs -> .wixobj
# ----------------------------
Write-Section "Compile (candle)"
Write-Info "Compiling Product.wxs -> $objProduct"
& candle `
    -nologo `
    -ext WixUIExtension `
    -dSourceDir="$DistDir" `
    -dAppVersion="$Ver" `
    -out "$objProduct" `
    "$Wxs"

Write-Info "Compiling AppFiles.wxs -> $objAppFiles"
& candle `
    -nologo `
    -dSourceDir="$DistDir" `
    -out "$objAppFiles" `
    "$appFilesWxs"

Require-File $objProduct  "candle output (Product.wixobj)"
Require-File $objAppFiles "candle output (AppFiles.wixobj)"
Write-Ok "Compiled .wxs -> .wixobj"

# ----------------------------
# 3) Link .wixobj -> .msi
# ----------------------------
Write-Section "Link (light)"
Write-Info "Linking -> $outMsi"

# -sval: suppress ICE validation errors (keep if you want “best-effort” builds)
& light `
    -nologo `
    -ext WixUIExtension `
    -sval `
    -out "$outMsi" `
    "$objProduct" "$objAppFiles"

Require-File $outMsi "final MSI"
Write-Ok "Built MSI: $outMsi"

if ($OpenOutput) {
    Write-Info "Opening output folder..."
    Start-Process -FilePath (Split-Path -Parent $outMsi)
}
