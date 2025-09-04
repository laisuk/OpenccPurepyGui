Set-Location -LiteralPath $PSScriptRoot
$UiPath = [System.IO.Path]::combine($PSScriptRoot, "form.ui")
$QrcPath = [System.IO.Path]::combine($PSScriptRoot, "resource.qrc")
if (Test-Path -LiteralPath $UiPath)
{
    $fileUI = Get-Item "./form.ui"
    $lastSaveDateUI = $fileUI.LastWriteTime
    "form.ui   : " + $lastSaveDateUI

    $fileUiPy = Get-Item "./ui_form.py"
    $lastSaveDateUiPy = $fileUiPy.LastWriteTime
    "ui_form.py: " + $lastSaveDateUiPy

    Write-Host "form.ui Newer than ui_form.py : " -NoNewline
    $IsUiNewer = ($lastSaveDateUI -gt $lastSaveDateUiPy)
    if ($IsUiNewer)
    {
        Write-Host $IsUiNewer -ForegroundColor Green
        & pyside6-uic form.ui -o ui_form.py
        Write-Host "ui_form.py updated." -ForegroundColor Blue
    }
    else
    {
        Write-Host $IsUiNewer -ForegroundColor Red
        Write-Host "No Ui update needed." -ForegroundColor Blue
    }
}
else
{
    Write-Host "form.ui not found." -ForegroundColor Blue
}

if ([System.IO.File]::Exists($QrcPath))
{
    $fileQrc = Get-Item "./resource.qrc"
    $lastSaveDateQrc = $fileQrc.LastWriteTime
    "resource.qrc  : " + $lastSaveDateQrc

    $fileQrcPy = Get-Item "./resource_rc.py"
    $lastSaveDateQrcPy = $fileQrcPy.LastWriteTime
    "resource_rc.py: " + $lastSaveDateQrcPy

    Write-Host "resource.qrc Newer than resource_rc.py : " -NoNewline
    $IsQrcNewer = ($lastSaveDateQrc -gt $lastSaveDateQrcPy)
    if ($IsQrcNewer)
    {
        Write-Host $IsQrcNewer -ForegroundColor Green
        & pyside6-rcc resource.qrc -o resource_rc.py
        Write-Host "resource_rc.py updated." -ForegroundColor Blue
    }
    else
    {
        Write-Host $IsQrcNewer -ForegroundColor Red
        Write-Host "No Qrc update needed." -ForegroundColor Blue
    }
}
else
{
    Write-Host "resource.qrc not found." -ForegroundColor Blue
}
