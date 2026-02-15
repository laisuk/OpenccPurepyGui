@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: Defaults
set "ONEFILE=0"                     :: 1 = --onefile, 0 = --standalone
set "RELEASE=1"                     :: 1 = add --lto=yes
set "CONSOLE=0"                     :: 1 = show console (debug); 0 = hide
set "CLEAN=0"                       :: 1 = remove previous *.build/*.dist
set "ASSUMEYES=1"                   :: 1 = --assume-yes-for-downloads
set "ENTRY=mainwindow.py"           :: entry script
set "OUTPUTNAME=OpenccPurepyGui.exe"
set "ICON=resource\openccpurepygui.ico"

:: Parse args
:parse
if "%~1"=="" goto afterparse
if /I "%~1"=="--onefile"          set "ONEFILE=1" & shift & goto parse
if /I "%~1"=="--standalone"       set "ONEFILE=0" & shift & goto parse
if /I "%~1"=="--release"          set "RELEASE=1" & shift & goto parse
if /I "%~1"=="--no-release"       set "RELEASE=0" & shift & goto parse
if /I "%~1"=="--console"          set "CONSOLE=1" & shift & goto parse
if /I "%~1"=="--no-console"       set "CONSOLE=0" & shift & goto parse
if /I "%~1"=="--clean"            set "CLEAN=1" & shift & goto parse
if /I "%~1"=="--assume-yes"       set "ASSUMEYES=1" & shift & goto parse
if /I "%~1"=="--no-assume-yes"    set "ASSUMEYES=0" & shift & goto parse
if /I "%~1"=="--entry"            set "ENTRY=%~2" & shift & shift & goto parse
if /I "%~1"=="--output-name"      set "OUTPUTNAME=%~2" & shift & shift & goto parse
if /I "%~1"=="--icon"             set "ICON=%~2" & shift & shift & goto parse
if /I "%~1"=="-h"  goto usage
if /I "%~1"=="--help" goto usage
echo Unknown option: %~1
goto usage

:afterparse
:: Basic checks (venv assumed already active)
if not exist "%ENTRY%" (
  echo ERROR: Entry file "%ENTRY%" not found.
  exit /b 1
)
if not exist "%ICON%" (
  echo WARN: Icon "%ICON%" not found. Continuing without a custom icon.
)

:: Clean previous builds
if "%CLEAN%"=="1" (
  for /d %%D in (*.build) do rmdir /s /q "%%D"
  for /d %%D in (*.dist)  do rmdir /s /q "%%D"
)

:: Common Nuitka args
set "COMMON=--enable-plugin=pyside6 --include-package=opencc_purepy --include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts --msvc=latest --output-filename=%OUTPUTNAME%"

if exist "%ICON%" (
  set "COMMON=!COMMON! --windows-icon-from-ico=%ICON%"
)

:: Hide console for GUI app
if "%CONSOLE%"=="0" (
  set "COMMON=!COMMON! --windows-console-mode=disable"
)

:: Release flags
if "%RELEASE%"=="1" (
  set "COMMON=!COMMON! --lto=yes"
)

:: Avoid interactive downloads
if "%ASSUMEYES%"=="1" (
  set "COMMON=!COMMON! --assume-yes-for-downloads"
)

echo Nuitka build starting...
echo   OneFile:      %ONEFILE%
echo   Release:      %RELEASE%
echo   Console:      %CONSOLE%
echo   OutputName:   %OUTPUTNAME%
echo   Entry:        %ENTRY%
echo.

if "%ONEFILE%"=="1" (
  python -m nuitka --onefile %COMMON% "%ENTRY%"
) else (
  python -m nuitka --standalone %COMMON% "%ENTRY%"
)

echo.
echo Build finished.
echo Tip: output is in "<entry>.dist\" (standalone) or alongside the script (onefile).
exit /b 0

:usage
echo.
echo Usage: %~nx0 [options]
echo   --onefile ^| --standalone
echo   --release ^| --no-release
echo   --console ^| --no-console
echo   --clean
echo   --assume-yes ^| --no-assume-yes
echo   --entry ^<path^>
echo   --output-name ^<name.exe^>
echo   --icon ^<path.ico^>
exit /b 2
