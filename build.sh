#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via flags or env PYTHON_EXE)
ONEFILE=0                  # 1 = --onefile, 0 = --standalone
RELEASE=1                  # 1 = add --lto=yes
CONSOLE=0                  # no-op on Linux; kept for parity
CLEAN=0                    # 1 = remove previous .build/.dist
ASSUME_YES=1               # 1 = --assume-yes-for-downloads
ENTRY="mainwindow.py"      # entry script
OUTPUT_NAME="OpenccPurepyGui"   # final binary name (no .exe on Linux)
ICON="resource/openccpurepygui.png"  # not used directly by Nuitka on Linux
PYTHON_EXE="${PYTHON_EXE:-python3}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --onefile / --standalone      Build as single binary or folder (default: --standalone)
  --release / --no-release      Enable LTO optimizations (default: --release)
  --console / --no-console      (no-op on Linux; kept for parity)
  --clean                       Remove previous *.build and *.dist folders
  --assume-yes / --no-assume-yes  Auto-yes for tool downloads (default: yes)
  --entry <path>                Entry Python file (default: $ENTRY)
  --output-name <name>          Output binary name (default: $OUTPUT_NAME)
  --icon <png>                  PNG icon (not used by Nuitka on Linux)
  --python-exe <cmd>            Python to run (default: $PYTHON_EXE)
  -h, --help                    Show this help
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onefile) ONEFILE=1; shift ;;
    --standalone) ONEFILE=0; shift ;;
    --release) RELEASE=1; shift ;;
    --no-release) RELEASE=0; shift ;;
    --console) CONSOLE=1; shift ;;
    --no-console) CONSOLE=0; shift ;;
    --clean) CLEAN=1; shift ;;
    --assume-yes) ASSUME_YES=1; shift ;;
    --no-assume-yes) ASSUME_YES=0; shift ;;
    --entry) ENTRY="${2:?}"; shift 2 ;;
    --output-name) OUTPUT_NAME="${2:?}"; shift 2 ;;
    --icon) ICON="${2:?}"; shift 2 ;;
    --python-exe) PYTHON_EXE="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

# Pick a Python if the chosen one isn't available
if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_EXE=python
  else
    echo "ERROR: Python executable not found (tried '$PYTHON_EXE' and 'python')." >&2
    exit 1
  fi
fi

# Entry check
if [[ ! -f "$ENTRY" ]]; then
  echo "ERROR: Entry file '$ENTRY' not found." >&2
  exit 1
fi

# Icon warning (not used directly by Nuitka on Linux)
if [[ ! -f "$ICON" ]]; then
  echo "WARN: Icon '$ICON' not found. Continuing without a custom icon."
fi

# Clean previous builds
if [[ "$CLEAN" == "1" ]]; then
  find . -maxdepth 1 -type d \( -name "*.build" -o -name "*.dist" \) -print0 | xargs -0r rm -rf
fi

# Common Nuitka args
common=(
  --enable-plugin=pyside6
  --include-package=opencc_purepy
  --include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts
  --output-filename="$OUTPUT_NAME"
)

# (Optional) include GUI resources directory if you have one:
# common+=( --include-data-dir=resource=resource )

# Release flags
if [[ "$RELEASE" == "1" ]]; then
  common+=( --lto=yes )
fi

# CI-friendly (no interactive prompts)
if [[ "$ASSUME_YES" == "1" ]]; then
  common+=( --assume-yes-for-downloads )
fi

echo "Nuitka build starting..."
echo "  OneFile:     $ONEFILE"
echo "  Release:     $RELEASE"
echo "  Console:     $CONSOLE (no-op on Linux)"
echo "  OutputName:  $OUTPUT_NAME"
echo "  Entry:       $ENTRY"
echo "  PythonExe:   $PYTHON_EXE"
echo

if [[ "$ONEFILE" == "1" ]]; then
  "$PYTHON_EXE" -m nuitka --onefile "${common[@]}" "$ENTRY"
else
  "$PYTHON_EXE" -m nuitka --standalone "${common[@]}" "$ENTRY"
fi

echo
echo "Build finished."
echo "Tip: output is in '<entry>.dist/' (standalone) or alongside the script (onefile)."
