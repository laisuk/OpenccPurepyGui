
#!/usr/bin/env bash
set -euo pipefail

# OpenccPurepyGui Linux Nuitka build.
# Run from the project root or invoke this script by path.

ONEFILE=0
RELEASE=1
CONSOLE=0
CLEAN=0
ASSUME_YES=1

ENTRY="mainwindow.py"
OUTPUT_NAME="OpenccPurepyGui"
PYTHON_EXE="${PYTHON_EXE:-python}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --onefile / --standalone       Build mode (default: standalone)
  --release / --no-release       Enable LTO (default: release)
  --console / --no-console       Console output (default: no-console)
  --clean                        Remove previous Nuitka build folders
  --assume-yes / --no-assume-yes  Automatic tool downloads
  --entry <path>                 Entry script (default: $ENTRY)
  --output-name <name>           Executable name (default: $OUTPUT_NAME)
  --python-exe <cmd>             Python interpreter (default: $PYTHON_EXE)
  -h, --help                     Show help
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

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
        --entry) ENTRY="${2:?Missing entry}"; shift 2 ;;
        --output-name) OUTPUT_NAME="${2:?Missing name}"; shift 2 ;;
        --python-exe) PYTHON_EXE="${2:?Missing Python}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

# Resolve the project root (build.sh lives in scripts/).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

[[ -f "$ENTRY" ]] || fail "Entry file not found: $ENTRY"
[[ -f VERSION ]] || fail "VERSION file not found"

VERSION="$(
    grep -vE '^[[:space:]]*(#|$)' VERSION |
        head -n 1 |
        tr -d '\r' || true
)"
[[ -n "$VERSION" ]] || fail "VERSION file is empty"
[[ "$VERSION" =~ ^[A-Za-z0-9._+-]+$ ]] ||
    fail "Invalid VERSION value: $VERSION"

# Prefer the active virtual environment.
if [[ -z "${PYTHON_EXE_OVERRIDE:-}" &&
      -z "${PYTHON_EXE_SET_BY_FLAG:-}" &&
      "$PYTHON_EXE" == "python" &&
      -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
fi

command -v "$PYTHON_EXE" >/dev/null 2>&1 ||
    fail "Python executable not found: $PYTHON_EXE"

"$PYTHON_EXE" -m nuitka --version >/dev/null ||
    fail "Nuitka is not available in the selected Python"

# Verify that the PDFium loader package is available.
[[ -f pdf_module/pdfium_loader.py ]] ||
    echo "WARNING: PDFium loader not found; verify PDF support."

case "$(uname -m)" in
    x86_64|amd64)
        PDFIUM_UNUSED_ARCH="linux-arm64"
        ;;
    aarch64|arm64)
        PDFIUM_UNUSED_ARCH="linux-x64"
        ;;
    *)
        fail "Unsupported Linux architecture: $(uname -m)"
        ;;
esac

# Clean generated folders only.
if [[ "$CLEAN" == "1" ]]; then
    find . -maxdepth 1 -type d \
        \( -name '*.build' \
        -o -name '*.dist' \
        -o -name '*.onefile-build' \) \
        -print0 | xargs -0r rm -rf --
fi

common=(
    --enable-plugin=pyside6

    --include-package=opencc_purepy
    --include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts

    # Bundle PDFium native library in its expected relative location.
    --include-package=pdf_module

    # QApplication reads VERSION at runtime.
    --include-data-files=VERSION=VERSION

    --output-filename="$OUTPUT_NAME"
)

# Python package imports are followed by Nuitka.
# PDFium .so libraries are intentionally not copied manually.
# Linux PDFium must be resolvable by the application's loader.

if [[ "$RELEASE" == "1" ]]; then
    common+=(--lto=yes)
fi

if [[ "$ASSUME_YES" == "1" ]]; then
    common+=(--assume-yes-for-downloads)
fi

if [[ "$ONEFILE" == "1" ]]; then
    mode=(
        --onefile
        "--onefile-tempdir-spec={CACHE_DIR}/OpenccPurepyGui/$VERSION/"
    )
else
    mode=(--standalone)
fi

echo "Nuitka build starting..."
echo "  Version:     $VERSION"
echo "  OneFile:     $ONEFILE"
echo "  Release:     $RELEASE"
echo "  Console:     $CONSOLE"
echo "  OutputName:  $OUTPUT_NAME"
echo "  Entry:       $ENTRY"
echo "  PythonExe:   $PYTHON_EXE"
echo "  PDFium:      Linux system/native loader"
echo

echo "Invoking: $PYTHON_EXE -m nuitka ${mode[*]} ${common[*]} $ENTRY"
"$PYTHON_EXE" -m nuitka "${mode[@]}" "${common[@]}" "$ENTRY"

base="$(basename "$ENTRY")"
base="${base%.*}"

if [[ "$ONEFILE" == "1" ]]; then
    output="$PROJECT_ROOT/$OUTPUT_NAME"
else
    output="$PROJECT_ROOT/$base.dist/$OUTPUT_NAME"
fi

# Remove PDFium native binaries for the other architecture.
DIST_DIR="$PROJECT_ROOT/${ENTRY##*/}"
DIST_DIR="${DIST_DIR%.*}.dist"

PDFIUM_UNUSED_DIR="$DIST_DIR/pdf_module/pdfium/$PDFIUM_UNUSED_ARCH"

if [[ -d "$PDFIUM_UNUSED_DIR" ]]; then
    echo "Removing unused PDFium architecture: $PDFIUM_UNUSED_ARCH"
    rm -rf -- "$PDFIUM_UNUSED_DIR"
fi

echo
echo "Build finished successfully."
echo "Version: $VERSION"
echo "Output:  $output"
