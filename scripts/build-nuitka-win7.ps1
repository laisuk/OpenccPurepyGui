# build-nuitka-win7.ps1
# Windows 7 x64 known-good Nuitka command (Python 3.8 + PySide6 6.1.3 + Nuitka 0.6.16–0.6.19)

python -m nuitka `
  --standalone `
  --enable-plugin=pyside6 `
  --include-package=opencc_purepy `
  --plugin-enable=multiprocessing `
  --include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts `
  --include-data-dir=pdf_module/pdfium/win-x64=pdf_module/pdfium/win-x64 `
  --windows-icon-from-ico=resource/openccpurepygui.ico `
  --windows-disable-console `
  mainwindow.py
