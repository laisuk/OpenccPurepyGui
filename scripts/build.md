# 🧩 OpenccPurepyGui Build Notes

## 🐧 Linux
```bash
python3 -m nuitka \
  --standalone \
  --enable-plugin=pyside6 \
  --include-package=opencc_purepy \
  --include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts \
  --output-filename=OpenccPurepyGui \
  --lto=yes \
  --assume-yes-for-downloads \
  mainwindow.py
  ```

**Notes:**
- Requires `patchelf` → `sudo apt install patchelf`
- May need Qt/X11 libraries:
  ```bash
  sudo apt install libgl1 libx11-xcb1 libxkbcommon-x11-0 libxcb-cursor0 libnss3
  ```
- Output folder: `mainwindow.dist/`
- Run: `./mainwindow.dist/OpenccPurepyGui`

---

## 🪟 Windows
```powershell
python -m nuitka `
  --standalone `
  --enable-plugin=pyside6 `
  --include-package=opencc_purepy `
  --include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts `
  --output-filename=OpenccPurepyGui.exe `
  --lto=yes `
  --assume-yes-for-downloads `
  mainwindow.py
```

**Notes:**
- Run from Developer PowerShell or Command Prompt
- Optional single-file build:
  ```powershell
  python -m nuitka --onefile ... mainwindow.py
  ```
- Output folder: `mainwindow.dist\`
- Executable: `mainwindow.dist\OpenccPurepyGui.exe`

---

## 🍎 macOS
```bash
python3 -m nuitka   --standalone   --enable-plugin=pyside6   --include-package=opencc_purepy   --include-data-dir=opencc_purepy/dicts=opencc_purepy/dicts   --output-filename=OpenccPurepyGui   --lto=yes   --assume-yes-for-downloads   mainwindow.py
```

**Notes:**
- Requires Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- Nuitka automatically embeds Qt plugins
- Output: `mainwindow.dist/OpenccPurepyGui.app`
- Run:
  ```bash
  ./mainwindow.dist/OpenccPurepyGui.app/Contents/MacOS/OpenccPurepyGui
  ```

---
