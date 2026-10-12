# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and uses
the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [1.4.0] - Unreleased

### Added

- Synced the embedded `opencc-purepy` core with upstream v1.5.0 development, adding Small Seal Script configs
  `s2seal`, `t2seal`, `seal2s`, and `seal2t` to the Python API and text/Office CLI.
- Added four Seal dictionary slots and their text/JSON data, including custom dictionary and `dictgen` support.
  Legacy dictionary files may omit the new slots. Existing indexed matching optimizations are retained.

### Fixed

- Moved the About dialog import to startup to avoid the lazy-import crash
  reproduced in OpenccPyo3Gui with Nuitka 4.3 / Python 3.13 / PySide6 6.12.
  OpenccPurepyGui contained the same import inside its About menu callback.
  Verified About, Copy Info, and OK in an LTO-enabled compiled smoke test (exit code `0`) and the interpreter; all 79
  existing tests passed.

- Replaced the Custom Dictionary Apply button's lambda callback with a named
  `_apply_rows` method decorated with `@Slot()`, preserving emission of the current
  dictionary rows while avoiding a compiled-callback compatibility issue.
    - **Context (2026-10-12):** this project contained the exact callback pattern
      associated with a confirmed Nuitka standalone startup segmentation fault in
      the sibling `OpenccPyo3Gui` project:
      `self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.rows()))`.
      The sibling executable built successfully but exited immediately with code `3`;
      a `--trace-execution` build localized the crash to this connection during
      widget construction, before the Apply button was clicked.
    - **Relevant versions:** this project's `.venv` uses Python `3.13.16`,
      PySide6/Shiboken `6.12.0`, and Nuitka `4.3`, matching the affected sibling
      environment. The sibling's working older environment used Python `3.8.10`
      and PySide6/Shiboken `6.6.3.1`, also with Nuitka `4.3`.
    - **Why this form:** `@Slot()` registers an explicit zero-argument Qt slot on
      the widget; connecting the named method avoids the failing compiled-lambda
      path. This is a compatibility workaround, not evidence that lambdas are
      generally invalid. The sibling fix changed both the callable form and slot
      registration, so the decorator alone has not been proven responsible.
      Python and PySide6 also changed together; the underlying defect has not been
      attributed to a specific component. The original crash was confirmed in
      `OpenccPyo3Gui`; this project receives the same fix for the shared pattern.
    - **Verification in this project:** all 79 existing `test_*.py` tests passed
      under `.venv`. An interpreted GUI smoke check constructed the main window
      and confirmed that clicking Apply emits the expected rows for both empty and
      populated tables. Rebuilt the standalone release using
      `scripts/build-nuitka.ps1 -PythonExe ./.venv/Scripts/python.exe` with default
      LTO enabled; the executable opened from `mainwindow.dist`, displayed its
      main window, and closed normally with exit code `0` and no captured output.
    - **Future troubleshooting:** capture executable stdout/stderr for immediate
      exits, compare Python/PySide6/Shiboken/Nuitka versions, and use a separate
      console-enabled `--trace-execution` build to identify the startup connection.
      Prefer named `@Slot(...)` methods for QObject callbacks when practical.
      See [Qt's signal and slot documentation](https://doc.qt.io/qtforpython-6.11/tutorials/basictutorial/signals_and_slots.html)
      and [Nuitka's segmentation-fault debugging guidance](https://nuitka.net/info/segfault.html).

- Fixed EPUB extraction of HTML named entities by converting them
  into XML-safe numeric character references, preventing XHTML
  parsing errors while preserving inline text spacing.

---

## [1.3.0] - 2026-10-05

### Added

- Added `normCompat()` and `detofu()` API functions.
- Added an editor font picker.
- Added automatic CJK text encoding detection when opening or dropping plain text files, with detected encodings
  reflected in the encoding selector.
- Added manual text encoding selection and reload support for UTF-8, GB18030/GBK, Big5/CP950, Big5-HKSCS, UTF-16 LE, and
  UTF-16 BE.
- Added direct Hong Kong phrase conversion configs and APIs: `t2hkp` / `OpenCC.t2hkp()` and `hk2tp` /
  `OpenCC.hk2tp()`.

### Changed

- Synced the embedded `opencc-purepy` package with upstream v1.4.4 development: shared normalization/conversion/DeTofu
  pipeline, Office/EPUB CLI options and filename conversion, transactional document output, and refined PPTX part
  selection.
- Adapted GUI batch Office conversion to the upstream text callback API while preserving punctuation conversion and
  fonts.
- Synced the embedded `opencc-purepy` v1.4.2 runtime refinements, including sequential custom dictionary application so
  repeated `-D` options preserve command-line order and late entries win as expected.
- Hardened the bundled CLI error handling, configuration and dictionary-slot normalization, DeTofu validation, and
  punctuation dictionary data.
- Flattened direct `t2twp` and `tw2tp` conversion from two dictionary passes to one using the Taiwan triple unions.
- Renamed the internal Taiwan and Hong Kong triple union keys to `TwTriple`, `TwRevTriple`, `HkTriple`, and
  `HkRevTriple`.
- Refactored `s2twp` from three conversion rounds to two rounds by combining Taiwan phrase and variant normalization
  into one round, matching upstream OpenCC behavior and improving conversion efficiency.
- Optimized dictionary matching with precomputed descending starter-indexed candidate-length tuples, substantially
  reducing conversion overhead.
- Optimized text reflow.
- Updated dictionary data.

---

## [1.2.2] - 2026-05-14

### Changed

- Updated dictionary data.
- Optimized `StarterUnion` preparation by replacing Python-level key checks with reversed `dict.update()` merging while
  preserving precedence semantics.
- Reduced `StarterUnion` cache preparation overhead during dictionary merge operations.

---

## [1.2.0] - 2026-02-18

### Added

- Added support for `text-embedded PDF`

### Changed

- Updated dictionary to v1.2.0.
- Code optimization.

---

## [1.1.0] - 2025-09-22

### Added

- Implement st_punctuations and ts_punctuations, dropped manual convert punctuations

---

## [1.0.0] – 2025-09-05

### Added

- Initial release of `OpenccPurepyGui` for `opencc-purepy`.
- Pure Python OpenCC-compatible engine for Traditional and Simplified Chinese text conversion.
- Supported standard OpenCC configs:
    - `s2t`, `s2tw`, `s2twp`, `s2hk`, `t2s`, `tw2s`, `tw2sp`, `hk2s`, `jp2t`, `t2jp`
- Support conversion of plain text, Office documents and Epub.
