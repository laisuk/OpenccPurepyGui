# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and uses
the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

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
