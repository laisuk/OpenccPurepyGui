# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and uses
the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

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