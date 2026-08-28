# unicode_compat.py

from pathlib import Path
from typing import Dict, Optional


_COMPAT_MAP: Optional[Dict[str, str]] = None
_UNICODE_COMPAT_MAP: Optional[Dict[str, str]] = None


def _load_map(filename: str) -> Dict[str, str]:
    path = Path(__file__).parent / "dicts" / filename
    mapping: Dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        source = parts[0]
        target = parts[1]

        if source and target:
            mapping[source] = target

    return mapping


def _get_compat_map() -> Dict[str, str]:
    global _COMPAT_MAP

    if _COMPAT_MAP is None:
        _COMPAT_MAP = _load_map("CJK_Compatibility_Ideographs.txt")

    return _COMPAT_MAP


def _get_unicode_compat_map() -> Dict[str, str]:
    global _UNICODE_COMPAT_MAP

    if _UNICODE_COMPAT_MAP is None:
        _UNICODE_COMPAT_MAP = _load_map("Unicode_Compatibility.txt")

    return _UNICODE_COMPAT_MAP


def _apply(text: Optional[str], mapping: Dict[str, str]) -> str:
    if not text:
        return text or ""

    translation = {
        ord(source): target
        for source, target in mapping.items()
    }

    return text.translate(translation)


def normalize_compat(text: Optional[str]) -> str:
    return _apply(text, _get_compat_map())


def normalize_unicode_compat(text: Optional[str]) -> str:
    return _apply(text, _get_unicode_compat_map())


def normalize_compat_extended(text: Optional[str]) -> str:
    # Same contract as your Rust implementation:
    # Unicode compatibility/allographs first, then CJK compatibility ideographs.
    return normalize_compat(normalize_unicode_compat(text))