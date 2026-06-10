from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union


class DeTofuLevel(IntEnum):
    ExtB = 0
    ExtC = 1
    ExtD = 2
    ExtE = 3
    ExtF = 4
    ExtG = 5
    ExtH = 6
    ExtI = 7


@dataclass(frozen=True)
class DeTofuEntry:
    tofu: str
    fallback: str
    extension: DeTofuLevel


_BUILTIN_ENTRIES: Optional[Tuple[DeTofuEntry, ...]] = None


def parse_level(value: str) -> DeTofuLevel:
    if value is None:
        raise TypeError("value must not be None")

    value = value.strip().lower()

    if value in ("all", "ext-b", "b"):
        return DeTofuLevel.ExtB
    if value in ("ext-c", "c"):
        return DeTofuLevel.ExtC
    if value in ("ext-d", "d"):
        return DeTofuLevel.ExtD  # see note below
    if value in ("ext-e", "e"):
        return DeTofuLevel.ExtE
    if value in ("ext-f", "f"):
        return DeTofuLevel.ExtF
    if value in ("ext-g", "g"):
        return DeTofuLevel.ExtG
    if value in ("ext-h", "h"):
        return DeTofuLevel.ExtH
    if value in ("ext-i", "i"):
        return DeTofuLevel.ExtI

    raise ValueError(
        "Supported DeTofu levels: all, ext-b, ext-c, ext-d, "
        "ext-e, ext-f, ext-g, ext-h, ext-i."
    )


def _parse_extension(value: str) -> Optional[DeTofuLevel]:
    value = value.strip()

    table = {
        "ExtB": DeTofuLevel.ExtB,
        "B": DeTofuLevel.ExtB,
        "b": DeTofuLevel.ExtB,
        "ExtC": DeTofuLevel.ExtC,
        "C": DeTofuLevel.ExtC,
        "c": DeTofuLevel.ExtC,
        "ExtD": DeTofuLevel.ExtD,
        "D": DeTofuLevel.ExtD,
        "d": DeTofuLevel.ExtD,
        "ExtE": DeTofuLevel.ExtE,
        "E": DeTofuLevel.ExtE,
        "e": DeTofuLevel.ExtE,
        "ExtF": DeTofuLevel.ExtF,
        "F": DeTofuLevel.ExtF,
        "f": DeTofuLevel.ExtF,
        "ExtG": DeTofuLevel.ExtG,
        "G": DeTofuLevel.ExtG,
        "g": DeTofuLevel.ExtG,
        "ExtH": DeTofuLevel.ExtH,
        "H": DeTofuLevel.ExtH,
        "h": DeTofuLevel.ExtH,
        "ExtI": DeTofuLevel.ExtI,
        "I": DeTofuLevel.ExtI,
        "i": DeTofuLevel.ExtI,
    }

    return table.get(value)


def _first_scalar(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    # Normal Python non-BMP char: already one scalar.
    first = value[0]

    # Defensive support for explicit UTF-16 surrogate-pair strings.
    if (
            0xD800 <= ord(first) <= 0xDBFF
            and len(value) >= 2
            and 0xDC00 <= ord(value[1]) <= 0xDFFF
    ):
        high = ord(first)
        low = ord(value[1])
        cp = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
        return chr(cp)

    return first


def _enumerate_scalars(text: str) -> Iterator[str]:
    i = 0
    while i < len(text):
        ch = text[i]

        if (
                0xD800 <= ord(ch) <= 0xDBFF
                and i + 1 < len(text)
                and 0xDC00 <= ord(text[i + 1]) <= 0xDFFF
        ):
            high = ord(ch)
            low = ord(text[i + 1])
            cp = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
            yield chr(cp)
            i += 2
        else:
            yield ch
            i += 1


def parse_entries(text: str) -> Tuple[DeTofuEntry, ...]:
    entries = []

    if not text:
        return tuple(entries)

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            continue

        tofu = _first_scalar(parts[0].strip())
        fallback = _first_scalar(parts[1].strip())
        extension = _parse_extension(parts[2])

        if tofu is None or fallback is None or extension is None:
            continue

        entries.append(DeTofuEntry(tofu, fallback, extension))

    return tuple(entries)


def _builtin_tofu_path() -> Path:
    return Path(__file__).resolve().parent / "dicts" / "TSCharactersTofu.txt"


def get_builtin_entries() -> Tuple[DeTofuEntry, ...]:
    global _BUILTIN_ENTRIES

    if _BUILTIN_ENTRIES is None:
        path = _builtin_tofu_path()

        if not path.exists():
            _BUILTIN_ENTRIES = tuple()
        else:
            _BUILTIN_ENTRIES = parse_entries(path.read_text(encoding="utf-8"))

    return _BUILTIN_ENTRIES


class DeTofuMap:
    def __init__(self, level: DeTofuLevel, mapping: Dict[str, str]) -> None:
        self._level = level
        self._map = mapping

    @classmethod
    def builtin(cls, level: DeTofuLevel) -> "DeTofuMap":
        mapping: Dict[str, str] = {}

        for entry in get_builtin_entries():
            if entry.extension >= level:
                mapping[entry.tofu] = entry.fallback

        return cls(level, mapping)

    def with_custom_file(self, path: Union[str, Path]) -> "DeTofuMap":
        text = Path(path).read_text(encoding="utf-8")
        return self._with_custom_entries(parse_entries(text))

    def with_custom_pairs(
            self,
            pairs: Union[
                Mapping[str, str],
                Iterable[Tuple[str, str]]
            ]
    ) -> "DeTofuMap":
        items = pairs.items() if isinstance(pairs, Mapping) else pairs

        for key, value in items:
            tofu = _first_scalar(key)
            fallback = _first_scalar(value)

            if tofu is not None and fallback is not None:
                self._map[tofu] = fallback

        return self

    def convert(self, text: Optional[str]) -> str:
        if not text or not self._map:
            return text or ""

        return "".join(self._map.get(ch, ch) for ch in _enumerate_scalars(text))

    def _with_custom_entries(self, entries: Iterable[DeTofuEntry]) -> "DeTofuMap":
        for entry in entries:
            if entry.extension >= self._level:
                self._map[entry.tofu] = entry.fallback

        return self


def detofu(
        text: Optional[str],
        level: Union[DeTofuLevel, str] = DeTofuLevel.ExtB,
) -> str:
    if isinstance(level, str):
        level = parse_level(level)

    return DeTofuMap.builtin(level).convert(text)
