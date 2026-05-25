from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple, Union, Mapping, Optional  # type checking

from .dict_slot import DictSlot, DictSlotLike

PathLike = Union[str, Path]
SlotPathMap = Mapping[DictSlotLike, PathLike]
SlotPairsMap = Optional[Mapping[DictSlotLike, Mapping[str, str]]]


class DictionaryMaxlength:
    """
    A container for OpenCC-compatible dictionaries with each represented
    as a (dict, max_length) tuple to optimize the longest match lookup.
    """
    # Immutable, subclass-overridable
    DICT_FIELDS: Tuple[str, ...] = (
        "st_characters", "st_phrases", "st_punctuations",
        "ts_characters", "ts_phrases", "ts_punctuations",
        "tw_phrases", "tw_phrases_rev",
        "tw_variants", "tw_variants_rev", "tw_variants_rev_phrases",
        "hk_variants", "hk_variants_rev", "hk_variants_rev_phrases",
        "jps_characters", "jps_phrases",
        "jp_variants", "jp_variants_rev",
    )

    def __init__(self):
        """
        Initialize all supported dictionary attributes to empty dicts with max_length = 0.
        """
        self.st_characters: Tuple[Dict[str, str], int] = ({}, 0)
        self.st_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.st_punctuations: Tuple[Dict[str, str], int] = ({}, 0)
        self.ts_characters: Tuple[Dict[str, str], int] = ({}, 0)
        self.ts_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.ts_punctuations: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_phrases_rev: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_variants: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_variants_rev: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_variants_rev_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_variants: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_variants_rev: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_variants_rev_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.jps_characters: Tuple[Dict[str, str], int] = ({}, 0)
        self.jps_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.jp_variants: Tuple[Dict[str, str], int] = ({}, 0)
        self.jp_variants_rev: Tuple[Dict[str, str], int] = ({}, 0)

    def __repr__(self):
        count = sum(bool(v[0]) for v in self.__dict__.values())
        return "<DictionaryMaxlength with {} loaded dicts>".format(count)

    @classmethod
    def new(cls):
        """
        Shortcut to load from precompiled JSON for fast startup.
        :return: DictionaryMaxlength instance
        """
        return cls.from_json()

    @staticmethod
    def _as_tuple(value):
        """
        Prefer canonical array form: [ { map }, max_length ].
        Accept legacy object form: {"map": {...}, "maxlength": ...} (with warning).
        """
        # Canonical list/tuple form
        if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], dict):
            return value[0], int(value[1])

        # Legacy object form
        if isinstance(value, dict) and "map" in value and "maxlength" in value:
            warnings.warn(
                "Dictionary slot loaded in legacy object form; prefer [ {map}, max ] array form.",
                DeprecationWarning,
                stacklevel=2,
            )
            return value["map"], int(value["maxlength"])

        # Fallback
        return {}, 0

    @classmethod
    def from_json(cls):
        """
        Load dictionary data from JSON, tolerant to multiple shapes:
        - Each dict field can be [map, max_length] OR {"map": ..., "maxlength": ...}.
        - Unknown/non-dictionary keys (e.g., 'starter_index', 'version') are ignored.
        """
        import json

        path = Path(__file__).parent / "dicts" / "dictionary_maxlength.json"
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        instance = cls()

        for name in cls.DICT_FIELDS:
            if name in raw_data:
                setattr(instance, name, cls._as_tuple(raw_data[name]))
            # else: keep constructor default ({}, 0)

        return instance

    @classmethod
    def from_dicts(
            cls,
            base_dir: Optional[PathLike] = None,
            paths: Optional[SlotPathMap] = None,
            overrides: Optional[SlotPathMap] = None,
            appends: Optional[SlotPathMap] = None,
    ) -> "DictionaryMaxlength":
        """
        Load dictionaries directly from text files in the 'dicts' folder.
        Each file should contain tab-separated mappings.
        :return: Populated DictionaryMaxlength instance
        """
        paths = cls._normalize_slot_path_map(paths)
        overrides = cls._normalize_slot_path_map(overrides)
        appends = cls._normalize_slot_path_map(appends)

        if base_dir is not None:
            cls.validate_dicts_dir(base_dir)

        instance = cls()
        default_paths = {
            'st_characters': "STCharacters.txt",
            'st_phrases': "STPhrases.txt",
            'st_punctuations': "STPunctuations.txt",
            'ts_characters': "TSCharacters.txt",
            'ts_phrases': "TSPhrases.txt",
            'ts_punctuations': "TSPunctuations.txt",
            'tw_phrases': "TWPhrases.txt",
            'tw_phrases_rev': "TWPhrasesRev.txt",
            'tw_variants': "TWVariants.txt",
            'tw_variants_rev': "TWVariantsRev.txt",
            'tw_variants_rev_phrases': "TWVariantsRevPhrases.txt",
            'hk_variants': "HKVariants.txt",
            'hk_variants_rev': "HKVariantsRev.txt",
            'hk_variants_rev_phrases': "HKVariantsRevPhrases.txt",
            'jps_characters': "JPShinjitaiCharacters.txt",
            'jps_phrases': "JPShinjitaiPhrases.txt",
            'jp_variants': "JPVariants.txt",
            'jp_variants_rev': "JPVariantsRev.txt",
        }

        # ------------------------------------------------------------------
        # Backward-compatible legacy behavior
        # ------------------------------------------------------------------

        mapping = default_paths.copy()

        if paths:
            mapping.update(paths)

        base = (
            Path(base_dir)
            if base_dir is not None
            else Path(__file__).parent / "dicts"
        )

        # ------------------------------------------------------------------
        # Resolve initial dictionary file paths
        # ------------------------------------------------------------------

        file_map = {
            attr: base / filename
            for attr, filename in mapping.items()
        }

        # ------------------------------------------------------------------
        # Apply full replacement overrides
        # ------------------------------------------------------------------

        if overrides:
            for attr, path in overrides.items():
                if attr not in file_map:
                    raise ValueError(
                        "Unknown dictionary slot: {}".format(attr)
                    )

                file_map[attr] = Path(path)

        # ------------------------------------------------------------------
        # Load base dictionaries
        # ------------------------------------------------------------------

        for attr, path in file_map.items():
            content = path.read_text(encoding="utf-8")

            setattr(
                instance,
                attr,
                cls.load_dictionary_maxlength(content),
            )

        # ------------------------------------------------------------------
        # Apply append dictionaries (late-comer wins)
        # ------------------------------------------------------------------

        if appends:
            for attr, path in appends.items():
                if not hasattr(instance, attr):
                    raise ValueError(
                        "Unknown dictionary slot: {}".format(attr)
                    )

                base_dict, base_max = getattr(instance, attr)

                content = Path(path).read_text(encoding="utf-8")

                append_dict, append_max = (
                    cls.load_dictionary_maxlength(content)
                )

                # Late-comer wins
                base_dict.update(append_dict)

                setattr(
                    instance,
                    attr,
                    (
                        base_dict,
                        max(base_max, append_max),
                    ),
                )

        return instance

    @staticmethod
    def load_dictionary_maxlength(content: str) -> Tuple[Dict[str, str], int]:
        """
        Load a dictionary from plain text and determine the max phrase length.

        :param content: Raw dictionary text (one mapping per line)
        :return: Tuple of dict and max key length
        """
        dictionary = {}
        max_length = 1

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)  # safer for whitespace in values
            if len(parts) == 2:
                phrase, translation = parts
                translation = translation.split()[0]  # take first space-separated part
                dictionary[phrase] = translation
                max_length = max(max_length, len(phrase))
            else:
                import warnings
                warnings.warn(f"Ignoring malformed dictionary line: {line}")

        return dictionary, max_length

    def with_custom_dicts(
            self,
            overrides: Optional[SlotPairsMap] = None,
            appends: Optional[SlotPairsMap] = None,
    ) -> "DictionaryMaxlength":
        """
        Apply in-memory custom dictionary pairs after this DictionaryMaxlength
        has already been loaded.

        Unlike OpenCC text dictionary files, this API preserves exact keys,
        including keys with spaces.

        override:
            Replace the whole slot.

        append:
            Merge into the existing slot. Duplicate keys use late-comer wins.
        """
        # self._ensure_mutable()

        normalized_overrides = {
            self._normalize_slot(slot): pairs
            for slot, pairs in (overrides or {}).items()
        }

        normalized_appends = {
            self._normalize_slot(slot): pairs
            for slot, pairs in (appends or {}).items()
        }

        for attr, pairs in normalized_overrides.items():
            if not hasattr(self, attr):
                raise ValueError("Unknown dictionary slot: {}".format(attr))

            new_dict = dict(pairs)
            max_len = max((len(k) for k in new_dict), default=0)
            setattr(self, attr, (new_dict, max_len))

        for attr, pairs in normalized_appends.items():
            if not hasattr(self, attr):
                raise ValueError("Unknown dictionary slot: {}".format(attr))

            base_dict, base_max = getattr(self, attr)
            merged = dict(base_dict)
            merged.update(pairs)

            append_max = max((len(k) for k in pairs), default=0)
            setattr(self, attr, (merged, max(base_max, append_max)))

        return self

    def with_custom_dict_files(
            self,
            overrides: Optional[SlotPathMap] = None,
            appends: Optional[SlotPathMap] = None,
    ) -> "DictionaryMaxlength":
        """
        Apply OpenCC-compatible custom dictionary files after this
        DictionaryMaxlength has already been loaded.

        Dictionary files follow the normal OpenCC whitespace-separated format.
        Keys are parsed from the first column, so leading spaces or embedded
        spaces in keys are not preserved. Use with_custom_dicts() for exact
        in-memory keys.
        """
        # self._ensure_mutable()

        normalized_overrides = self._normalize_slot_path_map(overrides) or {}
        normalized_appends = self._normalize_slot_path_map(appends) or {}

        for attr, path in normalized_overrides.items():
            if not hasattr(self, attr):
                raise ValueError("Unknown dictionary slot: {}".format(attr))

            content = Path(path).read_text(encoding="utf-8")
            setattr(self, attr, self.load_dictionary_maxlength(content))

        for attr, path in normalized_appends.items():
            if not hasattr(self, attr):
                raise ValueError("Unknown dictionary slot: {}".format(attr))

            base_dict, base_max = getattr(self, attr)

            content = Path(path).read_text(encoding="utf-8")
            append_dict, append_max = self.load_dictionary_maxlength(content)

            merged = dict(base_dict)
            merged.update(append_dict)

            setattr(self, attr, (merged, max(base_max, append_max)))

        return self

    @staticmethod
    def _normalize_slot(slot: DictSlotLike) -> str:
        if isinstance(slot, DictSlot):
            return slot.value

        if slot in DictSlot._value2member_map_:
            return slot

        try:
            return DictSlot.__members__[slot].value
        except KeyError:
            raise ValueError("Unknown dictionary slot: {}".format(slot)) from None

    @classmethod
    def _normalize_slot_path_map(
            cls,
            mapping: Optional[SlotPathMap],
    ) -> Optional[Dict[str, PathLike]]:
        if mapping is None:
            return None

        return {
            cls._normalize_slot(slot): path
            for slot, path in mapping.items()
        }

    @staticmethod
    def validate_dicts_dir(path: PathLike) -> None:
        """
        Validate an OpenCC dictionary directory.

        Ensures the directory exists and contains all required
        OpenCC dictionary files.

        :param path:
            Dictionary directory path.

        :raises FileNotFoundError:
            If the directory or required dictionary files are missing.
        """

        base = Path(path)

        if not base.is_dir():
            raise FileNotFoundError(
                "Dictionary directory does not exist: {}".format(base)
            )

        required_files = [
            "STCharacters.txt",
            "STPhrases.txt",
            "STPunctuations.txt",
            "TSCharacters.txt",
            "TSPhrases.txt",
            "TSPunctuations.txt",
            "TWPhrases.txt",
            "TWPhrasesRev.txt",
            "TWVariants.txt",
            "TWVariantsRev.txt",
            "TWVariantsRevPhrases.txt",
            "HKVariants.txt",
            "HKVariantsRev.txt",
            "HKVariantsRevPhrases.txt",
            "JPShinjitaiCharacters.txt",
            "JPShinjitaiPhrases.txt",
            "JPVariants.txt",
            "JPVariantsRev.txt",
        ]

        missing = [
            name for name in required_files
            if not (base / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Dictionary directory is missing required files:\n"
                + "\n".join(
                    "  - {}".format(name)
                    for name in missing
                )
            )

    def serialize_to_json(self, path: str, pretty: bool = False) -> None:
        """
        Serialize the current dictionary set to a stable JSON format.

        Shape:
          - Each dictionary field is serialized as: [ { <mapping> }, <max_length:int> ]
            where the mapping is sorted by (key length ASC, then key ASC).

        Fields are written in a fixed order:
            st_characters, st_phrases, st_punctuations,
            ts_characters, ts_phrases, ts_punctuations,
            tw_phrases, tw_phrases_rev,
            tw_variants, tw_variants_rev, tw_variants_rev_phrases,
            hk_variants, hk_variants_rev, hk_variants_rev_phrases,
            jps_characters, jps_phrases,
            jp_variants, jp_variants_rev

        Notes
        -----
        - `max_length` values are plain integers.
        - Output is UTF-8 with non-ASCII preserved.
        - By default output is compact; set `pretty=True` for human-readable formatting.
        """
        import json
        from pathlib import Path

        def as_array(tup):
            m, L = tup
            # Deterministic inner-map order: by key length, then key
            ordered = {k: m[k] for k in sorted(m, key=lambda k: (len(k), k))}
            return [ordered, int(L)]

        out = {name: as_array(getattr(self, name)) for name in type(self).DICT_FIELDS}

        # Ensure parent folder exists
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        with p.open("w", encoding="utf-8") as f:
            if pretty:
                json.dump(out, f, ensure_ascii=False, indent=2)
            else:
                json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
