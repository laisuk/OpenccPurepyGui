import warnings
from pathlib import Path
from typing import Dict, Tuple, Tuple as Tup  # type checking


class DictionaryMaxlength:
    """
    A container for OpenCC-compatible dictionaries with each represented
    as a (dict, max_length) tuple to optimize the longest match lookup.
    """
    # Immutable, subclass-overridable
    DICT_FIELDS: Tup[str, ...] = (
        "st_characters", "st_phrases",
        "ts_characters", "ts_phrases",
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
        self.ts_characters: Tuple[Dict[str, str], int] = ({}, 0)
        self.ts_phrases: Tuple[Dict[str, str], int] = ({}, 0)
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
    def from_dicts(cls):
        """
        Load dictionaries directly from text files in the 'dicts' folder.
        Each file should contain tab-separated mappings.
        :return: Populated DictionaryMaxlength instance
        """
        instance = cls()
        paths = {
            'st_characters': "STCharacters.txt",
            'st_phrases': "STPhrases.txt",
            'ts_characters': "TSCharacters.txt",
            'ts_phrases': "TSPhrases.txt",
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

        base = Path(__file__).parent / "dicts"
        for attr, filename in paths.items():
            content = (base / filename).read_text(encoding="utf-8")
            setattr(instance, attr, cls.load_dictionary_maxlength(content))

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

    def serialize_to_json(self, path: str, pretty: bool = False) -> None:
        """
        Serialize the current dictionary set to a stable JSON format.

        Shape:
          - Each dictionary field is serialized as: [ { <mapping> }, <max_length:int> ]
            where the mapping is sorted by (key length ASC, then key ASC).

        Fields are written in a fixed order:
            st_characters, st_phrases,
            ts_characters, ts_phrases,
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

