from pathlib import Path
from threading import Lock
from typing import Dict, Tuple, Union, Optional, Mapping, List

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
        "tw_variants_phrases", "tw_variants",
        "tw_variants_rev", "tw_variants_rev_phrases",
        "hk_phrases", "hk_phrases_rev",
        "hk_variants_phrases", "hk_variants",
        "hk_variants_rev", "hk_variants_rev_phrases",
        "jps_characters", "jps_characters_rev", "jps_phrases",
    )

    _provider = None
    _provider_lock = Lock()

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
        self.tw_variants_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_variants: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_variants_rev: Tuple[Dict[str, str], int] = ({}, 0)
        self.tw_variants_rev_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_phrases_rev: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_variants_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_variants: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_variants_rev: Tuple[Dict[str, str], int] = ({}, 0)
        self.hk_variants_rev_phrases: Tuple[Dict[str, str], int] = ({}, 0)
        self.jps_characters: Tuple[Dict[str, str], int] = ({}, 0)
        self.jps_characters_rev: Tuple[Dict[str, str], int] = ({}, 0)
        self.jps_phrases: Tuple[Dict[str, str], int] = ({}, 0)

        self._is_shared_provider = False

    def __repr__(self):
        count = sum(bool(v[0]) for v in self.__dict__.values())
        return "<DictionaryMaxlength with {} loaded dicts>".format(count)

    @classmethod
    def get_provider(cls):
        """
        Return a shared dictionary provider loaded from precompiled JSON.
        :return: DictionaryMaxlength instance
        """
        if cls._provider is None:
            with cls._provider_lock:
                if cls._provider is None:
                    cls._provider = cls.from_json()
                    cls._provider._is_shared_provider = True
        return cls._provider

    def _ensure_mutable(self) -> None:
        if getattr(self, "_is_shared_provider", False):
            raise RuntimeError(
                "Cannot modify the shared DictionaryMaxlength provider. "
                "Use DictionaryMaxlength.from_json() or from_dicts() to create "
                "a private dictionary instance before applying custom dictionaries."
            )

    @classmethod
    def new(cls):
        """
        Backward-compatible alias for the shared dictionary provider.
        :return: DictionaryMaxlength instance
        """
        return cls.get_provider()

    @classmethod
    def _as_tuple(cls, value: object) -> Tuple[Dict[str, str], int]:
        if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], dict)
                and isinstance(value[1], int)
        ):
            return value[0], value[1]

        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict):
            return value[0], int(value[1])

        if isinstance(value, dict) and "map" in value and "maxlength" in value:
            raw_map = value["map"]
            if isinstance(raw_map, dict):
                return raw_map, int(value["maxlength"])

        raise ValueError("Invalid dictionary slot format")

    @classmethod
    def from_json(cls, path: Optional[PathLike] = None) -> "DictionaryMaxlength":
        """
        Load dictionary data from a JSON file.

        The JSON file must use the serialized DictionaryMaxlength format:

            {
                "st_characters": [{...}, 1],
                "st_phrases": [{...}, 4],
                ...
            }

        If ``path`` is omitted, the built-in packaged
        ``dicts/dictionary_maxlength.json`` file is used.

        :param path:
            Optional custom JSON dictionary path.

        :return:
            Populated DictionaryMaxlength instance.
        """
        import json

        json_path = (
            Path(path)
            if path is not None
            else Path(__file__).parent / "dicts" / "dictionary_maxlength.json"
        )

        with json_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        instance = cls()

        valid_slots = set(cls.DICT_FIELDS)

        for key in raw_data:
            if key not in valid_slots:
                raise ValueError("Unknown dictionary slot: {}".format(key))

        for key in cls.DICT_FIELDS:
            if key not in raw_data:
                continue

            try:
                setattr(instance, key, cls._as_tuple(raw_data[key]))
            except (TypeError, ValueError):
                raise ValueError("Invalid dictionary format for key: {}".format(key))

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
        Load OpenCC dictionaries directly from plain-text dictionary files.

        By default, all dictionaries are loaded from the built-in ``dicts`` folder.

        This method supports three customization modes:

        1. Legacy custom directory loading (backward compatible)
           Use ``base_dir`` and/or ``paths`` to load dictionaries from
           another directory.

        2. Dictionary replacement (overrides)
           Replace specific dictionary slots with fully custom dictionary files.

        3. Dictionary extension (appends)
           Append additional custom entries on top of existing dictionaries.
           When duplicate keys exist, later entries override earlier ones
           ("late-comer wins").

        Precedence order:

            built-in < override < append

        Slot Keys
        ---------
        Dictionary slot mappings accept either:

        - :class:`DictSlot` (recommended)
        - legacy ``str`` keys (backward compatible)

        Examples of valid slot keys:

        >>> DictSlot.STPhrases
        >>> "st_phrases"

        Examples
        --------
        Load built-in dictionaries:

        >>> DictionaryMaxlength.from_dicts()

        Load dictionaries from another directory (legacy behavior):

        >>> DictionaryMaxlength.from_dicts("./my_dicts")

        Replace an entire dictionary using ``DictSlot``:

        >>> from opencc_purepy import DictSlot
        >>>
        >>> DictionaryMaxlength.from_dicts(
        ...     overrides={
        ...         DictSlot.STPhrases: "./company/STPhrases.txt",
        ...     }
        ... )

        Append additional custom terms:

        >>> DictionaryMaxlength.from_dicts(
        ...     appends={
        ...         DictSlot.STPhrases: "./custom/custom_terms.txt",
        ...         DictSlot.TWVariantsPhrases: "./custom/TWVariantsPhrases.txt",
        ...         DictSlot.HKVariantsPhrases: "./custom/HKVariantsPhrases.txt",
        ...     }
        ... )

        Legacy ``str`` keys remain supported:

        >>> DictionaryMaxlength.from_dicts(
        ...     appends={
        ...         "st_phrases": "./custom/custom_terms.txt",
        ...     }
        ... )

        Parameters
        ----------
        base_dir:
            Optional base directory for legacy dictionary loading.
            Defaults to the built-in ``dicts`` folder.

        paths:
            Optional dictionary slot -> filename mapping.

            Supports both :class:`DictSlot` and legacy ``str`` keys.

        overrides:
            Optional dictionary slot -> file path mapping used to fully
            replace individual dictionaries.

            Supports both :class:`DictSlot` and legacy ``str`` keys.

        appends:
            Optional dictionary slot -> file path mapping used to append
            additional entries to existing dictionaries.

            Supports both :class:`DictSlot` and legacy ``str`` keys.

        Returns
        -------
        DictionaryMaxlength
            A populated dictionary container.
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
            'tw_variants_phrases': "TWVariantsPhrases.txt",
            'tw_variants': "TWVariants.txt",
            'tw_variants_rev': "TWVariantsRev.txt",
            'tw_variants_rev_phrases': "TWVariantsRevPhrases.txt",
            'hk_phrases': "HKPhrases.txt",
            'hk_phrases_rev': "HKPhrasesRev.txt",
            'hk_variants_phrases': "HKVariantsPhrases.txt",
            'hk_variants': "HKVariants.txt",
            'hk_variants_rev': "HKVariantsRev.txt",
            'hk_variants_rev_phrases': "HKVariantsRevPhrases.txt",
            'jps_characters': "JPShinjitaiCharacters.txt",
            'jps_characters_rev': "JPShinjitaiCharactersRev.txt",
            'jps_phrases': "JPShinjitaiPhrases.txt",
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

        optional_slots = {
            "st_punctuations",
            "ts_punctuations",
            "hk_phrases",
            "hk_phrases_rev",
        }

        for attr, path in file_map.items():
            if attr in optional_slots and not path.is_file():
                setattr(instance, attr, ({}, 0))
                continue

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

        for line in content.strip().splitlines():
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue

            parts = stripped_line.split()
            if len(parts) >= 2:
                phrase, translation = parts[0], parts[1]
                dictionary[phrase] = translation
                max_length = max(max_length, len(phrase))
            else:
                import warnings
                warnings.warn("Ignoring malformed dictionary line: {}".format(line))

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
        self._ensure_mutable()

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
        self._ensure_mutable()

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
            "TSCharacters.txt",
            "TSPhrases.txt",
            "TWPhrases.txt",
            "TWPhrasesRev.txt",
            "TWVariantsPhrases.txt",
            "TWVariants.txt",
            "TWVariantsRev.txt",
            "TWVariantsRevPhrases.txt",
            "HKPhrases.txt",
            "HKPhrasesRev.txt",
            "HKVariantsPhrases.txt",
            "HKVariants.txt",
            "HKVariantsRev.txt",
            "HKVariantsRevPhrases.txt",
            "JPShinjitaiCharacters.txt",
            "JPShinjitaiCharactersRev.txt",
            "JPShinjitaiPhrases.txt",
        ]

        optional_files = {
            "HKPhrases.txt",
            "HKPhrasesRev.txt",
        }

        missing = [
            name for name in required_files
            if name not in optional_files and not (base / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Dictionary directory is missing required files:\n"
                + "\n".join(
                    "  - {}".format(name)
                    for name in missing
                )
            )

    def serialize_to_json(
            self,
            path: str,
            pretty: bool = True,
            sort_keys: bool = True,
    ) -> None:
        """
        Serialize the current dictionary set to JSON.

        Shape:
          - Each dictionary field is serialized as:
            [ { <mapping> }, <max_length:int> ]

        Field order follows ``DICT_FIELDS``.

        Parameters
        ----------
        path:
            Output JSON path. Parent folders are created automatically.
        pretty:
            If True, write indented JSON for readable diffs.
            If False, write compact/minified JSON.
        sort_keys:
            If True, sort dictionary entries lexically by key for deterministic,
            Git-friendly output. This is for reproducible output and review, not
            for deserialization speed.

        Notes
        -----
        - Output is UTF-8.
        - Non-ASCII characters are preserved.
        - No external JSON dependency is required.
        """
        import json
        from pathlib import Path
        from typing import Any

        def as_array(tup: Any) -> List[Any]:
            mapping, max_length = tup

            if sort_keys:
                ordered = {k: mapping[k] for k in sorted(mapping)}
            else:
                # Preserve current insertion/source order.
                ordered = dict(mapping)

            return [ordered, int(max_length)]

        out = {
            name: as_array(getattr(self, name))
            for name in type(self).DICT_FIELDS
        }

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with p.open("w", encoding="utf-8", newline="\n") as f:
            if pretty:
                json.dump(out, f, ensure_ascii=False, indent=2)
                f.write("\n")
            else:
                json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
