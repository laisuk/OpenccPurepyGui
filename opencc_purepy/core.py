import re
from enum import Enum
from multiprocessing import Pool, cpu_count

from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union, cast
from .detofu import DeTofuLevel, DeTofuMap, parse_level, detofu
from .dict_refs import DictRefs, StarterUnionLike
from .dictionary_lib import DictionaryMaxlength, PathLike, SlotPathMap
from .union_cache import UnionCache, UnionKey

_PunctuationTranslateTable = Mapping[int, Union[int, str, None]]

# Pre-compiled regex for better performance
STRIP_REGEX = re.compile(r"[!-/:-@\[-`{-~\t\n\v\f\r 0-9A-Za-z_著]")

DELIMITERS = frozenset(
    " \t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^_{}|~＝、。“”‘’『』「」﹁﹂—－（）《》〈〉？！…／＼︒︑︔︓︿﹀︹︺︙︐［﹇］﹈︕︖︰︳︴︽︾︵︶｛︷｝︸﹃﹄【︻】︼　～．，；：")

# Pre-computed punctuation mappings
PUNCT_S2T_MAP = str.maketrans({
    '“': '「',
    '”': '」',
    '‘': '『',
    '’': '』',
})

PUNCT_T2S_MAP = str.maketrans({
    '「': '“',
    '」': '”',
    '『': '‘',
    '』': '’',
})

# Punctuation conversion architecture during the 1.3.x union-cache transition:
#
# Dedicated union punctuation paths:
#   s2t, t2s, s2tw, tw2s, s2twp, tw2sp, s2hk, hk2s, s2hkp, hk2sp
# These route punctuation=True through explicit *_punct union configs.
#
# Legacy punctuation fallback paths:
#   t2tw, t2twp, tw2t, tw2tp, t2hk, hk2t, t2jp, jp2t
# These still run the post-processing helper below to preserve 1.3.x beta
# behavior until punctuation handling is fully unified.
UNION_PUNCTUATION_CONFIGS = (
    "s2t", "t2s", "s2tw", "tw2s", "s2twp", "tw2sp", "s2hk", "hk2s", "s2hkp", "hk2sp",
)
LEGACY_PUNCTUATION_FALLBACK_CONFIGS = (
    "t2tw", "t2twp", "tw2t", "tw2tp", "t2hk", "hk2t", "t2jp", "jp2t",
)


class OpenccConfig(Enum):
    S2T = "s2t"
    T2S = "t2s"
    S2TW = "s2tw"
    TW2S = "tw2s"
    S2TWP = "s2twp"
    TW2SP = "tw2sp"
    S2HK = "s2hk"
    HK2S = "hk2s"
    S2HKP = "s2hkp"
    HK2SP = "hk2sp"
    T2TW = "t2tw"
    TW2T = "tw2t"
    T2TWP = "t2twp"
    TW2TP = "tw2tp"
    T2HK = "t2hk"
    HK2T = "hk2t"
    T2JP = "t2jp"
    JP2T = "jp2t"

    value: str

    def to_canonical_name(self) -> str:
        """Return OpenCC canonical config name (e.g. 's2t')."""
        return self.value

    @classmethod
    def parse(cls, s: str) -> "OpenccConfig":
        if not isinstance(s, str):
            raise ValueError("Invalid config: {}".format(s))
        return cls(s.strip().lower())


_ConfigLike = Optional[Union[str, OpenccConfig]]


class OpenCC:
    """
    A pure-Python implementation of OpenCC for text conversion between
    different Chinese language variants using segmentation and replacement.
    """

    CONFIG_LIST = [c.value for c in OpenccConfig]

    def __init__(
            self,
            config: _ConfigLike = None,
            dictionary: Optional[DictionaryMaxlength] = None,
    ):
        """
        Initialize OpenCC with a given config.

        By default, OpenCC loads the shared packaged dictionary provider via
        ``DictionaryMaxlength.new()``.

        A custom ``DictionaryMaxlength`` instance may be injected directly for
        advanced use cases, testing, custom dictionary loading, or preloaded
        dictionary reuse.

        :param config:
            OpenCC configuration name. Defaults to ``s2t`` when omitted.

        :param dictionary:
            Optional custom dictionary container. If omitted, the built-in
            shared dictionary provider is used.
        """
        self._last_error = None
        self._config_cache: Dict[str, DictRefs] = {}
        self.config = self._normalize_config(config)

        try:
            self.dictionary = (
                dictionary
                if dictionary is not None
                else DictionaryMaxlength.new()
            )
        except Exception as e:
            self._last_error = str(e)
            self.dictionary = DictionaryMaxlength()

        self.delimiters = DELIMITERS
        self.union_cache = UnionCache(self.dictionary)

        escaped_delimiters = ''.join(map(re.escape, self.delimiters))
        self.delimiter_regex = re.compile(f'[{escaped_delimiters}]')

    @classmethod
    def from_dicts(
            cls,
            config: _ConfigLike = None,
            base_dir: Optional[PathLike] = None,
            paths: Optional[Dict[str, str]] = None,
            overrides: Optional[SlotPathMap] = None,
            appends: Optional[SlotPathMap] = None,
    ) -> "OpenCC":
        """
        Create an ``OpenCC`` instance using dictionaries loaded from text files.

        This is a convenience constructor around
        ``DictionaryMaxlength.from_dicts()``.

        It supports both backward-compatible full directory loading and newer
        flexible user dictionary modes.

        Customization modes:

        1. Legacy directory loading
           ``base_dir`` and ``paths`` may be used to load dictionaries from a
           custom directory.

        2. Full dictionary replacement
           ``overrides`` replaces selected dictionary slots with complete
           custom dictionary files.

        3. Dictionary append mode
           ``appends`` loads extra user entries after the base dictionaries.
           Duplicate keys are resolved by late-comer wins.

        Precedence order:

            built-in/base < override < append

        Examples
        --------
        Load built-in TXT dictionaries:

        >>> cc = OpenCC.from_dicts()

        Load all dictionaries from a custom directory:

        >>> cc = OpenCC.from_dicts(base_dir="./my_dicts")

        Replace a whole dictionary slot using ``DictSlot``:

        >>> from opencc_purepy import DictSlot
        >>>
        >>> cc = OpenCC.from_dicts(
        ...     overrides={
        ...         DictSlot.STPhrases: "./company/STPhrases.txt",
        ...     }
        ... )

        Legacy string slot names are also supported:

        >>> cc = OpenCC.from_dicts(
        ...     overrides={
        ...         "st_phrases": "./company/STPhrases.txt",
        ...     }
        ... )

        Append custom user terms:

        >>> cc = OpenCC.from_dicts(
        ...     appends={
        ...         "st_phrases": "./custom/custom_terms.txt",
        ...     }
        ... )

        Use another OpenCC config:

        >>> cc = OpenCC.from_dicts(
        ...     config="s2tw",
        ...     appends={
        ...         "st_phrases": "./custom/custom_terms.txt",
        ...         DictSlot.TWVariantsPhrases: "./custom/TWVariantsPhrases.txt",
        ...     }
        ... )

        :param config:
            OpenCC configuration name. Defaults to ``s2t`` when omitted.

        :param base_dir:
            Optional base directory for legacy dictionary loading.

        :param paths:
            Optional dictionary slot -> filename mapping.

            Supports both ``DictSlot`` and legacy ``str`` keys.

        :param overrides:
            Optional dictionary slot -> file path mapping for full replacement.

            Supports both ``DictSlot`` and legacy ``str`` keys.

        :param appends:
            Optional dictionary slot -> file path mapping for appended custom
            entries.

            Supports both ``DictSlot`` and legacy ``str`` keys.

        :return:
            ``OpenCC`` instance using the loaded dictionary container.
        """
        dictionary = DictionaryMaxlength.from_dicts(
            base_dir=base_dir,
            paths=paths,
            overrides=overrides,
            appends=appends,
        )

        return cls(
            config=config,
            dictionary=dictionary,
        )

    def _normalize_config(self, config: _ConfigLike) -> str:
        """
        Normalize config to canonical lowercase OpenCC name.

        Returns:
            str: Valid canonical config name.

        Raises:
            ValueError: If the config is not a supported OpenCC conversion name.
        """
        if config is None:
            return "s2t"

        if isinstance(config, OpenccConfig):
            self._last_error = None
            return config.value

        if isinstance(config, str):
            cfg = config.strip().lower()
            if cfg in self.CONFIG_LIST:
                self._last_error = None
                return cfg

            self._last_error = "Invalid config: {}".format(config)
            raise ValueError(self._last_error)

        self._last_error = "Invalid config: {}".format(config)
        raise ValueError(self._last_error)

    def set_config(self, config: _ConfigLike) -> None:
        """
        Set the conversion configuration.

        :param config: Configuration name or OpenccConfig enum
        """
        self.config = self._normalize_config(config)

    def get_config(self):
        """
        Get the current conversion config.

        :return: Current config string
        """
        return self.config

    @classmethod
    def supported_configs(cls):
        """
        Return a list of supported conversion config strings.

        :return: List of config names
        """
        return cls.CONFIG_LIST

    def get_last_error(self):
        """
        Retrieve the last error message, if any.

        :return: Error string or None
        """
        return self._last_error

    def get_split_ranges(self, text: str, inclusive: bool = False) -> List[Tuple[int, int]]:
        """
        Split the input into ranges of text between delimiters using regex.

        If `inclusive` is True:
            - Each (start, end) range includes the delimiter (like forward mmseg).
        If `inclusive` is False:
            - Each (start, end) range excludes the delimiter.
            - Delimiters are returned as separate (start, end) segments.

        :param text: Input string
        :param inclusive: Whether to include delimiters in the same segment
        :return: List of (start, end) index pairs
        """
        ranges = []
        start = 0
        for match in self.delimiter_regex.finditer(text):
            delim_start, delim_end = match.start(), match.end()
            if inclusive:
                # Include delimiter in the same range
                ranges.append((start, delim_end))
            else:
                # Exclude delimiter from main segment, and add as its own
                if delim_start > start:
                    ranges.append((start, delim_start))
                ranges.append((delim_start, delim_end))
            start = delim_end

        if start < len(text):
            ranges.append((start, len(text)))

        return ranges

    def segment_replace(
            self,
            text: str,
            dictionaries: List[Tuple[Dict[str, str], int]],
            max_word_length: int
    ) -> str:
        """
        Perform dictionary-based greedy replacement on segmented text (legacy path).

        This version is simplified to work cleanly with DictRefs normalization:
        - No StarterIndex / fast-path logic.
        - Accepts a round's `dictionaries` as List[(dict, max_len)] and a round
          `max_word_length` (already computed by DictRefs).
        - Keeps the existing parallelization behavior.

        Parameters
        ----------
        text : str
            The input text to be converted.
        dictionaries : list of (dict, int)
            Sequence of dictionaries and their respective maximum key lengths.
            The order determines replacement precedence (earlier wins).
        max_word_length : int
            Global maximum match length for this round (from DictRefs).

        Returns
        -------
        str
            Converted text.
        """
        if not text:
            return text

        # Split into segments (inclusive keeps delimiters attached to segments)
        ranges = self.get_split_ranges(text, inclusive=True)

        # Single segment → direct convert (avoids slicing/join overhead)
        if len(ranges) == 1 and ranges[0] == (0, len(text)):
            return OpenCC.convert_segment(text, dictionaries, max_word_length)

        # Parallel threshold
        total_length = len(text)
        use_parallel = len(ranges) > 1_000 and total_length >= 1_000_000

        if use_parallel:
            group_count = min(4, max(1, cpu_count()))
            groups = chunk_ranges(ranges, group_count)
            with Pool(processes=group_count) as pool:
                results = pool.map(
                    convert_range_group,
                    [(text, group, dictionaries, max_word_length, OpenCC.convert_segment) for group in groups],
                )
            return "".join(results)

        # Serial path
        return "".join(
            OpenCC.convert_segment(text[s:e], dictionaries, max_word_length)
            for (s, e) in ranges
        )

    @staticmethod
    def convert_segment(segment: str, dictionaries, max_word_length: int) -> str:
        """
        Apply dictionary replacements to a text segment using greedy max-length matching.

        :param segment: Text segment to convert
        :param dictionaries: List of (dict, max_length) tuples
        :param max_word_length: Maximum matching word length
        :return: Converted string
        """
        if not segment or (len(segment) == 1 and segment in DELIMITERS):
            return segment

        result = []
        i = 0
        n = len(segment)

        while i < n:
            remaining = n - i
            best_match = None
            best_length = 0

            # Try matches from longest to shortest
            for length in range(min(max_word_length, remaining), 0, -1):
                end = i + length
                word = segment[i:end]

                # Check all dictionaries for this word
                for dict_data, max_len in dictionaries:
                    if max_len < length:
                        continue

                    match = dict_data.get(word)
                    if match is not None:
                        best_match = match
                        best_length = length
                        break

                if best_match:
                    break

            if best_match is not None:
                result.append(best_match)
                i += best_length
            else:
                result.append(segment[i])
                i += 1

        return ''.join(result)

    def union_replace(self, text: str, union: StarterUnionLike) -> str:
        """
        Greedy replacement on segmented text using a cached StarterUnion.
        """
        if not text:
            return text

        if not getattr(union, "_indexed", False):
            union.build_starter_index()

        total_length = len(text)
        if total_length < 10_000:
            return OpenCC.convert_union_indexed(text, union)

        ranges = self.get_split_ranges(text, inclusive=True)
        if len(ranges) == 1 and ranges[0] == (0, len(text)):
            return OpenCC.convert_union_indexed(text, union)

        use_parallel = len(ranges) > 1_000 and total_length >= 1_000_000

        if use_parallel:
            group_count = min(4, max(1, cpu_count()))
            groups = chunk_ranges(ranges, group_count)
            with Pool(processes=group_count) as pool:
                results = pool.map(
                    convert_range_group_union,
                    [(text, group, union) for group in groups],
                )
            return "".join(results)

        return "".join(
            OpenCC.convert_union_indexed(text[s:e], union)
            for (s, e) in ranges
        )

    @staticmethod
    def convert_union(segment: str, union: StarterUnionLike) -> str:
        if not segment:
            return segment
        if not getattr(union, "_indexed", False):
            union.build_starter_index()
        return OpenCC.convert_union_indexed(segment, union)

    @staticmethod
    def convert_union_indexed(segment: str, union: StarterUnionLike) -> str:
        if not segment:
            return segment

        n = len(segment)
        i = 0
        merged_map = union.merged_map
        get = merged_map.get
        out = []
        append = out.append

        bmp_mask = union.bmp_mask
        bmp_cap = union.bmp_cap
        astral_mask = union.astral_mask
        astral_cap = union.astral_cap
        global_cap = int(union.cap) if union.cap else 0

        while i < n:
            starter = segment[i]
            code = ord(starter)
            remaining = n - i

            if code <= 0xFFFF:
                mask = bmp_mask[code]
                cap_here = bmp_cap[code]
            else:
                mask = astral_mask.get(starter, 0)
                cap_here = astral_cap.get(starter, 0)

            if not mask or not cap_here:
                append(starter)
                i += 1
                continue

            cap_eff = min(cap_here, remaining)
            if global_cap:
                cap_eff = min(cap_eff, global_cap)

            matched = False

            if cap_eff > 63:
                length = cap_eff
                while length >= 64:
                    replacement = get(segment[i:i + length])
                    if replacement is not None:
                        append(replacement)
                        i += length
                        matched = True
                        break
                    length -= 1

            if not matched:
                if cap_eff < 64:
                    mask &= (1 << cap_eff) - 1
                else:
                    mask &= (1 << 63) - 1

                while mask:
                    bit_index = mask.bit_length()
                    length = bit_index
                    replacement = get(segment[i:i + length])
                    if replacement is not None:
                        append(replacement)
                        i += length
                        matched = True
                        break
                    mask ^= 1 << (bit_index - 1)

            if not matched:
                append(starter)
                i += 1

        return "".join(out)

    def _get_dict_refs(self, config_key: str) -> DictRefs:
        """Get cached DictRefs for a config to avoid recreation."""
        cached = self._config_cache.get(config_key)
        if cached is not None:
            return cached

        if config_key == "s2t":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
        elif config_key == "s2t_punct":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T_PUNCT))
        elif config_key == "t2s":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.T2S))
        elif config_key == "t2s_punct":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.T2S_PUNCT))
        elif config_key == "s2tw":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwVariantsPair))
            )
        elif config_key == "s2tw_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T_PUNCT))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwVariantsPair))
            )
        elif config_key == "tw2s":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.TwRevPair))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S))
            )
        elif config_key == "tw2s_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.TwRevPair))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S_PUNCT))
            )
        elif config_key == "s2twp":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.S2TwpR2TwTriple))
            )
        elif config_key == "s2twp_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T_PUNCT))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.S2TwpR2TwTriple))
            )
        elif config_key == "tw2sp":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.Tw2SpR1TwRevTriple))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S))
            )
        elif config_key == "tw2sp_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.Tw2SpR1TwRevTriple))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S_PUNCT))
            )
        elif config_key == "s2hkp":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.S2HkpR2HkTriple))
            )
        elif config_key == "s2hkp_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T_PUNCT))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.S2HkpR2HkTriple))
            )
        elif config_key == "hk2sp":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.Hk2SpR1HkRevTriple))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S))
            )
        elif config_key == "hk2sp_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.Hk2SpR1HkRevTriple))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S_PUNCT))
            )
        elif config_key == "s2hk":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.HkVariantsPair))
            )
        elif config_key == "s2hk_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T_PUNCT))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.HkVariantsPair))
            )
        elif config_key == "hk2s":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.HkRevPair))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S))
            )
        elif config_key == "hk2s_punct":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.HkRevPair))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S_PUNCT))
            )
        elif config_key == "t2tw":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.TwVariantsPair))
        elif config_key == "t2twp":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.TwPhrasesOnly))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwVariantsPair))
            )
        elif config_key == "tw2t":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.TwRevPair))
        elif config_key == "tw2tp":
            refs = (
                DictRefs(self.union_cache.ensure_indexed(UnionKey.TwRevPair))
                .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwPhrasesRevOnly))
            )
        elif config_key == "t2hk":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.HkVariantsPair))
        elif config_key == "hk2t":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.HkRevPair))
        elif config_key == "t2jp":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.JpsCharactersRev))
        elif config_key == "jp2t":
            refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.JpsPair))
        else:
            raise ValueError(f"Unsupported config: {config_key}")

        self._config_cache[config_key] = refs
        return refs

    @staticmethod
    def _apply_punctuation(text: str, config_key: str, punctuation: bool) -> str:
        """
        Deprecated compatibility layer for legacy punctuation post-processing.

        Dedicated union punctuation paths currently exist for:
            s2t, t2s, s2tw, tw2s, s2twp, tw2sp, s2hk, hk2s

        Legacy fallback paths that still call this helper:
            t2tw, t2twp, tw2t, tw2tp, t2hk, hk2t, t2jp, jp2t

        Runtime behavior is intentionally preserved for 1.3.x beta users. Do
        not emit runtime deprecation warnings here; this is an internal
        migration note only.

        TODO:
            Revisit this helper once all supported punctuation conversions are
            represented by explicit union-cache punctuation paths.
        """
        if not punctuation:
            return text

        if config_key in ("t2s", "tw2s", "tw2sp", "hk2s"):
            translate_table = cast(_PunctuationTranslateTable, PUNCT_T2S_MAP)
        else:
            translate_table = cast(_PunctuationTranslateTable, PUNCT_S2T_MAP)
        return text.translate(translate_table)

    def s2t(self, input_text, punctuation=False):
        """
        Convert Simplified Chinese to Traditional Chinese.

        :param input_text: The source string in Simplified Chinese
        :param punctuation: Whether to convert punctuation
        :return: Transformed string in Traditional Chinese
        """
        refs = self._get_dict_refs("s2t_punct" if punctuation else "s2t")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def t2s(self, input_text, punctuation=False):
        """
        Convert Traditional Chinese to Simplified Chinese.

        :param input_text: The source string in Traditional Chinese
        :param punctuation: Whether to convert punctuation
        :return: Transformed string in Simplified Chinese
        """
        refs = self._get_dict_refs("t2s_punct" if punctuation else "t2s")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def s2tw(self, input_text, punctuation=False):
        """
        Convert Simplified Chinese to Traditional Chinese (Taiwan Standard).

        :param input_text: The source string
        :param punctuation: Whether to convert punctuation
        :return: Transformed string in Taiwan Traditional Chinese
        """
        refs = self._get_dict_refs("s2tw_punct" if punctuation else "s2tw")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def tw2s(self, input_text, punctuation=False):
        """
        Convert Traditional Chinese (Taiwan) to Simplified Chinese.

        :param input_text: The source string in Taiwan Traditional Chinese
        :param punctuation: Whether to convert punctuation
        :return: Transformed string in Simplified Chinese
        """
        refs = self._get_dict_refs("tw2s_punct" if punctuation else "tw2s")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def s2twp(self, input_text, punctuation=False):
        """
        Convert Simplified Chinese to Traditional (Taiwan) with phrase and variant normalization.

        Round 1: Simplified Chinese -> Traditional Chinese.
        Round 2: Taiwan phrase and variant normalization.

        :param input_text: The source string
        :param punctuation: Whether to convert punctuation
        :return: Transformed string
        """
        refs = self._get_dict_refs("s2twp_punct" if punctuation else "s2twp")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def tw2sp(self, input_text, punctuation=False):
        """
        Convert Traditional (Taiwan) with phrases to Simplified Chinese.

        :param input_text: The source string
        :param punctuation: Whether to convert punctuation
        :return: Transformed string
        """
        refs = self._get_dict_refs("tw2sp_punct" if punctuation else "tw2sp")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def s2hkp(self, input_text, punctuation=False):
        """
        Convert Simplified Chinese to Hong Kong Traditional with phrase and variant normalization.

        Round 1: Simplified Chinese -> Traditional Chinese.
        Round 2: Hong Kong phrase and variant normalization.

        :param input_text: The source string
        :param punctuation: Whether to convert punctuation
        :return: Transformed string
        """
        refs = self._get_dict_refs("s2hkp_punct" if punctuation else "s2hkp")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def hk2sp(self, input_text, punctuation=False):
        """
        Convert Hong Kong Traditional with phrases and variants to Simplified Chinese.

        :param input_text: Hong Kong Traditional Chinese input
        :param punctuation: Whether to convert punctuation
        :return: Simplified Chinese output
        """
        refs = self._get_dict_refs("hk2sp_punct" if punctuation else "hk2sp")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def s2hk(self, input_text, punctuation=False):
        """
        Convert Simplified Chinese to Traditional (Hong Kong Standard).

        :param input_text: Simplified Chinese input
        :param punctuation: Whether to convert punctuation
        :return: Transformed string
        """
        refs = self._get_dict_refs("s2hk_punct" if punctuation else "s2hk")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def hk2s(self, input_text, punctuation=False):
        """
        Convert Traditional (Hong Kong) to Simplified Chinese.

        :param input_text: Hong Kong Traditional Chinese input
        :param punctuation: Whether to convert punctuation
        :return: Simplified Chinese output
        """
        refs = self._get_dict_refs("hk2s_punct" if punctuation else "hk2s")
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)

    def t2tw(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Traditional Chinese to Taiwan Standard Traditional Chinese.
        """
        refs = self._get_dict_refs("t2tw")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "t2tw", punctuation)

    def t2twp(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Traditional Chinese to Taiwan Standard using phrase and variant mappings.
        """
        refs = self._get_dict_refs("t2twp")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "t2twp", punctuation)

    def tw2t(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Taiwan Traditional to general Traditional Chinese.
        """
        refs = self._get_dict_refs("tw2t")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "tw2t", punctuation)

    def tw2tp(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Taiwan Traditional to Traditional with phrase reversal.
        """
        refs = self._get_dict_refs("tw2tp")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "tw2tp", punctuation)

    def t2hk(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Traditional Chinese to Hong Kong variant.
        """
        refs = self._get_dict_refs("t2hk")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "t2hk", punctuation)

    def hk2t(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Hong Kong Traditional to standard Traditional Chinese.
        """
        refs = self._get_dict_refs("hk2t")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "hk2t", punctuation)

    def t2jp(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Traditional Chinese to Japanese variants.
        """
        refs = self._get_dict_refs("t2jp")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "t2jp", punctuation)

    def jp2t(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Japanese Shinjitai (modern Kanji) to Traditional Chinese.
        """
        refs = self._get_dict_refs("jp2t")
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace, validate_delegates=False)
        return OpenCC._apply_punctuation(output, "jp2t", punctuation)

    def convert(self, input_text: str, punctuation: bool = False) -> str:
        """
        Automatically dispatch to the appropriate conversion method based on `self.config`.

        :param input_text: The string to convert
        :param punctuation: Whether to apply punctuation conversion
        :return: Converted string or error message
        """
        if not input_text:
            self._last_error = "Input text is empty"
            return ""

        config = self.config.lower()
        try:
            if config == "s2t":
                return self.s2t(input_text, punctuation)
            elif config == "s2tw":
                return self.s2tw(input_text, punctuation)
            elif config == "s2twp":
                return self.s2twp(input_text, punctuation)
            elif config == "s2hkp":
                return self.s2hkp(input_text, punctuation)
            elif config == "s2hk":
                return self.s2hk(input_text, punctuation)
            elif config == "t2s":
                return self.t2s(input_text, punctuation)
            elif config == "t2tw":
                return self.t2tw(input_text, punctuation)
            elif config == "t2twp":
                return self.t2twp(input_text, punctuation)
            elif config == "t2hk":
                return self.t2hk(input_text, punctuation)
            elif config == "tw2s":
                return self.tw2s(input_text, punctuation)
            elif config == "tw2sp":
                return self.tw2sp(input_text, punctuation)
            elif config == "tw2t":
                return self.tw2t(input_text, punctuation)
            elif config == "tw2tp":
                return self.tw2tp(input_text, punctuation)
            elif config == "hk2s":
                return self.hk2s(input_text, punctuation)
            elif config == "hk2sp":
                return self.hk2sp(input_text, punctuation)
            elif config == "hk2t":
                return self.hk2t(input_text, punctuation)
            elif config == "jp2t":
                return self.jp2t(input_text, punctuation)
            elif config == "t2jp":
                return self.t2jp(input_text, punctuation)
            else:
                self._last_error = f"Invalid config: {config}"
                return self._last_error
        except Exception as e:
            self._last_error = f"Conversion failed: {e}"
            return self._last_error

    def st(self, input_text: str) -> str:
        """
        Convert Simplified Chinese characters only (no phrases).
        """
        if not input_text:
            return input_text

        dict_data = [self.dictionary.st_characters]
        return self.convert_segment(input_text, dict_data, 1)

    def ts(self, input_text: str) -> str:
        """
        Convert Traditional Chinese characters only (no phrases).
        """
        if not input_text:
            return input_text

        dict_data = [self.dictionary.ts_characters]
        return self.convert_segment(input_text, dict_data, 1)

    def zho_check(self, input_text: str) -> int:
        """
        Heuristically determine whether input text is Simplified or Traditional Chinese.
        Only a small prefix of the input is inspected for performance reasons.

        :param input_text: Input string
        :return: 0 = unknown, 1 = traditional, 2 = simplified
        """
        if not input_text:
            return 0

        sample = input_text[:1000]
        strip_text = STRIP_REGEX.sub("", sample)[:100]

        if strip_text != self.ts(strip_text):
            return 1
        elif strip_text != self.st(strip_text):
            return 2
        else:
            return 0

    # ------ DeTofu helpers ------

    @staticmethod
    def detofu(
            text: Optional[str],
            level: Union[DeTofuLevel, str] = DeTofuLevel.ExtB,
    ) -> str:
        """Apply DeTofu display-compatible fallbacks to mapped rare CJK extension characters.

        DeTofu is a display compatibility pass. It does not modify OpenCC
        conversion dictionaries, phrase matching, regional variant selection,
        script detection, or punctuation conversion.

        For converted text, apply DeTofu after convert().
        """
        return detofu(text, level)

    @staticmethod
    def detofu_with_custom_file(
            text: Optional[str],
            level: Union[DeTofuLevel, str],
            path: str,
    ) -> str:
        """Apply DeTofu using built-in mappings plus a custom fallback file.

        The file must be UTF-8 text using:

            tofu_char<TAB>fallback_char<TAB>extension

        Blank lines and lines beginning with '#' are ignored. Custom mappings
        override built-in mappings for the same tofu-risk character.
        """
        if path is None:
            raise TypeError("path must not be None")

        if isinstance(level, str):
            level = parse_level(level)

        return (
            DeTofuMap
            .builtin(level)
            .with_custom_file(path)
            .convert(text)
        )

    @staticmethod
    def detofu_with_custom_pairs(
            text: Optional[str],
            level: Union[DeTofuLevel, str],
            pairs: Union[Mapping[str, str], Iterable[Tuple[str, str]]],
    ) -> str:
        """Apply DeTofu using built-in mappings plus custom in-memory pairs.

        Keys are tofu-risk characters and values are display-compatible fallback
        characters. Only the first Unicode scalar from each key and value is used.
        Empty keys and values are ignored.
        """
        if pairs is None:
            raise TypeError("pairs must not be None")

        if isinstance(level, str):
            level = parse_level(level)

        return (
            DeTofuMap
            .builtin(level)
            .with_custom_pairs(pairs)
            .convert(text)
        )


def chunk_ranges(ranges: List[Tuple[int, int]], group_count: int) -> List[List[Tuple[int, int]]]:
    """
    Split a list of (start, end) index ranges into evenly sized chunks.

    This function divides the input list of ranges into approximately equal-sized sublists,
    useful for distributing work across multiple worker processes or threads.

    :param ranges: A list of (start, end) index tuples representing text segments.
    :param group_count: Number of groups to divide the ranges into (typically the number of worker processes).
    :return: A list of range groups, each being a list of (start, end) tuples.
    """
    chunk_size = (len(ranges) + group_count - 1) // group_count
    return [ranges[i:i + chunk_size] for i in range(0, len(ranges), chunk_size)]


def convert_range_group(args):
    """
    Convert a group of text segments using the provided conversion function.

    This function is designed for use with multiprocessing. It processes a group of
    (start, end) index ranges from the original input text, applies the dictionary-based
    segment conversion to each, and joins the results.

    :param args: A tuple containing:
        - text: The original input string.
        - group_ranges: A list of (start, end) index tuples for this group.
        - dictionaries: A list of (dictionary, max_length) tuples.
        - max_word_length: The maximum matching length used for dictionary lookup.
        - convert_segment_fn: A callable function to convert each segment.
    :return: A string representing the converted result for the group.
    """
    text, group_ranges, dictionaries, max_word_length, convert_segment_fn = args
    conv = convert_segment_fn  # local bind
    return ''.join(
        conv(text[start:end], dictionaries, max_word_length)
        for start, end in group_ranges
    )


def convert_range_group_union(args: Tuple[str, List[Tuple[int, int]], StarterUnionLike]) -> str:
    text, group_ranges, union = args
    conv = OpenCC.convert_union_indexed
    return ''.join(
        conv(text[start:end], union)
        for start, end in group_ranges
    )
