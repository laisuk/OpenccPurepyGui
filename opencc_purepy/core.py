import re
from multiprocessing import Pool, cpu_count
from typing import TYPE_CHECKING, Any, Optional, List, Tuple, Callable, Dict, Literal, Iterable

from .dict_refs import DictRefs, DictSlot
from .dictionary_lib import DictionaryMaxlength
from .union_cache import UnionKey

if TYPE_CHECKING:
    from .starter_union import StarterUnion
else:
    StarterUnion = Any  # type: ignore

# Pre-compiled regex for better performance
STRIP_REGEX = re.compile(r"[!-/:-@\[-`{-~\t\n\v\f\r 0-9A-Za-z_著]")

DELIMITERS = frozenset(
    " \t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^_{}|~＝、。“”‘’『』「」﹁﹂—－（）《》〈〉？！…／＼︒︑︔︓︿﹀︹︺︙︐［﹇］﹈︕︖︰︳︴︽︾︵︶｛︷｝︸﹃﹄【︻】︼　～．，；：")

# Pre-computed punctuation mappings - fallback for older Python versions
try:
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
    HAS_MAKETRANS = True
except (AttributeError, TypeError):
    # Fallback for Python < 3.0
    HAS_MAKETRANS = False
    PUNCT_S2T_MAP = {
        '“': '「',
        '”': '」',
        '‘': '『',
        '’': '』',
    }

    PUNCT_T2S_MAP = {
        '「': '“',
        '」': '”',
        '『': '‘',
        '』': '’',
    }


class OpenCC:
    """
    A pure-Python implementation of OpenCC for text conversion between
    different Chinese language variants using segmentation and replacement.
    """
    CONFIG_LIST = [
        "s2t", "t2s", "s2tw", "tw2s", "s2twp", "tw2sp", "s2hk", "hk2s",
        "t2tw", "tw2t", "t2twp", "tw2tp", "t2hk", "hk2t", "t2jp", "jp2t"
    ]

    def __init__(self, config=None):
        """
        Initialize OpenCC with a given config (default: s2t).

        :param config: Configuration name (optional)
        """
        self._last_error = None
        self._config_cache = {}

        if config in self.CONFIG_LIST:
            self.config = config
        else:
            self._last_error = "Invalid config: {}".format(config) if config else None
            self.config = "s2t"

        try:
            self.dictionary = DictionaryMaxlength.new()
        except Exception as e:
            self._last_error = str(e)
            self.dictionary = DictionaryMaxlength()

        from .union_cache import UnionCache
        self.union_cache = UnionCache(self.dictionary)
        self.delimiters = DELIMITERS
        # Escape special regex characters in delimiters
        escaped_delimiters = ''.join(map(re.escape, self.delimiters))
        self.delimiter_regex = re.compile(f'[{escaped_delimiters}]')

    def set_config(self, config):
        """
        Set the conversion configuration.

        :param config: One of OpenCC.CONFIG_LIST
        """
        if config in self.CONFIG_LIST:
            self.config = config
        else:
            self._last_error = "Invalid config: {}".format(config)
            self.config = "s2t"

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

    def union_replace(self, text: str, union) -> str:
        """
        Greedy replacement on segmented text using a StarterUnion
        (uses convert_segment_union internally).
        """
        if not text:
            return text

        # Ensure masks/caps are ready (no-op if already built)
        if not getattr(union, "_indexed", False):
            union.build_starter_index()

        ranges = self.get_split_ranges(text, inclusive=True)

        # Single segment → direct
        if len(ranges) == 1 and ranges[0] == (0, len(text)):
            return OpenCC.convert_union(text, union)

        # Parallel threshold (same as legacy)
        total_length = len(text)
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

        # Serial
        return "".join(
            OpenCC.convert_union(text[s:e], union)
            for (s, e) in ranges
        )

    @staticmethod
    def convert_union(segment: str, union) -> str:
        """
        Greedy longest-match conversion for a single segment using StarterUnion.
        Optimized to walk only the lengths that actually exist (mask bits).
        """
        if not segment:
            return segment

        # Ensure starter index is ready (no-op if already built)
        if not getattr(union, "_indexed", False):
            union.build_starter_index()

        s = segment
        n = len(s)
        i = 0

        merged_map = union.merged_map
        get = merged_map.get  # local bind
        out = []
        append = out.append

        bmp_mask = union.bmp_mask
        bmp_cap = union.bmp_cap
        a_mask = union.astral_mask
        a_cap = union.astral_cap
        global_cap = int(union.cap) if union.cap else 0

        while i < n:
            c0 = s[i]
            code = ord(c0)
            rem = n - i

            # Per-starter mask and cap
            if code <= 0xFFFF:
                mask = bmp_mask[code]
                cap_here = bmp_cap[code]
            else:
                mask = a_mask.get(c0, 0)
                cap_here = a_cap.get(c0, 0)

            if not mask or not cap_here:
                append(c0)
                i += 1
                continue

            # Effective cap for this position
            cap_eff = cap_here
            if global_cap and global_cap < cap_eff:
                cap_eff = global_cap
            if rem < cap_eff:
                cap_eff = rem

            matched = False

            # Handle lengths > 63 (rare; CAP bit is 63). If cap_eff > 63, we must try those exact
            # lengths first because the mask only tells us ">=64" via bit63, not which exact lengths.
            if cap_eff > 63:
                L = cap_eff
                while L >= 64:
                    repl = get(s[i:i + L])
                    if repl is not None:
                        append(repl)
                        i += L
                        matched = True
                        break
                    L -= 1

            if not matched:
                # Restrict mask to <= cap_eff
                m = mask
                if cap_eff < 64:
                    m &= (1 << cap_eff) - 1  # keep only bits [0..cap_eff-1]
                else:
                    m &= (1 << 63) - 1  # drop CAP bit; we already tried >=64 above

                # Walk set bits from longest to shortest
                while m:
                    bit_index = m.bit_length()  # 1..63
                    L = bit_index  # bit k -> length k
                    repl = get(s[i:i + L])
                    if repl is not None:
                        append(repl)
                        i += L
                        matched = True
                        break
                    # clear that bit
                    m ^= 1 << (bit_index - 1)

            if not matched:
                append(c0)
                i += 1

        return "".join(out)

    DictSlot = Tuple[dict, int]
    ConfigKey = Literal[
        "s2t", "t2s", "s2tw", "tw2s", "s2twp", "tw2sp", "s2hk", "hk2s",
        "t2tw", "t2twp", "tw2t", "tw2tp", "t2hk", "hk2t", "t2jp", "jp2t"
    ]

    def _get_legacy_dict_refs(self, config_key: str) -> Optional["DictRefs"]:
        """
        Legacy DictRefs builder (compat layer).
        Returns DictRefs whose rounds each contain lists of (dict, max_len) slots.
        Cached by config_key.
        """
        # cache hit
        cached = self._config_cache.get(config_key)
        if cached is not None:
            return cached

        d = self.dictionary  # assumes fields like st_phrases, tw_variants, ...

        # helpers to keep the table readable
        def R1(round1: List[DictSlot]) -> "DictRefs":
            return DictRefs(round1)

        def R2(round1: List[DictSlot], round2: List[DictSlot]) -> "DictRefs":
            return DictRefs(round1).with_round_2(round2)

        def R3(round1: List[DictSlot], round2: List[DictSlot], round3: List[DictSlot]) -> "DictRefs":
            return DictRefs(round1).with_round_2(round2).with_round_3(round3)

        table: Dict[str, Callable[[], "DictRefs"]] = {
            # -------- Base S/T --------
            "s2t": lambda: R1([d.st_phrases, d.st_characters]),
            "t2s": lambda: R1([d.ts_phrases, d.ts_characters]),

            # -------- Taiwan (TW) --------
            "s2tw": lambda: R2([d.st_phrases, d.st_characters], [d.tw_variants]),
            "tw2s": lambda: R2([d.tw_variants_rev_phrases, d.tw_variants_rev], [d.ts_phrases, d.ts_characters]),
            "s2twp": lambda: R3([d.st_phrases, d.st_characters], [d.tw_phrases], [d.tw_variants]),
            "tw2sp": lambda: R2([d.tw_phrases_rev, d.tw_variants_rev_phrases, d.tw_variants_rev],
                                [d.ts_phrases, d.ts_characters]),
            "t2tw": lambda: R1([d.tw_variants]),
            "t2twp": lambda: R2([d.tw_phrases], [d.tw_variants]),
            "tw2t": lambda: R1([d.tw_variants_rev_phrases, d.tw_variants_rev]),
            "tw2tp": lambda: R2([d.tw_variants_rev_phrases, d.tw_variants_rev], [d.tw_phrases_rev]),

            # -------- Hong Kong (HK) --------
            "s2hk": lambda: R2([d.st_phrases, d.st_characters], [d.hk_variants]),
            "hk2s": lambda: R2([d.hk_variants_rev_phrases, d.hk_variants_rev], [d.ts_phrases, d.ts_characters]),
            "t2hk": lambda: R1([d.hk_variants]),
            "hk2t": lambda: R1([d.hk_variants_rev_phrases, d.hk_variants_rev]),

            # -------- Japanese (JP) --------
            "t2jp": lambda: R1([d.jp_variants]),
            "jp2t": lambda: R1([d.jps_phrases, d.jps_characters, d.jp_variants_rev]),
        }

        builder = table.get(config_key)  # type: ignore[arg-type]
        if builder is None:
            return None

        refs = builder()
        self._config_cache[config_key] = refs
        return refs

    @staticmethod
    def _convert_punctuation_legacy(text, punct_map):
        """
        (Fallback punctuation conversion for older Python versions)
        Convert between Simplified and Traditional punctuation styles.

        :param text: Input text
        :param punct_map: Conversion punctuation map
        :return: Text with punctuation converted
        """
        result = []
        for char in text:
            result.append(punct_map.get(char, char))
        return ''.join(result)

    def s2t(self, input_text: str, punctuation: bool = False) -> str:
        """
        Simplified -> Traditional (no punctuation in dicts; optional separate pass).
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            if HAS_MAKETRANS:
                return output.translate(PUNCT_S2T_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_S2T_MAP)
        return output

    def t2s(self, input_text: str, punctuation: bool = False) -> str:
        """
        Traditional -> Simplified (no punctuation in dicts; optional separate pass).
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.T2S))
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            # reverse of S2T_MAP if you have one; else use your legacy map for T->S
            if HAS_MAKETRANS:
                return output.translate(PUNCT_T2S_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_T2S_MAP)
        return output

    def s2tw(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Simplified Chinese to Taiwan Standard Traditional Chinese.
        Round 1: S2T (no punctuation in union)
        Round 2: TW variants only
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwVariantsOnly))
        )
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            # handle punctuation separately (S->T)
            if HAS_MAKETRANS:
                return output.translate(PUNCT_S2T_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_S2T_MAP)

        return output

    def tw2s(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Traditional Chinese (Taiwan) to Simplified Chinese.

        Pipeline:
          1) TW reverse localization (variants + rev-phrases)
          2) T2S
          3) Optional punctuation T->S
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.TwRevPair))
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S))
        )

        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            if HAS_MAKETRANS:
                return output.translate(PUNCT_T2S_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_T2S_MAP)
        return output

    def s2twp(self, input_text: str, punctuation: bool = False) -> str:
        """
        Simplified -> Traditional (Taiwan) using S2T, then TW phrases, then TW variants.
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))  # round 1
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwPhrasesOnly))  # round 2
            .with_round_3(self.union_cache.ensure_indexed(UnionKey.TwVariantsOnly))  # round 3
        )
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            # Handle S->T punctuation as a separate pass
            if HAS_MAKETRANS:
                return output.translate(PUNCT_S2T_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_S2T_MAP)

        return output

    def tw2sp(self, input_text: str, punctuation: bool = False) -> str:
        """
        Traditional (Taiwan) -> Simplified (phrases+chars), optional T->S punctuation.

        Round 1: TW reverse triple (tw_phrases_rev, tw_variants_rev_phrases, tw_variants_rev)
        Round 2: T2S (phrases + characters)
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.Tw2SpR1TwRevTriple))
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S))
        )
        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            if HAS_MAKETRANS:
                return output.translate(PUNCT_T2S_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_T2S_MAP)
        return output

    def s2hk(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Simplified Chinese to Traditional (Hong Kong Standard).
        Pipeline:
          1) S2T
          2) HK variants (forward)
          3) Optional punctuation S->T
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.S2T))
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.HkVariantsOnly))
        )

        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            if HAS_MAKETRANS:
                return output.translate(PUNCT_S2T_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_S2T_MAP)
        return output

    def hk2s(self, input_text: str, punctuation: bool = False) -> str:
        """
        Convert Traditional (Hong Kong) to Simplified Chinese.

        Pipeline:
          1) HK reverse (variants + rev-phrases)
          2) T2S
          3) Optional punctuation T->S
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.HkRevPair))
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.T2S))
        )

        output = refs.apply_segment_replace(input_text, union_replace=self.union_replace)

        if punctuation:
            if HAS_MAKETRANS:
                return output.translate(PUNCT_T2S_MAP)
            return self._convert_punctuation_legacy(output, PUNCT_T2S_MAP)
        return output

    def t2tw(self, input_text: str) -> str:
        """
        Convert Traditional Chinese to Taiwan Standard Traditional Chinese (variants only).
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.TwVariantsOnly))
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

    def t2twp(self, input_text: str) -> str:
        """
        Convert Traditional Chinese to Taiwan Standard using phrase and variant mappings.
        Round 1: TW phrases only
        Round 2: TW variants only
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.TwPhrasesOnly))
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwVariantsOnly))
        )
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

    def tw2t(self, input_text: str) -> str:
        """
        Traditional (Taiwan) -> Traditional (normalize TW forms back to general T).
        Round 1: TW reverse pair (variants_rev_phrases + variants_rev)
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.TwRevPair))
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

    def tw2tp(self, input_text: str) -> str:
        """
        Traditional (Taiwan) -> Traditional (normalized) with extra reverse phrase pass.

        Round 1: TW reverse pair (variants_rev_phrases + variants_rev)
        Round 2: TW phrases_rev only
        """
        refs = (
            DictRefs(self.union_cache.ensure_indexed(UnionKey.TwRevPair))
            .with_round_2(self.union_cache.ensure_indexed(UnionKey.TwPhrasesRevOnly))
        )
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

    def t2hk(self, input_text: str) -> str:
        """
        Traditional -> Traditional (Hong Kong Standard).
        Round 1: HK variants only.
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.HkVariantsOnly))
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

    def hk2t(self, input_text: str) -> str:
        """
        Traditional (Hong Kong) -> Traditional (normalize HK forms back to general T).
        Round 1: HK reverse pair (variants_rev_phrases + variants_rev)
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.HkRevPair))
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

    def t2jp(self, input_text: str) -> str:
        """
        Traditional -> Japanese variants (Shinjitai) using JP variants only.
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.JpVariantsOnly))
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

    def jp2t(self, input_text: str) -> str:
        """
        Japanese (Shinjitai) -> Traditional Chinese.
        Round 1: JP reverse triple (jps_phrases, jps_characters, jp_variants_rev)
        """
        refs = DictRefs(self.union_cache.ensure_indexed(UnionKey.JpRevTriple))
        return refs.apply_segment_replace(input_text, union_replace=self.union_replace)

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
            elif config == "s2hk":
                return self.s2hk(input_text, punctuation)
            elif config == "t2s":
                return self.t2s(input_text, punctuation)
            elif config == "t2tw":
                return self.t2tw(input_text)
            elif config == "t2twp":
                return self.t2twp(input_text)
            elif config == "t2hk":
                return self.t2hk(input_text)
            elif config == "tw2s":
                return self.tw2s(input_text, punctuation)
            elif config == "tw2sp":
                return self.tw2sp(input_text, punctuation)
            elif config == "tw2t":
                return self.tw2t(input_text)
            elif config == "tw2tp":
                return self.tw2tp(input_text)
            elif config == "hk2s":
                return self.hk2s(input_text, punctuation)
            elif config == "hk2t":
                return self.hk2t(input_text)
            elif config == "jp2t":
                return self.jp2t(input_text)
            elif config == "t2jp":
                return self.t2jp(input_text)
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
        # return self.convert_segment(input_text, dict_data, 1)
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

        :param input_text: Input string
        :return: 0 = unknown, 1 = traditional, 2 = simplified
        """
        if not input_text:
            return 0

        stripped = STRIP_REGEX.sub("", input_text)
        strip_text = stripped[:100]

        if strip_text != self.ts(strip_text):
            return 1
        elif strip_text != self.st(strip_text):
            return 2
        else:
            return 0


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
    return ''.join(
        convert_segment_fn(text[start:end], dictionaries, max_word_length)
        for start, end in group_ranges
    )


def convert_range_group_union(args: Tuple[str, Iterable[Tuple[int, int]], "StarterUnion"]) -> str:
    text, group, union = args
    conv = OpenCC.convert_union  # local bind
    return "".join(conv(text[s:e], union) for (s, e) in group)
