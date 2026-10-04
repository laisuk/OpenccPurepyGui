"""
Pure-Python CJK/Unicode encoding detector.

Port of CjkEncodingDetector.cs.

Detection order:
BOM -> ASCII -> strict UTF-8 -> BOM-less UTF-16 heuristic ->
Big5/GB18030 statistical fallback -> Unknown.

GB2312 and GBK are reported as GB18030.

The Big5/GB18030 statistical fallback is derived from uchardet 0.0.5 /
Mozilla Universal Charset Detector. Preserve the corresponding upstream
licensing notices when redistributing this port.
"""
from dataclasses import dataclass
from typing import Optional

MAX_HEURISTIC_SAMPLE_SIZE = 128 * 1024
MIN_LEGACY_CONFIDENCE = 0.20
MIN_LEGACY_MARGIN = 0.05
STATE_START = 0
STATE_ERROR = 1
STATE_ITS_ME = 2


@dataclass(frozen=True)
class DetectionResult:
    encoding: Optional[str]
    name: str
    bom_size: int
    confidence: float


def detect_cjk_encoding(data) -> DetectionResult:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    data = memoryview(data).cast("B")
    if not data:
        return _unknown()

    if _starts_with(data, 0xEF, 0xBB, 0xBF):
        return DetectionResult("utf-8-sig", "UTF-8 BOM", 3, 1.0)
    if _starts_with(data, 0xFF, 0xFE):
        return DetectionResult("utf-16le", "UTF-16 LE BOM", 2, 1.0)
    if _starts_with(data, 0xFE, 0xFF):
        return DetectionResult("utf-16be", "UTF-16 BE BOM", 2, 1.0)

    kind = _classify_utf8(data)
    if kind is not None:
        if kind == "ascii":
            return DetectionResult("ascii", "ASCII", 0, 1.0)
        return DetectionResult("utf-8", "UTF-8", 0, 1.0)

    if _looks_like_utf16(data, 1):
        return DetectionResult("utf-16le", "UTF-16 LE", 0, 0.90)
    if _looks_like_utf16(data, 0):
        return DetectionResult("utf-16be", "UTF-16 BE", 0, 0.90)

    return _detect_chinese_legacy(data)


def _unknown():
    return DetectionResult(None, "Unknown", 0, 0.0)


def _starts_with(data, *values):
    return len(data) >= len(values) and all(data[i] == v for i, v in enumerate(values))


def _classify_utf8(data):
    i = 0
    end = len(data)
    ascii_only = True
    while i < end:
        c0 = data[i]
        if c0 <= 0x7F:
            i += 1
            continue
        ascii_only = False

        if 0xC2 <= c0 <= 0xDF:
            if i + 1 >= end or (data[i + 1] & 0xC0) != 0x80:
                return None
            i += 2
            continue

        if 0xE0 <= c0 <= 0xEF:
            if i + 2 >= end:
                return None
            c1, c2 = data[i + 1], data[i + 2]
            if (c2 & 0xC0) != 0x80:
                return None
            if c0 == 0xE0:
                if not 0xA0 <= c1 <= 0xBF: return None
            elif c0 == 0xED:
                if not 0x80 <= c1 <= 0x9F: return None
            elif (c1 & 0xC0) != 0x80:
                return None
            i += 3
            continue

        if 0xF0 <= c0 <= 0xF4:
            if i + 3 >= end:
                return None
            c1, c2, c3 = data[i + 1], data[i + 2], data[i + 3]
            if (c2 & 0xC0) != 0x80 or (c3 & 0xC0) != 0x80:
                return None
            if c0 == 0xF0:
                if not 0x90 <= c1 <= 0xBF: return None
            elif c0 == 0xF4:
                if not 0x80 <= c1 <= 0x8F: return None
            elif (c1 & 0xC0) != 0x80:
                return None
            i += 4
            continue
        return None
    return "ascii" if ascii_only else "utf8"


def _looks_like_utf16(data, zero_byte_index):
    count = len(data)
    if count < 4 or (count & 1):
        return False
    sample_count = min(count, MAX_HEURISTIC_SAMPLE_SIZE) & ~1
    zero_bytes = pairs = 0
    for i in range(0, sample_count, 2):
        first, second = data[i], data[i + 1]
        zero_byte = first if zero_byte_index == 0 else second
        other_byte = second if zero_byte_index == 0 else first
        if zero_byte == 0 and other_byte != 0:
            zero_bytes += 1
        pairs += 1
    return pairs != 0 and zero_bytes * 100 // pairs >= 60


def _detect_chinese_legacy(data):
    count = min(len(data), MAX_HEURISTIC_SAMPLE_SIZE)
    big5 = _probe_big5(data, count)
    gb = _probe_gb18030(data, count)
    best, second = max(big5, gb), min(big5, gb)
    if best < MIN_LEGACY_CONFIDENCE or best - second < MIN_LEGACY_MARGIN:
        return _unknown()
    if big5 > gb:
        return DetectionResult("big5", "Big5", 0, best)
    return DetectionResult("gb18030", "GB18030", 0, best)


class _DistributionAnalysis:
    MINIMUM_DATA_THRESHOLD = 4

    def __init__(self, bits, table_size, ratio, get_order):
        self.bits, self.table_size, self.ratio, self.get_order = bits, table_size, ratio, get_order
        self.total_chars = self.freq_chars = 0

    def handle_one_char(self, first, second):
        order = self.get_order(first, second)
        if order < 0: return
        self.total_chars += 1
        if order < self.table_size and _is_frequent(self.bits, order):
            self.freq_chars += 1

    def get_confidence(self):
        if self.total_chars <= 0 or self.freq_chars <= self.MINIMUM_DATA_THRESHOLD:
            return 0.01
        if self.total_chars == self.freq_chars:
            return 0.99
        ratio = self.freq_chars / ((self.total_chars - self.freq_chars) * self.ratio)
        return min(ratio, 0.99)


class _CodingStateMachine:
    def __init__(self, class_table, factor, state_table, char_len_table):
        self.class_table, self.factor = class_table, factor
        self.state_table, self.char_len_table = state_table, char_len_table
        self.current_state = STATE_START
        self.current_char_len = 0

    def next_state(self, value):
        byte_class = self.class_table[value]
        if self.current_state == STATE_START:
            self.current_char_len = self.char_len_table[byte_class]
        self.current_state = self.state_table[self.current_state * self.factor + byte_class]
        return self.current_state


def _feed_prober(data, count, machine, distribution):
    for i in range(count):
        state = machine.next_state(data[i])
        if state == STATE_ITS_ME: break
        if state != STATE_START: continue
        if machine.current_char_len == 2 and i > 0:
            distribution.handle_one_char(data[i - 1], data[i])


def _probe_big5(data, count):
    m = _CodingStateMachine(BIG5_CLASS_TABLE, 5, BIG5_STATE_TABLE, BIG5_CHAR_LEN_TABLE)
    d = _DistributionAnalysis(BIG5_FREQUENT_BITS, 5376, 0.75, _get_big5_order)
    _feed_prober(data, count, m, d)
    return d.get_confidence()


def _probe_gb18030(data, count):
    m = _CodingStateMachine(GB18030_CLASS_TABLE, 7, GB18030_STATE_TABLE, GB18030_CHAR_LEN_TABLE)
    d = _DistributionAnalysis(GB2312_FREQUENT_BITS, 3760, 0.90, _get_gb2312_order)
    _feed_prober(data, count, m, d)
    return d.get_confidence()


def _get_gb2312_order(first, second):
    return 94 * (first - 0xB0) + second - 0xA1 if first >= 0xB0 and second >= 0xA1 else -1


def _get_big5_order(first, second):
    if first < 0xA4: return -1
    if second >= 0xA1:
        return 157 * (first - 0xA4) + second - 0xA1 + 63
    return 157 * (first - 0xA4) + second - 0x40


def _is_frequent(bits, order):
    return (bits[order >> 5] & (1 << (order & 31))) != 0


BIG5_CLASS_TABLE = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0)
BIG5_STATE_TABLE = (1, 0, 0, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0)
BIG5_CHAR_LEN_TABLE = (0, 1, 1, 2, 0)
GB18030_CLASS_TABLE = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4,
                       5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0)
GB18030_STATE_TABLE = (1, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 4, 1, 0, 0, 1, 1, 1, 1,
                       1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0)
GB18030_CHAR_LEN_TABLE = (0, 1, 1, 1, 1, 1, 2)
BIG5_FREQUENT_BITS = (0x20BA8DE9, 0x40E91D50, 0x4B4C4826, 0x3012B810, 0xA08482CD, 0x17200A22, 0x2A264062, 0x14C0A142,
                      0x04891300, 0x0F2CC002, 0x198F8400, 0x03022704, 0x00070831, 0x44180E04, 0x05A00831, 0x00011142,
                      0x00000001, 0x00044000, 0x00888808, 0x01C00511, 0x80000082, 0xA0A10180, 0x07201001, 0x4074002A,
                      0x40005021, 0x20400008, 0x00000100, 0xA2000102, 0x00000002, 0x04308225, 0x02000000, 0x00800010,
                      0x21021640, 0x22002000, 0x04488038, 0x06090000, 0x00200200, 0x00022000, 0x00000423, 0x01804024,
                      0x02001000, 0x00400082, 0x00008000, 0x8004020A, 0x00000004, 0x20280008, 0x6D104000, 0x00C08000,
                      0x40008000, 0x00000000, 0x01000888, 0x00000000, 0x00A20250, 0x11010040, 0x00040000, 0x00100020,
                      0x00200100, 0x20000000, 0x00000020, 0x08010800, 0x21010000, 0x00008014, 0x00802A14, 0x00001010,
                      0x060000C0, 0x0108050C, 0x20801000, 0x20000080, 0x00008000, 0x00400008, 0x00839000, 0x01010180,
                      0x00000204, 0x00200000, 0x00018412, 0x001410D4, 0x20800003, 0x00810002, 0x40000240, 0x00000100,
                      0x00020000, 0x0000A008, 0x00000000, 0x06000000, 0x10040000, 0x20010100, 0x00000010, 0x00000048,
                      0x12040008, 0x08080000, 0xA1000280, 0x00008000, 0x00000010, 0x03A00000, 0x04000000, 0x00000028,
                      0x01001000, 0x00040000, 0x02000000, 0x00200810, 0x08000020, 0x40A86000, 0x20401000, 0x000020B8,
                      0x01040000, 0x00040000, 0x00026000, 0x00004200, 0x00000000, 0x20040000, 0x00000000, 0x04100000,
                      0x00010080, 0x00002C00, 0x04404000, 0x00012000, 0x00480000, 0x00040800, 0x00008000, 0x0000D800,
                      0x10000000, 0x00001042, 0x00001000, 0x00000000, 0x00000400, 0x00000000, 0x08000508, 0x00000000,
                      0x00000000, 0x00000000, 0x02800200, 0x00080410, 0x00000080, 0x00000000, 0x00000000, 0x01000000,
                      0x12000000, 0x00000080, 0x00000020, 0x20000000, 0x00000000, 0x50020008, 0x00000000, 0x02800200,
                      0x00000020, 0x00010000, 0x00000001, 0x00008000, 0x00000000, 0x00000000, 0x00000000, 0x00000210,
                      0x00000080, 0x00000000, 0x00000000, 0x00080000, 0x82800020, 0x00000000, 0x00000000, 0x000A8000,
                      0x00010000, 0x00010000, 0x00000000, 0x00180000, 0x00000001, 0x40800000, 0x01000010, 0x00220000)
GB2312_FREQUENT_BITS = (0x008A0000, 0x01010000, 0x08000800, 0x09204021, 0x00200020, 0x20002482, 0x05C00000, 0x04000213,
                        0x34200010, 0x00004001, 0x00008025, 0x20008000, 0x00000804, 0x10000424, 0x04028640, 0x23A60141,
                        0x1100A000, 0x48001000, 0x08008004, 0x000800C0, 0x020A0800, 0x01414020, 0x01004084, 0x20008000,
                        0x810C0601, 0x40204400, 0x814A09E1, 0x00400040, 0x00098320, 0x00004580, 0x47004000, 0x40000080,
                        0x038B2000, 0x00000005, 0x4C800402, 0xA0C4004C, 0x0612640A, 0x00000904, 0x00052024, 0x81120001,
                        0x286010C0, 0x00415002, 0x12010004, 0x92000005, 0x00200800, 0x08005480, 0x0080A000, 0x00081000,
                        0x80005000, 0x86007000, 0x14000288, 0x00083100, 0x00100200, 0x00040000, 0x00400030, 0x00000100,
                        0x43103000, 0x80010001, 0x04110400, 0x40400000, 0x000A0140, 0x40000002, 0x00000008, 0x00000000,
                        0x00008000, 0x00010400, 0x00802000, 0x00000848, 0x00010002, 0x04420000, 0x04910210, 0x64800640,
                        0x04400010, 0x01001800, 0x12000400, 0x4C300040, 0x56121080, 0x48062F97, 0x001000C7, 0x42800101,
                        0x000020A4, 0x00000004, 0x80008074, 0x80000000, 0x10081300, 0x19022000, 0x00000C40, 0x04808080,
                        0x40042001, 0x00202180, 0x04140102, 0x00410008, 0x82800208, 0x18189103, 0x00001145, 0x0008B41A,
                        0x40801080, 0x00080010, 0x00003010, 0x00600140, 0x01412020, 0x20259000, 0x81082013, 0x50000208,
                        0x00088284, 0x00400010, 0x021A0912, 0x00030004, 0x00002288, 0x04064000, 0x08900000, 0x248D4080,
                        0x2212491E, 0x0001AA0A, 0x0A440400, 0x08402002, 0x84412030, 0x00000180)
