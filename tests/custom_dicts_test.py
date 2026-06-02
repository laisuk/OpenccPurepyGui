# tests/custom_dicts_test.py

import unittest
from pathlib import Path

from opencc_purepy import OpenCC, DictSlot
from opencc_purepy.dictionary_lib import DictionaryMaxlength

_TEST_DIR = Path(__file__).resolve().parent
_CUSTOM_DICT = _TEST_DIR / "_custom_st_phrases.txt"


def _write_custom_dict() -> Path:
    _CUSTOM_DICT.write_text(
        "帕兰蒂尔\t柏蘭蒂爾\n",
        encoding="utf-8",
    )
    return _CUSTOM_DICT


def _cleanup_custom_dict() -> None:
    try:
        _CUSTOM_DICT.unlink()
    except FileNotFoundError:
        pass


class CustomDictsTest(unittest.TestCase):
    def tearDown(self) -> None:
        _cleanup_custom_dict()

    def test_opencc_from_dicts_appends_custom_st_phrase(self) -> None:
        custom_dict = _write_custom_dict()

        cc = OpenCC.from_dicts(
            config="s2t",
            appends={
                "st_phrases": custom_dict,
            },
        )

        self.assertEqual(
            cc.convert("帕兰蒂尔是一家公司"),
            "柏蘭蒂爾是一家公司",
        )

    def test_dictionary_from_dicts_appends_late_comer_wins(self) -> None:
        custom_dict = _write_custom_dict()

        dictionary = DictionaryMaxlength.from_dicts(
            appends={
                DictSlot.ST_PHRASES: custom_dict,
            },
        )

        st_phrases, max_len = dictionary.st_phrases

        self.assertEqual(st_phrases["帕兰蒂尔"], "柏蘭蒂爾")
        self.assertGreaterEqual(max_len, len("帕兰蒂尔"))

    def test_dictslot_forward_variant_phrase_enum_members_exist(self) -> None:
        self.assertEqual(DictSlot.TW_VARIANTS_PHRASES.value, "tw_variants_phrases")
        self.assertEqual(DictSlot.HK_VARIANTS_PHRASES.value, "hk_variants_phrases")

    def test_dictionary_from_dicts_loads_forward_variant_phrase_slots(self) -> None:
        dictionary = DictionaryMaxlength.from_dicts()

        tw_variants_phrases, tw_max_len = dictionary.tw_variants_phrases
        hk_variants_phrases, hk_max_len = dictionary.hk_variants_phrases

        self.assertEqual(tw_variants_phrases["喫茶小舖"], "喫茶小舖")
        self.assertEqual(hk_variants_phrases["喫茶小舖"], "喫茶小舖")
        self.assertGreaterEqual(tw_max_len, len("喫茶小舖"))
        self.assertGreaterEqual(hk_max_len, len("喫茶小舖"))

    def test_dictionary_with_custom_dicts_appends_forward_variant_phrase_slots(self) -> None:
        dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
            appends={
                DictSlot.TW_VARIANTS_PHRASES: {
                    "喫茶測試": "喫茶測試",
                },
                DictSlot.HK_VARIANTS_PHRASES: {
                    "喫茶測試": "喫茶測試",
                },
            },
        )

        self.assertEqual(OpenCC(config="t2tw", dictionary=dictionary).convert("喫茶測試"), "喫茶測試")
        self.assertEqual(OpenCC(config="t2hk", dictionary=dictionary).convert("喫茶測試"), "喫茶測試")

    def test_forward_variant_phrase_slots_precede_character_slots(self) -> None:
        self.assertEqual(OpenCC("t2tw").convert("喫茶小舖"), "喫茶小舖")
        self.assertEqual(OpenCC("s2tw").convert("喫茶小舖"), "喫茶小舖")
        self.assertEqual(OpenCC("t2hk").convert("喫茶小舖"), "喫茶小舖")
        self.assertEqual(OpenCC("s2hk").convert("喫茶小舖"), "喫茶小舖")

    def test_reverse_tw_hk_variant_behavior_remains_unchanged(self) -> None:
        self.assertEqual(OpenCC("tw2t").convert("吃口飯"), "喫口飯")
        self.assertEqual(OpenCC("hk2t").convert("吃口飯"), "喫口飯")


if __name__ == "__main__":
    unittest.main()
