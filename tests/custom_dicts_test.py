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


if __name__ == "__main__":
    unittest.main()