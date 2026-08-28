import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opencc_purepy import DictSlot, OpenCC
from opencc_purepy.__main__ import main as cli_main
from opencc_purepy.utils import (
    CustomDictSpec,
    custom_dict_specs_to_maps,
    parse_custom_dict_spec,
)


class DictionarySlotRefactorTest(unittest.TestCase):
    def test_dict_slot_parse_accepts_public_spellings(self) -> None:
        expected = DictSlot.HKPhrasesRev

        self.assertIs(DictSlot.parse(expected), expected)
        self.assertIs(DictSlot.parse("HKPhrasesRev"), expected)
        self.assertIs(DictSlot.parse("hkphrasesrev"), expected)
        self.assertIs(DictSlot.parse(" hk_phrases_rev "), expected)

    def test_dict_slot_parse_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "DictSlot or str"):
            # pyrefly: ignore [bad-argument-type]
            DictSlot.parse(42)

        with self.assertRaisesRegex(ValueError, "Unknown dictionary slot"):
            DictSlot.parse("not_a_slot")

    @patch("opencc_purepy.utils.Path.is_file", return_value=True)
    def test_custom_spec_preserves_windows_drive_path(self, _is_file) -> None:
        parsed = parse_custom_dict_spec(
            r"STPhrases:append:C:\dictionaries\custom.txt"
        )

        self.assertIs(parsed.slot, DictSlot.STPhrases)
        self.assertEqual(parsed.mode, "append")
        self.assertEqual(parsed.path, r"C:\dictionaries\custom.txt")

    def test_custom_specs_group_by_mode(self) -> None:
        overrides, appends = custom_dict_specs_to_maps([
            CustomDictSpec(DictSlot.STCharacters, "override", "chars.txt"),
            CustomDictSpec(DictSlot.STPhrases, "append", "phrases.txt"),
        ])

        self.assertEqual(overrides, {DictSlot.STCharacters: "chars.txt"})
        self.assertEqual(appends, {DictSlot.STPhrases: "phrases.txt"})

    def test_from_dict_files_appends_to_packaged_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "custom.txt"
            custom_path.write_text("帕兰蒂尔\t柏蘭蒂爾\n", encoding="utf-8")

            converter = OpenCC.from_dict_files(
                "s2t",
                [CustomDictSpec(DictSlot.STPhrases, "append", custom_path)],
            )

            self.assertEqual(converter.convert("帕兰蒂尔是一家公司"), "柏蘭蒂爾是一家公司")

    def test_convert_cli_accepts_custom_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            custom_path = temp_path / "custom.txt"
            input_path = temp_path / "input.txt"
            output_path = temp_path / "output.txt"
            custom_path.write_text("帕兰蒂尔\t柏蘭蒂爾\n", encoding="utf-8")
            input_path.write_text("帕兰蒂尔", encoding="utf-8")

            argv = [
                "opencc-purepy", "convert", "-c", "s2t",
                "-i", str(input_path), "-o", str(output_path),
                "-D", "STPhrases:append:{}".format(custom_path),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(cli_main(), 0)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "柏蘭蒂爾")


if __name__ == "__main__":
    unittest.main()
