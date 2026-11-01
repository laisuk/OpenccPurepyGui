"""Focused font restoration tests; no conversion engine or native module needed."""
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


_HELPER_PATH = Path(__file__).resolve().parents[1] / "opencc_purepy" / "office_helper.py"
_SPEC = importlib.util.spec_from_file_location("office_helper_under_test", _HELPER_PATH)
office_helper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(office_helper)


class TestOfficeFontRestoration(unittest.TestCase):
    def test_known_repeated_adjacent_and_multidigit_markers(self):
        fonts = {"__F_O_N_T_0__": "宋体", "__F_O_N_T_10__": "Arial"}
        self.assertEqual(
            office_helper._restore_fonts("前__F_O_N_T_10____F_O_N_T_0__ / __F_O_N_T_0__后", fonts),
            "前Arial宋体 / 宋体后",
        )

    def test_unknown_and_incomplete_markers_allow_later_known_markers(self):
        fonts = {"__F_O_N_T_0__": "宋体"}
        self.assertEqual(
            office_helper._restore_fonts(
                "__F_O_N_T_99__ __F_O_N_T_bad__F_O_N_T_0__ __F_O_N_T___F_O_N_T_0__ __F_O_N_T_",
                fonts,
            ),
            "__F_O_N_T_99__ __F_O_N_T_bad宋体 __F_O_N_T_宋体 __F_O_N_T_",
        )

    def test_restoration_is_nonrecursive(self):
        fonts = {"__F_O_N_T_0__": "__F_O_N_T_1__", "__F_O_N_T_1__": "__F_O_N_T_0__"}
        self.assertEqual(
            office_helper._restore_fonts("__F_O_N_T_0____F_O_N_T_1__", fonts),
            "__F_O_N_T_1____F_O_N_T_0__",
        )

    def test_empty_map_empty_input_and_plain_text(self):
        fonts = {"__F_O_N_T_0__": "宋体"}
        self.assertEqual(office_helper._restore_fonts("__F_O_N_T_0__", {}), "__F_O_N_T_0__")
        self.assertEqual(office_helper._restore_fonts("", fonts), "")
        self.assertEqual(office_helper._restore_fonts("普通文本", fonts), "普通文本")
        self.assertEqual(office_helper._restore_fonts("__F_O_N_T_0__", {"__F_O_N_T_0__": ""}), "")

    def test_docx_conversion_restores_fonts_without_recursion(self):
        xml = '<w:rFonts w:ascii="__F_O_N_T_1__" w:eastAsia="宋体"/><w:t>汉语</w:t>'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.docx"
            output = Path(directory) / "output.docx"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("word/document.xml", xml)
            success, message = office_helper.convert_office_doc(
                str(source), str(output), "docx", lambda text: text.replace("汉语", "漢語"), keep_font=True,
            )
            self.assertTrue(success, message)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("word/document.xml").decode("utf-8"), xml.replace("汉语", "漢語"))


if __name__ == "__main__":
    unittest.main()
