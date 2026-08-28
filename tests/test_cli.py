import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCli(unittest.TestCase):
    @staticmethod
    def _run(*args, input_text=None):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        return subprocess.run(
            [sys.executable, "-m", "opencc_purepy", *args],
            input=input_text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            env=env,
        )

    def test_convert_defaults_to_s2t_with_stdin_stdout(self):
        result = self._run("convert", input_text="汉字")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "漢字")
        self.assertIn("default 's2t'", result.stderr)

    def test_convert_repeated_custom_dicts_preserve_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.txt"
            second = Path(temp_dir) / "second.txt"
            first.write_text("汉字\t第一\n", encoding="utf-8")
            second.write_text("汉字\t第二\n", encoding="utf-8")
            result = self._run(
                "convert", "-c", "s2t",
                "-D", "STPhrases:append:{}".format(first),
                "-D", "STPhrases:append:{}".format(second),
                input_text="汉字",
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "第二")

    def test_convert_norm_compat(self):
        result = self._run(
            "convert", "-c", "t2s", "-n",
            input_text="天龍八部書裡的喬峰是契丹人",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "天龙八部书里的乔峰是契丹人")

    def test_convert_norm_compat_extended(self):
        result = self._run(
            "convert", "-c", "t2s", "-E",
            input_text="聼聼竒羙⽟䂖甁噐⾳",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "听听奇美玉石瓶器音")

    def test_convert_norm_compat_extended_takes_precedence(self):
        result = self._run(
            "convert", "-c", "t2s", "-n", "-E",
            input_text="聼聼竒羙⽟䂖甁噐⾳",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "听听奇美玉石瓶器音")

    def test_convert_reports_invalid_detofu_without_traceback(self):
        result = self._run("convert", "--detofu", "invalid", input_text="汉字")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Supported DeTofu levels", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_convert_reports_invalid_encoding_without_traceback(self):
        result = self._run("convert", "--in-enc", "not-an-encoding", input_text="汉字")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Conversion failed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_convert_requires_detofu_for_custom_fallback_file(self):
        result = self._run("convert", "--detofu-file", "missing.txt", input_text="汉字")
        self.assertEqual(result.returncode, 1)
        self.assertIn("requires --detofu", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_dictgen_reports_missing_directory_on_stderr(self):
        result = self._run("dictgen", "-d", "directory-that-does-not-exist")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Dictionary directory does not exist", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_dictgen_repeated_custom_dicts_preserve_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.txt"
            second = Path(temp_dir) / "second.txt"
            output = Path(temp_dir) / "dictionary.json"
            first.write_text("汉字\t第一\n", encoding="utf-8")
            second.write_text("汉字\t第二\n", encoding="utf-8")
            result = self._run(
                "dictgen", "-o", str(output), "--compact", "--no-sort",
                "-D", "STPhrases:append:{}".format(first),
                "-D", "STPhrases:append:{}".format(second),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["st_phrases"][0]["汉字"], "第二")

    def test_help_lists_configs_slots_and_format_override(self):
        convert_help = self._run("convert", "--help")
        office_help = self._run("office", "--help")
        self.assertEqual(convert_help.returncode, 0)
        self.assertIn("Supported configurations: s2t", convert_help.stdout)
        self.assertIn("s2t.", convert_help.stdout)
        self.assertIn("JPSCharactersRev", convert_help.stdout)
        self.assertIn("-n", convert_help.stdout)
        self.assertIn("--norm-compat", convert_help.stdout)
        self.assertIn("-E", convert_help.stdout)
        self.assertIn("--norm-compat-extended", convert_help.stdout)
        self.assertEqual(office_help.returncode, 0)
        self.assertIn("Document format override", office_help.stdout)
        self.assertNotIn("--keep-font", office_help.stdout)
        self.assertIn("--no-keep-font", office_help.stdout)


if __name__ == "__main__":
    unittest.main()
