import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from opencc_purepy import OpenCC
from opencc_purepy.office_helper import convert_office_doc
import test_cli
from workers.batch_worker import BatchWorker


class TestOfficeSync(unittest.TestCase):
    @staticmethod
    def make_docx(path, text):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", '<w:document xmlns:w="urn:word"><w:r><w:rPr><w:rFonts w:ascii="汉字"/></w:rPr><w:t>' + text + '</w:t></w:r></w:document>')

    def test_office_cli_normalizes_detofus_and_converts_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "龍汉字.docx"
            fallback = Path(temp_dir) / "fallback.txt"
            self.make_docx(source, "龍汉字𠀀")
            fallback.write_text("𠀀\t汉\tExtB\n", encoding="utf-8")
            result = test_cli.TestCli._run("office", "-i", str(source), "-c", "t2s", "-n", "--detofu", "--detofu-file", str(fallback), "-F")
            self.assertEqual(result.returncode, 0, result.stderr)
            with zipfile.ZipFile(Path(temp_dir) / "龙汉字_converted.docx") as archive:
                xml = archive.read("word/document.xml").decode("utf-8")
            self.assertIn("龙汉字汉", xml)
            self.assertIn('w:ascii="汉字"', xml)

    def test_office_cli_requires_detofu_for_custom_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.docx"
            self.make_docx(source, "汉字")
            result = test_cli.TestCli._run("office", "-i", str(source), "--detofu-file", "missing.txt")
            self.assertEqual(result.returncode, 1)
            self.assertIn("requires --detofu", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_failed_archive_publication_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.docx"
            output = Path(temp_dir) / "output.docx"
            self.make_docx(source, "汉字")
            output.write_bytes(b"existing output")
            with patch("opencc_purepy.office_helper._validate_zip_file", side_effect=zipfile.BadZipFile("invalid candidate")):
                success, message = convert_office_doc(str(source), str(output), "docx", OpenCC().convert)
            self.assertFalse(success)
            self.assertIn("invalid candidate", message)
            self.assertEqual(output.read_bytes(), b"existing output")
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_gui_batch_worker_converts_office_and_preserves_fonts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "汉字.docx"
            out_dir = Path(temp_dir) / "output"
            out_dir.mkdir()
            self.make_docx(source, "汉字“测试”")
            converter = OpenCC("s2t")
            worker = BatchWorker([str(source)], out_dir, converter, "s2t", True, False, False, False, True)
            messages = []
            worker.log.connect(messages.append)
            worker._process_one_file(1, 1, source)
            with zipfile.ZipFile(out_dir / "漢字_s2t.docx") as archive:
                xml = archive.read("word/document.xml").decode("utf-8")
            self.assertIn(converter.convert("汉字“测试”", True), xml)
            self.assertIn('w:ascii="汉字"', xml)
            self.assertTrue(any("Done." in message for message in messages))
