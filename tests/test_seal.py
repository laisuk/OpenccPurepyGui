import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import test_cli

from opencc_purepy import DictSlot, OpenCC, OpenccConfig
from opencc_purepy.dictionary_lib import DictionaryMaxlength
from opencc_purepy.union_cache import UnionKey


class TestSeal(unittest.TestCase):
    def test_builtin_conversion_and_dispatch(self):
        cc = OpenCC()
        seal = "\U0003ef7a\U0003fbb5"
        cases = [("s2seal", "汉字", seal), ("t2seal", "漢字", seal),
                 ("seal2s", seal, "汉字"), ("seal2t", seal, "漢字")]
        for config, source, expected in cases:
            with self.subTest(config=config):
                self.assertEqual(getattr(cc, config)(source), expected)
                self.assertEqual(cc.get_config(), "s2t")
                converter = OpenCC(OpenccConfig.parse(config.upper()))
                self.assertEqual(converter.convert(source), expected)
                self.assertEqual(getattr(cc, config)(""), "")
                self.assertEqual(getattr(cc, config)("ABC🙂"), "ABC🙂")

    def test_punctuation(self):
        cc = OpenCC()
        seal = cc.t2seal("漢")
        for config, source in [("s2seal", "汉"), ("t2seal", "漢")]:
            self.assertEqual(getattr(cc, config)("“" + source + "”", True), "「" + seal + "」")
        self.assertEqual(cc.seal2s("「" + seal + "」", True), "“汉”")
        self.assertEqual(cc.seal2t("“" + seal + "”", True), "「漢」")

    def test_txt_and_packaged_slots_match(self):
        raw = DictionaryMaxlength.from_dicts()
        packaged = DictionaryMaxlength.from_json()
        for slot in [DictSlot.SealCharacters, DictSlot.SealCharactersRev,
                     DictSlot.SealVariants, DictSlot.SealVariantsRev]:
            self.assertTrue(getattr(packaged, slot.value)[0])
            self.assertEqual(getattr(raw, slot.value), getattr(packaged, slot.value))

    def test_round_order_customization_and_cache(self):
        dictionary = DictionaryMaxlength()
        dictionary.st_characters = ({"s": "t"}, 1)
        dictionary.seal_variants = ({"t": "v"}, 1)
        dictionary.seal_characters_rev = ({"v": "𽀀"}, 1)
        dictionary.seal_characters = ({"𽀀": "v"}, 1)
        dictionary.seal_variants_rev = ({"v": "t"}, 1)
        dictionary.ts_characters = ({"t": "s"}, 1)
        cc = OpenCC(dictionary=dictionary)
        self.assertEqual(cc.s2seal("s"), "𽀀")
        self.assertEqual(cc.t2seal("t"), "𽀀")
        self.assertEqual(cc.seal2s("𽀀"), "s")
        self.assertEqual(cc.seal2t("𽀀"), "t")
        self.assertIs(cc._get_dict_refs("s2seal"), cc._get_dict_refs("s2seal"))
        self.assertIs(cc._get_dict_refs("s2seal").round_2,
                      cc.union_cache.get_union(UnionKey.SealVariantsOnly))
        self.assertIsNone(cc._get_dict_refs("t2seal").round_3)
        self.assertIsNotNone(cc._get_dict_refs("t2seal_punct").round_3)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "custom.txt"
            path.write_text("v\tX\n", encoding="utf-8")
            dictionary.with_custom_dict_files(overrides={DictSlot.SealCharactersRev: path})
            self.assertEqual(OpenCC(dictionary=dictionary).s2seal("s"), "X")

    def test_cli_accepts_seal_config(self):
        for command in ("convert", "office"):
            result = test_cli.TestCli._run(command, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            for config in ("s2seal", "t2seal", "seal2s", "seal2t"):
                self.assertIn(config, result.stdout)
            for slot in ("SealCharacters", "SealCharactersRev", "SealVariants", "SealVariantsRev"):
                self.assertIn(slot, result.stdout)

    def test_cli_converts_all_seal_configs_with_punctuation(self):
        seal = "\U0003ef7a\U0003fbb5"
        cases = [("s2seal", "“汉字”", "「" + seal + "」"),
                 ("t2seal", "“漢字”", "「" + seal + "」"),
                 ("seal2s", "「" + seal + "」", "“汉字”"),
                 ("seal2t", "“" + seal + "”", "「漢字」")]
        for config, source, expected in cases:
            with self.subTest(config=config):
                result = test_cli.TestCli._run("convert", "-c", config.upper(), "-p", input_text=source)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_cli_custom_seal_slot_and_dictgen(self):
        with tempfile.TemporaryDirectory() as temp:
            custom = Path(temp) / "custom.txt"
            # SealVariants maps 漢 to this intermediate character first.
            custom.write_text("\U00024049\tX\n", encoding="utf-8")
            spec = "SealCharactersRev:override:" + str(custom)
            result = test_cli.TestCli._run("convert", "-c", "t2seal", "-D", spec, input_text="漢")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "X")
            output = Path(temp) / "dictionary.json"
            result = test_cli.TestCli._run("dictgen", "-o", str(output), "-D", spec)
            self.assertEqual(result.returncode, 0, result.stderr)
            generated = DictionaryMaxlength.from_json(output)
            self.assertEqual(OpenCC("t2seal", dictionary=generated).convert("漢"), "X")
            for slot in (DictSlot.SealCharacters, DictSlot.SealCharactersRev,
                         DictSlot.SealVariants, DictSlot.SealVariantsRev):
                self.assertTrue(getattr(generated, slot.value)[0])

    def test_legacy_dictionaries_can_omit_seal_slots(self):
        source = Path(__file__).resolve().parents[1] / "opencc_purepy" / "dicts"
        with tempfile.TemporaryDirectory() as temp:
            legacy = Path(temp)
            for path in source.glob("*.txt"):
                if not path.name.startswith("Seal"):
                    shutil.copyfile(path, legacy / path.name)
            data = json.loads((source / "dictionary_maxlength.json").read_text(encoding="utf-8"))
            data = {key: value for key, value in data.items() if not key.startswith("seal_")}
            json_path = legacy / "dictionary.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            for dictionary in (DictionaryMaxlength.from_dicts(legacy), DictionaryMaxlength.from_json(json_path)):
                self.assertEqual(OpenCC(dictionary=dictionary).convert("汉字"), "漢字")
                self.assertEqual(dictionary.seal_characters, ({}, 0))
                self.assertEqual(OpenCC("t2seal", dictionary=dictionary).convert("漢字"), "漢字")

    def test_office_cli_preserves_seal_characters(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.docx"
            sealed = Path(temp) / "sealed.docx"
            restored = Path(temp) / "restored.docx"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("word/document.xml",
                                 '<w:document xmlns:w="urn:word"><w:t>汉字</w:t></w:document>')
            for input_path, output_path, config, expected in (
                    (source, sealed, "s2seal", "\U0003ef7a\U0003fbb5"),
                    (sealed, restored, "seal2s", "汉字")):
                result = test_cli.TestCli._run("office", "-i", str(input_path), "-o", str(output_path), "-c", config)
                self.assertEqual(result.returncode, 0, result.stderr)
                with zipfile.ZipFile(output_path) as archive:
                    self.assertIn(expected, archive.read("word/document.xml").decode("utf-8"))
