import unittest
import json
import os
import shutil
import tempfile
from pathlib import Path
from opencc_purepy import DictSlot
from opencc_purepy.utils import CustomDictSpec
from opencc_purepy.core import OpenCC, OpenccConfig
from opencc_purepy.dictionary_lib import DictionaryMaxlength
from opencc_purepy.union_cache import UnionKey
from opencc_purepy.__main__ import _config_arg, _format_arg


class TestOpenCC(unittest.TestCase):

    def setUp(self):
        self.converter = OpenCC("s2t")

    def test_s2t_conversion(self):
        simplified = "汉字转换测试：意大利的罗马城不是一天里就能建成的"
        result = self.converter.s2t(simplified)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "漢字轉換測試：意大利的羅馬城不是一天裡就能建成的")  # Expect some output

    def test_t2s_conversion(self):
        traditional = "漢字轉換測試：意大利的羅馬城不是一天裡就能建成的"
        self.converter.config = "t2s"
        result = self.converter.convert(traditional)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "汉字转换测试：意大利的罗马城不是一天里就能建成的")

    def test_s2twp_conversion(self):
        simplified = "汉字转换测试：意大利的罗马城不是一天里就能建成的"
        result = self.converter.s2twp(simplified)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "漢字轉換測試：義大利的羅馬城不是一天裡就能建成的")  # Expect some output

    def test_s2twp_applies_taiwan_phrase_and_variant_normalization(self):
        self.assertEqual(self.converter.s2twp("软件为"), "軟體為")
        self.assertEqual(self.converter.s2twp("软件众"), "軟體眾")

    def test_s2twp_uses_two_round_plan(self):
        refs = self.converter._get_dict_refs("s2twp")

        self.assertIsNotNone(refs.round_2)
        self.assertIsNone(refs.round_3)

    def test_direct_taiwan_phrase_configs_use_one_round_triple_unions(self):
        forward_refs = self.converter._get_dict_refs("t2twp")
        reverse_refs = self.converter._get_dict_refs("tw2tp")

        self.assertIs(forward_refs.round_1, self.converter.union_cache.get_union(UnionKey.TwTriple))
        self.assertIsNone(forward_refs.round_2)
        self.assertIsNone(forward_refs.round_3)
        self.assertIs(reverse_refs.round_1, self.converter.union_cache.get_union(UnionKey.TwRevTriple))
        self.assertIsNone(reverse_refs.round_2)
        self.assertIsNone(reverse_refs.round_3)

    def test_direct_hong_kong_phrase_configs_use_one_round_triple_unions(self):
        forward_refs = self.converter._get_dict_refs("t2hkp")
        reverse_refs = self.converter._get_dict_refs("hk2tp")

        self.assertIs(forward_refs.round_1, self.converter.union_cache.get_union(UnionKey.HkTriple))
        self.assertIsNone(forward_refs.round_2)
        self.assertIsNone(forward_refs.round_3)
        self.assertIs(reverse_refs.round_1, self.converter.union_cache.get_union(UnionKey.HkRevTriple))
        self.assertIsNone(reverse_refs.round_2)
        self.assertIsNone(reverse_refs.round_3)

    def test_direct_hong_kong_phrase_conversion_and_dispatch(self):
        self.assertEqual(self.converter.t2hkp("搜索服務器"), "搜尋伺服器")
        self.assertEqual(self.converter.hk2tp("搜尋伺服器"), "搜索服務器")
        self.assertEqual(OpenCC("t2hkp").convert("搜索服務器"), "搜尋伺服器")
        self.assertEqual(OpenCC("hk2tp").convert("搜尋伺服器"), "搜索服務器")

    def test_tw2sp_conversion(self):
        traditional = "漢字轉換測試：義大利的羅馬城不是一天裡就能建成的"
        self.converter.config = "tw2sp"
        result = self.converter.convert(traditional)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "汉字转换测试：意大利的罗马城不是一天里就能建成的")

    def test_s2hkp_conversion_applies_hong_kong_phrase_and_variant_normalization(self):
        result = OpenCC("s2hkp").convert("软件搜索")

        self.assertEqual(result, "軟件搜尋")

    def test_hk2sp_conversion_applies_hong_kong_phrase_and_variant_normalization(self):
        result = OpenCC("hk2sp").convert("軟件伺服器")

        self.assertEqual(result, "软件服务器")

    def test_t2jp_conversion_uses_shinjitai_reverse_characters(self):
        result = OpenCC("t2jp").convert("舊字體：廣國，讀賣。")

        self.assertEqual(result, "旧字体：広国，読売。")

    def test_jp2t_conversion_uses_shinjitai_phrases_and_characters(self):
        result = OpenCC("jp2t").convert("旧字体：広国，読売。")

        self.assertEqual(result, "舊字體：廣國，讀賣。")

    def test_custom_japanese_shinjitai_slots_append(self):
        dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
            appends={
                DictSlot.JPSCharactersRev: {
                    "傳": "伝",
                },
                DictSlot.JPSCharacters: {
                    "仮": "假",
                },
                DictSlot.JPSPhrases: {
                    "売仮": "賣假",
                },
            },
        )

        self.assertEqual(OpenCC("t2jp", dictionary=dictionary).convert("傳"), "伝")
        self.assertEqual(OpenCC("jp2t", dictionary=dictionary).convert("仮"), "假")
        self.assertEqual(OpenCC("jp2t", dictionary=dictionary).convert("売仮"), "賣假")

    def test_custom_japanese_shinjitai_slots_override(self):
        dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
            overrides={
                DictSlot.JPSCharactersRev: {
                    "廣": "广",
                },
                DictSlot.JPSCharacters: {
                    "広": "廣-custom",
                },
                DictSlot.JPSPhrases: {
                    "広国": "廣國-custom",
                },
            },
        )

        self.assertEqual(OpenCC("t2jp", dictionary=dictionary).convert("廣"), "广")
        self.assertEqual(OpenCC("jp2t", dictionary=dictionary).convert("広"), "廣-custom")
        self.assertEqual(OpenCC("jp2t", dictionary=dictionary).convert("広国"), "廣國-custom")

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            OpenCC("bad_config")

    def test_config_parse_normalizes_supported_names(self):
        self.assertEqual(OpenccConfig.parse(" S2T "), OpenccConfig.S2T)
        self.assertEqual(OpenccConfig.parse("s2hkp"), OpenccConfig.S2HKP)
        self.assertEqual(OpenccConfig.parse("hk2sp"), OpenccConfig.HK2SP)
        self.assertEqual(OpenccConfig.parse("t2hkp"), OpenccConfig.T2HKP)
        self.assertEqual(OpenccConfig.parse("hk2tp"), OpenccConfig.HK2TP)

    def test_supported_configs_include_hong_kong_phrase_configs(self):
        configs = OpenCC.supported_configs()

        self.assertIn("s2hkp", configs)
        self.assertIn("hk2sp", configs)
        self.assertIn("t2hkp", configs)
        self.assertIn("hk2tp", configs)

    def test_dictionary_slot_normalization_uses_public_parser(self):
        self.assertEqual(
            DictionaryMaxlength._normalize_slot("ST_PHRASES"),
            DictSlot.STPhrases.value,
        )
        self.assertEqual(
            DictionaryMaxlength._normalize_slot("stphrases"),
            DictSlot.STPhrases.value,
        )

    def test_available_slots_match_canonical_enum_order(self):
        expected = [slot.canonical_name() for slot in DictSlot]

        self.assertEqual(OpenCC.available_slots(), expected)
        self.assertEqual(len(expected), len(set(expected)))
        self.assertIn("JPSCharacters", expected)
        self.assertIn("JPSCharactersRev", expected)
        self.assertIn("JPSPhrases", expected)
        self.assertFalse(any("JPVariants" in name for name in expected))

    def test_cli_config_arg_normalizes_supported_names(self):
        self.assertEqual(_config_arg("s2TW"), "s2tw")
        self.assertEqual(_config_arg("S2t"), "s2t")
        self.assertEqual(_config_arg("T2HKP"), "t2hkp")
        self.assertEqual(_config_arg("HK2TP"), "hk2tp")

    def test_cli_format_arg_normalizes_supported_names(self):
        self.assertEqual(_format_arg("DOCX"), "docx")

    def test_convert_with_punctuation(self):
        simplified = "“汉字转换测试”"
        result = self.converter.s2t(simplified, punctuation=True)
        self.assertEqual(result, "「漢字轉換測試」")

    def test_t2s_convert_with_punctuation(self):
        traditional = "「漢字轉換測試」"
        result = OpenCC("t2s").convert(traditional, punctuation=True)
        self.assertEqual(result, "“汉字转换测试”")

    def test_punctuation_applies_to_variant_configs(self):
        converter = OpenCC("t2tw")
        result = converter.convert("“軟體”", punctuation=True)
        self.assertIn("「", result)
        self.assertIn("」", result)

    def test_segment_replace_matches_direct_conversion_for_short_punctuated_text(self):
        refs = self.converter._get_dict_refs("s2t")
        slots, cap = refs._normalize()[0]
        text = "汉字转换测试，意大利的罗马城不是一天里就能建成的。" * 20
        expected = OpenCC.convert_segment(text, slots, cap)

        result = self.converter.segment_replace(text, slots, cap)
        self.assertEqual(expected, result)

    def test_backward_compatible_old_json_without_punctuation_slots(self):
        old_json = {
            "st_characters": [{"汉": "漢"}, 1],
            "st_phrases": [{"汉字": "漢字"}, 2],
            "ts_characters": [{"漢": "汉"}, 1],
            "ts_phrases": [{"漢字": "汉字"}, 2],
        }

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(old_json, handle, ensure_ascii=False)
            path = handle.name

        try:
            dictionary = DictionaryMaxlength.from_json(path)
        finally:
            os.unlink(path)
        self.assertEqual(dictionary.st_punctuations, ({}, 0))
        self.assertEqual(dictionary.ts_punctuations, ({}, 0))

    def test_txt_loading_accepts_missing_hong_kong_phrase_files(self):
        source_dir = Path(__file__).resolve().parents[1] / "opencc_purepy" / "dicts"
        legacy_dir = Path(tempfile.mkdtemp())

        try:
            for path in source_dir.glob("*.txt"):
                if path.name not in {"HKPhrases.txt", "HKPhrasesRev.txt"}:
                    shutil.copy(path, legacy_dir / path.name)

            dictionary = DictionaryMaxlength.from_dicts(base_dir=legacy_dir)

            self.assertEqual(dictionary.hk_phrases, ({}, 0))
            self.assertEqual(dictionary.hk_phrases_rev, ({}, 0))
        finally:
            shutil.rmtree(legacy_dir)

    def test_custom_hong_kong_phrase_slots_append_and_override(self):
        dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
            appends={
                DictSlot.HKPhrases: {
                    "搜索測試": "搜尋測試",
                },
                DictSlot.HKPhrasesRev: {
                    "搜尋測試": "搜索測試",
                },
            },
        )

        self.assertEqual(OpenCC("s2hkp", dictionary=dictionary).convert("搜索测试"), "搜尋測試")
        self.assertEqual(OpenCC("hk2sp", dictionary=dictionary).convert("搜尋測試"), "搜索测试")

        dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
            overrides={
                DictSlot.HKPhrases: {
                    "搜索": "搵嘢",
                },
                DictSlot.HKPhrasesRev: {
                    "搵嘢": "搜索",
                },
            },
        )

        self.assertEqual(dictionary.hk_phrases, ({"搜索": "搵嘢"}, len("搜索")))
        self.assertEqual(dictionary.hk_phrases_rev, ({"搵嘢": "搜索"}, len("搵嘢")))
        self.assertEqual(OpenCC("s2hkp", dictionary=dictionary).convert("搜索"), "搵嘢")
        self.assertEqual(OpenCC("hk2sp", dictionary=dictionary).convert("搵嘢"), "搜索")

    def test_custom_hong_kong_phrase_slot_files_append_and_override(self):
        temp_dir = Path(tempfile.mkdtemp())

        try:
            hk_append = temp_dir / "HKPhrasesAppend.txt"
            hk_rev_append = temp_dir / "HKPhrasesRevAppend.txt"
            hk_override = temp_dir / "HKPhrasesOverride.txt"
            hk_rev_override = temp_dir / "HKPhrasesRevOverride.txt"

            hk_append.write_text("搜索測試\t搜尋測試\n", encoding="utf-8")
            hk_rev_append.write_text("搜尋測試\t搜索測試\n細路哥\t小男孩\n", encoding="utf-8")
            hk_override.write_text("搜索\t搵嘢\n", encoding="utf-8")
            hk_rev_override.write_text("搵嘢\t搜索\n", encoding="utf-8")

            cc = OpenCC.from_dicts(
                config="s2hkp",
                appends={
                    DictSlot.HKPhrases: hk_append,
                    DictSlot.HKPhrasesRev: hk_rev_append,
                },
            )

            self.assertEqual(cc.convert("搜索测试"), "搜尋測試")
            self.assertEqual(OpenCC.from_dicts(
                config="hk2sp",
                appends={
                    DictSlot.HKPhrases: hk_append,
                    DictSlot.HKPhrasesRev: hk_rev_append,
                },
            ).convert("搜尋測試"), "搜索测试")

            self.assertEqual(OpenCC.from_dict_files(
                config="hk2sp",
                specs=[
                    CustomDictSpec(DictSlot.HKPhrasesRev, "append", hk_rev_append),
                ],
            ).convert("這個細路哥很靈活"), "这个小男孩很灵活")

            dictionary = DictionaryMaxlength.from_dicts(
                overrides={
                    DictSlot.HKPhrases: hk_override,
                    DictSlot.HKPhrasesRev: hk_rev_override,
                },
            )

            self.assertEqual(dictionary.hk_phrases, ({"搜索": "搵嘢"}, len("搜索")))
            self.assertEqual(dictionary.hk_phrases_rev, ({"搵嘢": "搜索"}, len("搵嘢")))
        finally:
            shutil.rmtree(temp_dir)

    def test_union_cache_warm_path_reuses_indexed_union(self):
        cc = OpenCC("s2t")
        first = cc.convert("汉字转换测试")
        union = cc.union_cache.get_union(UnionKey.S2T)
        second = cc.convert("汉字转换测试")

        self.assertEqual(first, "漢字轉換測試")
        self.assertEqual(second, first)
        self.assertTrue(union.indexed)

    def test_zho_check(self):
        mixed = "這是一個測試test123"  # Should be treated as Traditional
        result = self.converter.zho_check(mixed)
        self.assertEqual(result, 1)  # Assert this is detected as Traditional

    def test_detofu_builtin(self):
        from opencc_purepy import OpenCC
        from opencc_purepy.detofu import DeTofuLevel

        cc = OpenCC("t2s")

        text = "儼驂騑於上路，訪風景於崇阿，𱁬"
        converted = cc.convert(text)

        result = cc.detofu(converted, DeTofuLevel.ExtB)

        self.assertEqual(result, "俨骖騑于上路，访风景于崇阿，𱁬")

    def test_detofu_with_custom_pairs(self):
        from opencc_purepy import OpenCC
        from opencc_purepy.detofu import DeTofuLevel

        cc = OpenCC("t2s")

        result = cc.detofu_with_custom_pairs(
            "𱁬",
            DeTofuLevel.ExtB,
            {"𱁬": "?"}
        )

        self.assertEqual(result, "?")

    def test_detofu_level_string(self):
        from opencc_purepy import OpenCC

        cc = OpenCC("t2s")

        result = cc.detofu_with_custom_pairs(
            "𱁬",
            "all",
            {"𱁬": "?"}
        )

        self.assertEqual(result, "?")

    def test_detofu_with_custom_file(self):
        import os
        import tempfile

        from opencc_purepy import OpenCC
        from opencc_purepy.detofu import DeTofuLevel

        cc = OpenCC("t2s")

        fd, path = tempfile.mkstemp(suffix=".txt", text=True)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("𱁬\t?\tExtB\n")

            result = cc.detofu_with_custom_file(
                "𱁬",
                DeTofuLevel.ExtB,
                path
            )

            self.assertEqual(result, "?")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_detofu_preserves_unknown_characters(self):
        from opencc_purepy import OpenCC

        cc = OpenCC("t2s")

        self.assertEqual(cc.detofu("abc𱁬xyz", "all"), "abc𱁬xyz")


if __name__ == "__main__":
    unittest.main()
