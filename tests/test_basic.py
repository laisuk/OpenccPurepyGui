import unittest
from opencc_purepy.core import OpenCC, OpenccConfig
from opencc_purepy.union_cache import UnionKey


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

    def test_direct_taiwan_phrase_configs_use_one_round_triple_unions(self):
        forward_refs = self.converter._get_dict_refs("t2twp")
        reverse_refs = self.converter._get_dict_refs("tw2tp")

        self.assertIs(forward_refs.round_1, self.converter.union_cache.get_union(UnionKey.TwTriple))
        self.assertIsNone(forward_refs.round_2)
        self.assertIs(reverse_refs.round_1, self.converter.union_cache.get_union(UnionKey.TwRevTriple))
        self.assertIsNone(reverse_refs.round_2)

    def test_direct_hong_kong_phrase_configs_use_one_round_triple_unions(self):
        forward_refs = self.converter._get_dict_refs("t2hkp")
        reverse_refs = self.converter._get_dict_refs("hk2tp")

        self.assertIs(forward_refs.round_1, self.converter.union_cache.get_union(UnionKey.HkTriple))
        self.assertIsNone(forward_refs.round_2)
        self.assertIs(reverse_refs.round_1, self.converter.union_cache.get_union(UnionKey.HkRevTriple))
        self.assertIsNone(reverse_refs.round_2)

    def test_direct_hong_kong_phrase_conversion_and_dispatch(self):
        self.assertEqual(self.converter.t2hkp("搜索服務器"), "搜尋伺服器")
        self.assertEqual(self.converter.hk2tp("搜尋伺服器"), "搜索服務器")
        self.assertEqual(OpenCC("t2hkp").convert("搜索服務器"), "搜尋伺服器")
        self.assertEqual(OpenCC("hk2tp").convert("搜尋伺服器"), "搜索服務器")

    def test_direct_hong_kong_phrase_configs_are_supported(self):
        self.assertEqual(OpenccConfig.parse("t2hkp"), OpenccConfig.T2HKP)
        self.assertEqual(OpenccConfig.parse("hk2tp"), OpenccConfig.HK2TP)

    def test_tw2sp_conversion(self):
        traditional = "漢字轉換測試：義大利的羅馬城不是一天裡就能建成的"
        self.converter.config = "tw2sp"
        result = self.converter.convert(traditional)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "汉字转换测试：意大利的罗马城不是一天里就能建成的")

    def test_invalid_config(self):
        with self.assertRaisesRegex(ValueError, r"Invalid config: bad_config"):
            OpenCC("bad_config")

    def test_convert_with_punctuation(self):
        simplified = "“汉字转换测试”"
        result = self.converter.s2t(simplified, punctuation=True)
        self.assertIn("「", result)
        self.assertIn("」", result)

    def test_zho_check(self):
        mixed = "這是一個測試test123"  # Should be treated as Traditional
        result = self.converter.zho_check(mixed)
        self.assertEqual(result, 1)  # Assert this is detected as Traditional


if __name__ == "__main__":
    unittest.main()
