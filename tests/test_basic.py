import unittest
from opencc_purepy.core import OpenCC


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

    def test_tw2sp_conversion(self):
        traditional = "漢字轉換測試：義大利的羅馬城不是一天裡就能建成的"
        self.converter.config = "tw2sp"
        result = self.converter.convert(traditional)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "汉字转换测试：意大利的罗马城不是一天里就能建成的")

    def test_invalid_config(self):
        converter = OpenCC("bad_config")
        # print(converter.config)
        result = converter.convert("测试")
        self.assertEqual("測試", result)  # s2t
        self.assertIn("Invalid config", converter.get_last_error())

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
