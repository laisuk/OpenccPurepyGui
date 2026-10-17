"""Behavior and compatibility coverage for indexed greedy conversion."""
import inspect
import pickle
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opencc_purepy.core import OpenCC, convert_range_group_union
from opencc_purepy.starter_union import StarterUnion


def legacy_union(union):
    return SimpleNamespace(**{key: value for key, value in vars(union).items()
                              if key != '_starter_lengths'})


class TestIndexedHotPath(unittest.TestCase):
    def test_constructor_signature_and_lazy_index_are_preserved(self):
        self.assertEqual(list(inspect.signature(StarterUnion.__init__).parameters),
                         ['self', 'merged_map', 'cap', 'bmp_mask', 'bmp_cap',
                          'astral_mask', 'astral_cap', 'indexed'])
        union = StarterUnion({'汉': '漢'}, 1)
        self.assertIsNone(union._starter_lengths)
        union.build_starter_index()
        index = union._starter_lengths
        union.build_starter_index()
        self.assertIs(union._starter_lengths, index)
        self.assertNotIn('_starter_lengths', repr(union))

    def test_long_keys_astral_starters_and_empty_replacements(self):
        mapping = {'汉': 'H', '汉字': '', '𠀀字': 'ASTRAL'}
        for length in (63, 64, 65, 70, 120):
            mapping['a' * length] = str(length)
        text = '汉字汉𠀀字!' + 'a' * 125
        for cap in (0, 1, 3, 63, 64, 65, 70, 120):
            union = StarterUnion(mapping, cap)
            union.build_starter_index()
            with self.subTest(cap=cap):
                self.assertEqual(OpenCC.convert_union_indexed(text, union),
                                 OpenCC.convert_union_indexed(text, legacy_union(union)))
        union = StarterUnion(mapping, 120)
        self.assertEqual(OpenCC.convert_union('汉字汉𠀀字!', union), 'HASTRAL!')

    def test_randomized_matching_against_original_mask_path(self):
        rng = random.Random(20261004)
        for _ in range(100):
            mapping = {''.join(rng.choice('ab汉𠀀') for _ in range(rng.randint(1, 12))):
                           rng.choice(['', 'X', '汉字']) for _ in range(80)}
            union = StarterUnion(mapping, rng.choice([0, 1, 5, 12]))
            union.build_starter_index()
            text = ''.join(rng.choice('ab汉𠀀!') for _ in range(300))
            self.assertEqual(OpenCC.convert_union_indexed(text, union),
                             OpenCC.convert_union_indexed(text, legacy_union(union)))

    def test_delimiter_behavior_at_short_text_threshold(self):
        cc = OpenCC()
        union = StarterUnion({'汉，字': 'CROSS', '汉': 'H', '字': 'Z'}, 3)
        short = 'x' * 9996 + '汉，字'
        long = 'x' * 9997 + '汉，字'
        self.assertEqual(cc.union_replace(short, union), 'x' * 9996 + 'CROSS')
        self.assertEqual(cc.union_replace(long, union), 'x' * 9997 + 'H，Z')

    def test_group_ranges_preserve_offsets_gaps_and_order(self):
        union = StarterUnion({'汉字': 'PAIR', '汉': 'H', '字': 'Z'}, 2)
        union.build_starter_index()
        text = 'xx汉字!汉字yy'
        ranges = [(5, 7), (2, 3), (3, 4), (2, 4), (4, 4)]
        self.assertEqual(convert_range_group_union((text, ranges, union)), 'PAIRHZPAIR')
        self.assertEqual(convert_range_group_union((text, [], union)), '')
        self.assertEqual(convert_range_group_union((text, ranges, legacy_union(union))),
                         'PAIRHZPAIR')
        unusual = [(-4, -2), (2, 100), (8, 2)]
        self.assertEqual(convert_range_group_union((text, unusual, union)),
                         convert_range_group_union((text, unusual, legacy_union(union))))

    def test_precedence_and_pickle_round_trip(self):
        union = StarterUnion.merge_precedence([
            ({'汉字': 'FIRST', '汉': ''}, 2), ({'汉字': 'SECOND', '字': 'Z'}, 2)])
        union.build_starter_index()
        restored = pickle.loads(pickle.dumps(union))
        self.assertEqual(OpenCC.convert_union_indexed('汉字汉字汉', restored), 'FIRSTFIRST')
        self.assertEqual(restored, union)
        # Older objects can arrive indexed without the new instance attribute.
        del restored._starter_lengths
        self.assertEqual(OpenCC.convert_union_indexed('汉字汉', restored), 'FIRST')

    def test_all_configs_and_punctuation_match_mask_path(self):
        text = (
                "汉字软件搜索后台发展头发，漢字軟體搜尋伺服器；𠀀😀「测试」。\n"
                * 400
        )
        self.assertGreaterEqual(len(text), 10000)

        for config in OpenCC.CONFIG_LIST:
            cc = OpenCC(config)

            for punctuation in (False, True):
                config_key = config + ("_punct" if punctuation else "")
                refs = cc._get_dict_refs(config_key)

                # Reference: legacy greedy dictionary lookup.
                expected = refs.apply_segment_replace(
                    text,
                    segment_replace=cc.segment_replace,
                    validate_delegates=False,
                )

                # Actual: optimized StarterUnion lookup.
                actual = cc.convert(text, punctuation)

                with self.subTest(
                        config=config,
                        punctuation=punctuation,
                ):
                    self.assertEqual(actual, expected)
                    self.assertFalse(
                        actual.startswith("Conversion failed:")
                    )

    def test_parallel_threshold_and_chunking_are_preserved(self):
        cc = OpenCC()
        union = StarterUnion({'汉': 'H'}, 1)
        below = 'x，' * 499999 + '汉'
        at_threshold = below + '，'
        with patch('opencc_purepy.core.Pool') as pool_type:
            pool = pool_type.return_value.__enter__.return_value
            pool.map.side_effect = lambda fn, grps: [fn(group) for group in grps]
            self.assertEqual(cc.union_replace(below, union), 'x，' * 499999 + 'H')
            pool_type.assert_not_called()
            self.assertEqual(cc.union_replace(at_threshold, union), 'x，' * 499999 + 'H，')
            pool_type.assert_called_once()
            groups = pool.map.call_args[0][1]
            self.assertGreater(len(groups), 0)
            self.assertLessEqual(len(groups), 4)


if __name__ == '__main__':
    unittest.main()
