"""Process eligibility, fallback, and real spawn/pickle regression coverage."""
import builtins
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opencc_purepy import OpenCC
from opencc_purepy import core
from opencc_purepy.starter_union import StarterUnion


class TestMultiprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cc = OpenCC()
        cls.mapping = {'汉字': '漢字', '汉': '漢'}
        cls.dictionaries = [(cls.mapping, 2)]
        cls.union = StarterUnion(cls.mapping, 2)
        cls.union.build_starter_index()
        cls.legacy = SimpleNamespace(**{k: v for k, v in vars(cls.union).items()
                                       if k != '_starter_lengths'})
        # Exactly 1,000,000 characters, with more than 1,000 ranges.
        cls.large = ('汉字' + 'x' * 997 + '，') * 1000
        cls.large += '，'

    def paths(self):
        return [lambda text: self.cc.segment_replace(text, self.dictionaries, 2),
                lambda text: self.cc.union_replace(text, self.union),
                lambda text: self.cc.union_replace(text, self.legacy)]

    def test_serial_and_threshold_boundaries(self):
        for convert in self.paths():
            with patch.object(core, '_load_pool') as load:
                for text in ('', '汉字', '汉字，汉', self.large[:999999],
                             ('汉' * 999 + '，') * 1000, '汉' * 1000001):
                    with self.subTest(length=len(text)):
                        self.assertEqual(convert(text), text.replace('汉', '漢'))
                load.assert_not_called()

    def test_serial_fallbacks(self):
        cases = [({}, {'return_value': 1}), ({}, {'return_value': None}),
                 ({}, {'return_value': 0}),
                 ({}, {'side_effect': NotImplementedError}),
                 ({}, {'side_effect': OSError}),
                 ({'platform': 'emscripten'}, {'side_effect': AssertionError}),
                 ({'platform': 'wasi'}, {'side_effect': AssertionError})]
        for platform, cpu in cases:
            with patch.object(core.sys, 'platform', platform.get('platform', 'win32')), \
                    patch.object(core.os, 'cpu_count', **cpu), \
                    patch.object(core, '_load_pool') as load:
                for convert in self.paths():
                    self.assertEqual(convert(self.large), self.large.replace('汉', '漢'))
                load.assert_not_called()

    def test_lazy_import_and_missing_process_support(self):
        original = builtins.__import__
        def no_process(name, *args, **kwargs):
            if name == 'multiprocessing':
                raise ModuleNotFoundError("No module named '_multiprocessing'")
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_process), \
                patch.object(core.os, 'cpu_count', return_value=4):
            for convert in self.paths():
                self.assertEqual(convert(self.large), self.large.replace('汉', '漢'))

    def test_fresh_emscripten_import_and_large_public_conversion(self):
        code = """
import builtins
import sys
sys.platform = 'emscripten'
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('multiprocessing', '_multiprocessing'):
        raise AssertionError('Process support must not be imported')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from opencc_purepy import OpenCC
cc = OpenCC()
assert len(cc.supported_configs()) == 24
text = '汉字，' * 400000
assert len(text) == 1200000
assert cc.convert(text) == '漢字，' * 400000
assert OpenCC('t2s').convert('漢字') == '汉字'
"""
        subprocess.run([sys.executable, '-B', '-c', code], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_worker_count_cap(self):
        for count, expected in ((1, 1), (2, 2), (4, 4), (64, 4)):
            with patch.object(core.os, 'cpu_count', return_value=count):
                self.assertEqual(core._parallel_worker_count(), expected)

    def test_real_native_pool_matches_serial(self):
        if core.sys.platform in ('emscripten', 'wasi'):
            self.skipTest('Native process support required')
        for convert in self.paths():
            with patch.object(core.os, 'cpu_count', return_value=1):
                serial = convert(self.large)
            from multiprocessing import Pool
            with patch.object(core.os, 'cpu_count', return_value=2), \
                    patch.object(core, '_load_pool', return_value=Pool) as load:
                self.assertEqual(convert(self.large), serial)
                load.assert_called_once_with()

    def test_pool_errors_are_not_silently_retried(self):
        for convert in self.paths():
            for worker_error in (False, True):
                with patch.object(core.os, 'cpu_count', return_value=2), \
                        patch('multiprocessing.Pool') as pool:
                    if worker_error:
                        pool.return_value.__enter__.return_value.map.side_effect = RuntimeError('worker')
                    else:
                        pool.side_effect = OSError('startup')
                    with self.assertRaises((RuntimeError, OSError)):
                        convert(self.large)


if __name__ == '__main__':
    unittest.main()
