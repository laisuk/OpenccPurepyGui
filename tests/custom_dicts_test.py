# tests/custom_dicts_test.py

from pathlib import Path
import shutil
import tempfile

from opencc_purepy import OpenCC, DictSlot
from opencc_purepy.dictionary_lib import DictionaryMaxlength

_TEST_DIR = Path(__file__).resolve().parent
_CUSTOM_DICT = _TEST_DIR / "_custom_st_phrases.txt"
_CUSTOM_OVERRIDE_DICT = _TEST_DIR / "_override_st_phrases.txt"
_CUSTOM_PUNCT_DICT = _TEST_DIR / "_custom_st_punctuations.txt"


def _write_custom_dict() -> Path:
    _CUSTOM_DICT.write_text(
        "帕兰蒂尔\t柏蘭蒂爾\n",
        encoding="utf-8",
    )
    return _CUSTOM_DICT


def _cleanup_custom_dict() -> None:
    for path in (_CUSTOM_DICT, _CUSTOM_OVERRIDE_DICT, _CUSTOM_PUNCT_DICT):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_opencc_from_dicts_appends_custom_st_phrase():
    custom_dict = _write_custom_dict()

    try:
        cc = OpenCC.from_dicts(
            config="s2t",
            appends={
                "st_phrases": custom_dict,
            },
        )

        assert cc.convert("帕兰蒂尔是一家公司") == "柏蘭蒂爾是一家公司"
    finally:
        _cleanup_custom_dict()


def test_dictionary_from_dicts_appends_late_comer_wins():
    custom_dict = _write_custom_dict()

    try:
        dictionary = DictionaryMaxlength.from_dicts(
            appends={
                DictSlot.STPhrases: custom_dict,
            },
        )

        st_phrases, max_len = dictionary.st_phrases

        assert st_phrases["帕兰蒂尔"] == "柏蘭蒂爾"
        assert max_len >= len("帕兰蒂尔")
    finally:
        _cleanup_custom_dict()


def test_opencc_from_dicts_append_late_comer_wins_existing_key():
    _CUSTOM_DICT.write_text(
        "汉字\t測字\n",
        encoding="utf-8",
    )

    try:
        cc = OpenCC.from_dicts(
            config="s2t",
            appends={
                DictSlot.STPhrases: _CUSTOM_DICT,
            },
        )

        assert cc.convert("汉字转换") == "測字轉換"
    finally:
        _cleanup_custom_dict()


def test_opencc_from_dicts_override_replaces_entire_slot():
    _CUSTOM_OVERRIDE_DICT.write_text(
        "帕兰蒂尔\t柏蘭蒂爾\n",
        encoding="utf-8",
    )

    try:
        cc = OpenCC.from_dicts(
            config="s2t",
            overrides={
                DictSlot.STPhrases: _CUSTOM_OVERRIDE_DICT,
            },
        )

        st_phrases, _max_len = cc.dictionary.st_phrases
        assert st_phrases == {"帕兰蒂尔": "柏蘭蒂爾"}
        assert cc.convert("帕兰蒂尔") == "柏蘭蒂爾"
        assert cc.convert("汉字") == "漢字"
    finally:
        _cleanup_custom_dict()


def test_dictslot_punctuation_append_is_used_by_punctuation_conversion():
    _CUSTOM_PUNCT_DICT.write_text(
        "！\t‼\n",
        encoding="utf-8",
    )

    try:
        cc = OpenCC.from_dicts(
            config="s2t",
            appends={
                DictSlot.STPunctuations: _CUSTOM_PUNCT_DICT,
            },
        )

        assert cc.convert("“汉字”！", punctuation=True) == "「漢字」‼"
    finally:
        _cleanup_custom_dict()


def test_dictslot_punctuation_enum_members_exist():
    assert DictSlot.STPunctuations.value == "st_punctuations"
    assert DictSlot.TSPunctuations.value == "ts_punctuations"


def test_dictslot_forward_variant_phrase_enum_members_exist():
    assert DictSlot.TWVariantsPhrases.value == "tw_variants_phrases"
    assert DictSlot.HKVariantsPhrases.value == "hk_variants_phrases"


def test_dictslot_japanese_shinjitai_enum_members_exist():
    assert DictSlot.JPSCharacters.value == "jps_characters"
    assert DictSlot.JPSCharactersRev.value == "jps_characters_rev"
    assert DictSlot.JPSPhrases.value == "jps_phrases"


def test_dictionary_from_dicts_loads_forward_variant_phrase_slots():
    dictionary = DictionaryMaxlength.from_dicts()

    tw_variants_phrases, tw_max_len = dictionary.tw_variants_phrases
    hk_variants_phrases, hk_max_len = dictionary.hk_variants_phrases

    assert tw_variants_phrases["喫茶小舖"] == "喫茶小舖"
    assert hk_variants_phrases["喫茶小舖"] == "喫茶小舖"
    assert tw_max_len >= len("喫茶小舖")
    assert hk_max_len >= len("喫茶小舖")


def test_dictionary_json_preserves_forward_variant_phrase_slots():
    dictionary = DictionaryMaxlength.from_dicts().with_custom_dicts(
        appends={
            DictSlot.TWVariantsPhrases: {
                "喫茶測試": "喫茶測試",
            },
            DictSlot.HKVariantsPhrases: {
                "喫茶測試": "喫茶測試",
            },
        },
    )

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        path = Path(handle.name)

    try:
        dictionary.serialize_to_json(str(path))
        loaded = DictionaryMaxlength.from_json(path)
    finally:
        path.unlink()

    assert loaded.tw_variants_phrases[0]["喫茶測試"] == "喫茶測試"
    assert loaded.hk_variants_phrases[0]["喫茶測試"] == "喫茶測試"


def test_dictionary_with_custom_dicts_appends_forward_variant_phrase_slots():
    dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
        appends={
            DictSlot.TWVariantsPhrases: {
                "喫茶測試": "喫茶測試",
            },
            DictSlot.HKVariantsPhrases: {
                "喫茶測試": "喫茶測試",
            },
        },
    )

    assert OpenCC(config="t2tw", dictionary=dictionary).convert("喫茶測試") == "喫茶測試"
    assert OpenCC(config="t2hk", dictionary=dictionary).convert("喫茶測試") == "喫茶測試"


def test_dictionary_with_custom_dicts_overrides_forward_variant_phrase_slots():
    dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
        overrides={
            DictSlot.TWVariantsPhrases: {
                "喫茶小舖": "喫茶小舖",
            },
            DictSlot.HKVariantsPhrases: {
                "喫茶小舖": "喫茶小舖",
            },
        },
    )

    assert dictionary.tw_variants_phrases == ({"喫茶小舖": "喫茶小舖"}, len("喫茶小舖"))
    assert dictionary.hk_variants_phrases == ({"喫茶小舖": "喫茶小舖"}, len("喫茶小舖"))


def test_forward_tw_variant_phrase_slot_precedes_character_slot():
    assert OpenCC("t2tw").convert("喫茶小舖") == "喫茶小舖"
    assert OpenCC("s2tw").convert("喫茶小舖") == "喫茶小舖"


def test_forward_hk_variant_phrase_slot_precedes_character_slot():
    assert OpenCC("t2hk").convert("喫茶小舖") == "喫茶小舖"
    assert OpenCC("s2hk").convert("喫茶小舖") == "喫茶小舖"


def test_reverse_tw_hk_variant_behavior_remains_unchanged():
    assert OpenCC("tw2t").convert("吃口飯") == "吃口飯"
    assert OpenCC("hk2t").convert("吃口飯") == "吃口飯"


def test_legacy_custom_dict_dir_without_punctuation_files_loads():
    legacy_dir = _TEST_DIR / "_legacy_dicts_without_punctuation"
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir)
    legacy_dir.mkdir()

    old_required_files = [
        "STCharacters.txt",
        "STPhrases.txt",
        "TSCharacters.txt",
        "TSPhrases.txt",
        "TWPhrases.txt",
        "TWPhrasesRev.txt",
        "TWVariantsPhrases.txt",
        "TWVariants.txt",
        "TWVariantsRev.txt",
        "TWVariantsRevPhrases.txt",
        "HKVariantsPhrases.txt",
        "HKVariants.txt",
        "HKVariantsRev.txt",
        "HKVariantsRevPhrases.txt",
        "JPShinjitaiCharacters.txt",
        "JPShinjitaiCharactersRev.txt",
        "JPShinjitaiPhrases.txt",
    ]

    try:
        for filename in old_required_files:
            (legacy_dir / filename).write_text("", encoding="utf-8")
        (legacy_dir / "STCharacters.txt").write_text("汉\t漢\n", encoding="utf-8")
        (legacy_dir / "TSCharacters.txt").write_text("漢\t汉\n", encoding="utf-8")

        dictionary = DictionaryMaxlength.from_dicts(base_dir=legacy_dir)

        assert dictionary.st_characters == ({"汉": "漢"}, 1)
        assert dictionary.st_punctuations == ({}, 0)
        assert dictionary.ts_punctuations == ({}, 0)
    finally:
        shutil.rmtree(legacy_dir)

def test_dictionary_with_custom_dicts_appends_exact_pairs_with_spaces():
    dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
        appends={
            DictSlot.STPhrases: {
                " 著": " 著",
            },
        },
    )

    cc = OpenCC(config="s2t", dictionary=dictionary)

    assert cc.convert("馬斯克 著") == "馬斯克 著"

def test_dictionary_with_custom_dicts_append_late_comer_wins_existing_key():
    dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
        appends={
            DictSlot.STPhrases: {
                "汉字": "測字",
            },
        },
    )

    cc = OpenCC(config="s2t", dictionary=dictionary)

    assert cc.convert("汉字转换") == "測字轉換"

def test_dictionary_with_custom_dicts_override_replaces_entire_slot():
    dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
        overrides={
            DictSlot.STPhrases: {
                "帕兰蒂尔": "柏蘭蒂爾",
            },
        },
    )

    st_phrases, max_len = dictionary.st_phrases

    assert st_phrases == {"帕兰蒂尔": "柏蘭蒂爾"}
    assert max_len == len("帕兰蒂尔")

    cc = OpenCC(config="s2t", dictionary=dictionary)

    assert cc.convert("帕兰蒂尔") == "柏蘭蒂爾"
    assert cc.convert("汉字") == "漢字"

def test_dictionary_with_custom_dict_files_appends_custom_st_phrase():
    custom_dict = _write_custom_dict()

    try:
        dictionary = DictionaryMaxlength.from_json().with_custom_dict_files(
            appends={
                DictSlot.STPhrases: custom_dict,
            },
        )

        cc = OpenCC(config="s2t", dictionary=dictionary)

        assert cc.convert("帕兰蒂尔是一家公司") == "柏蘭蒂爾是一家公司"
    finally:
        _cleanup_custom_dict()

def test_dictionary_with_custom_dict_files_override_replaces_entire_slot():
    _CUSTOM_OVERRIDE_DICT.write_text(
        "帕兰蒂尔\t柏蘭蒂爾\n",
        encoding="utf-8",
    )

    try:
        dictionary = DictionaryMaxlength.from_json().with_custom_dict_files(
            overrides={
                DictSlot.STPhrases: _CUSTOM_OVERRIDE_DICT,
            },
        )

        st_phrases, max_len = dictionary.st_phrases

        assert st_phrases == {"帕兰蒂尔": "柏蘭蒂爾"}
        assert max_len == len("帕兰蒂尔")

        cc = OpenCC(config="s2tw", dictionary=dictionary)

        assert cc.convert("帕兰蒂尔") == "柏蘭蒂爾"
        assert cc.convert("汉字") == "漢字"
    finally:
        _cleanup_custom_dict()

def test_dictionary_with_custom_dict_files_uses_opencc_file_contract_not_exact_space_key():
    _CUSTOM_DICT.write_text(
        " 著\t著\n",
        encoding="utf-8",
    )

    try:
        dictionary = DictionaryMaxlength.from_json().with_custom_dict_files(
            appends={
                DictSlot.STPhrases: _CUSTOM_DICT,
            },
        )

        st_phrases, _max_len = dictionary.st_phrases

        assert " 著" not in st_phrases
        assert st_phrases["著"] == "著"
    finally:
        _cleanup_custom_dict()

def test_dictionary_with_custom_dicts_supports_string_slot_names():
    dictionary = DictionaryMaxlength.from_json().with_custom_dicts(
        appends={
            "st_phrases": {
                "帕兰蒂尔": "柏蘭蒂爾",
            },
        },
    )

    cc = OpenCC(config="s2t", dictionary=dictionary)

    assert cc.convert("帕兰蒂尔") == "柏蘭蒂爾"

def test_dictionary_with_custom_dict_files_supports_string_slot_names():
    custom_dict = _write_custom_dict()

    try:
        dictionary = DictionaryMaxlength.from_json().with_custom_dict_files(
            appends={
                "st_phrases": custom_dict,
            },
        )

        cc = OpenCC(config="s2t", dictionary=dictionary)

        assert cc.convert("帕兰蒂尔") == "柏蘭蒂爾"
    finally:
        _cleanup_custom_dict()

def test_shared_provider_rejects_post_load_custom_dicts():
    provider = DictionaryMaxlength.get_provider()

    try:
        provider.with_custom_dicts(
            appends={
                DictSlot.STPhrases: {
                    "帕兰蒂尔": "柏蘭蒂爾",
                },
            },
        )
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert "shared DictionaryMaxlength provider" in str(exc)

def test_shared_provider_rejects_post_load_custom_dict_files():
    custom_dict = _write_custom_dict()
    provider = DictionaryMaxlength.get_provider()

    try:
        try:
            provider.with_custom_dict_files(
                appends={
                    DictSlot.STPhrases: custom_dict,
                },
            )
            raise AssertionError("Expected RuntimeError")
        except RuntimeError as exc:
            assert "shared DictionaryMaxlength provider" in str(exc)
    finally:
        _cleanup_custom_dict()
