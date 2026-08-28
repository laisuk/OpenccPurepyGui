from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
from typing import TYPE_CHECKING

from .dict_slot import DictSlot
from .dictionary_lib import PathLike, SlotPathMap

if TYPE_CHECKING:
    from typing import Literal

    CustomDictMode = Literal["append", "override"]
else:
    CustomDictMode = str


@dataclass(frozen=True)
class CustomDictSpec:
    """
    One custom dictionary file request.

    This is the user-facing, ungrouped form used by ``OpenCC.from_dict_files()``
    and by the CLI ``slot:mode:path`` syntax. It describes exactly one file:

    - ``slot``: the OpenCC dictionary slot to customize.
    - ``mode``: ``"append"`` to extend that slot, or ``"override"`` to replace
      that slot.
    - ``path``: the text dictionary file to load.

    Do not confuse this with the lower-level ``overrides`` / ``appends``
    ``SlotPathMap`` values accepted by dictionary loading APIs. Those maps are
    the grouped form: ``{slot: path}`` split by mode.
    """

    slot: DictSlot
    mode: CustomDictMode
    path: PathLike


def parse_custom_dict_spec(spec: str) -> CustomDictSpec:
    """
    Parse one ``--custom-dict`` value into a ``CustomDictSpec``.

    ``spec`` must use ``slot:mode:path`` syntax, for example
    ``"STPhrases:append:./UserDict.txt"``.

    This function returns the single-spec record. It does not check whether the
    file exists, and it does not group values into ``overrides`` and ``appends``
    maps. Use ``parse_custom_dict_specs()`` when parsing CLI values that should
    be passed directly to dictionary loading APIs.
    """
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Invalid --custom-dict spec {!r}. Expected slot:mode:path".format(spec)
        )

    slot, mode, path = (p.strip() for p in parts)

    if not slot:
        raise ValueError(
            "Invalid --custom-dict spec {!r}: slot is empty".format(spec)
        )

    mode = mode.lower()
    if mode not in ("append", "override"):
        raise ValueError(
            "Invalid --custom-dict mode {!r}. Expected append or override".format(mode)
        )

    if not path:
        raise ValueError(
            "Invalid --custom-dict spec {!r}: path is empty".format(spec)
        )

    if not Path(path).is_file():
        raise ValueError(
            "Custom dictionary file not found: {}".format(path)
        )

    return CustomDictSpec(
        slot=DictSlot.parse(slot),
        mode=mode,
        path=path,
    )


def parse_custom_dict_specs(
        specs: Optional[Iterable[str]],
) -> Tuple[SlotPathMap, SlotPathMap]:
    """
    Parse CLI ``--custom-dict`` strings into ``(overrides, appends)`` maps.

    This is the CLI-oriented helper. It accepts raw ``slot:mode:path`` strings,
    validates that each referenced file exists, and returns the grouped
    ``SlotPathMap`` form expected by ``DictionaryMaxlength`` customization APIs.

    Return value order is ``(overrides, appends)``.

    Because mappings can hold only one path per slot and mode, later records
    replace earlier records with the same slot. Use ``OpenCC.from_dict_files()``
    when repeated files or mixed modes must be applied sequentially.

    For API code that already has ``CustomDictSpec`` instances, use
    ``custom_dict_specs_to_maps()`` instead.
    """
    overrides: SlotPathMap = {}
    appends: SlotPathMap = {}

    for spec in specs or ():
        parsed = parse_custom_dict_spec(spec)

        if not Path(parsed.path).is_file():
            raise ValueError(
                "Custom dictionary file not found: {}".format(parsed.path)
            )

        if parsed.mode == "override":
            overrides[parsed.slot] = parsed.path
        else:
            appends[parsed.slot] = parsed.path

    return overrides, appends


def custom_dict_specs_to_maps(
        specs: Optional[Iterable[CustomDictSpec]],
) -> Tuple[SlotPathMap, SlotPathMap]:
    """
    Group ``CustomDictSpec`` records into ``(overrides, appends)`` maps.

    This is the API-oriented bridge from the user-facing spec form to the
    lower-level dictionary loader form. Unlike ``parse_custom_dict_specs()``,
    this function expects already-parsed ``CustomDictSpec`` objects and does not
    check file existence.

    Return value order is ``(overrides, appends)``.

    Because mappings can hold only one path per slot and mode, later records
    replace earlier records with the same slot. Use ``OpenCC.from_dict_files()``
    when repeated files or mixed modes must be applied sequentially.
    """
    overrides: SlotPathMap = {}
    appends: SlotPathMap = {}

    for spec in specs or ():
        if spec.mode == "override":
            overrides[spec.slot] = spec.path
        elif spec.mode == "append":
            appends[spec.slot] = spec.path
        else:
            raise ValueError("Invalid custom dictionary mode: {}".format(spec.mode))

    return overrides, appends


def ensure_distinct_paths(
        input_path: Optional[PathLike],
        output_path: Optional[PathLike],
) -> None:
    """Ensure input and output do not refer to the same normalized path."""
    import os
    if input_path is None or output_path is None:
        return

    input_norm = os.path.normcase(
        os.path.normpath(os.path.abspath(input_path))
    )
    output_norm = os.path.normcase(
        os.path.normpath(os.path.abspath(output_path))
    )

    if input_norm == output_norm:
        raise ValueError("Input and output files must be different.")
