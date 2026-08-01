import os
import sys

from .dictionary_lib import DictionaryMaxlength
from .utils import parse_custom_dict_spec

BLUE = "\033[1;34m"
RESET = "\033[0m"


def main(args):
    """
    Main entry point for the dictionary generation command-line tool.

    Generates and serializes dictionary data in the specified format.

    Args:
        args: Parsed command-line arguments with attributes:
            - format (str): Output format, currently supports 'json'.
            - output (str): Output file path or None for default.
            - dicts (str): Optional custom dictionary directory.
            - compact (bool): Whether to write compact JSON.
            - no_sort (bool): Whether to preserve dictionary insertion order.
            - custom_dict (list[str] | None): Ordered custom dictionary specs.

    Returns:
        int: Exit code (0 for success).
    """

    # ------------------------------------------------------------------
    # Validate custom dictionary directory
    # ------------------------------------------------------------------

    try:
        if args.dicts is not None:
            DictionaryMaxlength.validate_dicts_dir(args.dicts)

    except (OSError, ValueError) as e:
        print("Error: {}".format(e), file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Set default output file name
    # ------------------------------------------------------------------

    default_output = {
        "json": "dictionary_maxlength.json"
    }[args.format]

    output_file = args.output or default_output
    output_file_path = os.path.abspath(output_file)

    # ------------------------------------------------------------------
    # Generate dictionary data
    # ------------------------------------------------------------------

    try:
        dictionaries = DictionaryMaxlength.from_dicts(base_dir=args.dicts)
        specs = [parse_custom_dict_spec(s) for s in (args.custom_dict or [])]
        for spec in specs:
            if spec.mode == "override":
                dictionaries.with_custom_dict_files(overrides={spec.slot: spec.path})
            else:
                dictionaries.with_custom_dict_files(appends={spec.slot: spec.path})

        if args.format == "json":
            dictionaries.serialize_to_json(
                output_file_path,
                pretty=not args.compact,
                sort_keys=not args.no_sort,
            )
    except (OSError, UnicodeError, ValueError) as ex:
        print("❌  Dictionary generation failed: {}".format(ex), file=sys.stderr)
        return 1

    print(
        f"{BLUE}Dictionary saved in JSON format at: "
        f"{output_file_path}{RESET}"
    )

    return 0
