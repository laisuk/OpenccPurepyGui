import os
from .dictionary_lib import DictionaryMaxlength

BLUE = "\033[1;34m"
RESET = "\033[0m"

def main(args):
    """
    Main entry point for the dictionary generation command-line tool.

    Generates and serializes dictionary data in the specified format.

    Args:
        args: Parsed command-line arguments with attributes:
            - format (str): Output format, currently supports 'json'.
            - output (str|None): Output file path or None for default.
            - compact (bool): If True, write compact JSON (no indentation).

    Returns:
        int: Exit code (0 for success).
    """
    # Default output file per format
    default_output = {
        "json": "dictionary_maxlength.json"
    }[args.format]

    output_file = args.output or default_output
    output_file_path = os.path.abspath(output_file)

    # Generate dictionary data
    dictionaries = DictionaryMaxlength.from_dicts()

    if args.format == "json":
        # pretty = not compact
        dictionaries.serialize_to_json(output_file_path, pretty=not args.compact)
        print(f"{BLUE}Dictionary saved in JSON format at: {output_file_path}{RESET}")

    return 0
