import io
import os
import sys

from opencc_purepy import OpenCC
from opencc_purepy.utils import parse_custom_dict_spec


def main(args):
    """
    Main entry point for the OpenCC command-line conversion tool.

    Handles plain text conversion based on the provided arguments.

    Args:
        args: Parsed command-line arguments with attributes:
            - input (str): Input file path or None for stdin.
            - output (str): Output file path or None for stdout.
            - config (str): OpenCC conversion configuration.
            - punct (bool): Whether to convert punctuation.
            - detofu (str | None): Optional DeTofu compatibility level
              ("all", "ext-b", "ext-c", "ext-d", "ext-e",
              "ext-f", "ext-g", "ext-h", or "ext-i").
            - detofu_file (str | None): Optional UTF-8 custom DeTofu
              fallback mapping file. Requires --detofu.
            - in_enc (str): Input encoding (plain text only).
            - out_enc (str): Output encoding (plain text only).
            - custom_dict (list[str] | None): Ordered custom dictionary specs.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    if args.config is None:
        print("ℹ️  Config not specified. Use default 's2t'", file=sys.stderr)
        args.config = 's2t'

    if args.input and not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.detofu_file and not args.detofu:
        print("❌  --detofu-file requires --detofu", file=sys.stderr)
        return 1

    try:
        specs = [parse_custom_dict_spec(s) for s in (args.custom_dict or [])]
        opencc = OpenCC.from_dict_files(args.config, specs) if specs else OpenCC(args.config)
    except (OSError, UnicodeError, ValueError) as ex:
        print(f"❌  Invalid --custom-dict: {ex}", file=sys.stderr)
        return 1

    # Prompt user if input is from terminal
    if args.input is None and sys.stdin.isatty():
        print("Input text to convert, <Ctrl+Z>/<Ctrl+D> to submit:", file=sys.stderr)

    try:
        with io.open(args.input if args.input else 0, encoding=args.in_enc) as f:
            input_str = f.read()

        output_str = opencc.convert(input_str, args.punct)

        if args.detofu:
            if args.detofu_file:
                output_str = opencc.detofu_with_custom_file(
                    output_str,
                    args.detofu,
                    args.detofu_file,
                )
            else:
                output_str = opencc.detofu(output_str, args.detofu)

        with io.open(args.output if args.output else 1, 'w', encoding=args.out_enc) as f:
            f.write(output_str)
    except (OSError, UnicodeError, LookupError, ValueError) as ex:
        print("❌  Conversion failed: {}".format(ex), file=sys.stderr)
        return 1

    in_from = args.input if args.input else "<stdin>"
    out_to = args.output if args.output else "stdout"
    if sys.stderr.isatty():
        if not args.output and output_str and not output_str.endswith("\n"):
            print()
        # print(f"Conversion completed ({args.config}): {in_from} -> {out_to}", file=sys.stderr)
        status = f"Conversion completed ({args.config}"
        if args.detofu:
            status += f", detofu:{args.detofu}"
        status += f"): {in_from} -> {out_to}"
        print(status, file=sys.stderr)

    return 0
