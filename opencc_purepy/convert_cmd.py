import io
import sys

from opencc_purepy import OpenCC


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
            - in_enc (str): Input encoding (plain text only).
            - out_enc (str): Output encoding (plain text only).

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    if args.config is None:
        print("ℹ️  Config not specified. Use default 's2t'", file=sys.stderr)
        args.config = 's2t'

    # Plain text conversion fallback
    opencc = OpenCC(args.config)

    # Prompt user if input is from terminal
    if args.input is None and sys.stdin.isatty():
        print("Input text to convert, <Ctrl+Z>/<Ctrl+D> to submit:", file=sys.stderr)

    # Read input text (from file or stdin)
    with io.open(args.input if args.input else 0, encoding=args.in_enc) as f:
        input_str = f.read()

    # Perform conversion
    output_str = opencc.convert(input_str, args.punct)

    # Write output text (to file or stdout)
    with io.open(args.output if args.output else 1, 'w', encoding=args.out_enc) as f:
        f.write(output_str)

    in_from = args.input if args.input else "<stdin>"
    out_to = args.output if args.output else "stdout"
    if sys.stderr.isatty():
        print(f"Conversion completed ({args.config}): {in_from} -> {out_to}", file=sys.stderr)

    return 0
