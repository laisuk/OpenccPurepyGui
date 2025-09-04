"""
Command-line interface for OpenCC-based Office and EPUB document conversion.

This script acts as the entry point for converting Office documents (.docx, .xlsx, .pptx, .odt, .ods, .odp)
and EPUB files using the OpenCC conversion system, with support for punctuation conversion and font preservation.

It uses `opencc_purepy.OpenCC` and `convert_office_doc()` from `office_helper.py`.

Author:
    Laisuk Lai (https://github.com/laisuk)
"""

import os
import sys

from opencc_purepy import OpenCC
from opencc_purepy.office_helper import convert_office_doc, OFFICE_FORMATS


def main(args):
    """
    Main function for handling command-line arguments and executing document conversion.

    Args:
        args: An argparse-like namespace containing:
            - input (str): Path to input Office/EPUB file.
            - output (str): Path to output file (optional; will be inferred if not provided).
            - format (str): File format (e.g., docx, xlsx, etc.). If omitted, guessed from file extension.
            - config (str): OpenCC config key (e.g., 's2t', 't2s').
            - punct (bool): Whether to convert punctuation.
            - keep_font (bool): Whether to preserve font-family attributes during conversion.
            - auto_ext (bool): Whether to append correct extension to output if missing.

    Returns:
        int: Exit code (0 on success, 1 on failure).
    """
    if args.config is None:
        print("ℹ️  Config not specified. Use default 's2t'", file=sys.stderr)
        args.config = 's2t'

    input_file = args.input
    output_file = args.output
    office_format = args.format
    auto_ext = getattr(args, "auto_ext", False)
    config = args.config
    punct = args.punct
    keep_font = getattr(args, "keep_font", False)

    # Check for missing input/output files
    if not input_file and not output_file:
        print("❌  Input and output files are missing.", file=sys.stderr)
        return 1
    if not input_file:
        print("❌  Input file is missing.", file=sys.stderr)
        return 1

    # If output file is not specified, generate one based on input file
    if not output_file:
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        input_dir = os.path.dirname(input_file) or os.getcwd()
        ext = f".{office_format}" if auto_ext and office_format and office_format in OFFICE_FORMATS else \
            os.path.splitext(input_file)[1]
        output_file = os.path.join(input_dir, f"{input_name}_converted{ext}")
        print(f"ℹ️  Output file not specified. Using: {output_file}", file=sys.stderr)

    # Determine office format from file extension if not provided
    if not office_format:
        file_ext = os.path.splitext(input_file)[1].lower()
        if file_ext[1:] not in OFFICE_FORMATS:
            print(f"❌  Invalid Office file extension: {file_ext}", file=sys.stderr)
            print("   Valid extensions: .docx | .xlsx | .pptx | .odt | .ods | .odp | .epub", file=sys.stderr)
            return 1
        office_format = file_ext[1:]

    # Auto-append extension to output file if needed
    if auto_ext and output_file and not os.path.splitext(output_file)[1] and office_format in OFFICE_FORMATS:
        output_file += f".{office_format}"
        print(f"ℹ️  Auto-extension applied: {output_file}", file=sys.stderr)

    try:
        # Perform Office document conversion
        success, message = convert_office_doc(
            input_file,
            output_file,
            office_format,
            OpenCC(config),
            punct,
            keep_font,
        )
        if success:
            print(f"{message}\n📁  Output saved to: {os.path.abspath(output_file)}", file=sys.stderr)
            return 0
        else:
            print(f"❌  Conversion failed: {message}", file=sys.stderr)
            return 1
    except Exception as ex:
        print(f"❌  Error during Office document conversion: {str(ex)}", file=sys.stderr)
        return 1
