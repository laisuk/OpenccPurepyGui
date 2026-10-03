"""
Command-line handler for OpenCC-based Office and EPUB conversion.

The command is responsible for CLI validation, OpenCC construction, custom
dictionary loading, document format resolution, output-path generation, and
status/error reporting.

Document container processing is delegated to ``convert_office_doc()``.
Text transformation is delegated to the shared ``make_text_converter()``
pipeline:

    Normalize -> Convert -> DeTofu

This keeps Office/EPUB XML and ZIP handling independent of OpenCC-specific
conversion logic.
"""

import os
import sys
from pathlib import Path

from opencc_purepy import OpenCC
from opencc_purepy.office_helper import OFFICE_FORMATS, convert_office_doc
from opencc_purepy.utils import (
    ensure_distinct_paths,
    make_text_converter,
    parse_custom_dict_spec,
)


def main(args):
    """
    Run Office or EPUB document conversion.

    The command resolves the document format, validates paths and DeTofu
    arguments, constructs the configured ``OpenCC`` instance, loads custom
    dictionaries, builds the shared text converter, and passes that callable
    to ``convert_office_doc()``.

    The Office helper itself remains format-focused and receives only a
    ``str -> str`` transformation callback.

    Args:
        args: Parsed command-line arguments with attributes including:

            - ``input``: Input Office/EPUB file path.
            - ``output``: Optional output file path.
            - ``format``: Optional explicit input document format override.
            - ``config``: OpenCC conversion configuration.
            - ``punct``: Whether punctuation conversion is enabled.
            - ``norm_compat``: Whether to normalize CJK Compatibility
              Ideographs before conversion.
            - ``norm_compat_extended``: Whether to apply extended
              compatibility normalization before conversion.
            - ``detofu``: Optional DeTofu level.
            - ``detofu_file``: Optional UTF-8 custom DeTofu mapping file.
              Requires ``detofu``.
            - ``keep_font``: Whether font-family information should be
              preserved where supported.
            - ``custom_dict``: Optional repeated custom OpenCC dictionary
              specs in ``slot:mode:path`` form.

    Returns:
        int: ``0`` on success, otherwise ``1``.
    """
    if args.config is None:
        print("ℹ️  Config not specified. Use default 's2t'", file=sys.stderr)
        args.config = "s2t"

    input_file = args.input
    output_file = args.output
    office_format = args.format.lower() if args.format else None
    config = args.config
    keep_font = getattr(args, "keep_font", False)

    if not input_file and not output_file:
        print("❌  Input and output files are missing.", file=sys.stderr)
        return 1

    if not input_file:
        print("❌  Input file is missing.", file=sys.stderr)
        return 1

    if not Path(input_file).is_file():
        print(f"❌ Input file not found: {input_file}", file=sys.stderr)
        return 1

    if args.detofu_file is not None and args.detofu is None:
        print("❌  --detofu-file requires --detofu", file=sys.stderr)
        return 1

    try:
        ensure_distinct_paths(input_file, output_file)
    except ValueError as ex:
        print(f"❌  {ex}", file=sys.stderr)
        return 1

    if office_format:
        if office_format not in OFFICE_FORMATS:
            print(
                f"❌  Unsupported Office format: {args.format}",
                file=sys.stderr,
            )
            return 1
    else:
        file_ext = os.path.splitext(input_file)[1].lower().lstrip(".")
        if file_ext not in OFFICE_FORMATS:
            print(
                f"❌  Invalid Office file extension: .{file_ext or '(none)'}",
                file=sys.stderr,
            )
            print(
                "   Valid extensions: "
                ".docx | .xlsx | .pptx | .odt | .ods | .odp | .epub",
                file=sys.stderr,
            )
            return 1
        office_format = file_ext

    try:
        specs = [parse_custom_dict_spec(s) for s in (args.custom_dict or [])]
        opencc = (
            OpenCC.from_dict_files(config, specs)
            if specs
            else OpenCC(config)
        )
    except (OSError, UnicodeError, ValueError) as ex:
        print(f"❌  Invalid --custom-dict: {ex}", file=sys.stderr)
        return 1

    text_converter = make_text_converter(opencc, args)

    if not output_file:
        input_path = Path(input_file)
        input_dir = (
            input_path.parent
            if input_path.parent != Path("")
            else Path.cwd()
        )

        output_stem = input_path.stem
        if args.convert_filename:
            output_stem = text_converter(output_stem)

        output_path = (
                input_dir
                / f"{output_stem}_converted.{office_format}"
        )
        output_file = str(output_path)

        print(
            f"ℹ️  Output file not specified. Using: {output_path}",
            file=sys.stderr,
        )
    elif not os.path.splitext(output_file)[1]:
        output_file += f".{office_format}"
        print(
            f"ℹ️  Auto-extension applied: {output_file}",
            file=sys.stderr,
        )

    try:
        success, message = convert_office_doc(
            input_file,
            output_file,
            office_format,
            text_converter,
            keep_font,
        )

        if success:
            print(
                "{}\n📁  Output saved to: {}".format(
                    message,
                    os.path.abspath(output_file),
                ),
                file=sys.stderr,
            )
            return 0

        print(f"❌  Conversion failed: {message}", file=sys.stderr)
        return 1

    except (OSError, UnicodeError, ValueError) as ex:
        print(
            "❌  Error during Office document conversion: {}".format(ex),
            file=sys.stderr,
        )
        return 1
