from __future__ import print_function

import argparse
import sys

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    version = None
    PackageNotFoundError = Exception


def _get_version():
    """
    Return installed package version, or a fallback string if unavailable.
    """
    if version is None:
        return "unknown"

    try:
        return version("opencc-purepy")
    except PackageNotFoundError:
        return "unknown"


def _config_arg(value):
    from .core import OpenccConfig

    try:
        return OpenccConfig.parse(value).to_canonical_name()
    except ValueError:
        raise argparse.ArgumentTypeError("invalid conversion config: {}".format(value))


def _format_arg(value):
    from .office_helper import OFFICE_FORMATS

    normalized = value.strip().lower()
    if normalized in OFFICE_FORMATS:
        return normalized
    raise argparse.ArgumentTypeError("invalid office format: {}".format(value))


def _run_convert(args):
    from . import convert_cmd
    return convert_cmd.main(args)


def _run_office(args):
    from . import office_cmd
    return office_cmd.main(args)


def _run_dictgen(args):
    from . import dictgen_cmd
    return dictgen_cmd.main(args)


def main():
    """
    Main entry point for the opencc_purepy command-line interface.

    Sets up argument parsing for subcommands:
      - convert: Convert text using OpenCC.
      - office: Office documents and Epub converter.
      - dictgen: Generate dictionary files for OpenCC.

    Parses command-line arguments and dispatches to the appropriate subcommand handler.

    Returns:
        int: Exit code from the invoked subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="opencc_purepy",
        description="Pure Python OpenCC CLI with multiple tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {}".format(_get_version()),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- convert subcommand ----
    parser_convert = subparsers.add_parser(
        "convert",
        help="Convert text files using pure Python OpenCC",
    )
    parser_convert.add_argument("-i", "--input", metavar="<file>", help="Input file")
    parser_convert.add_argument("-o", "--output", metavar="<file>", help="Output file")
    parser_convert.add_argument(
        "-c",
        "--config",
        metavar="<conversion>",
        type=_config_arg,
        help="Conversion configuration",
    )
    parser_convert.add_argument(
        "-p",
        "--punct",
        action="store_true",
        default=False,
        help="Punctuation conversion: Enable/Disable",
    )
    parser_convert.add_argument("--in-enc", metavar="<encoding>", default="UTF-8", help="Input encoding")
    parser_convert.add_argument("--out-enc", metavar="<encoding>", default="UTF-8", help="Output encoding")
    parser_convert.set_defaults(func=_run_convert)

    # ---- office subcommand ----
    parser_office = subparsers.add_parser(
        "office",
        help="Convert Office files using pure Python OpenCC",
    )
    parser_office.add_argument("-i", "--input", metavar="<file>", help="Office document Input file")
    parser_office.add_argument("-o", "--output", metavar="<file>", help="Office document Output file")
    parser_office.add_argument(
        "-c",
        "--config",
        metavar="<conversion>",
        type=_config_arg,
        help="Conversion configuration",
    )
    parser_office.add_argument(
        "-p",
        "--punct",
        action="store_true",
        default=False,
        help="Enable punctuation conversion",
    )
    parser_office.add_argument(
        "--format",
        metavar="<format>",
        type=_format_arg,
        help="Target Office format (e.g., docx, xlsx, pptx, odt, epub)",
    )
    parser_office.add_argument(
        "--auto-ext",
        action="store_true",
        default=False,
        help="Auto-append extension to output file",
    )
    parser_office.add_argument(
        "--keep-font",
        action="store_true",
        default=True,
        help="Preserve font-family information in Office content (Default: True)",
    )
    parser_office.add_argument(
        "--no-keep-font",
        action="store_false",
        dest="keep_font",
        help="Do not preserve font-family information in Office content (Overrides --keep-font)",
    )
    parser_office.set_defaults(func=_run_office)

    # ---- dictgen subcommand ----
    parser_dictgen = subparsers.add_parser(
        "dictgen",
        help="Generate dictionary for pure Python OpenCC",
    )
    parser_dictgen.add_argument(
        "-f",
        "--format",
        choices=["json"],
        default="json",
        help="Dictionary format: [json]",
    )
    parser_dictgen.add_argument(
        "-o",
        "--output",
        metavar="<filename>",
        help="Write generated dictionary to <filename>. If not specified, a default filename is used.",
    )
    parser_dictgen.add_argument(
        '-c',
        '--compact',
        action='store_true', default=False,
        help='Enable non-indented JSON compact output')
    parser_dictgen.add_argument(
        "-d",
        "--dicts",
        metavar="<directory>",
        help=(
            "Load dictionaries from a custom directory instead of "
            "the built-in dicts folder."
        ),
    )

    parser_dictgen.set_defaults(func=_run_dictgen)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
