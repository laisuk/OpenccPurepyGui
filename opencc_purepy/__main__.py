from __future__ import print_function

import argparse
import sys
from . import convert_cmd, dictgen_cmd, office_cmd


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
        prog='opencc_purepy',
        description='Pure Python OpenCC CLI with multiple tools',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # ---- convert subcommand ----
    parser_convert = subparsers.add_parser('convert', help='Convert text files using pure Python OpenCC')
    parser_convert.add_argument('-i', '--input', metavar='<file>', help='Input file')
    parser_convert.add_argument('-o', '--output', metavar='<file>', help='Output file')
    parser_convert.add_argument('-c', '--config', metavar='<conversion>', help='Conversion configuration')
    parser_convert.add_argument('-p', '--punct', action='store_true', default=False,
                                help='Punctuation conversion: Enable/Disable')
    parser_convert.add_argument('--in-enc', metavar='<encoding>', default='UTF-8', help='Input encoding')
    parser_convert.add_argument('--out-enc', metavar='<encoding>', default='UTF-8', help='Output encoding')

    parser_convert.set_defaults(func=convert_cmd.main)

    # ---- Office SUBCOMMAND ----
    parser_office = subparsers.add_parser('office', help='Convert Office files using pure Python OpenCC')
    parser_office.add_argument('-i', '--input', metavar='<file>', help='Office document Input file')
    parser_office.add_argument('-o', '--output', metavar='<file>', help='Office document Output file')
    parser_office.add_argument('-c', '--config', metavar='<conversion>', help='Conversion configuration')
    parser_office.add_argument('-p', '--punct', action='store_true', default=False,
                               help='Enable punctuation conversion')
    parser_office.add_argument('--format', metavar='<format>',
                               help='Target Office format (e.g., docx, xlsx, pptx, odt, epub)')
    parser_office.add_argument('--auto-ext', action='store_true', default=False,
                               help='Auto-append extension to output file')
    parser_office.add_argument('--keep-font', action='store_true', default=True,
                               help='Preserve font-family information in Office content (Default: True)')
    parser_office.add_argument('--no-keep-font', action='store_false', dest='keep_font',
                               help='Do not preserve font-family information in Office content (Overrides --keep-font)')

    parser_office.set_defaults(func=office_cmd.main)

    # ---- dictgen subcommand ----
    parser_dictgen = subparsers.add_parser('dictgen', help='Generate dictionary for pure Python OpenCC')
    parser_dictgen.add_argument(
        "-f", "--format",
        choices=["json"],
        default="json",
        help="Dictionary format: [json]"
    )
    parser_dictgen.add_argument(
        "-o", "--output",
        metavar="<filename>",
        help="Write generated dictionary to <filename>. If not specified, a default filename is used."
    )
    parser_dictgen.add_argument('-c', '--compact', action='store_true', default=False,
                               help='Enable non-indented JSON compact output')
    parser_dictgen.set_defaults(func=dictgen_cmd.main)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    # Entry point for the CLI tool
    sys.exit(main())
