import codecs
import io
import os
import sys

from opencc_purepy import OpenCC
from opencc_purepy.utils import (
    ensure_distinct_paths,
    make_text_converter,
    parse_custom_dict_spec,
)


def main(args):
    """
    Run the plain-text OpenCC conversion command.

    This command owns CLI-specific behavior such as argument validation,
    OpenCC construction, custom dictionary loading, input/output handling,
    encoding validation, and status reporting.

    The actual text transformation is delegated to
    ``make_text_converter()``, which applies the shared pipeline:

        Normalize -> Convert -> DeTofu

    Args:
        args: Parsed command-line arguments with attributes including:

            - ``input``: Input file path, or ``None`` to read from stdin.
            - ``output``: Output file path, or ``None`` to write to stdout.
            - ``config``: OpenCC conversion configuration.
            - ``punct``: Whether punctuation conversion is enabled.
            - ``norm_compat``: Whether to normalize CJK Compatibility
              Ideographs before conversion.
            - ``norm_compat_extended``: Whether to apply the extended
              compatibility normalization pipeline before conversion.
            - ``detofu``: Optional DeTofu level.
            - ``detofu_file``: Optional UTF-8 custom DeTofu mapping file.
              Requires ``detofu``.
            - ``in_enc``: Input text encoding.
            - ``out_enc``: Output text encoding.
            - ``custom_dict``: Optional repeated custom dictionary specs in
              ``slot:mode:path`` form.

    Returns:
        int: ``0`` on success, otherwise ``1``.
    """
    if args.config is None:
        print("ℹ️  Config not specified. Use default 's2t'", file=sys.stderr)
        args.config = "s2t"

    if args.input and not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.detofu_file is not None and args.detofu is None:
        print("❌  --detofu-file requires --detofu", file=sys.stderr)
        return 1

    try:
        ensure_distinct_paths(args.input, args.output)
    except ValueError as ex:
        print(f"❌  {ex}", file=sys.stderr)
        return 1

    try:
        specs = [parse_custom_dict_spec(s) for s in (args.custom_dict or [])]
        opencc = (
            OpenCC.from_dict_files(args.config, specs)
            if specs
            else OpenCC(args.config)
        )
    except (OSError, UnicodeError, ValueError) as ex:
        print(f"❌  Invalid --custom-dict: {ex}", file=sys.stderr)
        return 1

    text_converter = make_text_converter(opencc, args)

    if args.input is None and sys.stdin.isatty():
        print(
            "Input text to convert, <Ctrl+Z>/<Ctrl+D> to submit:",
            file=sys.stderr,
        )

    try:
        if args.input:
            with io.open(args.input, "r", encoding=args.in_enc) as f:
                input_str = f.read()
        else:
            input_str = sys.stdin.buffer.read().decode(args.in_enc)

        output_str = text_converter(input_str)

        try:
            codecs.lookup(args.out_enc)
        except LookupError as ex:
            print(
                "❌ Invalid output encoding '{}': {}".format(args.out_enc, ex),
                file=sys.stderr,
            )
            return 1

        try:
            if args.output:
                with io.open(args.output, "w", encoding=args.out_enc) as f:
                    f.write(output_str)
            elif sys.stdout.isatty():
                sys.stdout.write(output_str)
                sys.stdout.flush()
            else:
                encoded = output_str.encode(args.out_enc)
                sys.stdout.buffer.write(encoded)
                sys.stdout.buffer.flush()
        except (OSError, UnicodeError) as ex:
            target = args.output or "<stdout>"
            print(
                "❌ Failed to write output '{}': {}".format(target, ex),
                file=sys.stderr,
            )
            return 1

    except (OSError, UnicodeError, LookupError, ValueError) as ex:
        print("❌  Conversion failed: {}".format(ex), file=sys.stderr)
        return 1

    in_from = args.input if args.input else "<stdin>"
    out_to = args.output if args.output else "stdout"

    if sys.stderr.isatty():
        if not args.output and output_str and not output_str.endswith("\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()

        status = f"Conversion completed ({args.config}"

        if args.norm_compat_extended:
            status += ", norm-compat-extended"
        elif args.norm_compat:
            status += ", norm-compat"

        if args.detofu is not None:
            status += f", detofu:{args.detofu}"

        if specs:
            custom_status = ",".join(
                f"{spec.slot.name}:{spec.mode}"
                for spec in specs
            )
            status += f", custom:{custom_status}"

        status += f"): {in_from} -> {out_to}"
        print(status, file=sys.stderr)

    return 0
