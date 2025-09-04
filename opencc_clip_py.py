#!/usr/bin/env python3
import argparse
import platform
import sys
from typing import List

from opencc_purepy import OpenCC

# Platform clipboard backends
if platform.system() == 'Windows':
    from clipboard_win import get_clipboard_text, set_clipboard_text
elif platform.system() == 'Linux':
    from clipboard_linux import get_clipboard_text, set_clipboard_text
else:
    # If you have a macOS backend, import it here.
    # from clipboard_darwin import get_clipboard_text, set_clipboard_text
    raise RuntimeError("Unsupported platform or missing clipboard backend")

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
RESET = "\033[0m"

CONFIGS: List[str] = [
    "s2t", "t2s", "s2tw", "tw2s", "s2twp", "tw2sp", "s2hk", "hk2s",
    "t2tw", "tw2t", "t2twp", "tw2t", "tw2tp", "t2hk", "hk2t", "t2jp", "jp2t",
    "auto"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="opencc-clip-py",
        description="OpenCC clipboard converter (reads from clipboard, converts, writes back)."
    )
    parser.add_argument(
        "-c", "--config",
        default="auto",
        choices=CONFIGS,
        help="OpenCC config (default: auto)"
    )
    parser.add_argument(
        "-p", "--punct",
        action="store_true",
        help="Also convert punctuation"
    )
    # if no args given, argparse still returns defaults → matches requested behavior
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config
    punctuation = args.punct

    # Display labels based on config (pre-auto; may change if auto-detect triggers)
    def labels_from_config(cfg: str):
        if cfg.startswith("s"):
            return "Simplified 简体", "Traditional 繁体"
        if cfg.startswith("jp"):
            return "Japanese Shinjitai 新字体", "Japanese Kyujitai 舊字體"
        if cfg.endswith("jp"):
            return "Japanese Kyujitai 舊字體", "Japanese Shinjitai 新字体"
        # default traditional→(maybe) simplified
        inp = "Traditional 繁体"
        out = "Simplified 简体" if "s" in cfg else "Traditional 繁体"
        return inp, out

    display_input_code, display_output_code = labels_from_config(config)

    # Read from clipboard
    input_text = get_clipboard_text()
    if input_text == "":
        print(f"{RED}Clipboard is empty{RESET}")
        return

    auto_detect = ""
    if config == "auto":
        auto_detect = " (auto)"
        text_code = OpenCC().zho_check(input_text)
        if text_code == 1:
            config = "t2s"
            display_input_code, display_output_code = "Traditional 繁体", "Simplified 简体"
        elif text_code == 2:
            config = "s2t"
            display_input_code, display_output_code = "Simplified 简体", "Traditional 繁体"
        else:
            # Fallback: keep original behavior
            config = "s2t"
            display_input_code, display_output_code = "Others 其它", "Others 其它"

    # Convert
    converter = OpenCC(config)
    output_text = converter.convert(input_text, punctuation)

    # Pretty print (trim to 200 chars for preview)
    display_input = input_text[:200]
    display_output = output_text[:200]
    etc = "..." if len(input_text) > 200 else ""

    print(f"Config: {BLUE}{config}{auto_detect}, {punctuation}{RESET}")
    print(f"{GREEN}== Clipboard Input text ({display_input_code}) =={YELLOW}\n{display_input}{etc}\n")
    print(f"{GREEN}== Clipboard Set Text ({display_output_code}) =={YELLOW}\n{display_output}{etc}{RESET}")
    print(f"{BLUE}(Total {len(output_text):,} chars converted){RESET}")

    # Write back to clipboard
    set_clipboard_text(output_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
