import platform
import sys

# import pyperclip as pc
from opencc_rs import OpenCC

if platform.system() == 'Windows':
    from clipboard_win import get_clipboard_text, set_clipboard_text
elif platform.system() == 'Linux':
    from clipboard_linux import get_clipboard_text, set_clipboard_text
else:
    pass

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


def main():
    config_list = ["s2t", "t2s", "s2tw", "tw2s", "s2twp", "tw2sp", "s2hk", "hk2s", "t2tw", "tw2t", "t2twp", "tw2t",
                   "tw2tp", "t2hk", "hk2t", "t2jp", "jp2t"]
    config = "auto"
    punctuation = False
    if len(sys.argv) > 1:
        if sys.argv[1].lower() not in config_list:
            config = "auto"
        else:
            config = sys.argv[1].lower()
        if len(sys.argv) >= 2:
            punctuation = True if sys.argv[2].lower() == "punct" else False

    if config[0] == "s":
        display_input_code = "Simplified 简体"
        display_output_code = "Traditional 繁体"
    elif config[0:2] == "jp":
        display_input_code = "Japanese Shinjitai 新字体"
        display_output_code = "Japanese Kyujitai 舊字體"
    elif config[-2:] == "jp":
        display_input_code = "Japanese Kyujitai 舊字體"
        display_output_code = "Japanese Shinjitai 新字体"
    else:
        display_input_code = "Traditional 繁体"
        if "s" in config:
            display_output_code = "Simplified 简体"
        else:
            display_output_code = "Traditional 繁体"

    # Paste from clipboard
    # input_text = pc.paste()
    # input_text = Tk().clipboard_get()
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
            display_input_code = "Traditional 繁体"
            display_output_code = "Simplified 简体"
        elif text_code == 2:
            config = "s2t"
            display_input_code = "Simplified 简体"
            display_output_code = "Traditional 繁体"
        else:
            config = "s2t"
            display_input_code = "Others 其它"
            display_output_code = "Others 其它"

    # Initialized conversion config
    converter = OpenCC(config)
    # Do conversion
    output_text = converter.convert(input_text, punctuation)

    display_input = input_text[0:200] if len(input_text) > 200 else input_text
    display_output = output_text[0:200] if len(
        output_text) > 200 else output_text
    etc = "..." if len(input_text) > 200 else ""

    # Print out input contents with color texts
    print(f"Config: {BLUE}{config}{auto_detect}, {punctuation}{RESET}")
    print(f"{GREEN}== Clipboard Input text ({display_input_code}) =={YELLOW}\n{display_input}{etc}\n")
    print(f"{GREEN}== Clipboard Set Text ({display_output_code}) =={YELLOW}\n{display_output}{etc}{RESET}")
    print(f"{BLUE}(Total {len(output_text):,} chars converted){RESET}")
    # Copy converted contents to clipboard
    # pc.copy(output_text)
    set_clipboard_text(output_text)


if __name__ == "__main__":
    main()
