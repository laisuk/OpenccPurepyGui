"""Decode plain-text batch inputs and previews with optional CJK detection."""

from __future__ import annotations

from helpers.cjk_encoding_detector import detect_cjk_encoding


def decode_plain_text(data: bytes, auto_detect: bool, *, preview: bool = False) -> tuple[str, str]:
    encoding = "utf-8"
    payload = data
    if auto_detect:
        # UTF-32 BOMs must precede UTF-16's overlapping little-endian BOM.
        if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
            encoding = "utf-32"
        elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        elif data.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8-sig"
        elif data:
            detected = detect_cjk_encoding(data)
            if detected.encoding is None and not preview:
                raise UnicodeError("Cannot detect text encoding. File skipped.")
            encoding = detected.encoding or "utf-8"
            if encoding == "big5":
                encoding = "cp950"
            if encoding == "shift_jis":
                encoding = "cp932"
            payload = data[detected.bom_size:]
    try:
        contents = payload.decode(encoding)
    except UnicodeError:
        if not preview:
            raise
        contents = payload.decode("utf-8", errors="replace")
    # Preserve the newline behavior of Python text-mode input.
    return contents.replace("\r\n", "\n").replace("\r", "\n"), encoding
