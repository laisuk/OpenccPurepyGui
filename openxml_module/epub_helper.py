from __future__ import annotations

import html
import io
import re
import xml.etree.ElementTree as eT
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import BadZipFile, LargeZipFile, ZipFile


# =============================================================================
# Public: format detection
# =============================================================================

def is_epub(path: str) -> bool:
    """
    C# EpubHelper.IsEpub equivalent:

    - must exist
    - must end with .epub
    - must contain META-INF/container.xml
    """
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() != ".epub":
        return False

    try:
        with ZipFile(p, "r") as zf:
            return _zip_has(zf, "META-INF/container.xml")
    except (BadZipFile, LargeZipFile, OSError):
        return False


def _zip_has(zf: ZipFile, name: str) -> bool:
    try:
        zf.getinfo(name)
        return True
    except KeyError:
        return False


# =============================================================================
# Public: EPUB extraction
# =============================================================================

def extract_epub_all_text(
    epub_path: str,
    *,
    include_part_headings: bool = False,
    normalize_newlines: bool = True,
    skip_nav_documents: bool = True,
) -> str:
    """
    C# EpubHelper.ExtractEpubAllText equivalent.

    - Reads META-INF/container.xml -> OPF
    - Parses OPF manifest + spine
    - Extracts plaintext from XHTML/HTML items in spine
    - Optional: skip nav documents
    - Normalizes newlines + clamps excessive blank lines
    """
    with ZipFile(epub_path, "r") as zf:
        opf_path = _find_opf_path(zf)
        if opf_path is None:
            raise ValueError("container.xml has no OPF rootfile. Not a valid .epub?")

        opf_dir = _get_dir(opf_path)
        manifest, spine = _load_opf(zf, opf_path)

        out: List[str] = []

        for idref in spine:
            item = manifest.get(idref)
            if item is None:
                continue

            if not _looks_like_html(item.media_type, item.href):
                continue

            if skip_nav_documents and item.is_nav:
                continue

            full_name = _combine_zip_path(opf_dir, item.href)
            try:
                xml_bytes = zf.read(full_name)
            except KeyError:
                continue

            if include_part_headings:
                if out and not _ends_with_newline_chunks(out):
                    out.append("\n")
                out.append(f"=== {full_name} ===\n")

            chapter_text = _extract_xhtml_text(xml_bytes)
            out.append(chapter_text)

            if not _ends_with_newline_chunks(out):
                out.append("\n")
            out.append("\n")  # blank line between spine docs

    text = "".join(out)

    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _normalize_excess_blank_lines(text)
    return text


# =============================================================================
# container.xml -> OPF path
# =============================================================================

_CONTAINER_XML = "META-INF/container.xml"


def _find_opf_path(zf: ZipFile) -> Optional[str]:
    """
    Mirrors C# FindOpfPath:
    scan container.xml for <rootfile full-path=""...">
    """
    try:
        xml_bytes = zf.read(_CONTAINER_XML)
    except KeyError:
        return None

    # container.xml is usually small; a simple parse is fine.
    # It may contain namespaces; we do not hard-require them.
    try:
        root = eT.fromstring(xml_bytes)
    except eT.ParseError:
        # Try decoding and stripping potential BOM/oddities
        s = xml_bytes.decode("utf-8", errors="replace")
        root = eT.fromstring(s.encode("utf-8"))

    for elem in root.iter():
        if _local_name(elem.tag) != "rootfile":
            continue
        full_path = elem.get("full-path") or elem.get("fullpath")
        if full_path and full_path.strip():
            return full_path.strip()

    return None


# =============================================================================
# OPF parsing (manifest + spine)
# =============================================================================

@dataclass(frozen=True)
class _ManifestItem:
    href: str
    media_type: str
    is_nav: bool


def _load_opf(zf: ZipFile, opf_path: str) -> Tuple[Dict[str, _ManifestItem], List[str]]:
    """
    Mirrors C# LoadOpf:
    - manifest: id -> { href, media-type, isNav(properties contains 'nav') }
    - spine: ordered list of idref
    """
    try:
        opf_bytes = zf.read(opf_path)
    except KeyError as e:
        raise ValueError(f"OPF not found: {opf_path}") from e

    # OPF may include DOCTYPE; sanitize similarly
    opf_bytes = _sanitize_xml_like_bytes(opf_bytes)

    try:
        root = eT.fromstring(opf_bytes)
    except eT.ParseError:
        # As a fallback, decode/encode again
        s = opf_bytes.decode("utf-8", errors="replace")
        root = eT.fromstring(s.encode("utf-8"))

    manifest: Dict[str, _ManifestItem] = {}
    spine: List[str] = []

    for elem in root.iter():
        name = _local_name(elem.tag)

        if name == "item":
            id_ = elem.get("id")
            href = elem.get("href")
            mt = elem.get("media-type") or ""

            if id_ and href:
                props = elem.get("properties") or ""
                is_nav = any(p.lower() == "nav" for p in props.split())
                manifest[id_] = _ManifestItem(href=href, media_type=mt, is_nav=is_nav)

        elif name == "itemref":
            idref = elem.get("idref")
            if idref:
                spine.append(idref)

    return manifest, spine


def _looks_like_html(media_type: str, href: str) -> bool:
    """
    Mirrors C# LooksLikeHtml:
    tolerant checks by media-type or extension.
    """
    mt = (media_type or "").strip()
    if not mt:
        return _has_html_ext(href)

    mt_low = mt.lower()
    if mt_low == "application/xhtml+xml":
        return True
    if mt_low == "text/html":
        return True
    if "html" in mt_low:
        return True
    return _has_html_ext(href)


def _has_html_ext(href: str) -> bool:
    h = (href or "").lower()
    return h.endswith(".xhtml") or h.endswith(".html") or h.endswith(".htm")


# =============================================================================
# XHTML -> plain text (XmlReader-like behavior)
# =============================================================================

_SKIP_ELEMENTS = {"script", "style", "head", "svg", "math", "noscript"}

_BLOCK_ELEMENTS = {
    "p", "div", "section", "article", "blockquote", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "hr",
}

_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE | re.DOTALL)
_XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>", re.IGNORECASE)
_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]+);")


def _extract_xhtml_text(xhtml_bytes: bytes) -> str:
    """
    Mirrors C# ExtractXhtmlText(Stream):
    - skip depth inside skip-elements
    - block elements -> EnsureParagraphBreak
    - <br> -> newline
    - text normalized: no runaway whitespace, no CJK smart spacing
    """
    # Sanitize DOCTYPE + named entities to make ET parsing reliable
    xhtml_bytes = _sanitize_xhtml_bytes(xhtml_bytes)

    sb: List[str] = []
    skip_depth = 0

    # Stream parse to avoid huge DOM build
    for ev, elem in eT.iterparse(io.BytesIO(xhtml_bytes), events=("start", "end")):
        name = _local_name(elem.tag).lower()

        if ev == "start":
            if name in _SKIP_ELEMENTS:
                skip_depth += 1
                # empty element: immediately pop
                if elem.text is None and len(elem) == 0 and elem.tail is None:
                    skip_depth -= 1
                continue

            if skip_depth > 0:
                continue

            if name in _BLOCK_ELEMENTS:
                _ensure_paragraph_break(sb)

            if name == "br":
                sb.append("\n")

        else:  # end
            if skip_depth > 0:
                if name in _SKIP_ELEMENTS:
                    skip_depth -= 1
                elem.clear()
                continue

            # Text content
            if elem.text:
                _append_normalized_text(sb, elem.text)

            # End element block break
            if name in _BLOCK_ELEMENTS:
                _ensure_paragraph_break(sb)

            # Tail text
            if elem.tail:
                _append_normalized_text(sb, elem.tail)

            elem.clear()

    text = "".join(sb)
    # C# post-fixes
    text = text.replace("\u00AD", "")   # soft hyphen
    text = text.replace("\u00A0", " ")  # nbsp
    return text


def _append_normalized_text(sb: List[str], t: str) -> None:
    """
    Mirrors C# AppendNormalizedText(StringBuilder, string):
    - preserve internal spaces
    - avoid runaway whitespace
    """
    for c in t:
        if c.isspace():
            if not sb:
                continue
            last = sb[-1]
            if last and last[-1] in (" ", "\n", "\r", "\t"):
                continue
            sb.append(" ")
        else:
            sb.append(c)


def _ensure_paragraph_break(sb: List[str]) -> None:
    """
    Mirrors C# EnsureParagraphBreak:
    - trim trailing spaces/tabs
    - ensure blank-line separation
    """
    _trim_trailing_spaces(sb)
    if not sb:
        return

    # if already blank line at end, keep
    if _ends_with_blank_line(sb):
        return

    # ensure ends with '\n'
    if not _ends_with_newline_chunks(sb):
        sb.append("\n")

    # blank line separation
    sb.append("\n")


def _trim_trailing_spaces(sb: List[str]) -> None:
    while sb:
        chunk = sb[-1]
        if not chunk:
            sb.pop()
            continue
        # trim only at end of the last chunk
        i = len(chunk)
        while i > 0 and chunk[i - 1] in (" ", "\t"):
            i -= 1
        if i == len(chunk):
            break
        if i == 0:
            sb.pop()
        else:
            sb[-1] = chunk[:i]
            break


def _ends_with_newline_chunks(chunks: List[str]) -> bool:
    if not chunks:
        return True
    last = chunks[-1]
    if not last:
        return True
    return last.endswith("\n") or last.endswith("\r")


def _ends_with_blank_line(chunks: List[str]) -> bool:
    """
    True if the buffer ends with '\n\n'.
    """
    if not chunks:
        return True
    tail = "".join(chunks[-3:])  # small window
    return tail.endswith("\n\n")


# =============================================================================
# Sanitizers: make XML parser tolerant like C# XmlReader(DtdProcessing.Ignore)
# =============================================================================

def _sanitize_xml_like_bytes(b: bytes) -> bytes:
    """
    For OPF/container: remove DOCTYPE if present.
    """
    s = b.decode("utf-8", errors="replace")
    s = _DOCTYPE_RE.sub("", s)
    return s.encode("utf-8")


def _sanitize_xhtml_bytes(b: bytes) -> bytes:
    """
    XHTML often contains:
    - <!DOCTYPE ...> with entity definitions
    - named entities like &nbsp;

    ElementTree cannot resolve named entities without DTD,
    so we:
    - strip DOCTYPE
    - convert named entities to Unicode via html.unescape
      (but only safely for named entities; keep numeric entities intact)
    """
    s = b.decode("utf-8", errors="replace")

    # Strip DOCTYPE (XmlReader ignores it)
    s = _DOCTYPE_RE.sub("", s)

    # Convert named entities to Unicode to avoid undefined entity parse errors.
    # We only replace &name; patterns; numeric entities remain as-is.
    def repl(m: re.Match[str]) -> str:
        ent = m.group(0)  # like "&nbsp;"
        return html.unescape(ent)

    s = _ENTITY_RE.sub(repl, s)

    return s.encode("utf-8")


# =============================================================================
# Zip path helpers (same semantics as C#)
# =============================================================================

def _get_dir(path: str) -> str:
    idx = path.rfind("/")
    return "" if idx < 0 else path[: idx + 1]


def _combine_zip_path(dir_: Optional[str], href: Optional[str]) -> str:
    raw = (dir_ or "") + (href or "")
    raw = raw.replace("\\", "/")

    parts = [p for p in raw.split("/") if p]
    stack: List[str] = []

    for p in parts:
        if p == ".":
            continue
        if p == "..":
            if stack:
                stack.pop()
            continue
        stack.append(p)

    return "/".join(stack)


def _local_name(tag: str) -> str:
    # "{ns}p" -> "p"
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


# =============================================================================
# Post-processing
# =============================================================================

def _normalize_excess_blank_lines(s: str) -> str:
    """
    Keep at most 2 consecutive newlines (same as C# NormalizeExcessBlankLines).
    """
    out: List[str] = []
    nl = 0
    for c in s:
        if c == "\n":
            nl += 1
            if nl <= 2:
                out.append(c)
        else:
            nl = 0
            out.append(c)
    return "".join(out)
