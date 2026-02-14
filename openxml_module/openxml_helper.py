from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import re
import xml.etree.ElementTree as eT


# =============================================================================
# Public: format detection
# =============================================================================

def is_docx(path: str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() != ".docx":
        return False

    try:
        with ZipFile(p, "r") as zf:
            return (
                    _zip_has(zf, "word/document.xml")
                    and _zip_has(zf, "[Content_Types].xml")
            )
    except OSError:
        return False


def is_odt(path: str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() != ".odt":
        return False

    try:
        with ZipFile(p, "r") as zf:
            if not _zip_has(zf, "content.xml"):
                return False

            # Optional mimetype verification ( the best effort)
            if not _zip_has(zf, "mimetype"):
                return True

            mt = zf.read("mimetype").decode("ascii", errors="ignore").strip()
            return mt == "application/vnd.oasis.opendocument.text"
    except OSError:
        return False


def _zip_has(zf: ZipFile, name: str) -> bool:
    try:
        zf.getinfo(name)
        return True
    except KeyError:
        return False


# =============================================================================
# Public: DOCX extraction
# =============================================================================

def extract_docx_all_text(
        docx_path: str,
        *,
        include_part_headings: bool = False,
        normalize_newlines: bool = True,
        include_numbering: bool = True,
) -> str:
    """
    C#-equivalent of OpenXmlHelper.ExtractDocxAllText().

    Notes:
    - Extracts: document, footnotes, endnotes, comments, headers*, footers*
    - Resets numbering counters per part
    - Numbering: minimal/common only (bullet + %1..%9 expansion) and switchable
    """
    with ZipFile(docx_path, "r") as zf:
        ctx = NumberingContext.load(zf) if include_numbering else None

        parts: List[str] = []
        parts.extend([
            "word/document.xml",
            "word/footnotes.xml",
            "word/endnotes.xml",
            "word/comments.xml",
        ])

        names = zf.namelist()

        headers = sorted(
            (n for n in names if n.lower().startswith("word/header") and n.lower().endswith(".xml")),
            key=lambda s: s.lower(),
        )

        footers = sorted(
            (n for n in names if n.lower().startswith("word/footer") and n.lower().endswith(".xml")),
            key=lambda s: s.lower(),
        )

        parts.extend(headers)
        parts.extend(footers)

        # distinct (case-insensitive)
        seen = set()
        uniq_parts: List[str] = []
        for n in parts:
            k = n.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq_parts.append(n)

        out: List[str] = []

        for part_name in uniq_parts:
            try:
                xml_bytes = zf.read(part_name)
            except KeyError:
                continue

            if include_part_headings:
                if out and not _ends_with_newline_chunks(out):
                    out.append("\n")
                out.append(f"=== {part_name} ===\n")

            if ctx is not None:
                ctx.reset_counters_for_part()

            out.append(_extract_wordprocessingml_text(xml_bytes, ctx))

            if not _ends_with_newline_chunks(out):
                out.append("\n")

        result = "".join(out)
        if normalize_newlines:
            result = result.replace("\r\n", "\n").replace("\r", "\n")
        return result


# =============================================================================
# DOCX: WordprocessingML streaming extraction (C#-equivalent behavior)
# =============================================================================

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W_TAG = f"{{{_W_NS}}}"


def _extract_wordprocessingml_text(xml_bytes: bytes, ctx: Optional["NumberingContext"]) -> str:
    """
    Mirrors the C# XmlReader state machine closely.

    - Table rows are flattened into tab-separated lines.
    - Paragraph ends always emit '\n' (unless skipping separator footnotes/endnotes).
    - Prefix emitted once per paragraph, lazily on first content (t/tab/br/cr).
    """
    sb: List[str] = []

    in_table = False
    in_row = False
    in_cell = False
    current_row_cells: Optional[List[str]] = None
    current_cell: Optional[List[str]] = None  # list[str] buffer for the cell

    in_paragraph = False
    para_prefix_emitted = False
    para_num_id: Optional[int] = None
    para_ilvl: Optional[int] = None
    para_style_id: Optional[str] = None

    in_footnote = False
    in_endnote = False
    skip_this_note = False

    def current_target() -> List[str]:
        return current_cell if (in_cell and current_cell is not None) else sb

    def emit_prefix_if_needed() -> None:
        nonlocal para_prefix_emitted
        if not in_paragraph or para_prefix_emitted:
            return
        if ctx is None:
            return

        num_id, ilvl = ctx.resolve_num(para_num_id, para_ilvl, para_style_id)
        if num_id is None or ilvl is None:
            return

        prefix = ctx.next_prefix(num_id, ilvl)
        if not prefix:
            return

        current_target().append(prefix)
        para_prefix_emitted = True

    def should_skip_note_element(element: eT.Element) -> bool:
        # matches C# ShouldSkipNoteElement(XmlReader)
        type_ = _w_attr(element, "type")
        if type_ and type_.lower() in ("separator", "continuationseparator"):
            return True

        id_str = _w_attr(element, "id")
        if id_str is not None:
            try:
                return int(id_str) <= 0
            except ValueError:
                pass

        return False

    # iterparse stream
    events = ("start", "end")
    for ev, elem in eT.iterparse(_bytes_io(xml_bytes), events=events):
        if not elem.tag.startswith(_W_TAG):
            continue

        name = elem.tag[len(_W_TAG):]

        if ev == "start":
            if name == "footnote":
                in_footnote = True
                skip_this_note = should_skip_note_element(elem)

            elif name == "endnote":
                in_endnote = True
                skip_this_note = should_skip_note_element(elem)

            elif name == "tbl":
                in_table = True

            elif name == "tr":
                if in_table:
                    in_row = True
                    current_row_cells = []

            elif name == "tc":
                if in_row:
                    in_cell = True
                    current_cell = []

            elif name == "p":
                in_paragraph = True
                para_prefix_emitted = False
                para_num_id = None
                para_ilvl = None
                para_style_id = None

            elif name == "pStyle":
                if in_paragraph:
                    v = _w_attr(elem, "val")
                    if v:
                        para_style_id = v

            elif name == "numId":
                if in_paragraph:
                    v = _w_attr(elem, "val")
                    if v is not None:
                        try:
                            para_num_id = int(v)
                        except ValueError:
                            pass

            elif name == "ilvl":
                if in_paragraph:
                    v = _w_attr(elem, "val")
                    if v is not None:
                        try:
                            para_ilvl = int(v)
                        except ValueError:
                            pass

            elif name == "t":
                # in C#, ReadElementContentAsString happens at start; we mimic at end
                pass

            elif name == "tab":
                if skip_this_note and (in_footnote or in_endnote):
                    continue
                emit_prefix_if_needed()
                current_target().append("\t")

            elif name in ("br", "cr"):
                if skip_this_note and (in_footnote or in_endnote):
                    continue
                emit_prefix_if_needed()
                current_target().append("\n")

        else:  # ev == "end"
            if name == "t":
                if skip_this_note and (in_footnote or in_endnote):
                    elem.clear()
                    continue
                text = elem.text or ""
                if text:
                    emit_prefix_if_needed()
                    current_target().append(text)

            elif name == "p":
                if not (skip_this_note and (in_footnote or in_endnote)):
                    current_target().append("\n")
                in_paragraph = False

            elif name == "tc":
                if in_cell and current_row_cells is not None and current_cell is not None:
                    cell_text = "".join(current_cell)
                    current_row_cells.append(_trim_trailing_newlines(cell_text))
                    current_cell = None
                    in_cell = False

            elif name == "tr":
                if in_row and current_row_cells is not None:
                    sb.append("\t".join(current_row_cells))
                    sb.append("\n")
                    current_row_cells = None
                    in_row = False

            elif name == "tbl":
                if in_table:
                    if not _ends_with_newline_chunks(sb):
                        sb.append("\n")
                    in_table = False

            elif name == "footnote":
                in_footnote = False
                skip_this_note = False

            elif name == "endnote":
                in_endnote = False
                skip_this_note = False

            elem.clear()

    return "".join(sb)


def _w_attr(elem: eT.Element, local: str) -> Optional[str]:
    # WordprocessingML attributes are in the same w namespace in the C# code.
    return elem.get(f"{{{_W_NS}}}{local}") or elem.get(f"w:{local}") or elem.get(local)


def _bytes_io(b: bytes):
    # tiny helper to keep imports minimal
    import io
    return io.BytesIO(b)


# =============================================================================
# Public: ODT extraction (C#-equivalent behavior)
# =============================================================================

_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_TEXT_TAG = f"{{{_TEXT_NS}}}"
_TABLE_TAG = f"{{{_TABLE_NS}}}"


def extract_odt_all_text(odt_path: str, *, normalize_newlines: bool = True) -> str:
    with ZipFile(odt_path, "r") as zf:
        try:
            xml_bytes = zf.read("content.xml")
        except KeyError as e:
            raise ValueError("content.xml not found. Not a valid .odt?") from e

    text = _extract_odf_content_xml(xml_bytes)

    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text


def _extract_odf_content_xml(xml_bytes: bytes) -> str:
    sb: List[str] = []

    list_level = 0

    in_table = False
    in_row = False
    in_cell = False
    row_cells: Optional[List[str]] = None
    cell_buf: Optional[List[str]] = None

    in_paragraph = False
    prefix_emitted = False

    def target() -> List[str]:
        return cell_buf if (in_cell and cell_buf is not None) else sb

    def emit_list_prefix_if_needed() -> None:
        nonlocal prefix_emitted
        if not in_paragraph or prefix_emitted:
            return
        if list_level > 0:
            target().append(" " * ((list_level - 1) * 2))
            target().append("- ")
        prefix_emitted = True

    def append_text(s: str) -> None:
        if not s:
            return
        emit_list_prefix_if_needed()
        target().append(s)

    for ev, elem in eT.iterparse(_bytes_io(xml_bytes), events=("start", "end")):
        tag = elem.tag

        if ev == "start":
            # text namespace
            if tag.startswith(_TEXT_TAG):
                name = tag[len(_TEXT_TAG):]

                if name == "list":
                    list_level += 1

                elif name in ("p", "h"):
                    in_paragraph = True
                    prefix_emitted = False
                    emit_list_prefix_if_needed()

                elif name == "tab":
                    emit_list_prefix_if_needed()
                    target().append("\t")

                elif name == "line-break":
                    emit_list_prefix_if_needed()
                    target().append("\n")

                elif name == "s":
                    emit_list_prefix_if_needed()
                    c_attr = elem.get(f"{{{_TEXT_NS}}}c") or elem.get("text:c") or elem.get("c")
                    count = 1
                    if c_attr:
                        try:
                            n = int(c_attr)
                            if n > 0:
                                count = n
                        except ValueError:
                            pass
                    target().append(" " * count)

            # table namespace
            if tag.startswith(_TABLE_TAG):
                name = tag[len(_TABLE_TAG):]

                if name == "table":
                    in_table = True

                elif name == "table-row":
                    if in_table:
                        in_row = True
                        row_cells = []

                elif name == "table-cell":
                    if in_row:
                        in_cell = True
                        cell_buf = []

        else:  # end
            # text nodes: in ElementTree, they appear as .text/.tail
            if elem.text:
                append_text(elem.text)

            if tag.startswith(_TEXT_TAG):
                name = tag[len(_TEXT_TAG):]

                if name == "list":
                    if list_level > 0:
                        list_level -= 1

                elif name in ("p", "h"):
                    target().append("\n")
                    in_paragraph = False

            if tag.startswith(_TABLE_TAG):
                name = tag[len(_TABLE_TAG):]

                if name == "table-cell":
                    if in_cell and row_cells is not None and cell_buf is not None:
                        cell_text = "".join(cell_buf)
                        row_cells.append(_trim_trailing_newlines(cell_text))
                        cell_buf = None
                        in_cell = False

                elif name == "table-row":
                    if in_row and row_cells is not None:
                        sb.append("\t".join(row_cells))
                        sb.append("\n")
                        row_cells = None
                        in_row = False

                elif name == "table":
                    if in_table:
                        if not _ends_with_newline_chunks(sb):
                            sb.append("\n")
                        in_table = False

            # tail text
            if elem.tail:
                append_text(elem.tail)

            elem.clear()

    return "".join(sb)


# =============================================================================
# Shared small helpers (C#-equivalent)
# =============================================================================

def _ends_with_newline_chunks(chunks: List[str]) -> bool:
    if not chunks:
        return True
    last = chunks[-1]
    if not last:
        return True
    return last.endswith("\n") or last.endswith("\r")


def _trim_trailing_newlines(s: str) -> str:
    i = len(s)
    while i > 0 and s[i - 1] in ("\n", "\r"):
        i -= 1
    return s if i == len(s) else s[:i]


# =============================================================================
# DOCX numbering support (minimal/common only, switchable)
# =============================================================================

@dataclass
class _LevelDef:
    num_fmt: str = ""
    lvl_text: str = ""


class NumberingContext:
    """
    C#-equivalent minimal numbering:

    - numId -> abstractNumId
    - abstractNumId -> ilvl -> (numFmt, lvlText)
    - styleId -> (numId, ilvl)
    - counters per numId (9 levels)
    """

    _re_pct = re.compile(r"%([1-9])")

    def __init__(self) -> None:
        self._num_to_abstract: Dict[int, int] = {}
        self._abstract_levels: Dict[int, Dict[int, _LevelDef]] = {}
        self._style_num: Dict[str, Tuple[int, int]] = {}
        self._counters: Dict[int, List[int]] = {}

    # ---- lifecycle ----

    def reset_counters_for_part(self) -> None:
        self._counters.clear()

    @classmethod
    def load(cls, zf: ZipFile) -> "NumberingContext":
        ctx = cls()
        ctx._load_numbering(zf)
        ctx._load_styles(zf)
        return ctx

    # ---- resolve + next prefix ----

    def resolve_num(
            self,
            direct_num_id: Optional[int],
            direct_ilvl: Optional[int],
            style_id: Optional[str],
    ) -> Tuple[Optional[int], Optional[int]]:
        if direct_num_id is not None and direct_ilvl is not None:
            return direct_num_id, direct_ilvl

        if style_id:
            v = self._style_num.get(style_id)
            if v is not None:
                return v[0], v[1]

        return None, None

    def next_prefix(self, num_id: int, ilvl: int) -> str:
        # clamp like C#
        if ilvl < 0:
            ilvl = 0
        elif ilvl > 8:
            ilvl = 8

        abs_id = self._num_to_abstract.get(num_id)
        if abs_id is None:
            return ""

        lvls = self._abstract_levels.get(abs_id)
        if not lvls:
            return ""

        defn = lvls.get(ilvl)
        if defn is None:
            return ""

        counters = self._counters.get(num_id)
        if counters is None:
            counters = [0] * 9
            self._counters[num_id] = counters

        counters[ilvl] += 1
        for d in range(ilvl + 1, len(counters)):
            counters[d] = 0

        # common numbering only: bullet + %n replacement
        if defn.num_fmt.lower() == "bullet":
            return "• "

        lvl_text = defn.lvl_text or "%1."

        def repl(m: re.Match[str]) -> str:
            k = ord(m.group(1)) - ord("1")  # 0..8
            v = counters[k]
            if v <= 0:
                v = 1
            return str(v)

        prefix = self._re_pct.sub(repl, lvl_text)

        # normalize like C#
        prefix = prefix.replace("\t", " ").replace("\u00A0", " ")
        if prefix and not prefix[-1].isspace():
            prefix += " "
        return prefix

    # ---- load numbering.xml ----

    def _load_numbering(self, zf: ZipFile) -> None:
        try:
            xml_bytes = zf.read("word/numbering.xml")
        except KeyError:
            return

        current_abs: Optional[int] = None
        current_lvl: Optional[int] = None

        for ev, elem in eT.iterparse(_bytes_io(xml_bytes), events=("start", "end")):
            if not elem.tag.startswith(_W_TAG):
                continue
            name = elem.tag[len(_W_TAG):]

            if ev == "start":
                if name == "num":
                    num_id_str = _w_attr(elem, "numId")
                    try:
                        this_num_id = int(num_id_str) if num_id_str is not None else None
                    except ValueError:
                        this_num_id = None

                    if this_num_id is not None:
                        # scan subtree (like C# ReadSubtree) for abstractNumId
                        # Here, we just rely on parsing and capture at end when we see abstractNumId.
                        elem._openxmlhelper_numid = this_num_id  # type: ignore[attr-defined]

                elif name == "abstractNum":
                    abs_id_str = _w_attr(elem, "abstractNumId")
                    try:
                        abs_id = int(abs_id_str) if abs_id_str is not None else None
                    except ValueError:
                        abs_id = None

                    if abs_id is not None:
                        current_abs = abs_id
                        self._abstract_levels.setdefault(abs_id, {})

                elif name == "lvl":
                    if current_abs is not None:
                        ilvl_str = _w_attr(elem, "ilvl")
                        try:
                            ilvl = int(ilvl_str) if ilvl_str is not None else None
                        except ValueError:
                            ilvl = None

                        if ilvl is not None:
                            current_lvl = ilvl
                            self._abstract_levels[current_abs].setdefault(ilvl, _LevelDef())


                elif name == "abstractNumId":
                    # handled in second pass (_parse_num_to_abstract)
                    pass

                elif name == "numFmt":
                    if current_abs is not None and current_lvl is not None:
                        v = _w_attr(elem, "val") or ""
                        self._abstract_levels[current_abs][current_lvl].num_fmt = v

                elif name == "lvlText":
                    if current_abs is not None and current_lvl is not None:
                        v = _w_attr(elem, "val") or ""
                        self._abstract_levels[current_abs][current_lvl].lvl_text = v

            elif ev == "end":
                if name == "abstractNum":
                    current_abs = None
                    current_lvl = None
                elif name == "lvl":
                    current_lvl = None

                elem.clear()

        # Robust second pass to build numId->abstractNumId mapping (clean + deterministic)
        self._num_to_abstract.update(_parse_num_to_abstract(xml_bytes))

    # ---- load styles.xml ----

    def _load_styles(self, zf: ZipFile) -> None:
        try:
            xml_bytes = zf.read("word/styles.xml")
        except KeyError:
            return

        current_style_id: Optional[str] = None
        style_num_id: Optional[int] = None
        style_ilvl: Optional[int] = None

        for ev, elem in eT.iterparse(_bytes_io(xml_bytes), events=("start", "end")):
            if not elem.tag.startswith(_W_TAG):
                continue
            name = elem.tag[len(_W_TAG):]

            if ev == "start":
                if name == "style":
                    current_style_id = _w_attr(elem, "styleId")
                    style_num_id = None
                    style_ilvl = None

                elif name == "numId" and current_style_id is not None:
                    v = _w_attr(elem, "val")
                    if v is not None:
                        try:
                            style_num_id = int(v)
                        except ValueError:
                            pass

                elif name == "ilvl" and current_style_id is not None:
                    v = _w_attr(elem, "val")
                    if v is not None:
                        try:
                            style_ilvl = int(v)
                        except ValueError:
                            pass

            else:  # end
                if name == "style":
                    if current_style_id and style_num_id is not None and style_ilvl is not None:
                        self._style_num[current_style_id] = (style_num_id, style_ilvl)
                    current_style_id = None
                    style_num_id = None
                    style_ilvl = None

                elem.clear()


def _parse_num_to_abstract(numbering_xml_bytes: bytes) -> Dict[int, int]:
    """
    Clean pass that matches the C# behavior of:
    <w:num w:numId="X"> ... <w:abstractNumId w:val="Y" /> ...
    """
    out: Dict[int, int] = {}
    current_num_id: Optional[int] = None

    for ev, elem in eT.iterparse(_bytes_io(numbering_xml_bytes), events=("start", "end")):
        if not elem.tag.startswith(_W_TAG):
            continue
        name = elem.tag[len(_W_TAG):]

        if ev == "start":
            if name == "num":
                s = _w_attr(elem, "numId")
                try:
                    current_num_id = int(s) if s is not None else None
                except ValueError:
                    current_num_id = None

            elif name == "abstractNumId" and current_num_id is not None:
                v = _w_attr(elem, "val")
                try:
                    abs_id = int(v) if v is not None else None
                except ValueError:
                    abs_id = None
                if abs_id is not None:
                    out[current_num_id] = abs_id

        elif ev == "end":
            if name == "num":
                current_num_id = None
            elem.clear()

    return out
