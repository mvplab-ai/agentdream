"""Markdown / plain-text folder (or single file) memory store.

Granularity rules (documented in README):

* A file that starts with YAML-ish front matter (``---`` ... ``---``), like
  Claude Code's per-memory files, is ONE entry: content = body, metadata =
  front matter fields.
* Any other ``.md`` / ``.markdown`` / ``.txt`` file is split into one entry per
  top-level bullet / numbered item (continuation and nested lines belong to
  the bullet) and one entry per free paragraph. Headings are not entries; they
  become the entry ``key`` (``"Heading > Subheading"``).
* Entry ids are ``<relative path>#<n>`` (block entries) or ``<relative path>``
  (front matter files), so they are stable as long as the file is unchanged.

Writing re-renders only what changed: untouched files are copied byte for
byte, edited entries are re-rendered in place with their original bullet
marker, removed entries are dropped, and new entries are inserted at the end
of their section (or into ``dream-insights.md`` when they have no path).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..models import MemoryEntry
from .base import AdapterError, MemoryAdapter

TEXT_EXTS = {".md", ".markdown", ".txt"}
DEFAULT_NEW_FILE = "dream-insights.md"
_BULLET = re.compile(r"^( {0,3})([-*+]|\d+[.)])(\s+)(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?))?\b")


@dataclass
class _Block:
    entry: MemoryEntry
    start: int
    end: int  # exclusive
    marker: str  # "- " or "" for paragraphs
    indent: str = ""


@dataclass
class _ParsedFile:
    rel: str
    text: str
    lines: list[str]
    frontmatter: str | None = None
    blocks: list[_Block] = field(default_factory=list)


def _parse_frontmatter(raw: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        m = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip().strip("'\"")
    return meta


def _date_in(text: str) -> str | None:
    m = _DATE.search(text)
    if not m:
        return None
    return m.group(1) + (f"T{m.group(2)}" if m.group(2) else "")


def parse_markdown(rel: str, text: str) -> _ParsedFile:
    lines = text.splitlines()
    pf = _ParsedFile(rel=rel, text=text, lines=lines)
    fm = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.S)
    if fm:
        pf.frontmatter = text[: fm.end()]
        meta = _parse_frontmatter(fm.group(1))
        body = text[fm.end():].strip("\n")
        e = MemoryEntry(
            id=rel, content=body.strip(), path=rel, key=meta.get("name"),
            metadata={**meta, "_layout": "file"},
            created_at=meta.get("created") or meta.get("created_at"),
            updated_at=meta.get("updated") or meta.get("updated_at") or meta.get("date") or _date_in(body),
            source=rel,
        )
        if e.content:
            pf.blocks.append(_Block(e, 0, len(lines), ""))
        return pf

    headings: list[tuple[int, str]] = []
    i = 0
    n = 0
    in_fence = False

    def section() -> str | None:
        return " > ".join(h for _, h in headings) or None

    def add(start: int, end: int, marker: str, content: str, indent: str = "") -> None:
        nonlocal n
        content = content.strip("\n")
        if not content.strip():
            return
        n += 1
        e = MemoryEntry(
            id=f"{rel}#{n}", content=content, path=rel, key=section(),
            metadata={"_layout": "block", "section": section()} if section() else {"_layout": "block"},
            updated_at=_date_in(content), source=f"{rel}:{start + 1}",
        )
        pf.blocks.append(_Block(e, start, end, marker, indent))

    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            # A fenced code block is kept as one paragraph entry.
            start = i
            i += 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                i += 1
            i = min(i + 1, len(lines))
            add(start, i, "", "\n".join(lines[start:i]))
            continue
        hm = _HEADING.match(line)
        if hm and not in_fence:
            level = len(hm.group(1))
            headings[:] = [(lv, h) for lv, h in headings if lv < level]
            headings.append((level, hm.group(2)))
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        bm = _BULLET.match(line)
        if bm:
            indent, mark, space, first = bm.groups()
            marker = mark + space
            start = i
            body = [first]
            i += 1
            cont_indent = len(indent) + len(marker)
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    # blank line: continue only if followed by an indented continuation
                    if i + 1 < len(lines) and lines[i + 1].startswith(" " * cont_indent) and lines[i + 1].strip():
                        body.append("")
                        i += 1
                        continue
                    break
                if _BULLET.match(nxt) or _HEADING.match(nxt) or _FENCE.match(nxt):
                    break
                if nxt.startswith(" ") or nxt.startswith("\t"):
                    body.append(nxt[cont_indent:] if nxt.startswith(" " * cont_indent) else nxt.strip())
                    i += 1
                    continue
                # lazy continuation line
                body.append(nxt.strip())
                i += 1
            add(start, i, marker, "\n".join(body), indent)
            continue
        # paragraph
        start = i
        while i < len(lines) and lines[i].strip() and not _HEADING.match(lines[i]) and not _BULLET.match(lines[i]) \
                and not _FENCE.match(lines[i]):
            i += 1
        add(start, i, "", "\n".join(lines[start:i]))
    return pf


def _render_block(content: str, marker: str, indent: str) -> list[str]:
    lines = content.split("\n")
    if not marker:
        return lines
    pad = " " * (len(indent) + len(marker))
    out = [indent + marker + lines[0]]
    out.extend((pad + ln) if ln else "" for ln in lines[1:])
    return out


class MarkdownAdapter(MemoryAdapter):
    name = "markdown"

    def __init__(self, path, **options):
        super().__init__(path, **options)
        self.single_file = self.path.is_file()
        self._cache: dict[str, _ParsedFile] | None = None

    @classmethod
    def detect(cls, path: Path) -> bool:
        if path.is_file():
            return path.suffix.lower() in TEXT_EXTS
        if path.is_dir():
            return any(p.suffix.lower() in TEXT_EXTS for p in cls._iter_files(path))
        return False

    @staticmethod
    def _iter_files(root: Path):
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if p.is_file():
                yield p

    def _files(self) -> dict[str, _ParsedFile]:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            raise AdapterError(f"memory store not found: {self.path}")
        out: dict[str, _ParsedFile] = {}
        if self.single_file:
            files = [(self.path.name, self.path)]
        else:
            files = [(str(p.relative_to(self.path).as_posix()), p) for p in self._iter_files(self.path)
                     if p.suffix.lower() in TEXT_EXTS]
        for rel, p in files:
            text = p.read_text(encoding="utf-8", errors="replace")
            out[rel] = parse_markdown(rel, text)
        self._cache = out
        return out

    def load(self) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        order = 0
        for pf in self._files().values():
            for b in pf.blocks:
                e = b.entry.copy()
                e.metadata["_order"] = order
                order += 1
                entries.append(e)
        return entries

    def output_path(self, store_dir):
        d = Path(store_dir)
        return d / self.path.name if self.single_file else d

    def write(self, entries: list[MemoryEntry], dest) -> Path:
        dest = Path(dest)
        self._check_dest(dest)
        files = self._files()
        by_path: dict[str, list[MemoryEntry]] = {}
        default_new = self.path.name if self.single_file else DEFAULT_NEW_FILE
        for e in entries:
            rel = e.path if (e.path and (e.path in files or not self.single_file)) else default_new
            by_path.setdefault(rel, []).append(e)

        rendered: dict[str, str] = {}
        for rel, pf in files.items():
            rendered_text = self._render_file(pf, by_path.get(rel, []))
            if rendered_text is not None:
                rendered[rel] = rendered_text
        for rel, ents in by_path.items():
            if rel in files:
                continue
            rendered[rel] = self._render_new_file(ents)

        if self.single_file:
            text = rendered.get(self.path.name, "")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(dest)
            return dest

        tmpdir = dest.with_name(dest.name + ".tmp-write")
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True)
        # copy non-memory files verbatim
        for p in self._iter_files(self.path):
            rel = p.relative_to(self.path).as_posix()
            if p.suffix.lower() not in TEXT_EXTS:
                (tmpdir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tmpdir / rel)
        for rel, text in rendered.items():
            target = tmpdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = self.path / rel
            if rel in files and text == files[rel].text:
                shutil.copy2(src, target)
            else:
                target.write_text(text, encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        tmpdir.replace(dest)
        return dest

    # -- rendering helpers
    def _render_file(self, pf: _ParsedFile, new_entries: list[MemoryEntry]) -> str | None:
        new_by_id = {e.id: e for e in new_entries}
        orig_ids = {b.entry.id for b in pf.blocks}
        if pf.frontmatter is not None:
            e = new_by_id.get(pf.rel)
            extras = [x for x in new_entries if x.id not in orig_ids]
            if e is None and not extras:
                return None if pf.blocks else pf.text
            if e is not None and pf.blocks and e.content == pf.blocks[0].entry.content and not extras:
                return pf.text
            body = e.content if e is not None else ""
            for x in extras:
                body += ("\n\n" if body else "") + "- " + x.content.replace("\n", "\n  ")
            return pf.frontmatter + ("\n" if not pf.frontmatter.endswith("\n") else "") + body.strip("\n") + "\n"

        unchanged = (len(new_entries) == len(pf.blocks) and
                     all(b.entry.id in new_by_id and new_by_id[b.entry.id].content == b.entry.content
                         for b in pf.blocks))
        if unchanged:
            return pf.text
        extras = [x for x in new_entries if x.id not in orig_ids]
        marker = next((b.marker for b in pf.blocks if b.marker), "- ")
        if marker and marker[0].isdigit():
            marker = "- "
        # where to insert extras: after last block of the same section, else end
        insert_after: dict[int, list[MemoryEntry]] = {}
        tail: list[MemoryEntry] = []
        for x in extras:
            idx = None
            for bi, b in enumerate(pf.blocks):
                if x.key and b.entry.key == x.key:
                    idx = bi
            if idx is None:
                tail.append(x)
            else:
                insert_after.setdefault(idx, []).append(x)

        out: list[str] = []
        pos = 0
        lines = pf.lines
        for bi, b in enumerate(pf.blocks):
            out.extend(lines[pos:b.start])
            cur = new_by_id.get(b.entry.id)
            if cur is not None:
                if cur.content == b.entry.content:
                    out.extend(lines[b.start:b.end])
                else:
                    out.extend(_render_block(cur.content, b.marker, b.indent))
            else:
                # drop the block; if it was a paragraph, also drop one following blank line
                if not b.marker and b.end < len(lines) and not lines[b.end].strip():
                    pos = b.end + 1
                    for x in insert_after.get(bi, []):
                        out.extend(_render_block(x.content, marker, ""))
                    continue
            for x in insert_after.get(bi, []):
                out.extend(_render_block(x.content, b.marker or marker, b.indent if b.marker else ""))
            pos = b.end
        out.extend(lines[pos:])
        if tail:
            while out and not out[-1].strip():
                out.pop()
            if out:
                out.append("")
            by_key: dict[str | None, list[MemoryEntry]] = {}
            for x in tail:
                by_key.setdefault(x.key, []).append(x)
            for key, xs in by_key.items():
                if key:
                    out.append(f"## {key.split(' > ')[-1]}")
                    out.append("")
                for x in xs:
                    out.extend(_render_block(x.content, marker, ""))
                out.append("")
        text = "\n".join(out)
        while text.endswith("\n\n"):
            text = text[:-1]
        if not text.endswith("\n"):
            text += "\n"
        return text

    def _render_new_file(self, entries: list[MemoryEntry]) -> str:
        out = ["# Dream insights", "", "<!-- Added by agentdream. Review before relying on these notes. -->", ""]
        by_key: dict[str | None, list[MemoryEntry]] = {}
        for e in entries:
            by_key.setdefault(e.key, []).append(e)
        for key, xs in by_key.items():
            if key:
                out.extend([f"## {key}", ""])
            for x in xs:
                out.extend(_render_block(x.content, "- ", ""))
            out.append("")
        return "\n".join(out).rstrip("\n") + "\n"
