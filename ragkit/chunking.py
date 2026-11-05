"""Two chunking strategies so we can ablate them."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc: str
    text: str


def fixed_chunks(doc: str, text: str, size: int = 500, overlap: int = 100) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    out, step, i = [], size - overlap, 0
    for start in range(0, max(len(text), 1), step):
        piece = text[start:start + size].strip()
        if piece:
            out.append(Chunk(f"{doc}#{i}", doc, piece))
            i += 1
        if start + size >= len(text):
            break
    return out


def paragraph_chunks(doc: str, text: str, size: int = 700) -> list[Chunk]:
    """Pack whole paragraphs up to `size` chars; keep the nearest heading as context."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf, heading, i = [], "", "", 0

    def flush():
        nonlocal buf, i
        if buf.strip():
            out.append(Chunk(f"{doc}#{i}", doc, (f"[{heading}] " if heading else "") + buf.strip()))
            i += 1
        buf = ""

    for p in paras:
        if p.startswith("#"):
            flush()
            heading = p.lstrip("# ").strip()
            continue
        if buf and len(buf) + len(p) > size:
            flush()
        buf += p + "\n\n"
    flush()
    return out


STRATEGIES = {"fixed": fixed_chunks, "paragraph": paragraph_chunks}


def load_corpus(path: str | Path, strategy: str = "paragraph") -> list[Chunk]:
    fn = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for f in sorted(Path(path).rglob("*")):
        if f.suffix.lower() in {".md", ".txt", ".rst"}:
            chunks += fn(f.relative_to(path).as_posix(), f.read_text(encoding="utf-8", errors="ignore"))
    return chunks
