from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import fitz

from app.config import SETTINGS


_CLEAN_NAME_RE = re.compile(r"[^0-9A-Za-zА-Яа-я._-]+")


def sanitize_filename(filename: str) -> str:
    name = (filename or "document.pdf").strip().replace(" ", "_")
    name = _CLEAN_NAME_RE.sub("_", name)
    return name or "document.pdf"


def save_upload(file_bytes: bytes, filename: str) -> Tuple[str, Path]:
    file_id = str(uuid.uuid4())
    safe_name = sanitize_filename(filename)
    path = SETTINGS.raw_dir / f"{file_id}_{safe_name}"
    path.write_bytes(file_bytes)
    return file_id, path


def extract_pdf_text(path: Path) -> Tuple[str, List[Dict]]:
    doc = fitz.open(path)
    pages: List[Dict] = []
    all_text: List[str] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        pages.append({"page": page_num, "text": text})
        if text:
            all_text.append(text)

    return "\n\n".join(all_text), pages


def _chunk_text(page_text: str, max_chars: int, overlap: int) -> List[str]:
    if len(page_text) <= max_chars:
        return [page_text]

    chunks: List[str] = []
    start = 0

    while start < len(page_text):
        end = min(len(page_text), start + max_chars)
        chunk = page_text[start:end]

        # Стараемся не резать внутри слова или предложения.
        if end < len(page_text):
            boundary = max(chunk.rfind("\n"), chunk.rfind("."), chunk.rfind(" "))
            if boundary > int(max_chars * 0.6):
                end = start + boundary + 1
                chunk = page_text[start:end]

        chunks.append(chunk.strip())
        if end == len(page_text):
            break

        start = max(0, end - overlap)

    return [c for c in chunks if c]


def build_semantic_chunks(
    pages: List[Dict],
    max_chars: int,
    overlap: int,
    min_chunk_chars: int,
) -> List[Dict]:
    chunks: List[Dict] = []

    for page in pages:
        page_number = page["page"]
        text = (page.get("text") or "").strip()
        if not text:
            continue

        offset = 0
        for idx, chunk_text in enumerate(_chunk_text(text, max_chars=max_chars, overlap=overlap)):
            if len(chunk_text) < min_chunk_chars and idx > 0:
                if chunks:
                    chunks[-1]["text"] = (chunks[-1]["text"] + "\n" + chunk_text).strip()
                continue

            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number,
                    "source_ref": f"page_{page_number}_offset_{offset}",
                }
            )
            offset += max(1, len(chunk_text) - overlap)

    return chunks
