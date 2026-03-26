from __future__ import annotations

from app.generator import _parse_test_cases
from app.ingest import build_semantic_chunks


def test_parse_test_cases_from_markdown_json() -> None:
    raw = """
```json
[
  {
    "title": "Проверка входа",
    "preconditions": "Пользователь зарегистрирован",
    "steps": ["Открыть форму", "Ввести данные", "Нажать вход"],
    "expected_result": "Пользователь вошел",
    "scenario_type": "positive",
    "source_ref": "page_1_offset_0",
    "priority": "high"
  }
]
```
"""
    parsed = _parse_test_cases(raw)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Проверка входа"
    assert parsed[0]["scenario_type"] == "positive"



def test_build_semantic_chunks_generates_refs() -> None:
    pages = [{"page": 1, "text": "A" * 2100}]
    chunks = build_semantic_chunks(
        pages=pages,
        max_chars=1000,
        overlap=100,
        min_chunk_chars=50,
    )

    assert len(chunks) >= 2
    assert all("source_ref" in chunk for chunk in chunks)
    assert all(chunk["page"] == 1 for chunk in chunks)
