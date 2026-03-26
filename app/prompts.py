from __future__ import annotations

from typing import Dict, List

SYSTEM_PROMPT = """
Ты senior QA analyst. Формируй тест-кейсы строго по контексту документа.
Ответ должен быть ТОЛЬКО JSON-массивом, без markdown и без комментариев.

Структура каждого объекта:
- title: string
- preconditions: string
- steps: array[string]
- expected_result: string
- scenario_type: one of [positive, negative, boundary]
- source_ref: string в формате page_<N>_offset_<M>
- priority: one of [low, medium, high]

Обязательные правила:
1. Не выдумывай сущности и поля, которых нет в контексте.
2. Не дублируй кейсы: title должен быть уникален.
3. Каждый кейс должен быть атомарным (1 проверка = 1 кейс).
4. expected_result должен быть проверяемым и наблюдаемым.
5. source_ref бери только из предоставленного контекста.
6. Если данных недостаточно — верни меньше кейсов, но без галлюцинаций.
""".strip()



def render_prompt(
    context_blocks: List[Dict],
    user_prompt: str,
    max_cases: int,
    include_negative: bool,
    include_boundary: bool,
) -> str:
    context = "\n\n".join(
        f"[source_ref={block['source_ref']}; page={block.get('page')}] {block['text']}"
        for block in context_blocks
    )

    scenario_requirements = ["positive"]
    if include_negative:
        scenario_requirements.append("negative")
    if include_boundary:
        scenario_requirements.append("boundary")

    return f"""{SYSTEM_PROMPT}

Дополнительный акцент пользователя:
{user_prompt}

Ограничения на генерацию:
- Максимум кейсов: {max_cases}
- Обязательные типы сценариев: {", ".join(scenario_requirements)}
- Язык: русский

Контекст документа:
{context}

Сгенерируй JSON-массив тест-кейсов.
"""
