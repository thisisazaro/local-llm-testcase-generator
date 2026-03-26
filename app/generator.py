from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Tuple

import requests

from app.config import SETTINGS
from app.prompts import render_prompt


class LLMError(RuntimeError):
    pass


class ParsingError(RuntimeError):
    pass


RE_WORD = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")
RE_SOURCE_REF = re.compile(r"page_(\d+)_offset_(\d+)")
RE_PAGE = re.compile(r"page\s*=\s*(\d+)")



def _mock_response() -> str:
    demo = [
        {
            "title": "Успешный вход активного пользователя",
            "preconditions": "Существует активная учетная запись пользователя",
            "steps": [
                "Открыть страницу входа",
                "Ввести корректный логин и пароль",
                "Нажать кнопку входа",
            ],
            "expected_result": "Пользователь авторизован и видит главную страницу",
            "scenario_type": "positive",
            "source_ref": "page_1_offset_0",
            "priority": "high",
        },
        {
            "title": "Ошибка входа при неверном пароле",
            "preconditions": "Существует активная учетная запись пользователя",
            "steps": [
                "Открыть страницу входа",
                "Ввести корректный логин и неверный пароль",
                "Нажать кнопку входа",
            ],
            "expected_result": "Показано сообщение об ошибке аутентификации, вход не выполнен",
            "scenario_type": "negative",
            "source_ref": "page_1_offset_0",
            "priority": "high",
        },
    ]
    return json.dumps(demo, ensure_ascii=False)



def _call_ollama(prompt: str) -> str:
    url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": SETTINGS.llm_model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": SETTINGS.llm_temperature,
            "num_predict": SETTINGS.llm_max_tokens,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=SETTINGS.llm_timeout_seconds)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(f"Ollama call failed: {exc}") from exc

    data = resp.json()
    answer = data.get("response")
    if not answer:
        raise LLMError(f"Ollama returned empty response: {data}")
    return str(answer)



def _call_openai_compatible(prompt: str) -> str:
    url = f"{SETTINGS.openai_compat_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": SETTINGS.llm_model_name,
        "temperature": SETTINGS.llm_temperature,
        "max_tokens": SETTINGS.llm_max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    headers = {"Content-Type": "application/json"}
    if SETTINGS.openai_compat_api_key:
        headers["Authorization"] = f"Bearer {SETTINGS.openai_compat_api_key}"

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=SETTINGS.llm_timeout_seconds,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(f"OpenAI-compatible call failed: {exc}") from exc

    data = resp.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response format: {data}") from exc



def call_local_llm(prompt: str) -> str:
    provider = SETTINGS.llm_provider.lower().strip()

    if provider == "ollama":
        return _call_ollama(prompt)

    if provider in {"openai_compatible", "vllm"}:
        return _call_openai_compatible(prompt)

    if provider == "mock":
        return _mock_response()

    if SETTINGS.allow_mock_fallback:
        return _mock_response()

    raise LLMError(
        f"Unknown LLM_PROVIDER={SETTINGS.llm_provider}. "
        "Expected one of: mock, ollama, openai_compatible"
    )



def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()



def _extract_balanced_json(text: str) -> str:
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start < 0:
            continue

        depth = 0
        in_string = False
        escaped = False

        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue

            if ch == "\\":
                escaped = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

    raise ParsingError("Could not find balanced JSON in model output")



def _normalize_test_case(item: Dict[str, Any]) -> Dict[str, Any]:
    def first_non_empty(keys: List[str]) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    title = first_non_empty(["title", "name", "test_case", "case_title"])
    expected_result = first_non_empty(
        ["expected_result", "expected", "result", "expected_outcome", "expectedOutcome"]
    )

    steps_raw = item.get("steps")
    if steps_raw is None:
        for alias in ("test_steps", "procedure", "actions", "step_list"):
            alias_value = item.get(alias)
            if alias_value is not None:
                steps_raw = alias_value
                break
    if steps_raw is None:
        steps_raw = []

    if not title and not expected_result and not steps_raw:
        raise ParsingError("Model returned incomplete testcase fields")

    if isinstance(steps_raw, str):
        steps = [step.strip() for step in steps_raw.split("\n") if step.strip()]
    else:
        steps = [str(step).strip() for step in steps_raw if str(step).strip()]

    if not title:
        if steps:
            title = f"Проверка: {steps[0][:90]}"
        else:
            title = "Проверка по требованию из документа"

    if not steps:
        steps = ["Выполнить проверку согласно требованиям документа."]

    if not expected_result:
        expected_result = "Система должна вернуть результат согласно требованиям документа."

    scenario_type = str(item.get("scenario_type", "positive")).strip().lower()
    if scenario_type not in {"positive", "negative", "boundary"}:
        scenario_type = "positive"

    priority = str(item.get("priority", "medium")).strip().lower()
    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    return {
        "title": title,
        "preconditions": str(item.get("preconditions", "")).strip() or None,
        "steps": steps,
        "expected_result": expected_result,
        "scenario_type": scenario_type,
        "source_ref": str(item.get("source_ref", "")).strip() or None,
        "priority": priority,
    }



def _parse_test_cases(raw_answer: str) -> List[Dict[str, Any]]:
    cleaned = _strip_markdown_fences(raw_answer)
    try:
        json_blob = _extract_balanced_json(cleaned)
    except ParsingError:
        json_blob = cleaned

    candidates = [json_blob]
    # Попытка починить типичный "почти JSON" (висячие запятые).
    candidates.append(re.sub(r",\s*([}\]])", r"\1", json_blob))

    payload = None
    last_json_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_json_error = exc
            continue

    # Фолбэк для python-подобного формата (single quotes/True/None).
    if payload is None:
        for candidate in candidates:
            try:
                payload = ast.literal_eval(candidate)
                break
            except (ValueError, SyntaxError):
                continue

    if payload is None:
        raise ParsingError(f"Invalid JSON from model: {last_json_error}")

    if isinstance(payload, dict) and "test_cases" in payload:
        payload = payload["test_cases"]

    if not isinstance(payload, list):
        raise ParsingError("Model output must be a JSON array or {test_cases: []}")

    normalized = []
    skipped_incomplete = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            normalized.append(_normalize_test_case(item))
        except ParsingError:
            skipped_incomplete += 1
            continue

    if not normalized:
        if skipped_incomplete > 0:
            raise ParsingError("Model returned only incomplete testcase fields")
        raise ParsingError("No valid test cases found in model output")

    return normalized



def _tokenize(text: str) -> set:
    return {m.group(0).lower() for m in RE_WORD.finditer(text or "")}



def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)



def _normalize_source_ref(source_ref: str | None, context_blocks: List[Dict]) -> str | None:
    if not context_blocks:
        return None

    fallback = str(context_blocks[0].get("source_ref") or "").strip() or None
    raw = (source_ref or "").strip()
    if not raw:
        return fallback

    direct = RE_SOURCE_REF.search(raw)
    if direct:
        return f"page_{direct.group(1)}_offset_{direct.group(2)}"

    page_match = RE_PAGE.search(raw)
    if page_match:
        page = int(page_match.group(1))
        for block in context_blocks:
            if int(block.get("page") or -1) == page:
                candidate = str(block.get("source_ref") or "").strip()
                if candidate:
                    return candidate

    return fallback



def _case_grounding_score(case: Dict[str, Any], context_tokens: set) -> float:
    case_text = " ".join(
        [
            str(case.get("title", "")),
            str(case.get("preconditions", "")),
            str(case.get("expected_result", "")),
            " ".join(case.get("steps", [])),
        ]
    )
    case_tokens = _tokenize(case_text)
    if not case_tokens:
        return 0.0
    return len(case_tokens & context_tokens) / len(case_tokens)



def _is_duplicate(case: Dict[str, Any], selected: List[Dict[str, Any]]) -> bool:
    title = str(case.get("title", "")).strip().lower()
    sig = _tokenize(f"{case.get('title', '')} {case.get('expected_result', '')}")

    for existing in selected:
        existing_title = str(existing.get("title", "")).strip().lower()
        if title and title == existing_title:
            return True

        existing_sig = _tokenize(
            f"{existing.get('title', '')} {existing.get('expected_result', '')}"
        )
        if _jaccard(sig, existing_sig) >= 0.82:
            return True

    return False



def _enforce_required_scenarios(
    cases: List[Dict[str, Any]], include_negative: bool, include_boundary: bool
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []

    required = ["positive"]
    if include_negative:
        required.append("negative")
    if include_boundary:
        required.append("boundary")

    by_scenario = {"positive": [], "negative": [], "boundary": []}
    for case in cases:
        scenario = str(case.get("scenario_type", "positive")).lower()
        if scenario in by_scenario:
            by_scenario[scenario].append(case)

    selected: List[Dict[str, Any]] = []
    for scenario in required:
        if by_scenario.get(scenario):
            selected.append(by_scenario[scenario][0])
        else:
            warnings.append(f"No {scenario} testcases were generated")

    for case in cases:
        if case in selected:
            continue
        selected.append(case)

    return selected, warnings



def _postprocess_test_cases(
    parsed: List[Dict[str, Any]],
    context_blocks: List[Dict],
    max_cases: int,
    include_negative: bool,
    include_boundary: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    context_text = "\n".join(str(block.get("text") or "") for block in context_blocks)
    context_tokens = _tokenize(context_text)

    normalized = []
    for case in parsed:
        case["source_ref"] = _normalize_source_ref(case.get("source_ref"), context_blocks)
        normalized.append(case)

    selected: List[Dict[str, Any]] = []
    low_grounding_pool: List[Tuple[float, Dict[str, Any]]] = []
    removed_duplicates = 0
    removed_low_grounding = 0

    for case in normalized:
        if _is_duplicate(case, selected):
            removed_duplicates += 1
            continue

        grounding_score = _case_grounding_score(case, context_tokens)
        if grounding_score < 0.04:
            removed_low_grounding += 1
            low_grounding_pool.append((grounding_score, case))
            continue

        selected.append(case)

    required_count = 1 + (1 if include_negative else 0) + (1 if include_boundary else 0)
    if len(selected) < required_count and low_grounding_pool:
        needed = required_count - len(selected)
        low_grounding_pool.sort(key=lambda x: x[0], reverse=True)
        for _, case in low_grounding_pool[:needed]:
            selected.append(case)

    selected, scenario_warnings = _enforce_required_scenarios(
        selected,
        include_negative=include_negative,
        include_boundary=include_boundary,
    )

    final_cases = selected[:max_cases]
    scenario_distribution = {"positive": 0, "negative": 0, "boundary": 0}
    for case in final_cases:
        scenario = str(case.get("scenario_type", "positive")).lower()
        if scenario in scenario_distribution:
            scenario_distribution[scenario] += 1

    report = {
        "total_before_postprocess": len(parsed),
        "total_after_postprocess": len(final_cases),
        "removed_duplicates": removed_duplicates,
        "removed_low_grounding": removed_low_grounding,
        "scenario_distribution": scenario_distribution,
        "warnings": scenario_warnings,
    }
    return final_cases, report



def generate_test_cases(
    context_blocks: List[Dict],
    user_prompt: str,
    max_cases: int,
    include_negative: bool,
    include_boundary: bool,
) -> Tuple[List[Dict], Dict[str, Any]]:
    prompt = render_prompt(
        context_blocks=context_blocks,
        user_prompt=user_prompt,
        max_cases=max_cases,
        include_negative=include_negative,
        include_boundary=include_boundary,
    )
    raw = call_local_llm(prompt)
    parsed = _parse_test_cases(raw)
    final_cases, report = _postprocess_test_cases(
        parsed=parsed,
        context_blocks=context_blocks,
        max_cases=max_cases,
        include_negative=include_negative,
        include_boundary=include_boundary,
    )

    if not final_cases:
        # Не роняем генерацию полностью, если постпроцесс вычистил все:
        # возвращаем исходный нормализованный набор и фиксируем warning.
        fallback_cases = parsed[:max_cases]
        if not fallback_cases:
            raise ParsingError("No valid test cases remained after quality postprocessing")
        report["warnings"].append(
            "Postprocess removed all cases; returned parsed fallback cases"
        )
        return fallback_cases, report

    return final_cases, report
