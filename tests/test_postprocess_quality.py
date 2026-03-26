from __future__ import annotations

from fastapi.testclient import TestClient

from app.generator import _postprocess_test_cases
from app.main import app



def test_postprocess_removes_duplicates_and_normalizes_source_ref() -> None:
    parsed = [
        {
            "title": "Успешный вход",
            "preconditions": "Пользователь зарегистрирован",
            "steps": ["Открыть форму", "Ввести логин и пароль"],
            "expected_result": "Вход выполнен",
            "scenario_type": "positive",
            "source_ref": "page_3_offset_10; page=3",
            "priority": "high",
        },
        {
            "title": "Успешный вход",
            "preconditions": "Пользователь зарегистрирован",
            "steps": ["Открыть форму", "Ввести логин и пароль"],
            "expected_result": "Пользователь вошел в систему",
            "scenario_type": "positive",
            "source_ref": "page=3",
            "priority": "high",
        },
        {
            "title": "Ошибка входа",
            "preconditions": "Пользователь существует",
            "steps": ["Ввести неверный пароль"],
            "expected_result": "Показана ошибка",
            "scenario_type": "negative",
            "source_ref": "unknown",
            "priority": "medium",
        },
        {
            "title": "Completely unrelated synthetic case",
            "preconditions": "Alpha Beta Gamma",
            "steps": ["Omega Delta"],
            "expected_result": "Zeta Kappa",
            "scenario_type": "boundary",
            "source_ref": "page_99_offset_0",
            "priority": "low",
        },
    ]

    context_blocks = [
        {
            "text": "Пользователь вводит логин и пароль. При неверном пароле система показывает ошибку.",
            "source_ref": "page_3_offset_10",
            "page": 3,
        }
    ]

    final_cases, report = _postprocess_test_cases(
        parsed=parsed,
        context_blocks=context_blocks,
        max_cases=10,
        include_negative=True,
        include_boundary=False,
    )

    assert len(final_cases) == 2
    assert report["removed_duplicates"] >= 1
    assert report["removed_low_grounding"] >= 1
    assert all(case["source_ref"] == "page_3_offset_10" for case in final_cases)



def test_evaluate_endpoint_returns_metrics() -> None:
    client = TestClient(app)
    payload = {
        "generated_cases": [
            {
                "title": "Успешный вход",
                "preconditions": "Пользователь активен",
                "steps": ["Открыть форму", "Ввести валидные данные", "Нажать вход"],
                "expected_result": "Пользователь авторизован",
                "scenario_type": "positive",
                "source_ref": "page_1_offset_0",
                "priority": "high",
            }
        ]
    }

    resp = client.post("/evaluate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body
    assert "overall_score" in body["metrics"]
