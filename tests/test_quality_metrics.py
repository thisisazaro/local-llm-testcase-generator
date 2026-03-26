from __future__ import annotations

from evaluation.quality_metrics import compute_quality_metrics


def test_compute_quality_metrics_has_reasonable_bounds() -> None:
    generated = [
        {
            "title": "Успешный вход",
            "preconditions": "Пользователь активен",
            "steps": ["Открыть страницу", "Ввести данные", "Нажать вход"],
            "expected_result": "Авторизация успешна",
            "scenario_type": "positive",
            "source_ref": "page_1_offset_0",
            "priority": "high",
        },
        {
            "title": "Ошибка при неверном пароле",
            "preconditions": "Пользователь активен",
            "steps": ["Открыть страницу", "Ввести неверный пароль", "Нажать вход"],
            "expected_result": "Показана ошибка",
            "scenario_type": "negative",
            "source_ref": "page_1_offset_120",
            "priority": "high",
        },
    ]

    reference = [
        {
            "title": "Успешный вход пользователя",
            "expected_result": "Система выполняет авторизацию"
        }
    ]

    metrics = compute_quality_metrics(generated, reference)

    assert 0.0 <= metrics["overall_score"] <= 1.0
    assert metrics["total_cases"] == 2
    assert metrics["valid_cases"] == 2
