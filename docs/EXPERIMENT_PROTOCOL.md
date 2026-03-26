# Протокол эксперимента

## Гипотеза
Локальная LLM с retrieval по аналитическим документам формирует тест-кейсы с качеством не ниже порогового уровня `overall_score >= 0.70`.

## Дизайн эксперимента
1. Подготовить 3-10 документов требований (PDF) из разных доменов.
2. Для каждого документа определить эталонный набор тест-кейсов (`reference_test_cases`) от QA-эксперта.
3. Сгенерировать тест-кейсы сервисом с фиксированными параметрами.
4. Сравнить результат с эталоном по метрикам.

## Метрики
- `structural_validity_rate`: доля кейсов, где соблюдена структура.
- `traceability_rate`: доля кейсов с корректной ссылкой `source_ref`.
- `redundancy_rate`: доля дублей по заголовкам.
- `semantic_coverage`: семантическая близость к эталону (Jaccard по токенам title+expected_result).
- `overall_score`: интегральный показатель 0..1.

## Команды
```bash
# 1) запуск API
uvicorn app.main:app --reload

# 2) после загрузки документов и получения file_id — подготовить датасет
cp evaluation/experiment_dataset.template.json evaluation/experiment_dataset.json

# 3) запуск эксперимента
python3 evaluation/run_experiment.py \
  --dataset evaluation/experiment_dataset.json \
  --api-base-url http://localhost:8000 \
  --output-dir evaluation/out

# 4) генерация отчета
python3 evaluation/build_report.py \
  --summary evaluation/out/summary.json \
  --results evaluation/out/results.json \
  --output evaluation/out/report.md
```

## Критерий успеха
- `overall_score >= 0.70` в среднем по сценариям;
- `structural_validity_rate >= 0.95`;
- `traceability_rate >= 0.90`.
