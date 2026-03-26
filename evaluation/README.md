# Экспериментальная оценка качества

## Входные данные
`evaluation/experiment_dataset.json` (скопируйте из шаблона `experiment_dataset.template.json`).

## Запуск
```bash
python3 evaluation/run_experiment.py \
  --dataset evaluation/experiment_dataset.json \
  --api-base-url http://localhost:8000 \
  --output-dir evaluation/out
```

## Отчёт
```bash
python3 evaluation/build_report.py \
  --summary evaluation/out/summary.json \
  --results evaluation/out/results.json \
  --output evaluation/out/report.md
```

## Что оценивается
- структурная валидность тест-кейсов;
- трассируемость к исходному документу;
- дублирование;
- базовая семантическая близость к эталону.
