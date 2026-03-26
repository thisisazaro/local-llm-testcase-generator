# Local LLM Test Case Generator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19229746.svg)](https://doi.org/10.5281/zenodo.19229746)

Локальный сервис генерации тест-кейсов по аналитическим документам (PDF) с RAG-пайплайном и контролем качества.

Проект предназначен для демонстрации на защите и для пилотного использования в QA-процессе:
- загрузка документа требований;
- извлечение и индексирование контекста;
- генерация структурированных тест-кейсов локальной LLM;
- автоматическая оценка качества результата.

## Ключевые возможности
- FastAPI API: `/health`, `/upload`, `/generate`, `/evaluate`.
- Полностью локальный inference без облачных LLM API.
- Retrieval по PDF (Chroma + embeddings).
- Устойчивый парсинг ответа модели (включая «почти JSON» ответы).
- Постобработка качества:
  - дедупликация;
  - нормализация `source_ref`;
  - фильтрация слабой привязки к контексту;
  - контроль сценариев `positive/negative/boundary`.
- Встроенные метрики качества в `quality_report`.
- Экспериментальный контур и автогенерация отчета.
- Контур дообучения LoRA/QLoRA.

## Архитектура
1. `POST /upload`
- принимает PDF;
- извлекает текст;
- режет на чанки;
- индексирует чанки в векторном хранилище.

2. `POST /generate`
- извлекает релевантные чанки (`top_k`);
- формирует prompt;
- вызывает локальную LLM;
- парсит/нормализует JSON;
- возвращает `test_cases` + `quality_report`.

3. `POST /evaluate`
- принимает произвольный набор `generated_cases`;
- считает метрики качества;
- опционально сравнивает с `reference_cases`.

## Структура репозитория
- `app/` — backend приложения.
- `evaluation/` — скрипты эксперимента и отчёта.
- `training/` — подготовка данных и обучение LoRA/QLoRA.
- `docs/` — материалы для защиты.
- `configs/` — docker compose.

## Требования
- Python 3.9+
- macOS/Linux
- Для локальной LLM:
  - Ollama (рекомендуется для демонстрации)
  - или vLLM/OpenAI-compatible endpoint

## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Запуск API:
```bash
uvicorn app.main:app --reload
```

Swagger UI:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Проверка:
```bash
curl http://127.0.0.1:8000/health
```

## Запуск с локальной LLM (Ollama)

Установить и запустить Ollama:
```bash
ollama serve
```

Скачать модель:
```bash
ollama pull qwen2.5:7b-instruct
```

Переменные окружения:
```bash
export LLM_PROVIDER=ollama
export LLM_MODEL_NAME=qwen2.5:7b-instruct
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export LLM_TEMPERATURE=0.1
```

Эмбеддинги (рекомендуется):
```bash
pip install sentence-transformers
export EMBEDDING_PROVIDER=sentence_transformers
export EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Важно:
- Если в `/health` видно `embedding_provider: hash`, значит fallback-режим, качество retrieval ниже.

## E2E сценарий через Swagger
1. `GET /health`
- убедиться, что:
  - `llm_provider = ollama`;
  - `embedding_provider = sentence_transformers`.

2. `POST /upload`
- загрузить PDF;
- сохранить `file_id` из ответа.

3. `POST /generate`
- передать `file_id`;
- получить `test_cases` и `quality_report`.

Рекомендуемый body:
```json
{
  "file_id": "<FILE_ID_FROM_UPLOAD>",
  "user_prompt": "Сформируй 12 атомарных тест-кейсов по API: строго 4 positive, 4 negative, 4 boundary; без дублей title; у каждого кейса 3-5 шагов и конкретные входные данные; для negative укажи ожидаемый тип/код ошибки; ничего не выдумывать вне контекста.",
  "top_k": 12,
  "max_cases": 12,
  "include_negative": true,
  "include_boundary": true
}
```

4. `POST /evaluate`
- передать только массив `test_cases` из `/generate`:
```json
{
  "generated_cases": [
    {
      "title": "...",
      "preconditions": "...",
      "steps": ["..."],
      "expected_result": "...",
      "scenario_type": "positive",
      "source_ref": "page_1_offset_0",
      "priority": "medium"
    }
  ],
  "reference_cases": []
}
```

## Интерпретация quality_report
Ключевые поля:
- `overall_score` — интегральная оценка (0..1).
- `negative_ratio` / `boundary_ratio` — баланс типов сценариев.
- `avg_steps` — средняя детализация кейса.
- `removed_duplicates` — сколько дублей удалено.
- `warnings` — проблемные зоны генерации.

Практические цели для защиты:
- `overall_score >= 0.82`
- `negative_ratio >= 0.25`
- `boundary_ratio >= 0.25`
- `avg_steps >= 3`

## Экспериментальный контур
```bash
cp evaluation/experiment_dataset.template.json evaluation/experiment_dataset.json
# Заполните file_id и reference_test_cases

python3 evaluation/run_experiment.py \
  --dataset evaluation/experiment_dataset.json \
  --api-base-url http://127.0.0.1:8000 \
  --output-dir evaluation/out

python3 evaluation/build_report.py \
  --summary evaluation/out/summary.json \
  --results evaluation/out/results.json \
  --output evaluation/out/report.md
```

Артефакты:
- `evaluation/out/results.json`
- `evaluation/out/summary.json`
- `evaluation/out/report.md`

## Дообучение (LoRA/QLoRA)
Подробно: `training/README.md`

Быстрый путь:
```bash
python3 training/prepare_dataset.py --raw training/raw_pairs.json --out training/train.jsonl
python3 training/qlora_train.py --base-model Qwen/Qwen2.5-7B-Instruct --dataset training/train.jsonl --output-dir training/adapters/testcase-lora --epochs 2 --use-4bit
```

## Docker
```bash
docker compose -f configs/docker-compose.yml up --build
```

## Troubleshooting

### 1) `422 json_invalid` на `/generate`
Причина: невалидный JSON в request body (обычно переносы строк в `user_prompt`).
Решение:
- передавать `user_prompt` одной строкой;
- или экранировать переносы как `\n`.

### 2) `422 Model returned incomplete testcase fields`
Причина: LLM вернула неполные элементы кейсов.
Решение:
- повторить запрос с более конкретным prompt;
- уменьшить `temperature` до `0.1`;
- увеличить `top_k` до `12`.

### 3) `embedding_provider: hash` в `/health`
Причина: не установлен `sentence-transformers`.
Решение:
```bash
pip install sentence-transformers
export EMBEDDING_PROVIDER=sentence_transformers
```
Перезапустить API и заново выполнить `/upload`.

### 4) Низкий `overall_score` при малом числе кейсов
Причина: нерепрезентативный sample (`total_cases` 1-3).
Решение:
- генерировать 10-12 кейсов;
- передавать полный массив в `/evaluate`.

## Соответствие требованиям (локальная LLM)
- inference выполняется локально;
- внешний облачный LLM API не требуется;
- для строгого академического соответствия пункту «самостоятельно обученная и настроенная» рекомендуется показать LoRA-дообучение и сравнение `base vs tuned`.

## Доступ клиенту к демо
Рекомендуемый production-lite путь:
1. Код в GitHub (`private` репозиторий).
2. Сервис на вашем сервере (uvicorn/docker + ollama).
3. Доступ через Tailscale (`serve`/`funnel`) или другой tunnel.
4. Клиенту отправить:
- URL Swagger (`/docs`);
- краткий сценарий: `upload -> generate -> evaluate`;
- ограничения по данным (не загружать чувствительные документы без auth).

## Материалы для защиты
- `docs/PROJECT_ARCHITECTURE.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RISK_AND_ECONOMICS.md`
- `docs/DEFENSE_DEMO_SCRIPT.md`
