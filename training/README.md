# Контур дообучения (LoRA/QLoRA)

## 1) Подготовка исходных пар
Создайте файл `training/raw_pairs.json` по шаблону `training/raw_pairs.example.json`.

## 2) Подготовка JSONL
```bash
python3 training/prepare_dataset.py --raw training/raw_pairs.json --out training/train.jsonl
```

## 3) Запуск обучения
```bash
python3 training/qlora_train.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --dataset training/train.jsonl \
  --output-dir training/adapters/testcase-lora \
  --epochs 2 \
  --use-4bit
```

## Примечания
- Для обучения нужен GPU.
- Результат: директория адаптера LoRA, которую можно подключить к локальному инференсу.
