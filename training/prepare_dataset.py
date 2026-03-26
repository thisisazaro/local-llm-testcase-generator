from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROMPT_TEMPLATE = (
    "Контекст требований:\n{context}\n\n"
    "Задача:\n{instruction}\n\n"
    "Требуемый формат ответа:\nJSON-массив тест-кейсов"
)



def _load_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array")
    return data



def _validate_item(item: Dict[str, Any], idx: int) -> None:
    for field in ("context", "instruction", "output"):
        if field not in item:
            raise ValueError(f"Item #{idx} missing required field: {field}")



def _to_record(item: Dict[str, Any]) -> Dict[str, str]:
    instruction = PROMPT_TEMPLATE.format(
        context=str(item["context"]).strip(),
        instruction=str(item["instruction"]).strip(),
    )
    output = json.dumps(item["output"], ensure_ascii=False)
    return {"instruction": instruction, "output": output}



def prepare_dataset(raw_path: Path, output_path: Path) -> None:
    items = _load_items(raw_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items, start=1):
            _validate_item(item, idx)
            record = _to_record(item)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Prepared {len(items)} records -> {output_path}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare JSONL dataset for SFT/LoRA")
    parser.add_argument(
        "--raw",
        default=Path("training/raw_pairs.json"),
        type=Path,
        help="Path to source pairs JSON",
    )
    parser.add_argument(
        "--out",
        default=Path("training/train.jsonl"),
        type=Path,
        help="Path to output jsonl",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    prepare_dataset(raw_path=args.raw, output_path=args.out)
