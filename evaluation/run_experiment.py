from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import requests

from quality_metrics import compute_quality_metrics



def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")
    return data



def _call_generate(api_base_url: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "file_id": scenario["file_id"],
        "user_prompt": scenario.get("prompt", "Сформируй полный набор тест-кейсов"),
        "top_k": scenario.get("top_k", 8),
        "max_cases": scenario.get("max_cases", 20),
        "include_negative": scenario.get("include_negative", True),
        "include_boundary": scenario.get("include_boundary", True),
    }

    url = f"{api_base_url.rstrip('/')}/generate"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()



def run(dataset_path: Path, api_base_url: str, output_dir: Path) -> None:
    scenarios = _load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    score_values = []

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id", f"scenario_{len(results)+1}")
        reference_cases = scenario.get("reference_test_cases")

        row: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "file_id": scenario.get("file_id"),
            "prompt": scenario.get("prompt"),
            "status": "ok",
        }

        try:
            response = _call_generate(api_base_url=api_base_url, scenario=scenario)
            generated_cases = response.get("test_cases", [])

            metrics = compute_quality_metrics(
                generated_cases=generated_cases,
                reference_cases=reference_cases,
            )

            row["response"] = response
            row["metrics"] = metrics
            score_values.append(metrics["overall_score"])

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)

        results.append(row)

    successful = [item for item in results if item.get("status") == "ok"]

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_path": str(dataset_path),
        "api_base_url": api_base_url,
        "total_scenarios": len(results),
        "successful_scenarios": len(successful),
        "failed_scenarios": len(results) - len(successful),
        "mean_overall_score": round(mean(score_values), 4) if score_values else 0.0,
        "results_file": str((output_dir / "results.json").resolve()),
    }

    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run testcase generation experiment")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to experiment dataset JSON",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="FastAPI base URL",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("evaluation/out"),
        type=Path,
        help="Where to write results",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(dataset_path=args.dataset, api_base_url=args.api_base_url, output_dir=args.output_dir)
