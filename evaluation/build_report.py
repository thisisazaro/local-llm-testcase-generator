from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List



def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def _render_scenario_row(item: Dict[str, Any]) -> str:
    scenario_id = item.get("scenario_id", "-")
    status = item.get("status", "-")
    if status != "ok":
        return f"| {scenario_id} | error | - | - | - | {item.get('error', '')} |"

    metrics = item.get("metrics", {})
    return (
        f"| {scenario_id} | ok | {metrics.get('overall_score', 0)} | "
        f"{metrics.get('structural_validity_rate', 0)} | "
        f"{metrics.get('traceability_rate', 0)} | - |"
    )



def build_report(summary_path: Path, results_path: Path, output_path: Path) -> None:
    summary = _load_json(summary_path)
    results: List[Dict[str, Any]] = _load_json(results_path)

    lines = []
    lines.append("# Отчёт по эксперименту генерации тест-кейсов")
    lines.append("")
    lines.append(f"Сформирован: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Сводка")
    lines.append("")
    lines.append(f"- Всего сценариев: {summary.get('total_scenarios', 0)}")
    lines.append(f"- Успешно: {summary.get('successful_scenarios', 0)}")
    lines.append(f"- Ошибок: {summary.get('failed_scenarios', 0)}")
    lines.append(f"- Средний интегральный score: {summary.get('mean_overall_score', 0)}")
    lines.append("")
    lines.append("## Детализация по сценариям")
    lines.append("")
    lines.append("| Scenario | Status | Overall score | Structural | Traceability | Notes |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for item in results:
        lines.append(_render_scenario_row(item))

    lines.append("")
    lines.append("## Интерпретация")
    lines.append("")
    lines.append(
        "- Score >= 0.80: высокое качество прототипа для пилотного внедрения с QA-ревью."
    )
    lines.append(
        "- Score 0.65..0.79: приемлемый уровень, рекомендуется усилить retrieval/prompt и пост-валидацию."
    )
    lines.append(
        "- Score < 0.65: требуется доработка датасета, модели и правил генерации перед внедрением."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build markdown report from experiment results")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/out/report.md"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    build_report(summary_path=args.summary, results_path=args.results, output_path=args.output)
