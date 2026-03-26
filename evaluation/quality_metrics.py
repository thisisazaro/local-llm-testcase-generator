from __future__ import annotations

import re
from statistics import mean
from typing import Dict, List

RE_WORD = re.compile(r"[a-zA-Zа-яА-Я0-9]+")
REQUIRED_FIELDS = {
    "title",
    "preconditions",
    "steps",
    "expected_result",
    "scenario_type",
    "source_ref",
    "priority",
}



def _tokenize(text: str) -> set:
    return {m.group(0).lower() for m in RE_WORD.finditer(text or "")}



def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)



def _is_structurally_valid(case: Dict) -> bool:
    if not isinstance(case, dict):
        return False

    if not REQUIRED_FIELDS.issubset(case.keys()):
        return False

    if not str(case.get("title", "")).strip():
        return False
    if not str(case.get("expected_result", "")).strip():
        return False

    steps = case.get("steps")
    if not isinstance(steps, list) or not any(str(step).strip() for step in steps):
        return False

    scenario_type = str(case.get("scenario_type", "")).strip().lower()
    if scenario_type not in {"positive", "negative", "boundary"}:
        return False

    priority = str(case.get("priority", "")).strip().lower()
    if priority not in {"low", "medium", "high"}:
        return False

    return True



def _redundancy_rate(cases: List[Dict]) -> float:
    if not cases:
        return 0.0

    normalized_titles = [str(case.get("title", "")).strip().lower() for case in cases]
    unique_count = len(set(t for t in normalized_titles if t))
    total = len(normalized_titles)
    if total == 0:
        return 0.0

    duplicates = max(0, total - unique_count)
    return duplicates / total



def _semantic_coverage(generated: List[Dict], reference: List[Dict]) -> float:
    if not generated or not reference:
        return 0.0

    ref_vectors = [
        _tokenize(f"{item.get('title', '')} {item.get('expected_result', '')}") for item in reference
    ]

    scores = []
    for gen in generated:
        gen_vector = _tokenize(f"{gen.get('title', '')} {gen.get('expected_result', '')}")
        if not gen_vector:
            scores.append(0.0)
            continue
        best = max((_jaccard(gen_vector, ref) for ref in ref_vectors), default=0.0)
        scores.append(best)

    return mean(scores) if scores else 0.0



def compute_quality_metrics(generated_cases: List[Dict], reference_cases: List[Dict] | None = None) -> Dict:
    total = len(generated_cases)
    valid_flags = [_is_structurally_valid(case) for case in generated_cases]
    valid_count = sum(1 for flag in valid_flags if flag)

    valid_cases = [case for case, flag in zip(generated_cases, valid_flags) if flag]

    avg_steps = mean([len(case.get("steps", [])) for case in valid_cases]) if valid_cases else 0.0
    traceable = sum(1 for case in valid_cases if str(case.get("source_ref", "")).strip())
    traceability_rate = (traceable / len(valid_cases)) if valid_cases else 0.0

    scenario_counts = {"positive": 0, "negative": 0, "boundary": 0}
    for case in valid_cases:
        scenario = str(case.get("scenario_type", "")).strip().lower()
        if scenario in scenario_counts:
            scenario_counts[scenario] += 1

    total_valid = len(valid_cases)
    negative_ratio = scenario_counts["negative"] / total_valid if total_valid else 0.0
    boundary_ratio = scenario_counts["boundary"] / total_valid if total_valid else 0.0

    redundancy = _redundancy_rate(valid_cases)

    semantic_coverage = 0.0
    if reference_cases is not None:
        semantic_coverage = _semantic_coverage(valid_cases, reference_cases)

    structural_rate = (valid_count / total) if total else 0.0

    # Композитный score 0..1
    score_components = [
        0.35 * structural_rate,
        0.20 * traceability_rate,
        0.20 * (1.0 - redundancy),
        0.10 * min(avg_steps / 5.0, 1.0),
        0.15 * semantic_coverage,
    ]
    overall_score = sum(score_components)

    return {
        "total_cases": total,
        "valid_cases": valid_count,
        "structural_validity_rate": round(structural_rate, 4),
        "avg_steps": round(avg_steps, 2),
        "traceability_rate": round(traceability_rate, 4),
        "negative_ratio": round(negative_ratio, 4),
        "boundary_ratio": round(boundary_ratio, 4),
        "redundancy_rate": round(redundancy, 4),
        "semantic_coverage": round(semantic_coverage, 4),
        "overall_score": round(overall_score, 4),
    }
