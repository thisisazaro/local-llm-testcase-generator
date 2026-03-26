from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class UploadResponse(BaseModel):
    file_id: str
    chunks: int
    status: str


class GenerateRequest(BaseModel):
    file_id: str = Field(..., min_length=8)
    user_prompt: str = Field(default="Сформируй полный набор тест-кейсов", min_length=3)
    top_k: int = Field(default=8, ge=1, le=30)
    max_cases: int = Field(default=20, ge=1, le=100)
    include_negative: bool = True
    include_boundary: bool = True


class TestCase(BaseModel):
    title: str = Field(..., min_length=3)
    preconditions: Optional[str] = None
    steps: List[str]
    expected_result: str = Field(..., min_length=3)
    scenario_type: str
    source_ref: Optional[str] = None
    priority: Optional[str] = None

    @validator("steps")
    def steps_must_not_be_empty(cls, value: List[str]) -> List[str]:
        clean = [step.strip() for step in value if isinstance(step, str) and step.strip()]
        if not clean:
            raise ValueError("steps must contain at least one non-empty step")
        return clean


class GenerateResponse(BaseModel):
    file_id: str
    test_cases: List[TestCase]
    context_used: int
    model_provider: str
    model_name: str
    quality_report: Dict


class EvaluateRequest(BaseModel):
    generated_cases: List[Dict]
    reference_cases: Optional[List[Dict]] = None


class EvaluateResponse(BaseModel):
    metrics: Dict
