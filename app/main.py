from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import SETTINGS
from app.generator import LLMError, ParsingError, generate_test_cases
from app.ingest import build_semantic_chunks, extract_pdf_text, save_upload
from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    UploadResponse,
)
from app.vector_store import COLLECTION_NAME, add_chunks, count_chunks, search
from evaluation.quality_metrics import compute_quality_metrics

app = FastAPI(title=SETTINGS.app_name)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": SETTINGS.llm_provider,
        "llm_model": SETTINGS.llm_model_name,
        "embedding_provider": SETTINGS.embedding_provider,
        "vector_collection": COLLECTION_NAME,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        file_id, path = save_upload(file_bytes, filename)
        _, pages = extract_pdf_text(path)
        chunks = build_semantic_chunks(
            pages=pages,
            max_chars=SETTINGS.chunk_max_chars,
            overlap=SETTINGS.chunk_overlap,
            min_chunk_chars=SETTINGS.min_chunk_chars,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {exc}") from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in the PDF")

    add_chunks(file_id=file_id, chunks=chunks)
    return UploadResponse(file_id=file_id, chunks=len(chunks), status="indexed")


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    indexed_chunks = count_chunks(req.file_id)
    if indexed_chunks == 0:
        raise HTTPException(status_code=404, detail="file_id not found in vector index")

    top_k = min(req.top_k, indexed_chunks)
    context_blocks = search(file_id=req.file_id, query=req.user_prompt, top_k=top_k)
    if not context_blocks:
        raise HTTPException(status_code=404, detail="No context found for this file_id")

    try:
        test_cases, generation_report = generate_test_cases(
            context_blocks=context_blocks,
            user_prompt=req.user_prompt,
            max_cases=req.max_cases,
            include_negative=req.include_negative,
            include_boundary=req.include_boundary,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ParsingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    quality_metrics = compute_quality_metrics(generated_cases=test_cases)
    quality_report = {**quality_metrics, **generation_report}

    return GenerateResponse(
        file_id=req.file_id,
        test_cases=test_cases,
        context_used=len(context_blocks),
        model_provider=SETTINGS.llm_provider,
        model_name=SETTINGS.llm_model_name,
        quality_report=quality_report,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    metrics = compute_quality_metrics(
        generated_cases=req.generated_cases,
        reference_cases=req.reference_cases,
    )
    return EvaluateResponse(metrics=metrics)
