"""
------------
All FastAPI route definitions.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.models import (
    ConfigDetailResponse,
    ConfigListResponse,
    ExtractionResponse,
    GroundTruthResponse,
    HealthResponse,
    ValidationResponse,
)
from api.services import CMSVSService

router = APIRouter()


def get_service() -> CMSVSService:
    return CMSVSService.get()


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health and configuration status."""
    svc = get_service()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        nvidia_key_set=svc.nvidia_key_set,
        configs_available=svc.list_configs(),
    )


# ── Configs ────────────────────────────────────────────────────────────────────

@router.get("/configs", response_model=ConfigListResponse, tags=["Config"])
async def list_configs() -> ConfigListResponse:
    """List all available configuration files."""
    svc = get_service()
    return ConfigListResponse(configs=svc.list_configs())


@router.get(
    "/configs/{config_name}",
    response_model=ConfigDetailResponse,
    tags=["Config"],
)
async def get_config(config_name: str) -> ConfigDetailResponse:
    """
    Get full details of a configuration file including all sections
    and entity definitions.
    """
    svc = get_service()
    try:
        return svc.get_config_detail(config_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Extraction ─────────────────────────────────────────────────────────────────

@router.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_entities(
    file: UploadFile = File(..., description="PDF or image document"),
    config_name: str = Form(..., description="Config name (e.g. funsd_ner_config)"),
    confidence_threshold: float = Form(
        default=0.75,
        description="Minimum confidence for review flagging",
    ),
) -> ExtractionResponse:
    """
    Extract named entities from a single document.

    Accepts PDF or image files. Routes to the appropriate pipeline
    (RAG for PDF, direct MLLM for images) automatically.
    """
    svc = get_service()

    if not svc.nvidia_key_set:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA_API_KEY is not configured on the server.",
        )

    # Save upload to temp file preserving extension
    suffix = Path(file.filename or "doc.png").suffix.lower()
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return svc.extract_document(
            file_path=tmp_path,
            config_name=config_name,
            confidence_threshold=confidence_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Validation ──────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_documents(
    doc_a: UploadFile = File(..., description="Document A (PDF or image)"),
    doc_b: UploadFile = File(..., description="Document B (PDF or image)"),
    config_name: str = Form(..., description="Config name"),
    doc_a_name: str = Form(default="Document A", description="Label for Document A"),
    doc_b_name: str = Form(default="Document B", description="Label for Document B"),
    confidence_threshold: float = Form(
        default=0.75,
        description="Minimum confidence for review flagging",
    ),
) -> ValidationResponse:
    """
    Validate a document pair end-to-end.

    Extracts entities from both documents then runs section-wise
    Chain-of-Thought semantic validation.

    Returns per-entity validation status, reasoning, and confidence scores.
    """
    svc = get_service()

    if not svc.nvidia_key_set:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA_API_KEY is not configured on the server.",
        )

    suffix_a = Path(doc_a.filename or "doc_a.png").suffix.lower()
    suffix_b = Path(doc_b.filename or "doc_b.png").suffix.lower()

    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a) as ta,
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b) as tb,
    ):
        ta.write(await doc_a.read())
        tb.write(await doc_b.read())
        path_a, path_b = ta.name, tb.name

    try:
        return svc.validate_documents(
            doc_a_path=path_a,
            doc_b_path=path_b,
            config_name=config_name,
            doc_a_name=doc_a_name or doc_a.filename or "Document A",
            doc_b_name=doc_b_name or doc_b.filename or "Document B",
            confidence_threshold=confidence_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(path_a).unlink(missing_ok=True)
        Path(path_b).unlink(missing_ok=True)


@router.post(
    "/validate/gt",
    response_model=GroundTruthResponse,
    tags=["Validation"],
)
async def validate_documents_groundtruth(
    doc_a: UploadFile = File(..., description="Document A (PDF or image)"),
    doc_b: UploadFile = File(..., description="Document B (PDF or image)"),
    config_name: str = Form(..., description="Config name"),
    doc_a_name: str = Form(default="Document A"),
    doc_b_name: str = Form(default="Document B"),
    confidence_threshold: float = Form(default=0.75),
) -> GroundTruthResponse:
    """
    Validate a document pair and return output in M2 ground truth format.

    Used for direct comparison against dataset ground truth files
    for M5 evaluation metrics.
    """
    svc = get_service()

    if not svc.nvidia_key_set:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA_API_KEY is not configured on the server.",
        )

    suffix_a = Path(doc_a.filename or "doc_a.png").suffix.lower()
    suffix_b = Path(doc_b.filename or "doc_b.png").suffix.lower()

    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a) as ta,
        tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b) as tb,
    ):
        ta.write(await doc_a.read())
        tb.write(await doc_b.read())
        path_a, path_b = ta.name, tb.name

    try:
        return svc.validate_documents_gt(
            doc_a_path=path_a,
            doc_b_path=path_b,
            config_name=config_name,
            doc_a_name=doc_a_name or doc_a.filename or "Document A",
            doc_b_name=doc_b_name or doc_b.filename or "Document B",
            confidence_threshold=confidence_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(path_a).unlink(missing_ok=True)
        Path(path_b).unlink(missing_ok=True)