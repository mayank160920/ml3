"""
FastAPI application entry point for the CMSVS pipeline.

Endpoints
---------
POST /extract          — Extract entities from a single document
POST /validate         — Validate a document pair end-to-end
POST /validate/gt      — Validate and return M2 ground truth format
GET  /health           — Health check
GET  /configs          — List available configuration files
GET  /configs/{name}   — Get config details (sections + entities)

Run with:
    cd cmsvs
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

print(Path(__file__).parent.parent / ".env")

# ── Add src/ to Python path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# # ── Add src/ to Python path ───────────────────────────────────────────────────
# ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT / "src"))


from api.models import (
    ConfigCreateResponse,
    ConfigDetailResponse,
    ConfigListResponse,
    DeleteConfigResponse,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    MarkdownDetailResponse,
    MarkdownListResponse,
    ValidationResponse,
)
from api.services import CMSVSService

# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CMSVS API",
        description=(
            "Configurable Multimodal Semantic Validation System — "
            "Document Intelligence API"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow Streamlit frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach routes
    from api.routes import router
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )