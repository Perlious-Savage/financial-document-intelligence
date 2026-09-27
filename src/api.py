"""FastAPI service.

Three endpoints, because three are enough to demonstrate the pipeline:

  POST /extract                    document in, structured fields out
  POST /analyze                    the full pipeline: extract, validate, route
  GET  /documents/{id}/similar     semantic retrieval over the indexed corpus

Runs CPU-only. Extraction is delegated to whichever backend MODEL_BACKEND selects,
so the container starts and serves with no GPU present.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Must run before importing .extract, which reads MODEL_BACKEND at import time; loading
# afterwards silently ignores the backend configured in .env.
# Explicit path: find_dotenv() walks the call stack and fails in some embedding contexts.
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402

from .extract import BACKEND, extract  # noqa: E402
from .graph import analyze  # noqa: E402
from .schemas import AnalysisResult, ExtractedReceipt  # noqa: E402

ARTIFACTS = ROOT / "artifacts"

# Documents are untrusted input. Cap the size before anything parses them.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAGIC_PREFIXES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"%PDF-")

app = FastAPI(
    title="Financial Document Intelligence",
    description=(
        "Structured extraction from financial documents, with deterministic validation "
        "and confidence-based routing to human review."
    ),
    version="0.1.0",
)


async def _read_upload(file: UploadFile) -> bytes:
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "Empty file.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.")
    # Check the bytes, not the filename extension.
    if not payload.startswith(MAGIC_PREFIXES):
        raise HTTPException(415, "Unsupported file type. Expected JPEG, PNG or PDF.")
    return payload


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_backend": BACKEND}


@app.get("/metrics")
def metrics() -> dict:
    """Serve the measured results, or say plainly that none exist yet."""
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        return {"available": False, "detail": "No training run has completed yet."}
    return {"available": True, **json.loads(path.read_text())}


@app.post("/extract", response_model=ExtractedReceipt)
async def extract_endpoint(file: UploadFile = File(...)) -> ExtractedReceipt:
    payload = await _read_upload(file)
    return extract(file.filename or "upload", payload)


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_endpoint(file: UploadFile = File(...)) -> AnalysisResult:
    payload = await _read_upload(file)
    receipt = extract(file.filename or "upload", payload)
    return analyze(receipt)


@app.get("/documents/{doc_id}/similar")
def similar(doc_id: str, k: int = 5) -> dict:
    from .index import find_similar

    try:
        return {"doc_id": doc_id, "results": find_similar(doc_id, k)}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Retrieval failed: {exc}") from exc
