"""
Document extraction endpoint for STREAM.

This module provides the API endpoint for extracting content (text + images)
from uploaded documents. The frontend calls this endpoint when a user
attaches a document (PDF, DOCX, XLSX, PPTX, or text/code file) to a chat.

ARCHITECTURE
============

The extraction flow:

    User drops file in chat input
        │
        ▼
    Frontend sends file to POST /v1/documents/extract
        │
        ▼
    Backend extracts text + images (this module)
        │
        ▼
    Frontend receives structured content parts
        │
        ▼
    Frontend shows attachment chip with preview
        │
        ▼
    User sends message → extracted content + question
    are assembled into OpenAI multimodal message format
        │
        ▼
    Sent via existing POST /v1/chat/completions pipeline

WHY BACKEND EXTRACTION?
=======================

We extract on the backend rather than in the browser for several reasons:

1. Binary formats: PDF/DOCX/XLSX/PPTX require specialized libraries
   (PyMuPDF, python-docx, etc.) that don't run in the browser.

2. Security: Parsing complex binary formats in the browser would require
   loading large WASM bundles and could expose XSS attack vectors.

3. Consistency: Same extraction code runs whether the user is in
   desktop mode or server (Docker) mode.

4. Image extraction: Embedded images in documents need to be pulled
   out, compressed, and base64-encoded — easier on the server.

For simple text files (.txt, .py, .md), we COULD do it client-side with
FileReader, but using the backend for everything keeps the architecture
simple and the code DRY.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from stream.middleware.utils.document_extractor import (
    MAX_DOCUMENT_SIZE,
    MAX_DOCUMENTS_PER_MESSAGE,
    SUPPORTED_EXTENSIONS,
    extract_document,
)

logger = logging.getLogger(__name__)

# Create FastAPI router for document-related endpoints
router = APIRouter()


@router.post("/documents/extract")
async def extract_document_endpoint(file: UploadFile = File(...)):
    """Extract text and images from an uploaded document.

    This endpoint accepts a single file upload and returns the extracted
    content as a structured JSON response. The frontend calls this once
    per attached document (up to MAX_DOCUMENTS_PER_MESSAGE per chat message).

    Request:
        POST /v1/documents/extract
        Content-Type: multipart/form-data
        Body: file=<uploaded file>

    Response (200 OK):
        {
            "filename": "report.pdf",
            "file_type": "pdf",
            "file_size": 1234567,
            "content_parts": [
                {"type": "text", "text": "Page 1 content...", "image_base64": null, "image_mime": null},
                {"type": "image", "text": null, "image_base64": "...", "image_mime": "image/jpeg"}
            ],
            "text_preview": "First 500 chars of text...",
            "total_text_length": 15234,
            "image_count": 3,
            "page_count": 12,
            "warnings": ["Sheet 'Data' has 1500 rows. Only first 500 included."]
        }

    Errors:
        400: Unsupported file type or extraction failure
        413: File too large (> MAX_DOCUMENT_SIZE)
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    # Validate file extension before reading the entire file into memory
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Supported formats: text, code, PDF, DOCX, XLSX, PPTX."
            ),
        )

    # Read file contents
    file_bytes = await file.read()

    # Check file size (FastAPI/Starlette doesn't enforce this by default)
    if len(file_bytes) > MAX_DOCUMENT_SIZE:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large: {size_mb:.1f} MB. "
                f"Maximum allowed size is {MAX_DOCUMENT_SIZE / (1024 * 1024):.0f} MB."
            ),
        )

    try:
        result = await extract_document(file_bytes, file.filename)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception(f"[Documents] Extraction failed for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract document: {e}",
        ) from e


@router.get("/documents/supported-formats")
async def get_supported_formats():
    """Return the list of supported file extensions and limits.

    The frontend uses this to configure the file input's accept attribute
    and to display supported formats in the UI.

    Response (200 OK):
        {
            "extensions": [".txt", ".pdf", ".docx", ...],
            "max_file_size_mb": 25,
            "max_files_per_message": 10
        }
    """
    return {
        "extensions": sorted(SUPPORTED_EXTENSIONS),
        "max_file_size_mb": MAX_DOCUMENT_SIZE / (1024 * 1024),
        "max_files_per_message": MAX_DOCUMENTS_PER_MESSAGE,
    }
