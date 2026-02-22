# Document Upload — Technical Report

## Overview

STREAM supports uploading documents (PDF, Word, Excel, PowerPoint, and text/code files) directly into the chat conversation. When a user attaches a document, STREAM extracts its content — text and embedded images — and sends it to the LLM in a format it can understand.

This document explains **how** the feature works, **why** each design decision was made, and **where** to find the relevant code.

---

## Table of Contents

1. [Architecture](#architecture)
2. [The Interleaving Problem](#the-interleaving-problem)
3. [Supported Formats](#supported-formats)
4. [Backend: Document Extraction](#backend-document-extraction)
5. [Backend: API Endpoint](#backend-api-endpoint)
6. [Frontend: Upload Flow](#frontend-upload-flow)
7. [Frontend: Chat Display](#frontend-chat-display)
8. [Lakeshore Size Warnings](#lakeshore-size-warnings)
9. [Scanned PDF Detection](#scanned-pdf-detection)
10. [Image Compression](#image-compression)
11. [Testing](#testing)
12. [Code File Reference](#code-file-reference)
13. [Limits and Constraints](#limits-and-constraints)
14. [Future Considerations](#future-considerations)

---

## Architecture

The document upload feature uses a **backend extraction** architecture. When a user attaches a document, the raw file is sent to the backend, which extracts text and images and returns structured content to the frontend.

```
User drops file in chat input
    │
    ▼
Frontend sends file to POST /v1/documents/extract
    │
    ▼
Backend detects file type by extension
    │
    ├── .txt/.md/.py/.js/...  → Read as UTF-8 text
    ├── .pdf                  → PyMuPDF: text blocks + images per page
    ├── .docx                 → python-docx: paragraphs + tables + inline images
    ├── .xlsx                 → openpyxl: sheet data as markdown tables
    └── .pptx                 → python-pptx: slide text + images
    │
    ▼
Returns structured ExtractionResult:
    { content_parts: [{type: "text", text: "..."}, {type: "image", ...}], ... }
    │
    ▼
Frontend shows attachment chip with preview
    │
    ▼
User sends message → content parts assembled into
OpenAI multimodal format → existing chat pipeline
```

### Why Backend Extraction?

We chose backend extraction over client-side extraction for several reasons:

1. **Binary format parsing**: PDF, DOCX, XLSX, and PPTX are complex binary formats. The libraries that parse them (PyMuPDF, python-docx, openpyxl, python-pptx) are Python libraries that don't run in the browser.

2. **Security**: Parsing complex binary formats in the browser would require loading large WebAssembly bundles and could expose cross-site scripting (XSS) attack vectors through malformed documents.

3. **Consistency**: The same extraction code runs whether the user is in desktop mode (pywebview) or server mode (Docker), eliminating browser-specific edge cases.

4. **Image extraction**: Embedded images in documents need to be extracted, resized, compressed, and base64-encoded. This is more efficient on the server.

For simple text files (.txt, .py, .md), we *could* read them with the browser's `FileReader` API, but using the backend for everything keeps the architecture simple and the codebase DRY (Don't Repeat Yourself).

---

## The Interleaving Problem

This is the core technical challenge of document upload.

Consider a PDF page that looks like this:

```
"Figure 3 shows the temperature trend over the last decade..."

[IMAGE: line chart showing temperature data]

"Table 2 summarizes the regional differences:"

| Region | Avg Temp | Change |
|--------|----------|--------|
| North  | 12.3°C   | +1.2°  |
| South  | 18.7°C   | +0.8°  |
```

If we naively extract all text as one blob and all images separately, the LLM receives:

```
Text: "Figure 3 shows... Table 2 summarizes..."
Image 1: (line chart)
```

The LLM can't connect "Figure 3" to the actual chart because they're separated. It doesn't know which image corresponds to which reference in the text.

### Our Solution: Ordered Content Parts

We extract content **in document order**, producing a list of `ContentPart` objects that alternate between text and images:

```python
[
    ContentPart(type="text", text="Figure 3 shows the temperature trend..."),
    ContentPart(type="image", image_base64="...", image_mime="image/png"),
    ContentPart(type="text", text="Table 2 summarizes:\n| Region | Avg Temp |..."),
]
```

This maps directly to the **OpenAI multimodal content format**:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Figure 3 shows the temperature trend..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "Table 2 summarizes:\n| Region | Avg Temp |..."},
    {"type": "text", "text": "What trends do you see in this data?"}
  ]
}
```

The LLM now sees text and images in the exact order a human reader would — "Figure 3 shows..." is immediately followed by the figure, maintaining the reference.

### Why This Format Works

The OpenAI multimodal format (also used by Claude, Gemma, and other vision models) was designed for exactly this use case. Each content block in the array is processed sequentially by the model's attention mechanism, so the model "reads" the document in order, just like a human would.

STREAM's existing multimodal pipeline already handles this format for pasted images, so document content flows through the same code path with zero modifications to the streaming, routing, or model invocation logic.

---

## Supported Formats

### Text and Code Files (Read Directly)

These are the simplest — we read the file bytes as UTF-8 text. Code files get wrapped in markdown code fences with the appropriate language tag so the LLM knows it's looking at code.

| Extension | Language Tag | Notes |
|-----------|-------------|-------|
| `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.log`, `.yaml`, `.toml`, `.ini` | (none) | Plain text, read as-is |
| `.py` | `python` | Python source code |
| `.js` | `javascript` | JavaScript source |
| `.ts` | `typescript` | TypeScript source |
| `.jsx`, `.tsx` | `jsx`, `tsx` | React components |
| `.java` | `java` | Java source |
| `.cpp`, `.c`, `.h`, `.hpp` | `cpp`, `c` | C/C++ source |
| `.go` | `go` | Go source |
| `.rs` | `rust` | Rust source |
| `.rb` | `ruby` | Ruby source |
| `.sh`, `.bash` | `bash` | Shell scripts |
| `.sql` | `sql` | SQL queries |
| `.swift`, `.kt`, `.scala`, `.php`, `.css`, `.scss`, `.lua`, `.dart`, `.jl` | (various) | Other languages |

### Binary Document Formats (Library Extraction)

| Format | Library | What We Extract |
|--------|---------|-----------------|
| `.pdf` | PyMuPDF (`fitz`) | Text blocks per page, embedded images, scanned pages as images |
| `.docx` | `python-docx` | Paragraphs, tables (as markdown), inline images |
| `.xlsx` | `openpyxl` | Sheet data as markdown tables |
| `.pptx` | `python-pptx` | Slide text, embedded images per slide |

---

## Backend: Document Extraction

**File**: `stream/middleware/utils/document_extractor.py`

This is the core module. It defines:

### Data Types

- **`ContentPart`**: A single piece of extracted content (text or image). Has a `to_openai_format()` method that converts it to the exact format the OpenAI API expects.

- **`ExtractionResult`**: The complete result of extracting a document. Contains the ordered list of `ContentPart` objects plus metadata (filename, file size, text preview, page count, image count, warnings).

### Main Entry Point

```python
async def extract_document(file_bytes: bytes, filename: str) -> ExtractionResult:
```

This function:
1. Validates file size (max 25 MB)
2. Detects file type by extension
3. Routes to the appropriate extractor
4. Generates metadata (text preview, counts)
5. Returns the `ExtractionResult`

### Format-Specific Extractors

#### `_extract_text_file()`
Reads bytes as UTF-8 (with latin-1 fallback). Wraps code files in markdown code fences using `_ext_to_language()` for syntax context.

#### `_extract_pdf()`
Uses PyMuPDF to process each page:
1. Get page text with `page.get_text("text")`
2. If text length < 50 chars → scanned page → render as image (see [Scanned PDF Detection](#scanned-pdf-detection))
3. Otherwise, extract text and embedded images with `_extract_pdf_page_images()`
4. Add page markers (`--- Page N ---`) for navigation

#### `_extract_docx()`
Uses python-docx to iterate through the document body's XML elements in order:
- `<w:p>` (paragraphs) → Extract text, check for inline images via `wp:inline` and `wp:anchor` elements
- `<w:tbl>` (tables) → Convert to markdown with `_table_to_markdown()`
- Images are extracted from DOCX relationships (`doc.part.rels`)

#### `_extract_xlsx()`
Uses openpyxl to read each sheet:
- Iterates rows with `ws.iter_rows(values_only=True)`
- Converts to markdown table format with header separators
- Warns if a sheet has > 500 rows (truncates to save context)

#### `_extract_pptx()`
Uses python-pptx to process each slide:
- Extracts text from all `has_text_frame` shapes
- Extracts tables from `has_table` shapes
- Extracts images from `MSO_SHAPE_TYPE.PICTURE` shapes
- Adds slide markers (`--- Slide N ---`)

---

## Backend: API Endpoint

**File**: `stream/middleware/routes/documents.py`

Two endpoints on the `/v1` prefix:

### `POST /v1/documents/extract`

Accepts a `multipart/form-data` file upload. Returns the extracted content as JSON.

**Request**:
```
POST /v1/documents/extract
Content-Type: multipart/form-data
Body: file=<uploaded file>
```

**Response** (200 OK):
```json
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
```

**Error Responses**:
| Status | When |
|--------|------|
| 400 | Unsupported file type, extraction failure |
| 413 | File too large (> 25 MB) |
| 500 | Unexpected extraction error |

### `GET /v1/documents/supported-formats`

Returns the list of supported file extensions and limits. Used by the frontend to configure the file input's `accept` attribute.

**Router Registration** (`stream/middleware/app.py`):
```python
app.include_router(documents_router, prefix="/v1", tags=["Documents"])
```

---

## Frontend: Upload Flow

### File Selection

**File**: `frontends/react/src/components/input/ChatInput.tsx`

Users can attach files through:
1. **Unified upload button** (📎 paperclip icon) — opens a file picker that accepts ALL file types (images, documents, code, text)
2. **Camera button** (📷 icon) — opens webcam modal on desktop, native camera app on mobile
3. **Drag and drop** — drop any files onto the chat input area
4. **Paste** — Ctrl+V / Cmd+V for images from clipboard

The paperclip button shows a badge with the total count of all attachments (images + documents). The `handleFiles()` function automatically routes files based on type:
- Images (`file.type.startsWith('image/')`) → client-side compression → `images[]` state
- Documents (supported extensions) → backend extraction via `/v1/documents/extract` → `documents[]` state

This unified approach means users don't need to think about which upload button to use — one button handles everything.

### Upload State Management

Each document goes through a lifecycle tracked by the `DocumentAttachment.status` field:
- `'uploading'` — file is being sent to the backend for extraction
- `'ready'` — extraction complete, content available
- `'error'` — extraction failed (shows error message)

A placeholder `DocumentAttachment` with `status: 'uploading'` is added immediately so the user sees feedback. When extraction completes (or fails), the placeholder is replaced with the full result.

### Warning Behavior

Warnings are shown as a persistent amber banner above the attachment area. They are **not auto-dismissed** — they stay visible until the user resolves the underlying issue:

| Warning Trigger | Message | Clears When |
|-----------------|---------|-------------|
| File count at limit (10) and user tries to add more | "Maximum 10 documents per message. Remove some to add more." | User removes files to get below 10 |
| Partial acceptance (e.g., dropped 15 files but only 3 slots) | "Only 3 of 15 files were added. Maximum 10 documents per message." | User removes files to get below 10 |
| Total extracted text > 100K characters | "Total document content is ~150K characters. Very large documents may reduce response quality..." | User removes files until content is under 100K chars |

This design ensures the user always sees and understands the constraint, rather than having a warning flash and disappear before they can act on it.

### Document Chips

The `DocumentChipStrip` component displays compact chips for each attached document:
- **Uploading**: Yellow chip with spinner — "Extracting..."
- **Ready**: Blue chip with file icon — shows filename, size, page count, image count
- **Error**: Red chip with warning icon — shows error message

Each chip has an ✕ button to remove the attachment.

### API Client

**File**: `frontends/react/src/api/documents.ts`

- `extractDocument(file)` — sends file to `/v1/documents/extract`, returns `DocumentAttachment`
- `documentsToContentBlocks(documents)` — converts extracted content to OpenAI format for the API
- `isSupportedDocument(file)` — checks if a file extension is supported
- `isImageFile(file)` — distinguishes images from documents
- `formatFileSize(bytes)` — displays "1.2 MB" or "450 KB"

### Message Assembly

**File**: `frontends/react/src/api/stream.ts`

When sending a message with documents, the content blocks are assembled in this order:
1. **Document content** (text + extracted images) — prepended first
2. **User's text message** — the actual question
3. **Pasted/uploaded images** — appended last

This ordering ensures the LLM sees the document context before the user's question about it, similar to how you'd read a document before answering questions about it.

### Stripping Old Document Content

Only the **last user message** includes full document content. Older messages get a compact reference instead:

```
[Attached: report.pdf — 344.5 KB, 5 pages, 11.6K chars]
[Attached: 3 image(s)]

what is this?
```

Without this optimization, every old message's extracted text and images would be re-sent with every new message, causing:
- **Massive input token counts** — $0.10+ per message for conversations with many documents
- **30+ second latency** — processing thousands of unnecessary old tokens
- **Context window overflow** — quickly exceeding model limits

The LLM's previous responses about those documents remain in the conversation history, so it still has context. This is the same approach used by `strip_old_images()` on the backend for Lakeshore payloads.

**Note**: This stripping is a stopgap. The proper long-term solution is **rolling summarization**, where older exchanges (including document discussions) get compressed into a summary like *"User uploaded report.pdf. Key findings were A, B, C."* This preserves meaning without re-sending raw content. Production systems like Claude and ChatGPT handle this through context management — either sliding windows, summarization, or very large context windows. When STREAM implements rolling summarization, it will subsume this stripping logic.

---

## Frontend: Chat Display

### Message Component

**File**: `frontends/react/src/components/chat/Message.tsx`

Document attachments are shown as collapsible previews in user messages, similar to Claude's file display.

#### `CollapsibleDocumentPreview` Component

**Collapsed** (default): Shows a compact row with:
- Expand arrow (▸ / ▾)
- File icon
- Filename
- Metadata (file size, page count, image count, character count)

**Expanded**: Shows:
- First ~500 characters of extracted text (monospace font, scrollable)
- "... N more characters" indicator if content is truncated
- Any extraction warnings (e.g., sheet row truncation)

This design keeps the chat clean (no giant text blocks) while letting users verify what was extracted.

### Type Definitions

**File**: `frontends/react/src/types/message.ts`

New types added:
- `DocumentContentPart` — a single text or image piece
- `DocumentAttachment` — full attachment with metadata, status, and content parts

The `Message` interface now includes an optional `documents?: DocumentAttachment[]` field.

### Store Updates

**File**: `frontends/react/src/stores/chatStore.ts`

`addUserMessage()` now accepts an optional `documents` parameter. Documents are stored in the message object alongside images, and persisted to IndexedDB for conversation history.

---

## Lakeshore Size Warnings

**File**: `frontends/react/src/components/chat/ChatContainer.tsx`

Globus Compute has a 10 MB payload limit (documented in `stream/middleware/utils/globus_compute_client.py`). STREAM uses a 6 MB threshold (`LAKESHORE_MAX_IMAGE_BYTES`) as a safety margin.

When a message has documents + images that exceed 6 MB total:

1. **Tier = "lakeshore"** (explicit): Hard block with a warning message telling the user to switch to Local or Cloud tier, or reduce attachments.

2. **Tier = "auto"**: Soft info message — "Attachments total X MB (over 6 MB) — Lakeshore will be skipped for this message. Routing to Local or Cloud instead."

The size estimation includes:
- Image data: base64-encoded bytes
- Document content: all `image_base64` fields + text content lengths

---

## Scanned PDF Detection

**File**: `stream/middleware/utils/document_extractor.py`

Some PDFs are scanned documents — they're essentially images with no extractable text (think: a scanned paper or a photographed whiteboard).

### Detection

For each page, we check the extracted text length:
```python
if len(page_text) < SCANNED_PAGE_CHAR_THRESHOLD:  # 50 chars
    # This page is likely scanned
```

Pages with fewer than 50 characters of text are treated as scanned.

### Handling

Instead of returning no content for these pages, we **render the entire page as an image** using PyMuPDF:

```python
zoom = SCANNED_PAGE_DPI / 72  # 150 DPI / 72 DPI default
matrix = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=matrix)
img_bytes = pix.tobytes(output="jpeg", jpg_quality=85)
```

This image is then sent to the vision-capable LLM, which reads the content — essentially using the LLM as an OCR engine. This is actually *better* than traditional OCR because:

1. The LLM understands layout, tables, and figures
2. It handles handwriting and unusual fonts better
3. No separate OCR library dependency (like Tesseract)
4. The same pipeline works for all content types

### Limits

To prevent huge payloads from documents with many scanned pages:
- Maximum 10 scanned pages are rendered as images
- Beyond that, a warning is included in the extraction result

---

## Image Compression

**File**: `stream/middleware/utils/document_extractor.py` → `_compress_image()`

Embedded images from documents can be very large (e.g., a 4000×3000 photo in a DOCX). We compress them using the same approach as STREAM's image upload pipeline:

1. **Skip tiny images**: Images smaller than 50×50 pixels (icons, bullets) are returned as-is — they're not worth sending to the LLM.

2. **Resize**: Images larger than 1024px in either dimension are scaled down (using Lanczos resampling for quality).

3. **Format conversion**: RGBA images are converted to RGB. All images are saved as JPEG at 85% quality.

4. **Error handling**: If compression fails (corrupt image), the original bytes are returned as a fallback.

---

## Testing

**File**: `tests/test_document_extractor.py`

47 tests covering:

| Test Class | Tests | What It Validates |
|-----------|-------|-------------------|
| `TestContentPart` | 4 | OpenAI format conversion, edge cases |
| `TestExtractionResult` | 1 | JSON serialization |
| `TestTextFileExtraction` | 6 | Plain text, code files, CSV, UTF-8 fallback |
| `TestExtToLanguage` | 4 | File extension → language tag mapping |
| `TestTableToMarkdown` | 3 | Table conversion, empty tables, pipe escaping |
| `TestImageCompression` | 3 | Resize, skip tiny, RGBA conversion |
| `TestExtractDocument` | 6 | Main entry point, metadata, size limit, unsupported |
| `TestPDFExtraction` | 3 | Text PDF, multi-page, scanned detection |
| `TestDOCXExtraction` | 2 | Paragraphs, tables |
| `TestXLSXExtraction` | 3 | Simple data, multi-sheet, empty sheets |
| `TestPPTXExtraction` | 1 | Slide text |
| `TestSupportedExtensions` | 4 | Extension sets, common formats |
| `TestDocumentsAPI` | 3 | Router registration, endpoint paths |
| `TestConstants` | 4 | Size limits, thresholds |

Tests create documents **in-memory** (no disk fixtures) using the same libraries:
- PyMuPDF creates PDFs with `fitz.open()` / `doc.new_page()` / `page.insert_text()`
- python-docx creates DOCX with `Document()` / `doc.add_paragraph()`
- openpyxl creates XLSX with `Workbook()` / `ws.append()`
- python-pptx creates PPTX with `Presentation()` / `prs.slides.add_slide()`

Run tests:
```bash
python -m pytest tests/test_document_extractor.py -v
```

---

## Code File Reference

| File | Purpose |
|------|---------|
| `stream/middleware/utils/document_extractor.py` | Core extraction logic for all formats |
| `stream/middleware/routes/documents.py` | API endpoints (`/v1/documents/extract`, `/v1/documents/supported-formats`) |
| `stream/middleware/app.py` | Router registration |
| `frontends/react/src/api/documents.ts` | Frontend API client and utilities |
| `frontends/react/src/api/stream.ts` | Message assembly with document content blocks + old content stripping |
| `frontends/react/src/components/input/ChatInput.tsx` | Unified file upload (images + documents), drag-and-drop, camera, attachment chips, warnings |
| `frontends/react/src/components/input/ImageUpload.tsx` | Image compression, camera modal, camera strategy detection |
| `frontends/react/src/components/chat/Message.tsx` | Collapsible document preview in chat messages |
| `frontends/react/src/types/message.ts` | `DocumentAttachment`, `DocumentContentPart` types |
| `frontends/react/src/stores/chatStore.ts` | Updated `addUserMessage()` with documents parameter |
| `frontends/react/src/components/chat/ChatContainer.tsx` | Lakeshore size warnings, message send flow |
| `tests/test_document_extractor.py` | 47 unit tests |
| `pyproject.toml` | Dependencies: PyMuPDF, python-docx, openpyxl, python-pptx |

---

## Limits and Constraints

| Limit | Value | Why |
|-------|-------|-----|
| Max file size | 25 MB | Matches Claude's limit; prevents memory issues |
| Max files per message | 10 | UX guardrail; warns the user visibly when hit |
| Max scanned pages rendered | 10 | Each rendered page is ~200-500 KB as JPEG |
| Scanned page DPI | 150 | Balance between readability and file size |
| Max image dimension | 1024 px | Same as image upload; saves tokens and payload |
| JPEG quality | 85% | Good quality with significant compression |
| Scanned page threshold | 50 chars | Pages with less text are likely image-only |
| XLSX row truncation | 500 rows | Prevents massive context from large spreadsheets |
| Lakeshore payload warning | 6 MB | Safety margin under 10 MB Globus Compute limit |
| Large content warning | 100K chars | Warns that quality may degrade with huge documents |
| Text preview length | 500 chars | Enough for users to verify extraction quality |

---

## Future Considerations

1. **Rolling Summarization**: When STREAM implements conversation summarization, document content should be included in the summary. Long documents may need their own summarization before being added to the conversation context.

2. **Document-Aware Routing**: The complexity judge could consider document presence and size when routing. A user asking about a 50-page PDF probably needs a capable model.

3. **Caching**: For repeated conversations about the same document, we could cache extraction results to avoid re-processing. The document's hash could serve as a cache key.

4. **Streaming Extraction**: For very large documents, we could stream extraction progress to the frontend instead of waiting for the full result.

5. **Additional Formats**: Future support could include:
   - `.epub` (e-books)
   - `.odt` / `.ods` / `.odp` (LibreOffice formats)
   - `.rtf` (Rich Text Format)
   - `.msg` / `.eml` (email files)
