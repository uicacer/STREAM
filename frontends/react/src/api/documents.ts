/**
 * documents.ts - Document Upload & Extraction API Client
 * ======================================================
 *
 * This module handles uploading documents to the STREAM backend for
 * content extraction (text + images). The extraction happens server-side
 * because parsing binary formats (PDF, DOCX, etc.) requires specialized
 * Python libraries that don't run in the browser.
 *
 * FLOW:
 *   1. User drops/selects a file in the chat input
 *   2. Frontend calls extractDocument() → sends file to backend
 *   3. Backend extracts text + images in document order
 *   4. Frontend receives structured content → shows attachment chip
 *   5. When user sends message, content parts are assembled into
 *      the OpenAI multimodal format for the chat pipeline
 *
 * WHY NOT EXTRACT IN THE BROWSER?
 *   - Binary formats need PyMuPDF, python-docx, openpyxl, etc.
 *   - Security: parsing untrusted files is safer on the backend
 *   - Consistency: same code for desktop and Docker modes
 */

import type { DocumentAttachment, DocumentContentPart } from '../types'

/**
 * Raw response from the /v1/documents/extract endpoint.
 *
 * This mirrors the Python ExtractionResult.to_dict() output.
 * We map it to a DocumentAttachment for frontend use.
 */
interface ExtractionResponse {
  filename: string
  file_type: string
  file_size: number
  content_parts: DocumentContentPart[]
  text_preview: string
  total_text_length: number
  image_count: number
  page_count: number
  warnings: string[]
}

/**
 * Extract content from an uploaded document.
 *
 * Sends the file to the backend's /v1/documents/extract endpoint,
 * which uses format-specific libraries (PyMuPDF for PDF, python-docx
 * for DOCX, etc.) to extract text and images in document order.
 *
 * @param file - The File object from a file input or drag-and-drop
 * @returns A DocumentAttachment ready to be stored in the message
 *
 * @example
 *   const file = event.dataTransfer.files[0]
 *   const doc = await extractDocument(file)
 *   setDocuments(prev => [...prev, doc])
 */
export async function extractDocument(file: File): Promise<DocumentAttachment> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch('/v1/documents/extract', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null)
    const detail = errorBody?.detail || `Upload failed (HTTP ${response.status})`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }

  const data: ExtractionResponse = await response.json()

  return {
    id: crypto.randomUUID(),
    filename: data.filename,
    fileType: data.file_type,
    fileSize: data.file_size,
    contentParts: data.content_parts,
    textPreview: data.text_preview,
    totalTextLength: data.total_text_length,
    imageCount: data.image_count,
    pageCount: data.page_count,
    warnings: data.warnings,
    status: 'ready',
  }
}

/**
 * Convert document attachments to OpenAI multimodal content blocks.
 *
 * This takes the extracted content parts from one or more documents and
 * converts them to the ContentBlock format that the chat pipeline already
 * handles. This lets documents flow through the EXACT same path as
 * pasted images — no changes needed to the streaming, routing, or
 * model invocation code.
 *
 * The output is prepended to the user's message content when building
 * the API request in stream.ts.
 *
 * @param documents - Array of extracted document attachments
 * @returns Array of ContentBlock objects in the OpenAI format
 */
export function documentsToContentBlocks(
  documents: DocumentAttachment[]
): Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }> {
  const blocks: Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }> = []

  for (const doc of documents) {
    if (doc.status !== 'ready') continue

    for (const part of doc.contentParts) {
      if (part.type === 'text' && part.text) {
        blocks.push({ type: 'text' as const, text: part.text })
      } else if (part.type === 'image' && part.image_base64 && part.image_mime) {
        blocks.push({
          type: 'image_url' as const,
          image_url: { url: `data:${part.image_mime};base64,${part.image_base64}` },
        })
      }
    }
  }

  return blocks
}

/**
 * Check if a file is a supported document type.
 *
 * This is used by the frontend to validate files before sending them
 * to the backend. We check both by extension and by MIME type.
 *
 * @param file - The File object to check
 * @returns true if the file type is supported for extraction
 */
export function isSupportedDocument(file: File): boolean {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase()

  const supportedExtensions = new Set([
    // Text files
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
    '.log', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.env',
    '.tex', '.bib', '.rst',
    // Code files
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
    '.h', '.hpp', '.go', '.rs', '.rb', '.sh', '.bash', '.sql',
    '.r', '.m', '.swift', '.kt', '.kts', '.scala', '.php',
    '.css', '.scss', '.less', '.sass',
    '.lua', '.pl', '.pm', '.zig', '.v', '.dart', '.jl',
    // Binary document formats
    '.pdf', '.docx', '.xlsx', '.pptx',
  ])

  return supportedExtensions.has(ext)
}

/**
 * Check if a file is an image (handled separately from documents).
 *
 * Images are processed client-side (compressed, base64-encoded) and
 * stored in the message's images[] array. Documents go through the
 * backend extraction pipeline instead.
 *
 * @param file - The File object to check
 * @returns true if the file is an image
 */
export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/')
}

/**
 * Format a file size in bytes to a human-readable string.
 *
 * @param bytes - File size in bytes
 * @returns Formatted string like "1.2 MB" or "450 KB"
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
