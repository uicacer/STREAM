/**
 * ChatInput.tsx - Text Input with Image, Document Upload, and Web Search Toggle
 * ==============================================================================
 *
 * This component handles all user input for the chat:
 * - Text area that grows with content
 * - Image upload button (pick from files)
 * - Document upload button (PDF, DOCX, XLSX, PPTX, text, code files)
 * - Camera capture button (take a photo on mobile)
 * - Web search toggle (globe icon to enable internet search)
 * - Paste images from clipboard (Ctrl+V / Cmd+V)
 * - Drag and drop images AND documents from desktop/file manager
 * - Image thumbnails shown ABOVE the input
 * - Document attachment chips shown ABOVE the input
 * - Send button that transforms to stop button during streaming
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 *
 * LAYOUT (top to bottom):
 *   ┌──────────────────────────────────────────┐
 *   │ [📄 report.pdf] [📊 data.xlsx]            │  ← DocumentChipStrip
 *   │ [img1] [img2] [img3]  3 images           │  ← ImagePreviewStrip
 *   ├──────────────────────────────────────────┤
 *   │ 📎 📷 🌐 │ Type your message...   │ ▶    │  ← Upload + Camera + Globe + textarea + send
 *   └──────────────────────────────────────────┘
 *
 * The paperclip (📎) button opens a unified file picker that accepts
 * ALL file types — images, PDFs, documents, code, text. Files are
 * automatically routed: images go through client-side compression,
 * documents go through backend extraction.
 */

import { useState, useRef, useEffect, useCallback, KeyboardEvent, ClipboardEvent, DragEvent } from 'react'
import { Send, Square, Globe, Paperclip, Camera, FileText, Loader2, X, AlertTriangle, Info } from 'lucide-react'
import { cn } from '../../lib/utils'
import { ImagePreviewStrip, compressImage, getCameraStrategy, CameraModal } from './ImageUpload'
import { useSettingsStore } from '../../stores/settingsStore'
import { extractDocument, isSupportedDocument, isImageFile, formatFileSize } from '../../api/documents'
import type { DocumentAttachment } from '../../types'

// Maximum number of documents per message (UX guardrail)
const MAX_DOCUMENTS_PER_MESSAGE = 10

// If total extracted text exceeds this, warn the user that quality may degrade.
// ~100K chars is roughly 25K tokens — a significant chunk of most context windows.
const LARGE_CONTENT_CHAR_THRESHOLD = 100_000

interface ChatInputProps {
  /** Called when user sends a message (text + optional images + optional documents) */
  onSend: (message: string, images?: string[], documents?: DocumentAttachment[]) => void
  onStop?: () => void
  isStreaming: boolean
  placeholder?: string
}

export function ChatInput({
  onSend,
  onStop,
  isStreaming,
  placeholder = "Type your message..."
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const [images, setImages] = useState<string[]>([])
  const [documents, setDocuments] = useState<DocumentAttachment[]>([])
  const [docWarning, setDocWarning] = useState<string | null>(null)
  const [showCamera, setShowCamera] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const webSearch = useSettingsStore(state => state.webSearch)
  const setWebSearch = useSettingsStore(state => state.setWebSearch)

  // Auto-resize textarea while respecting min/max height from CSS
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${textarea.scrollHeight}px`
    }
  }, [value])

  // Clear warnings automatically when the user removes enough files.
  // The warning stays visible until the user takes action — no auto-dismiss.
  useEffect(() => {
    if (docWarning && documents.length < MAX_DOCUMENTS_PER_MESSAGE) {
      const totalChars = documents
        .filter(d => d.status === 'ready')
        .reduce((sum, d) => sum + d.totalTextLength, 0)
      if (totalChars <= LARGE_CONTENT_CHAR_THRESHOLD) {
        setDocWarning(null)
      }
    }
  }, [documents, docWarning])

  /**
   * Handle file uploads — routes images to the image pipeline and
   * documents to the backend extraction pipeline.
   *
   * File routing:
   *   - image/*  →  client-side compress → images[] state
   *   - .pdf/.docx/etc  →  POST /v1/documents/extract → documents[] state
   *   - unsupported  →  silently ignored
   */
  const handleFiles = useCallback(async (files: File[]) => {
    const imageFiles: File[] = []
    const docFiles: File[] = []

    for (const file of files) {
      if (isImageFile(file)) {
        imageFiles.push(file)
      } else if (isSupportedDocument(file)) {
        docFiles.push(file)
      }
    }

    // Compress and add images (existing pipeline)
    if (imageFiles.length > 0) {
      const newImages: string[] = []
      for (const file of imageFiles) {
        try {
          const compressed = await compressImage(file)
          newImages.push(compressed)
        } catch (err) {
          console.error('[ChatInput] Failed to compress image:', err)
        }
      }
      if (newImages.length > 0) {
        setImages(prev => [...prev, ...newImages])
      }
    }

    // Extract documents via backend
    if (docFiles.length > 0) {
      // Check file count limit — warn the user visibly instead of silently dropping
      if (documents.length >= MAX_DOCUMENTS_PER_MESSAGE) {
        setDocWarning(
          `Maximum ${MAX_DOCUMENTS_PER_MESSAGE} documents per message. ` +
          `Remove some to add more.`
        )
        return
      }

      const slotsAvailable = MAX_DOCUMENTS_PER_MESSAGE - documents.length
      let skippedCount = 0

      if (docFiles.length > slotsAvailable) {
        skippedCount = docFiles.length - slotsAvailable
        docFiles.splice(slotsAvailable)
      }

      for (const file of docFiles) {
        // Check 25 MB per-file size limit
        if (file.size > 25 * 1024 * 1024) {
          const errorDoc: DocumentAttachment = {
            id: crypto.randomUUID(),
            filename: file.name,
            fileType: file.name.split('.').pop() || '',
            fileSize: file.size,
            contentParts: [],
            textPreview: '',
            totalTextLength: 0,
            imageCount: 0,
            pageCount: 0,
            warnings: [],
            status: 'error',
            error: `File too large (${formatFileSize(file.size)}). Maximum is 25 MB.`,
          }
          setDocuments(prev => [...prev, errorDoc])
          continue
        }

        // Add a placeholder while uploading
        const placeholderId = crypto.randomUUID()
        const placeholderDoc: DocumentAttachment = {
          id: placeholderId,
          filename: file.name,
          fileType: file.name.split('.').pop() || '',
          fileSize: file.size,
          contentParts: [],
          textPreview: '',
          totalTextLength: 0,
          imageCount: 0,
          pageCount: 0,
          warnings: [],
          status: 'uploading',
        }
        setDocuments(prev => [...prev, placeholderDoc])

        // Send to backend for extraction
        try {
          const result = await extractDocument(file)
          setDocuments(prev => {
            const updated = prev.map(d =>
              d.id === placeholderId ? { ...result, id: placeholderId } : d
            )

            // After each extraction, check total content size and warn if large
            const totalChars = updated
              .filter(d => d.status === 'ready')
              .reduce((sum, d) => sum + d.totalTextLength, 0)

            if (totalChars > LARGE_CONTENT_CHAR_THRESHOLD) {
              setDocWarning(
                `Total document content is ~${Math.round(totalChars / 1000)}K characters. ` +
                `Very large documents may reduce response quality or exceed context limits.`
              )
            }

            return updated
          })
        } catch (err) {
          const errorMsg = err instanceof Error ? err.message : 'Extraction failed'
          setDocuments(prev =>
            prev.map(d =>
              d.id === placeholderId
                ? { ...d, status: 'error' as const, error: errorMsg }
                : d
            )
          )
        }
      }

      // Show warning about skipped files (after all processing)
      if (skippedCount > 0) {
        setDocWarning(
          `Only ${slotsAvailable} of ${slotsAvailable + skippedCount} files were added. ` +
          `Maximum ${MAX_DOCUMENTS_PER_MESSAGE} documents per message.`
        )
      }
    }
  }, [documents.length])

  const handleSubmit = () => {
    const trimmedValue = value.trim()
    const readyDocs = documents.filter(d => d.status === 'ready')
    const hasContent = trimmedValue || images.length > 0 || readyDocs.length > 0

    if (!hasContent || isStreaming) return

    const messageText = trimmedValue || (images.length > 0 ? '(image)' : '(document)')
    onSend(
      messageText,
      images.length > 0 ? images : undefined,
      readyDocs.length > 0 ? readyDocs : undefined
    )
    setValue('')
    setImages([])
    setDocuments([])
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handlePaste = useCallback(async (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items
    if (!items) return

    const imageFiles: File[] = []
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (file) imageFiles.push(file)
      }
    }

    if (imageFiles.length === 0) return

    e.preventDefault()

    const newImages: string[] = []
    for (const file of imageFiles) {
      try {
        const compressed = await compressImage(file)
        newImages.push(compressed)
      } catch (err) {
        console.error('[ChatInput] Failed to compress pasted image:', err)
      }
    }

    if (newImages.length > 0) {
      setImages(prev => [...prev, ...newImages])
    }
  }, [])

  const handleRemoveImage = useCallback((index: number) => {
    setImages(prev => prev.filter((_, i) => i !== index))
  }, [])

  const handleRemoveDocument = useCallback((docId: string) => {
    setDocuments(prev => {
      const updated = prev.filter(d => d.id !== docId)
      // Clear the warning if the user removed files below the limit
      if (updated.length < MAX_DOCUMENTS_PER_MESSAGE) {
        setDocWarning(null)
      }
      return updated
    })
  }, [])

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      handleFiles(files)
    }
    e.target.value = ''
  }, [handleFiles])

  const handleCameraClick = useCallback(() => {
    const strategy = getCameraStrategy()
    switch (strategy) {
      case 'native-camera':
        cameraInputRef.current?.click()
        break
      case 'webcam-modal':
        setShowCamera(true)
        break
      case 'file-picker':
        fileInputRef.current?.click()
        break
    }
  }, [])

  const handleCameraCapture = useCallback((dataUrl: string) => {
    setImages(prev => [...prev, dataUrl])
    setShowCamera(false)
  }, [])

  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.dataTransfer.types.includes('Files')) {
      setIsDragging(true)
    }
  }, [])

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length === 0) return
    handleFiles(files)
  }, [handleFiles])

  const handleButtonClick = () => {
    if (isStreaming) {
      onStop?.()
    } else {
      handleSubmit()
    }
  }

  const readyDocs = documents.filter(d => d.status === 'ready')
  const canSend = value.trim() || images.length > 0 || readyDocs.length > 0
  const hasUploading = documents.some(d => d.status === 'uploading')

  // Determine placeholder text based on attachments
  const getPlaceholder = () => {
    if (images.length > 0 && documents.length > 0) return "Ask about the images and documents..."
    if (images.length > 0) return "Add a message about the image(s)..."
    if (documents.length > 0) return "Ask about the document(s)..."
    return placeholder
  }

  return (
    <div
      className={cn(
        "border-t bg-background px-4 py-4 md:px-6 lg:px-8 transition-colors",
        isDragging && "bg-primary/5 border-t-primary/50"
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="max-w-4xl mx-auto">
        {/* Document warning banner — shown when file limit is hit or content is very large */}
        {docWarning && (
          <div className="flex items-start gap-2 px-3 py-2 mb-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-700 dark:text-amber-400 text-xs">
            <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span className="flex-1">{docWarning}</span>
          </div>
        )}

        {/* Document attachment chips — shown ABOVE images when documents are attached */}
        {documents.length > 0 && (
          <DocumentChipStrip documents={documents} onRemove={handleRemoveDocument} />
        )}

        {/* Image preview strip — shown ABOVE the input row when images are attached */}
        <ImagePreviewStrip images={images} onRemove={handleRemoveImage} />

        {/* Input row: [icon group] [textarea] [send button] */}
        <div className="relative flex items-center gap-2">
          {/* Icon group: unified upload + camera + globe — tightly spaced */}
          <div className="flex items-center gap-1">
            {/* Unified upload button — accepts images, documents, code, everything */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming || hasUploading}
              className={cn(
                "relative p-2 rounded-xl transition-all",
                (images.length > 0 || documents.length > 0)
                  ? "text-blue-500 bg-blue-500/10 hover:bg-blue-500/20"
                  : "hover:bg-muted text-muted-foreground",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
              aria-label="Upload file"
              title="Upload file (images, PDF, DOCX, XLSX, PPTX, text, code)"
            >
              <Paperclip className="w-5 h-5" />
              {(images.length + documents.length) > 0 && (
                <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-blue-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                  {images.length + documents.length}
                </span>
              )}
            </button>

            {/* Hidden file input — accepts ALL supported file types including images */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,.txt,.md,.csv,.json,.xml,.html,.htm,.log,.yaml,.yml,.toml,.ini,.cfg,.env,.tex,.bib,.rst,.py,.js,.ts,.jsx,.tsx,.java,.cpp,.c,.h,.hpp,.go,.rs,.rb,.sh,.bash,.sql,.r,.m,.swift,.kt,.scala,.php,.css,.scss,.less,.sass,.lua,.pl,.pm,.zig,.dart,.jl,.pdf,.docx,.xlsx,.pptx"
              className="hidden"
              onChange={handleFileInputChange}
            />

            {/* Camera button — opens webcam modal or native camera on mobile */}
            <button
              onClick={handleCameraClick}
              disabled={isStreaming}
              className={cn(
                "p-2 rounded-xl transition-all",
                "hover:bg-muted text-muted-foreground",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
              aria-label="Take a photo"
              title="Take a photo"
            >
              <Camera className="w-5 h-5" />
            </button>

            {/* Hidden input for native camera capture on touch devices */}
            <input
              ref={cameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFileInputChange}
              className="hidden"
            />

            {/* Web search toggle */}
            <button
              onClick={() => setWebSearch(!webSearch)}
              disabled={isStreaming}
              className={cn(
                "relative p-2 rounded-xl transition-all",
                webSearch
                  ? "text-blue-500 bg-blue-500/10 hover:bg-blue-500/20"
                  : "hover:bg-muted text-muted-foreground",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
              aria-label={webSearch ? "Disable web search" : "Enable web search"}
              title={webSearch ? "Web search enabled — click to disable" : "Enable web search"}
            >
              <Globe className="w-5 h-5" />
              {webSearch && (
                <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full" />
              )}
            </button>
          </div>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={getPlaceholder()}
            disabled={isStreaming}
            rows={1}
            className={cn(
              "flex-1 resize-none rounded-xl border-2 border-muted-foreground/30 bg-background px-4 py-2.5",
              "focus:outline-none focus:ring-2 focus:ring-ring focus:border-primary",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "text-base",
              "max-h-40 overflow-y-auto"
            )}
          />

          <button
            onClick={handleButtonClick}
            disabled={(!isStreaming && !canSend) || hasUploading}
            className={cn(
              "p-3 rounded-xl transition-all",
              isStreaming
                ? "bg-red-500 hover:bg-red-600 text-white"
                : canSend
                  ? "bg-primary text-primary-foreground hover:bg-primary/90"
                  : "bg-muted text-muted-foreground",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
            aria-label={isStreaming ? "Stop generating" : "Send message"}
          >
            {isStreaming ? (
              <Square className="w-5 h-5 fill-current" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>

        <p className="hidden md:block text-[11px] text-muted-foreground/70 mt-1.5 text-center">
          Enter to send · Shift+Enter for new line · Paste or drag & drop files
        </p>
        <p className="text-[11px] text-muted-foreground/60 mt-1 text-center">
          AI can make mistakes. Please verify important information and cited sources.
        </p>
      </div>

      {/* Webcam modal for desktop camera capture */}
      {showCamera && (
        <CameraModal
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  )
}


// =============================================================================
// DocumentChipStrip — Compact attachment chips for uploaded documents
// =============================================================================

/**
 * DocumentChipStrip shows compact chips for each attached document,
 * similar to how Claude shows file attachments. Each chip displays:
 * - File icon (colored by status: blue=ready, yellow=uploading, red=error)
 * - Filename (truncated if long)
 * - File size
 * - Remove button
 *
 * This keeps the chat input area clean while showing all attachments.
 */
function DocumentChipStrip({
  documents,
  onRemove,
}: {
  documents: DocumentAttachment[]
  onRemove: (id: string) => void
}) {
  return (
    <div className="flex flex-wrap gap-2 mb-2">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm border transition-colors",
            doc.status === 'uploading' && "bg-yellow-500/10 border-yellow-500/30 text-yellow-700 dark:text-yellow-400",
            doc.status === 'ready' && "bg-blue-500/10 border-blue-500/30 text-blue-700 dark:text-blue-400",
            doc.status === 'error' && "bg-red-500/10 border-red-500/30 text-red-700 dark:text-red-400",
          )}
        >
          {/* Status icon */}
          {doc.status === 'uploading' ? (
            <Loader2 className="w-4 h-4 animate-spin flex-shrink-0" />
          ) : doc.status === 'error' ? (
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          ) : (
            <FileText className="w-4 h-4 flex-shrink-0" />
          )}

          {/* Filename + metadata */}
          <div className="flex flex-col min-w-0">
            <span className="truncate max-w-[200px] font-medium text-xs">
              {doc.filename}
            </span>
            <span className="text-[10px] opacity-70">
              {doc.status === 'uploading' ? 'Extracting...' :
               doc.status === 'error' ? doc.error :
               `${formatFileSize(doc.fileSize)}${doc.pageCount > 0 ? ` · ${doc.pageCount} pages` : ''}${doc.imageCount > 0 ? ` · ${doc.imageCount} images` : ''}`}
            </span>
          </div>

          {/* Remove button */}
          <button
            onClick={() => onRemove(doc.id)}
            className="ml-1 p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors flex-shrink-0"
            aria-label={`Remove ${doc.filename}`}
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  )
}
