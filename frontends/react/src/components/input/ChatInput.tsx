/**
 * ChatInput.tsx - Text Input with Image Upload for Sending Messages
 * ==================================================================
 *
 * This component handles user input for the chat:
 * - Text area that grows with content
 * - Image upload button (pick from files)
 * - Camera capture button (take a photo on mobile)
 * - Paste images from clipboard (Ctrl+V / Cmd+V)
 * - Image thumbnails shown ABOVE the input (not beside it)
 * - Send button that transforms to stop button during streaming
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 *
 * LAYOUT (top to bottom):
 *   ┌──────────────────────────────────────┐
 *   │ [img1] [img2] [img3]  3 images       │  ← ImagePreviewStrip
 *   ├──────────────────────────────────────┤
 *   │ 📎 📷 │ Type your message...   │ ▶  │  ← Upload + Camera + textarea + send
 *   └──────────────────────────────────────┘
 */

import { useState, useRef, useEffect, useCallback, KeyboardEvent, ClipboardEvent } from 'react'
import { Send, Square } from 'lucide-react'
import { cn } from '../../lib/utils'
import { ImageUpload, ImagePreviewStrip, compressImage } from './ImageUpload'

interface ChatInputProps {
  /** Called when user sends a message (text + optional images) */
  onSend: (message: string, images?: string[]) => void
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
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${textarea.scrollHeight}px`
    }
  }, [value])

  const handleSubmit = () => {
    const trimmedValue = value.trim()
    if ((!trimmedValue && images.length === 0) || isStreaming) return
    onSend(trimmedValue || '(image)', images.length > 0 ? images : undefined)
    setValue('')
    setImages([])
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

  const handleButtonClick = () => {
    if (isStreaming) {
      onStop?.()
    } else {
      handleSubmit()
    }
  }

  const canSend = value.trim() || images.length > 0

  return (
    <div className="border-t bg-background px-4 py-4 md:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Image preview strip — shown ABOVE the input row when images are attached */}
        <ImagePreviewStrip images={images} onRemove={handleRemoveImage} />

        {/* Input row: upload button + textarea + send/stop button */}
        <div className="relative flex items-end gap-2">
          <ImageUpload
            images={images}
            onImagesChange={setImages}
            disabled={isStreaming}
          />

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={images.length > 0
              ? "Add a message about the image(s)..."
              : placeholder
            }
            disabled={isStreaming}
            rows={1}
            className={cn(
              "flex-1 resize-none rounded-xl border-2 border-muted-foreground/30 bg-background px-4 py-3",
              "focus:outline-none focus:ring-2 focus:ring-ring focus:border-primary",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "text-base",
              "max-h-32 overflow-y-auto"
            )}
          />

          <button
            onClick={handleButtonClick}
            disabled={!isStreaming && !canSend}
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

        <p className="hidden md:block text-xs text-muted-foreground mt-2 text-center">
          Enter to send, Shift+Enter for new line, Ctrl+V to paste images
        </p>
      </div>
    </div>
  )
}
