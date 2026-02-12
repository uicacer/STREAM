/**
 * ChatInput.tsx - Text Input for Sending Messages
 * ================================================
 *
 * This component handles user input for the chat:
 * - Text area that grows with content
 * - Send button that transforms to stop button during streaming
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 */

import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { Send, Square } from 'lucide-react'
import { cn } from '../../lib/utils'

interface ChatInputProps {
  onSend: (message: string) => void
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
    if (!trimmedValue || isStreaming) return
    onSend(trimmedValue)
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleButtonClick = () => {
    if (isStreaming) {
      onStop?.()
    } else {
      handleSubmit()
    }
  }

  return (
    <div className="border-t bg-background px-4 py-4 md:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="relative flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
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

          {/* Send/Stop button - transforms based on streaming state */}
          <button
            onClick={handleButtonClick}
            disabled={!isStreaming && !value.trim()}
            className={cn(
              "p-3 rounded-xl transition-all",
              isStreaming
                ? "bg-red-500 hover:bg-red-600 text-white"
                : value.trim()
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
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
