/**
 * ChatContainer.tsx - Main Chat Area
 * ===================================
 *
 * This is the heart of the chat interface. It:
 * - Displays all messages in a scrollable list
 * - Shows the typing indicator while AI responds
 * - Auto-scrolls to new messages
 * - Handles the streaming response display
 */

import { useEffect, useRef, useState } from 'react'
import { Message } from './Message'
import { TypingIndicator } from './TypingIndicator'
import { ChatInput } from '../input/ChatInput'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { streamChat, isReasoningModel } from '../../api/stream'
import { AlertTriangle } from 'lucide-react'

export function ChatContainer() {
  const messages = useChatStore(state => state.messages)
  const isStreaming = useChatStore(state => state.isStreaming)
  const currentResponse = useChatStore(state => state.currentResponse)
  const currentThinking = useChatStore(state => state.currentThinking)
  const streamMetadata = useChatStore(state => state.streamMetadata)

  // Actions from chat store
  const addUserMessage = useChatStore(state => state.addUserMessage)
  const startStreaming = useChatStore(state => state.startStreaming)
  const appendToken = useChatStore(state => state.appendToken)
  const appendThinking = useChatStore(state => state.appendThinking)
  const setMetadata = useChatStore(state => state.setMetadata)
  const finishStreaming = useChatStore(state => state.finishStreaming)
  const stopStreaming = useChatStore(state => state.stopStreaming)
  const setAbortController = useChatStore(state => state.setAbortController)
  const pendingQuery = useChatStore(state => state.pendingQuery)
  const setPendingQuery = useChatStore(state => state.setPendingQuery)

  // Settings
  const getSettings = useSettingsStore(state => state.getSettings)

  // Error state
  const [error, setError] = useState<string | null>(null)

  // Ref for auto-scroll
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentResponse])

  // Handle pending queries
  useEffect(() => {
    if (pendingQuery && !isStreaming) {
      handleSend(pendingQuery)
      setPendingQuery(null)
    }
  }, [pendingQuery, isStreaming])

  /**
   * Handle sending a message
   */
  const handleSend = async (content: string) => {
    addUserMessage(content)
    startStreaming()

    // Create AbortController for this request
    const controller = new AbortController()
    setAbortController(controller)

    const settings = getSettings()
    console.log('[ChatContainer] Sending with settings:', settings)

    const allMessages = [
      ...messages,
      { id: '', role: 'user' as const, content, createdAt: '' }
    ]

    setError(null)

    try {
      await streamChat(allMessages, settings, {
        onToken: (token) => {
          appendToken(token)
        },
        onMetadata: (meta) => {
          setMetadata(meta)
        },
        onThinking: (thought) => {
          appendThinking(thought)
        },
        onComplete: () => {
          finishStreaming()
        },
        onError: (err) => {
          console.error('[ChatContainer] Stream error:', err)
          setError(err)
          finishStreaming()
        },
      }, controller.signal)
    } catch (err) {
      // Check if this was an abort (user cancelled)
      if (err instanceof Error && err.name === 'AbortError') {
        console.log('[ChatContainer] Stream aborted by user')
        return
      }
      const errorMsg = err instanceof Error ? err.message : 'Failed to send message'
      console.error('[ChatContainer] Failed to send message:', err)
      setError(errorMsg)
      finishStreaming()
    }
  }

  /**
   * Handle stop button click
   */
  const handleStop = () => {
    stopStreaming()
  }

  const isReasoning = streamMetadata?.model
    ? isReasoningModel(streamMetadata.model)
    : false

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}

          {/* Error display */}
          {error && (
            <div className="flex items-start gap-2 p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Streaming response */}
          {isStreaming && (
            <>
              {!currentResponse && (
                <TypingIndicator
                  tier={streamMetadata?.tier}
                  isThinking={isReasoning && !!currentThinking}
                />
              )}

              {currentResponse && (
                <Message
                  message={{
                    id: 'streaming',
                    role: 'assistant',
                    content: currentResponse,
                    thinking: currentThinking || undefined,
                    metadata: streamMetadata || undefined,
                    createdAt: new Date().toISOString(),
                  }}
                  isStreaming={true}
                />
              )}
            </>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Chat input with stop button */}
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isStreaming={isStreaming}
        placeholder="Type your message..."
      />
    </div>
  )
}
