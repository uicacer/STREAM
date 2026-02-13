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
import { ContextLimitDialog, parseContextError, type ContextErrorInfo } from './ContextLimitDialog'
import { AuthErrorMessage, parseAuthError, type AuthErrorInfo } from './AuthErrorMessage'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { useHealthStore } from '../../stores/healthStore'
import { streamChat, isReasoningModel } from '../../api/stream'
import { summarizeConversation } from '../../api/summarize'
import { AlertTriangle, ArrowDown } from 'lucide-react'

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
  const clearChat = useChatStore(state => state.clearChat)
  const trimHistory = useChatStore(state => state.trimHistory)

  // Settings
  const getSettings = useSettingsStore(state => state.getSettings)
  const setTier = useSettingsStore(state => state.setTier)
  const userSelectedTier = useSettingsStore(state => state.tier)

  // Error state
  const [error, setError] = useState<string | null>(null)

  // Context limit dialog state
  const [contextError, setContextError] = useState<ContextErrorInfo | null>(null)
  const [pendingRetryQuery, setPendingRetryQuery] = useState<string | null>(null)
  const [isSummarizing, setIsSummarizing] = useState(false)

  // Auth error dialog state
  const [authError, setAuthError] = useState<AuthErrorInfo | null>(null)

  // Health status for Cloud tier availability
  const cloudStatus = useHealthStore(state => state.cloud)
  const cloudAvailable = cloudStatus?.available ?? false

  // NOTE: Auth error dialog only shows when a request actually fails,
  // not proactively on startup based on health checks.

  // Scroll state
  const [isNearBottom, setIsNearBottom] = useState(true)
  const [showScrollButton, setShowScrollButton] = useState(false)

  // Refs for scroll handling
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  // Track if user has manually scrolled away from bottom
  const userScrolledAwayRef = useRef(false)

  // Handle scroll events to track position
  const handleScroll = () => {
    const container = scrollContainerRef.current
    if (!container) return

    const threshold = 100 // pixels from bottom to consider "at bottom"
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight
    const atBottom = distanceFromBottom < threshold

    // If user scrolls away from bottom during streaming, remember it
    if (!atBottom && isStreaming) {
      userScrolledAwayRef.current = true
    }

    // If user scrolls back to bottom, reset the flag
    if (atBottom) {
      userScrolledAwayRef.current = false
    }

    setIsNearBottom(atBottom)
    setShowScrollButton(!atBottom)
  }

  // Scroll to bottom function
  const scrollToBottom = () => {
    const container = scrollContainerRef.current
    if (container) {
      container.scrollTop = container.scrollHeight
      userScrolledAwayRef.current = false
      setShowScrollButton(false)
    }
  }

  // Auto-scroll for new messages (not during token streaming)
  useEffect(() => {
    // Only auto-scroll when a new message is added, not on every token
    if (!userScrolledAwayRef.current) {
      scrollToBottom()
    }
  }, [messages])

  // Auto-scroll during streaming only if user hasn't scrolled away
  useEffect(() => {
    if (isStreaming && !userScrolledAwayRef.current && isNearBottom) {
      const container = scrollContainerRef.current
      if (container) {
        // Use instant scroll during streaming to avoid jank
        container.scrollTop = container.scrollHeight
      }
    }
  }, [currentResponse, isStreaming, isNearBottom])

  // Reset scroll flag when streaming ends
  useEffect(() => {
    if (!isStreaming) {
      userScrolledAwayRef.current = false
    }
  }, [isStreaming])

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

    // Filter messages for API:
    // - Exclude messages marked as 'summarized' (they're just for display)
    // - Include summary markers (they contain the context)
    // - Include all non-summarized messages
    const messagesForApi = messages.filter(m => !m.summarized || m.isSummaryMarker)

    const allMessages = [
      ...messagesForApi,
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
          // Check if this is a context limit error
          const ctxError = parseContextError(err)
          if (ctxError) {
            setContextError(ctxError)
            setPendingRetryQuery(content) // Save query for retry
            finishStreaming()
            return
          }
          // Check if this is an auth/subscription error
          const authErr = parseAuthError(err)
          if (authErr) {
            setAuthError(authErr)
            setPendingRetryQuery(content) // Save query for retry with different tier
            finishStreaming()
            return
          }
          // Generic error
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

  /**
   * Context limit dialog handlers
   */
  const handleContextNewChat = () => {
    clearChat()
    setContextError(null)
    setPendingRetryQuery(null)
  }

  const handleContextTrimHistory = () => {
    trimHistory() // Keep only last user message
    setContextError(null)
    // Retry the query with trimmed history
    if (pendingRetryQuery) {
      const query = pendingRetryQuery
      setPendingRetryQuery(null)
      // Use setTimeout to let state update before retry
      setTimeout(() => handleSend(query), 100)
    }
  }

  const handleContextUseCloud = () => {
    setTier('cloud')
    setContextError(null)
    // Retry the query with cloud tier
    if (pendingRetryQuery) {
      const query = pendingRetryQuery
      setPendingRetryQuery(null)
      // Use setTimeout to let tier change propagate
      setTimeout(() => handleSend(query), 100)
    }
  }

  const handleContextSummarize = async () => {
    if (!pendingRetryQuery) return

    setIsSummarizing(true)

    try {
      // Summarize the conversation using Cloud tier
      const summary = await summarizeConversation(messages)

      if (summary) {
        // Mark all existing messages as summarized (kept for display, excluded from API)
        const summarizedMessages = messages.map(m => ({
          ...m,
          summarized: true,
        }))

        // Create a summary marker message
        const summaryMarker = {
          id: crypto.randomUUID(),
          role: 'assistant' as const,
          content: `**[Conversation Summary]**\n\n${summary}\n\n---\n*Previous messages are shown above for reference but won't be sent to the AI.*`,
          createdAt: new Date().toISOString(),
          isSummaryMarker: true,
        }

        // Keep all messages + add summary marker
        useChatStore.getState().setMessages([...summarizedMessages, summaryMarker])

        setContextError(null)
        const query = pendingRetryQuery
        setPendingRetryQuery(null)
        setIsSummarizing(false)

        // Retry the query with summarized history
        setTimeout(() => handleSend(query), 100)
      } else {
        // Summarization failed - show error
        setIsSummarizing(false)
        setError('Failed to summarize conversation. Please try another option.')
        setContextError(null)
        setPendingRetryQuery(null)
      }
    } catch (err) {
      console.error('[ChatContainer] Summarization error:', err)
      setIsSummarizing(false)
      setError('Failed to summarize conversation. Please try another option.')
      setContextError(null)
      setPendingRetryQuery(null)
    }
  }

  const handleContextClose = () => {
    setContextError(null)
    setPendingRetryQuery(null)
  }

  /**
   * Auth error handlers (inline message)
   */
  const handleAuthSwitchTier = (newTier: 'local' | 'lakeshore') => {
    setTier(newTier)
    setAuthError(null)
    // Retry the query with new tier
    if (pendingRetryQuery) {
      const query = pendingRetryQuery
      setPendingRetryQuery(null)
      // Use setTimeout to let tier change propagate
      setTimeout(() => handleSend(query), 100)
    }
  }

  const handleAuthDismiss = () => {
    setAuthError(null)
    setPendingRetryQuery(null)
  }

  const isReasoning = streamMetadata?.model
    ? isReasoningModel(streamMetadata.model)
    : false

  return (
    <div className="flex-1 flex flex-col h-full relative">
      {/* Messages area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8"
      >
        <div className="max-w-6xl mx-auto space-y-6">
          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}

          {/* Generic error display */}
          {error && (
            <div className="flex items-start gap-2 p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Auth error - inline message with action buttons */}
          {authError && (
            <AuthErrorMessage
              error={authError}
              onSwitchTier={handleAuthSwitchTier}
              onDismiss={handleAuthDismiss}
            />
          )}

          {/* Streaming response */}
          {isStreaming && (
            <>
              {!currentResponse && (
                <TypingIndicator
                  tier={streamMetadata?.tier}
                  userSelectedTier={userSelectedTier}
                  complexity={streamMetadata?.complexity}
                  isThinking={isReasoning && !!currentThinking}
                  fallback={streamMetadata?.fallback}
                  originalTier={streamMetadata?.original_tier}
                  unavailableTiers={streamMetadata?.unavailable_tiers}
                />
              )}

              {currentResponse && (
                <>
                  {/* Fallback warning while streaming */}
                  {streamMetadata?.fallback && streamMetadata?.original_tier && (
                    <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-600 dark:text-yellow-400 mb-2">
                      <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                      <span className="text-sm font-medium">
                        {/* Show all unavailable tiers if available, otherwise just original_tier */}
                        {streamMetadata.unavailable_tiers && streamMetadata.unavailable_tiers.length > 0
                          ? `${streamMetadata.unavailable_tiers.map(t => t.charAt(0).toUpperCase() + t.slice(1)).join(' and ')} unavailable — using ${streamMetadata.tier.charAt(0).toUpperCase() + streamMetadata.tier.slice(1)} instead`
                          : `${streamMetadata.original_tier.charAt(0).toUpperCase() + streamMetadata.original_tier.slice(1)} unavailable — using ${streamMetadata.tier.charAt(0).toUpperCase() + streamMetadata.tier.slice(1)} instead`
                        }
                      </span>
                    </div>
                  )}
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
                </>
              )}
            </>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Scroll to bottom button - appears when user scrolls up */}
      {showScrollButton && (
        <button
          onClick={() => scrollToBottom()}
          className="absolute bottom-24 left-1/2 -translate-x-1/2 p-3 rounded-full bg-primary text-primary-foreground shadow-lg hover:bg-primary/90 transition-all z-10"
          aria-label="Scroll to bottom"
        >
          <ArrowDown className="w-5 h-5" />
        </button>
      )}

      {/* Chat input with stop button */}
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isStreaming={isStreaming}
        placeholder="Type your message..."
      />

      {/* Context limit dialog - shows when conversation exceeds model's limit */}
      {contextError && (
        <ContextLimitDialog
          error={contextError}
          onNewChat={handleContextNewChat}
          onTrimHistory={handleContextTrimHistory}
          onSummarize={handleContextSummarize}
          onUseCloud={handleContextUseCloud}
          onClose={handleContextClose}
          cloudAvailable={cloudAvailable}
          isSummarizing={isSummarizing}
        />
      )}

    </div>
  )
}
