/**
 * chatStore.ts - Global State for Chat Messages
 * ==============================================
 *
 * WHAT IS ZUSTAND?
 * ----------------
 * Zustand (German for "state") is a state management library for React.
 * It stores data that needs to be shared across multiple components.
 *
 * WHY DO WE NEED GLOBAL STATE?
 * Without it, if ChatInput and MessageList both need "messages":
 *   - You'd pass props down through every component (prop drilling)
 *   - Or use React Context (verbose, re-renders everything)
 *
 * With Zustand:
 *   - Any component can access/update the same data
 *   - Only components using that specific data re-render
 *   - Much simpler API than Redux
 *
 * HOW ZUSTAND WORKS:
 * 1. create() - Creates a store with initial state and actions
 * 2. useStore() - Hook to access state in components
 * 3. set() - Function to update state (triggers re-renders)
 *
 * EXAMPLE:
 *   // In any component:
 *   const messages = useChatStore(state => state.messages)
 *   const addMessage = useChatStore(state => state.addUserMessage)
 *
 *   // When addMessage is called, all components using "messages" update
 */

import { create } from 'zustand'
import type { Message, StreamMetadata } from '../types'
import { useConversationStore } from './conversationStore'

/**
 * ChatState - The shape of our chat store
 *
 * This interface defines:
 * 1. State (data we're storing)
 * 2. Actions (functions to modify the data)
 *
 * In Zustand, state and actions live together in one object.
 */
interface ChatState {
  // ============= STATE (Data) =============

  /**
   * All messages in the current conversation
   * This is what gets displayed in the chat window
   */
  messages: Message[]

  /**
   * Is the AI currently generating a response?
   * Used to show typing indicator and disable input
   */
  isStreaming: boolean

  /**
   * AbortController for cancelling the current stream
   */
  abortController: AbortController | null

  /**
   * The AI's thinking process (builds up token by token)
   * Only populated for reasoning models (Claude Sonnet 4, o1, etc.)
   */
  currentThinking: string

  /**
   * The AI's response (builds up token by token)
   * This is what the user sees appearing character by character
   */
  currentResponse: string

  /**
   * Metadata about the current stream (tier, model, etc.)
   * Received early in the stream before tokens arrive
   */
  streamMetadata: StreamMetadata | null

  // ============= ACTIONS (Functions) =============

  /**
   * Add a user's message to the conversation
   * Called when user presses Send/Enter
   */
  addUserMessage: (content: string) => void

  /**
   * Start streaming - prepare for incoming tokens
   * Resets current response and sets isStreaming to true
   */
  startStreaming: () => void

  /**
   * Append a token to the current response
   * Called for each token received from SSE stream
   */
  appendToken: (token: string) => void

  /**
   * Append to the thinking content (reasoning models)
   */
  appendThinking: (thought: string) => void

  /**
   * Set the stream metadata (tier, model info)
   */
  setMetadata: (meta: StreamMetadata) => void

  /**
   * Finish streaming - convert current response to a message
   * Called when SSE stream sends [DONE]
   */
  finishStreaming: () => void

  /**
   * Stop the current stream (user cancelled)
   */
  stopStreaming: () => void

  /**
   * Set the abort controller for the current stream
   */
  setAbortController: (controller: AbortController | null) => void

  /**
   * Load messages (e.g., when switching conversations)
   */
  setMessages: (messages: Message[]) => void

  /**
   * Clear all messages (start fresh)
   */
  clearChat: () => void

  /**
   * Pending query (e.g., from example query button)
   * ChatContainer picks this up and sends it
   */
  pendingQuery: string | null
  setPendingQuery: (query: string | null) => void
}

/**
 * useChatStore - The Zustand store for chat state
 *
 * create<ChatState>() creates a hook that:
 * - Returns state and actions
 * - Re-renders components when accessed state changes
 *
 * The function receives (set, get):
 * - set: Updates state (triggers re-renders)
 * - get: Gets current state without subscribing
 */
export const useChatStore = create<ChatState>((set, get) => ({
  // ============= Initial State =============
  messages: [],
  isStreaming: false,
  abortController: null,
  currentThinking: '',
  currentResponse: '',
  streamMetadata: null,

  // ============= Action Implementations =============

  addUserMessage: (content) => {
    /**
     * Add a user message to the chat
     *
     * This does TWO things:
     * 1. Update Zustand state (instant UI update)
     * 2. Save to IndexedDB via conversationStore (persistent)
     */
    const message: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      createdAt: new Date().toISOString(),
    }

    // Update UI immediately
    set((state) => ({
      messages: [...state.messages, message],
    }))

    // Persist to IndexedDB (async, but we don't wait for it)
    try {
      useConversationStore.getState().saveMessage(message)
    } catch (err) {
      console.error('[chatStore] Failed to save user message to IndexedDB:', err)
    }
  },

  startStreaming: () => set({
    /**
     * Prepare for a new response:
     * - Set streaming flag (shows typing indicator)
     * - Clear any previous partial response
     */
    isStreaming: true,
    currentResponse: '',
    currentThinking: '',
    streamMetadata: null,
  }),

  appendToken: (token) => set((state) => ({
    /**
     * Concatenate the new token to the existing response.
     * Called potentially hundreds of times per response!
     *
     * String concatenation is fine here - JavaScript engines
     * optimize this pattern well for repeated small appends.
     */
    currentResponse: state.currentResponse + token,
  })),

  appendThinking: (thought) => set((state) => ({
    currentThinking: state.currentThinking + thought,
  })),

  setMetadata: (meta) => set((state) => ({
    // MERGE metadata instead of replacing!
    // Backend sends metadata in multiple events:
    // 1. Initial event: {tier, model}
    // 2. Final event: {cost, duration}
    // We need to merge them to have complete metadata
    streamMetadata: state.streamMetadata
      ? { ...state.streamMetadata, ...meta }
      : meta,
  })),

  finishStreaming: () => {
    /**
     * Stream complete! Convert the accumulated response into a message.
     *
     * This does:
     * 1. Turn off streaming flag
     * 2. Create assistant message from currentResponse
     * 3. Update Zustand state (instant UI update)
     * 4. Save to IndexedDB (persistent storage)
     *
     * get() - Zustand's way to access current state inside actions
     * Unlike set(), get() doesn't subscribe or trigger re-renders
     */
    const state = get()

    const message: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: state.currentResponse,
      thinking: state.currentThinking || undefined,
      metadata: state.streamMetadata || undefined,
      createdAt: new Date().toISOString(),
    }

    // Update UI immediately
    set({
      isStreaming: false,
      messages: [...state.messages, message],
      currentResponse: '',
      currentThinking: '',
    })

    // Persist to IndexedDB (async, but we don't wait for it)
    try {
      useConversationStore.getState().saveMessage(message)
    } catch (err) {
      console.error('[chatStore] Failed to save assistant message to IndexedDB:', err)
    }
  },

  setMessages: (messages) => set({ messages }),

  clearChat: () => set({
    messages: [],
    currentResponse: '',
    currentThinking: '',
    streamMetadata: null,
  }),

  // Pending query (for example queries from sidebar)
  pendingQuery: null,
  setPendingQuery: (query) => set({ pendingQuery: query }),

  // Abort controller for cancelling streams
  setAbortController: (controller) => set({ abortController: controller }),

  stopStreaming: () => {
    /**
     * Stop the current stream (user cancelled)
     *
     * This does:
     * 1. Abort the fetch request
     * 2. Save whatever we have so far as a message
     * 3. Reset streaming state
     */
    const state = get()

    // Abort the request
    if (state.abortController) {
      state.abortController.abort()
    }

    // If we have partial content, save it as a message
    if (state.currentResponse) {
      const message: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: state.currentResponse + '\n\n*[Generation stopped]*',
        thinking: state.currentThinking || undefined,
        metadata: state.streamMetadata || undefined,
        createdAt: new Date().toISOString(),
      }

      set({
        isStreaming: false,
        abortController: null,
        messages: [...state.messages, message],
        currentResponse: '',
        currentThinking: '',
      })

      // Persist to IndexedDB
      try {
        useConversationStore.getState().saveMessage(message)
      } catch (err) {
        console.error('[chatStore] Failed to save stopped message to IndexedDB:', err)
      }
    } else {
      // No content yet, just reset
      set({
        isStreaming: false,
        abortController: null,
        currentResponse: '',
        currentThinking: '',
      })
    }
  },
}))
