/**
 * conversationStore.ts - Global State for Conversation Management
 * ================================================================
 *
 * This store manages:
 * - List of all conversations (shown in sidebar)
 * - Currently active conversation
 * - Creating, renaming, starring, deleting conversations
 *
 * HOW IT WORKS WITH IndexedDB:
 * 1. On app load: Fetch conversation list from IndexedDB
 * 2. User actions: Update both Zustand state AND IndexedDB
 * 3. IndexedDB is the "source of truth" for persistence
 * 4. Zustand state is the "live" version the UI reads from
 *
 * WHY TWO LAYERS?
 * - IndexedDB: Slow but persistent (survives refresh)
 * - Zustand: Fast but ephemeral (lives in memory)
 *
 * We keep them in sync so:
 * - UI is always responsive (reads from Zustand)
 * - Data is never lost (written to IndexedDB)
 */

import { create } from 'zustand'
import type { Conversation, Message } from '../types'
import {
  getAllConversations,
  getConversation,
  createConversation as dbCreateConversation,
  updateConversation as dbUpdateConversation,
  deleteConversation as dbDeleteConversation,
  getMessagesForConversation,
  addMessage as dbAddMessage,
} from '../lib/db'

/**
 * ConversationState - The shape of our conversation store
 */
interface ConversationState {
  // ============= STATE (Data) =============

  /**
   * All conversations, sorted by most recent first
   * This is what populates the sidebar
   */
  conversations: Conversation[]

  /**
   * The currently active conversation ID
   * null means "new conversation" (no messages yet)
   */
  activeConversationId: string | null

  /**
   * Is the conversation list loading from IndexedDB?
   * Used to show loading spinner on startup
   */
  isLoading: boolean

  // ============= ACTIONS (Functions) =============

  /**
   * Load all conversations from IndexedDB
   * Called once on app startup
   */
  loadConversations: () => Promise<void>

  /**
   * Create a new conversation
   * Called when user sends first message in a new chat
   * Returns the new conversation ID
   */
  createConversation: (title: string) => Promise<string>

  /**
   * Switch to a different conversation
   * Returns the messages for that conversation
   */
  switchConversation: (id: string) => Promise<Message[]>

  /**
   * Start a new conversation (clear current)
   * Sets activeConversationId to null
   */
  startNewConversation: () => void

  /**
   * Rename a conversation
   */
  renameConversation: (id: string, title: string) => Promise<void>

  /**
   * Toggle starred status
   */
  toggleStar: (id: string) => Promise<void>

  /**
   * Delete a conversation
   */
  deleteConversation: (id: string) => Promise<void>

  /**
   * Save a message to the current conversation
   * Also creates the conversation if it's new
   */
  saveMessage: (message: Message) => Promise<void>

  /**
   * Get the current active conversation (if any)
   */
  getActiveConversation: () => Conversation | undefined
}

/**
 * useConversationStore - Zustand store for conversation management
 */
export const useConversationStore = create<ConversationState>((set, get) => ({
  // ============= Initial State =============
  conversations: [],
  activeConversationId: null,
  isLoading: true,

  // ============= Action Implementations =============

  loadConversations: async () => {
    /**
     * Fetch all conversations from IndexedDB
     * This runs once when the app starts
     */
    try {
      set({ isLoading: true })
      const conversations = await getAllConversations()
      set({ conversations, isLoading: false })
      console.log('[ConversationStore] Loaded', conversations.length, 'conversations')
    } catch (error) {
      console.error('[ConversationStore] Failed to load conversations:', error)
      set({ isLoading: false })
    }
  },

  createConversation: async (title: string) => {
    /**
     * Create a new conversation in IndexedDB and add to state
     *
     * TITLE GENERATION:
     * We use the first ~50 chars of the first message as the title.
     * Later, user can rename it to something more meaningful.
     */
    const truncatedTitle = title.length > 50 ? title.slice(0, 47) + '...' : title
    const conversation = await dbCreateConversation(truncatedTitle)

    set((state) => ({
      /**
       * Add new conversation to the FRONT of the list
       * (most recent first)
       */
      conversations: [conversation, ...state.conversations],
      activeConversationId: conversation.id,
    }))

    console.log('[ConversationStore] Created conversation:', conversation.id)
    return conversation.id
  },

  switchConversation: async (id: string) => {
    /**
     * Switch to a different conversation
     *
     * 1. Update active ID in state
     * 2. Load messages from IndexedDB
     * 3. Return messages so chatStore can display them
     */
    set({ activeConversationId: id })

    const messages = await getMessagesForConversation(id)
    console.log('[ConversationStore] Switched to conversation:', id, 'with', messages.length, 'messages')

    return messages
  },

  startNewConversation: () => {
    /**
     * Clear active conversation - ready for new chat
     * The actual conversation isn't created until first message
     */
    set({ activeConversationId: null })
    console.log('[ConversationStore] Started new conversation')
  },

  renameConversation: async (id: string, title: string) => {
    /**
     * Rename a conversation
     *
     * 1. Update in IndexedDB (persistent)
     * 2. Update in Zustand state (UI updates immediately)
     */
    await dbUpdateConversation(id, { title })

    set((state) => ({
      conversations: state.conversations.map((conv) =>
        conv.id === id ? { ...conv, title } : conv
      ),
    }))

    console.log('[ConversationStore] Renamed conversation:', id, 'to:', title)
  },

  toggleStar: async (id: string) => {
    /**
     * Toggle the starred status of a conversation
     */
    const conversation = await getConversation(id)
    if (!conversation) return

    const newStarred = !conversation.starred
    await dbUpdateConversation(id, { starred: newStarred })

    set((state) => ({
      /**
       * Sort conversations: starred first, then by updatedAt
       */
      conversations: state.conversations
        .map((conv) =>
          conv.id === id ? { ...conv, starred: newStarred } : conv
        )
        .sort((a, b) => {
          // Starred conversations first
          if (a.starred !== b.starred) {
            return a.starred ? -1 : 1
          }
          // Then by updatedAt (most recent first)
          return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
        }),
    }))

    console.log('[ConversationStore] Toggled star for:', id, 'to:', newStarred)
  },

  deleteConversation: async (id: string) => {
    /**
     * Delete a conversation and all its messages
     *
     * If this is the active conversation, clear it
     */
    await dbDeleteConversation(id)

    set((state) => ({
      conversations: state.conversations.filter((conv) => conv.id !== id),
      // Clear active if we deleted the active one
      activeConversationId:
        state.activeConversationId === id ? null : state.activeConversationId,
    }))

    console.log('[ConversationStore] Deleted conversation:', id)
  },

  saveMessage: async (message: Message) => {
    /**
     * Save a message to the current conversation
     *
     * If no active conversation exists, create one first!
     * This handles the "first message creates the conversation" flow.
     */
    try {
      let conversationId = get().activeConversationId

      // No active conversation? Create one from the message content
      if (!conversationId) {
        // Use the message content as the title (for user messages)
        const title = message.role === 'user'
          ? message.content
          : 'New conversation'

        conversationId = await get().createConversation(title)
      }

      // Save message to IndexedDB
      await dbAddMessage(conversationId, message)

      // Update the conversation's metadata in state
      set((state) => ({
        conversations: state.conversations.map((conv) =>
          conv.id === conversationId
            ? {
                ...conv,
                messageCount: conv.messageCount + 1,
                lastMessage: message.content.slice(0, 100),
                updatedAt: new Date().toISOString(),
              }
            : conv
        ),
      }))
    } catch (error) {
      // Don't let IndexedDB errors crash the app
      // The message will still display in the UI (from Zustand state)
      console.error('[ConversationStore] Failed to save message:', error)
    }
  },

  getActiveConversation: () => {
    const { conversations, activeConversationId } = get()
    return conversations.find((c) => c.id === activeConversationId)
  },
}))
