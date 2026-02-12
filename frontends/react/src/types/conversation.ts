/**
 * conversation.ts - Type Definitions for Conversations
 * =====================================================
 *
 * A "conversation" is a collection of messages (like a chat thread).
 * Users can have multiple conversations, star favorites, rename them, etc.
 *
 * This is similar to how ChatGPT/Claude show a list of past chats
 * in the sidebar that you can switch between.
 */

/**
 * Conversation - A chat thread containing multiple messages
 *
 * Think of this like:
 * - Gmail: A conversation is like an email thread
 * - Slack: A conversation is like a DM or channel
 * - ChatGPT: A conversation is one chat in the sidebar
 */
export interface Conversation {
  /**
   * Unique identifier for this conversation
   * Used to load/save/delete the right conversation
   */
  id: string

  /**
   * Display name shown in the sidebar
   * - Auto-generated from first message: "What is quantum computing?"
   * - Or user-renamed: "Physics homework help"
   */
  title: string

  /**
   * Whether the user starred/favorited this conversation
   * Starred conversations appear at the top of the list
   */
  starred: boolean

  /**
   * When this conversation was first started
   * ISO 8601 format: "2024-01-15T10:30:00.000Z"
   */
  createdAt: string

  /**
   * When this conversation was last modified
   * Used to sort conversations (most recent first)
   * Also used for sync conflict resolution (last-write-wins)
   */
  updatedAt: string

  /**
   * How many messages are in this conversation
   * Useful for showing "12 messages" in the UI
   */
  messageCount: number

  /**
   * Optional: Preview of the last message
   * Shown in the sidebar to help identify conversations
   * Example: "The answer to your question about..."
   *
   * Truncated to ~50 characters typically
   */
  lastMessage?: string
}
