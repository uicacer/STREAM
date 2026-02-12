/**
 * ConversationItem.tsx - Single Conversation in Sidebar
 * ======================================================
 *
 * Displays one conversation with:
 * - Title (truncated if too long)
 * - Time ago (e.g., "2 hours ago")
 * - Star indicator
 * - Hover actions (rename, delete, star toggle)
 *
 * INTERACTION DESIGN:
 * - Click anywhere: Select this conversation
 * - Hover: Show action buttons
 * - Click action button: Perform that action (stops propagation)
 */

import { useState } from 'react'
import { Star, Trash2, Pencil } from 'lucide-react'
import type { Conversation } from '../../types'
import { useConversationStore } from '../../stores/conversationStore'
import { useChatStore } from '../../stores/chatStore'

interface ConversationItemProps {
  conversation: Conversation
  isActive: boolean
  onClick: () => void
}

/**
 * Format a date as relative time
 *
 * Examples:
 * - "Just now" (< 1 minute)
 * - "5 minutes ago"
 * - "2 hours ago"
 * - "Yesterday"
 * - "3 days ago"
 * - "Jan 15" (> 7 days, same year)
 * - "Jan 15, 2023" (different year)
 */
function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return `${diffDays}d ago`

  // Format as date
  const options: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric' }
  if (date.getFullYear() !== now.getFullYear()) {
    options.year = 'numeric'
  }
  return date.toLocaleDateString(undefined, options)
}

export function ConversationItem({ conversation, isActive, onClick }: ConversationItemProps) {
  /**
   * Local state for UI interactions
   *
   * isHovered: Show action buttons on hover
   * isEditing: Show rename input
   * editTitle: Current value of rename input
   */
  const [isHovered, setIsHovered] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState(conversation.title)

  /**
   * Get actions from stores
   */
  const toggleStar = useConversationStore((state) => state.toggleStar)
  const renameConversation = useConversationStore((state) => state.renameConversation)
  const deleteConversation = useConversationStore((state) => state.deleteConversation)
  const clearChat = useChatStore((state) => state.clearChat)

  /**
   * Handle starring/unstarring
   *
   * e.stopPropagation() prevents the click from also
   * triggering the parent onClick (selecting conversation)
   */
  const handleStar = (e: React.MouseEvent) => {
    e.stopPropagation()
    toggleStar(conversation.id)
  }

  /**
   * Handle delete click
   */
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation()

    // Confirm before deleting
    if (window.confirm('Delete this conversation?')) {
      deleteConversation(conversation.id)
      // If this was the active conversation, clear the chat
      if (isActive) {
        clearChat()
      }
    }
  }

  /**
   * Handle rename click - enter edit mode
   */
  const handleRenameClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setEditTitle(conversation.title)
    setIsEditing(true)
  }

  /**
   * Handle rename submit
   */
  const handleRenameSubmit = () => {
    if (editTitle.trim() && editTitle !== conversation.title) {
      renameConversation(conversation.id, editTitle.trim())
    }
    setIsEditing(false)
  }

  /**
   * Handle rename input key press
   * Enter: Submit
   * Escape: Cancel
   */
  const handleRenameKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleRenameSubmit()
    } else if (e.key === 'Escape') {
      setIsEditing(false)
      setEditTitle(conversation.title)
    }
  }

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`
        group relative flex items-center gap-2 p-2 rounded-lg cursor-pointer
        transition-colors
        ${isActive
          ? 'bg-primary/10 text-primary'
          : 'hover:bg-muted text-foreground'
        }
      `}
    >
      {/**
       * Star indicator (always visible if starred)
       */}
      {conversation.starred && !isHovered && (
        <Star className="w-3 h-3 text-yellow-500 fill-yellow-500 flex-shrink-0" />
      )}

      {/**
       * Conversation info (title + time)
       */}
      <div className="flex-1 min-w-0">
        {isEditing ? (
          /**
           * Rename input mode
           */
          <input
            type="text"
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onBlur={handleRenameSubmit}
            onKeyDown={handleRenameKeyDown}
            onClick={(e) => e.stopPropagation()}
            autoFocus
            className="w-full px-1 py-0.5 text-sm bg-background border rounded
                       focus:outline-none focus:ring-1 focus:ring-primary"
          />
        ) : (
          /**
           * Normal display mode
           */
          <>
            <div className="text-sm truncate font-medium">
              {conversation.title}
            </div>
            <div className="text-xs text-muted-foreground flex items-center gap-1">
              <span>{formatTimeAgo(conversation.updatedAt)}</span>
              {conversation.messageCount > 0 && (
                <>
                  <span>•</span>
                  <span>{conversation.messageCount} msgs</span>
                </>
              )}
            </div>
          </>
        )}
      </div>

      {/**
       * Action buttons (visible on hover)
       *
       * These appear when user hovers over the conversation.
       * Click handlers stop propagation to prevent selecting.
       */}
      {isHovered && !isEditing && (
        <div className="flex items-center gap-0.5">
          <button
            onClick={handleStar}
            className="p-1 rounded hover:bg-background/80 transition-colors"
            title={conversation.starred ? 'Unstar' : 'Star'}
          >
            <Star
              className={`w-3.5 h-3.5 ${
                conversation.starred
                  ? 'text-yellow-500 fill-yellow-500'
                  : 'text-muted-foreground'
              }`}
            />
          </button>
          <button
            onClick={handleRenameClick}
            className="p-1 rounded hover:bg-background/80 transition-colors"
            title="Rename"
          >
            <Pencil className="w-3.5 h-3.5 text-muted-foreground" />
          </button>
          <button
            onClick={handleDelete}
            className="p-1 rounded hover:bg-background/80 transition-colors"
            title="Delete"
          >
            <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive" />
          </button>
        </div>
      )}
    </div>
  )
}
