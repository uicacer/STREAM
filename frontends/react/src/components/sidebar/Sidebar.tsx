/**
 * Sidebar.tsx - Conversation List and Settings Sidebar
 * =====================================================
 *
 * This component displays two sections:
 * 1. CONVERSATIONS: List of all past conversations
 * 2. SETTINGS: Tier selector, judge strategy, temperature, stats
 *
 * DESIGN PATTERN:
 * Similar to ChatGPT/Claude sidebar where you can:
 * - See all your past chats
 * - Click to switch between them
 * - Star favorites, rename, delete
 * - Adjust settings
 */

import { useEffect, useState } from 'react'
import { Plus, MessageSquare, Star, Settings, History } from 'lucide-react'
import { useConversationStore } from '../../stores/conversationStore'
import { useChatStore } from '../../stores/chatStore'
import { ConversationItem } from './ConversationItem'
import { SettingsPanel } from './SettingsPanel'

/**
 * Sidebar Props
 *
 * onClose: Called when user clicks the X button (mobile only)
 * isMobile: Whether we're in mobile view (sidebar overlays content)
 * onExampleQuery: Called when user clicks an example query
 */
interface SidebarProps {
  onClose?: () => void
  isMobile?: boolean
  onExampleQuery?: (query: string) => void
}

export function Sidebar({ onClose, isMobile: _isMobile, onExampleQuery }: SidebarProps) {
  /**
   * Local state for which tab is active
   * 'history' = conversation list
   * 'settings' = settings panel
   */
  const [activeTab, setActiveTab] = useState<'history' | 'settings'>('settings')

  /**
   * Get state from conversation store
   */
  const conversations = useConversationStore((state) => state.conversations)
  const activeConversationId = useConversationStore((state) => state.activeConversationId)
  const isLoading = useConversationStore((state) => state.isLoading)

  /**
   * Get actions from stores
   */
  const loadConversations = useConversationStore((state) => state.loadConversations)
  const startNewConversation = useConversationStore((state) => state.startNewConversation)
  const switchConversation = useConversationStore((state) => state.switchConversation)
  const clearChat = useChatStore((state) => state.clearChat)
  const setMessages = useChatStore((state) => state.setMessages)

  /**
   * Load conversations from IndexedDB on mount
   */
  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  /**
   * Handle clicking "New Conversation"
   */
  const handleNewConversation = () => {
    clearChat()
    startNewConversation()
    onClose?.()
  }

  /**
   * Handle clicking on a conversation in the list
   */
  const handleSelectConversation = async (id: string) => {
    if (id === activeConversationId) {
      onClose?.()
      return
    }

    const messages = await switchConversation(id)
    setMessages(messages)
    onClose?.()
  }

  /**
   * Handle example query click
   */
  const handleExampleQuery = (query: string) => {
    onExampleQuery?.(query)
    onClose?.()
  }

  /**
   * Separate starred and unstarred conversations
   */
  const starredConversations = conversations.filter((c) => c.starred)
  const regularConversations = conversations.filter((c) => !c.starred)

  return (
    <div className="h-full flex flex-col bg-muted/50">
      {/**
       * Tab Switcher
       */}
      <div className="flex border-b">
        <button
          onClick={() => setActiveTab('settings')}
          className={`flex-1 flex items-center justify-center gap-2 px-3 py-3 text-sm font-medium transition-colors
            ${activeTab === 'settings'
              ? 'text-primary border-b-2 border-primary bg-background'
              : 'text-muted-foreground hover:text-foreground'
            }`}
        >
          <Settings className="w-4 h-4" />
          <span>Settings</span>
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`flex-1 flex items-center justify-center gap-2 px-3 py-3 text-sm font-medium transition-colors
            ${activeTab === 'history'
              ? 'text-primary border-b-2 border-primary bg-background'
              : 'text-muted-foreground hover:text-foreground'
            }`}
        >
          <History className="w-4 h-4" />
          <span>History</span>
          {conversations.length > 0 && (
            <span className="text-xs bg-muted px-1.5 py-0.5 rounded-full">
              {conversations.length}
            </span>
          )}
        </button>
      </div>

      {/**
       * Tab Content
       */}
      {activeTab === 'settings' ? (
        /**
         * Settings Panel
         */
        <div className="flex-1 overflow-y-auto p-3">
          <SettingsPanel onExampleQuery={handleExampleQuery} />
        </div>
      ) : (
        /**
         * Conversation History
         */
        <>
          {/* New Conversation Button */}
          <div className="p-3 border-b">
            <button
              onClick={handleNewConversation}
              className="w-full flex items-center gap-2 px-3 py-2 rounded-lg
                         bg-primary text-primary-foreground
                         hover:bg-primary/90 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span className="text-sm font-medium">New conversation</span>
            </button>
          </div>

          {/* Conversation List */}
          <div className="flex-1 overflow-y-auto p-2">
            {isLoading ? (
              <div className="flex items-center justify-center p-4 text-muted-foreground">
                <span className="text-sm">Loading conversations...</span>
              </div>
            ) : conversations.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-4 text-muted-foreground">
                <MessageSquare className="w-8 h-8 mb-2 opacity-50" />
                <span className="text-sm">No conversations yet</span>
                <span className="text-xs mt-1">Start chatting to create one!</span>
              </div>
            ) : (
              <div className="space-y-1">
                {/* Starred conversations */}
                {starredConversations.length > 0 && (
                  <div className="mb-3">
                    <div className="flex items-center gap-1 px-2 py-1 text-xs text-muted-foreground">
                      <Star className="w-3 h-3" />
                      <span>Starred</span>
                    </div>
                    {starredConversations.map((conversation) => (
                      <ConversationItem
                        key={conversation.id}
                        conversation={conversation}
                        isActive={conversation.id === activeConversationId}
                        onClick={() => handleSelectConversation(conversation.id)}
                      />
                    ))}
                  </div>
                )}

                {/* Regular conversations */}
                {regularConversations.length > 0 && (
                  <div>
                    {starredConversations.length > 0 && (
                      <div className="flex items-center gap-1 px-2 py-1 text-xs text-muted-foreground">
                        <MessageSquare className="w-3 h-3" />
                        <span>Recent</span>
                      </div>
                    )}
                    {regularConversations.map((conversation) => (
                      <ConversationItem
                        key={conversation.id}
                        conversation={conversation}
                        isActive={conversation.id === activeConversationId}
                        onClick={() => handleSelectConversation(conversation.id)}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {/**
       * Sidebar Footer
       */}
      <div className="p-3 border-t bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10">
        <div className="text-center">
          <span className="text-sm font-semibold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            🌊 STREAM
          </span>
          <span className="text-xs text-muted-foreground ml-1">v1.0</span>
        </div>
        <p className="text-[10px] text-muted-foreground text-center mt-1">
          Smart Tiered Routing Engine for AI Models
        </p>
      </div>
    </div>
  )
}
