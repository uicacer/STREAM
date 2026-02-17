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
import { MessageSquare, Star, Settings, History } from 'lucide-react'
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
  const switchConversation = useConversationStore((state) => state.switchConversation)
  const setMessages = useChatStore((state) => state.setMessages)

  /**
   * Load conversations from IndexedDB on mount
   */
  useEffect(() => {
    loadConversations()
  }, [loadConversations])

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
      <div className="flex bg-background">
        <button
          onClick={() => setActiveTab('settings')}
          className={`flex-1 flex items-center justify-center gap-2 px-3 py-3 text-sm font-medium transition-colors
            ${activeTab === 'settings'
              ? 'text-primary bg-muted/50 rounded-t-lg'
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
              ? 'text-primary bg-muted/50 rounded-t-lg'
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
          {/* Conversation List */}
          <div className="flex-1 overflow-y-auto p-2 pt-3">
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

    </div>
  )
}
