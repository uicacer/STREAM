/**
 * App.tsx - Root Component of the Application
 * ============================================
 *
 * This is the top-level component that structures the entire app.
 * Think of it as the "frame" that holds everything together.
 *
 * LAYOUT STRUCTURE:
 * ┌─────────────────────────────────────────────────────┐
 * │ Header (logo, title, menu button)                   │
 * ├──────────────┬──────────────────────────────────────┤
 * │   Sidebar    │         Chat Container               │
 * │ (conversations│    (messages + input)               │
 * │    list)     │                                      │
 * │              │                                      │
 * └──────────────┴──────────────────────────────────────┘
 *
 * RESPONSIVE BEHAVIOR:
 * - Desktop: Sidebar always visible on left
 * - Mobile: Sidebar hidden, toggle with menu button
 *
 * INITIALIZATION FLOW:
 * 1. App mounts → Fetch /v1/config from backend
 * 2. Apply backend defaults to settingsStore (if first-time user)
 * 3. Load conversations from IndexedDB
 * 4. Render the chat interface
 */

import { useEffect, useState, useRef, useCallback } from 'react'
import { ChatContainer } from './components/chat/ChatContainer'
import { Sidebar } from './components/sidebar/Sidebar'
import { useSettingsStore } from './stores/settingsStore'
import { useConversationStore } from './stores/conversationStore'
import { useChatStore } from './stores/chatStore'
import { fetchConfig } from './api/config'
import { Bot, Loader2, Menu, X, PanelLeftClose, PanelLeft, Plus } from 'lucide-react'
import { TierStatus } from './components/header/TierStatus'

/**
 * App Component
 *
 * The root component rendered by main.tsx.
 * Handles initialization and layout.
 */
export default function App() {
  /**
   * State: Is the app still initializing (fetching config)?
   *
   * We show a loading state until config is fetched.
   * This prevents the UI from showing stale/wrong defaults.
   */
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  /**
   * State: Is the mobile sidebar open?
   *
   * On mobile, the sidebar is hidden by default and opens
   * as an overlay when user clicks the menu button.
   */
  const [sidebarOpen, setSidebarOpen] = useState(false)

  /**
   * State: Is the desktop sidebar collapsed?
   */
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  /**
   * State: Sidebar width (for resizing)
   * Default is 320px, min is 250px, max is 450px
   */
  const [sidebarWidth, setSidebarWidth] = useState(320)
  const isResizing = useRef(false)

  /**
   * Handle sidebar resize drag
   */
  const handleMouseDown = useCallback(() => {
    isResizing.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return
    const newWidth = Math.min(Math.max(e.clientX, 250), 450)
    setSidebarWidth(newWidth)
  }, [])

  const handleMouseUp = useCallback(() => {
    isResizing.current = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  /**
   * Attach resize listeners
   */
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [handleMouseMove, handleMouseUp])

  /**
   * Get the initialization functions from stores
   */
  const initializeFromBackend = useSettingsStore((state) => state.initializeFromBackend)
  const loadConversations = useConversationStore((state) => state.loadConversations)
  const startNewConversation = useConversationStore((state) => state.startNewConversation)
  const setPendingQuery = useChatStore((state) => state.setPendingQuery)
  const clearChat = useChatStore((state) => state.clearChat)

  /**
   * Handle starting a new conversation
   */
  const handleNewChat = () => {
    clearChat()
    startNewConversation()
  }

  /**
   * Fetch backend config and load conversations on mount
   *
   * useEffect with empty dependency array [] runs once on mount.
   * This is the right place for initialization logic.
   */
  useEffect(() => {
    async function init() {
      try {
        // Step 1: Fetch configuration from backend
        const config = await fetchConfig()

        // Step 2: Initialize settings store with backend defaults
        // This only applies if user hasn't set preferences yet
        initializeFromBackend(config.defaults)

        console.log('[App] Initialized with backend config:', config.version)
      } catch (err) {
        // Config fetch failed - use fallback defaults in store
        console.warn('[App] Failed to fetch config, using fallbacks:', err)
        setError('Could not connect to server. Using offline mode.')
      }

      try {
        // Step 3: Load saved conversations from IndexedDB
        await loadConversations()
      } catch (err) {
        console.warn('[App] Failed to load conversations:', err)
      }

      // Always finish loading, even if something failed
      // The app can still work with fallback defaults
      setIsLoading(false)
    }

    init()
  }, [initializeFromBackend, loadConversations])

  /**
   * Loading state - show spinner while fetching config
   */
  if (isLoading) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-background">
        <div className="flex items-center gap-3">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <span className="text-lg text-muted-foreground">Loading STREAM...</span>
        </div>
      </div>
    )
  }

  return (
    /**
     * Main container - full viewport height
     */
    <div className="h-screen flex flex-col bg-background">
      {/**
       * Error banner (if config fetch failed)
       */}
      {error && (
        <div className="bg-yellow-500/10 border-b border-yellow-500/20 px-4 py-2 text-sm text-yellow-600 dark:text-yellow-400">
          {error}
        </div>
      )}

      {/**
       * Header Bar
       */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-20">
        <div className="flex items-center justify-between h-14 md:h-16 px-4 md:px-6">
          {/* Left side: Menu button (mobile) + Logo */}
          <div className="flex items-center gap-2">
            {/**
             * Mobile menu button
             *
             * md:hidden = only visible on screens smaller than 768px
             * Toggles the sidebar overlay
             */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="md:hidden p-2 -ml-2 rounded-lg hover:bg-muted transition-colors"
              aria-label="Toggle menu"
            >
              {sidebarOpen ? (
                <X className="w-5 h-5" />
              ) : (
                <Menu className="w-5 h-5" />
              )}
            </button>

            {/* Logo and title */}
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                <Bot className="w-5 h-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-lg md:text-xl font-semibold">STREAM</h1>
                <p className="text-xs text-muted-foreground hidden md:block">
                  Smart Tiered Routing Engine for AI Models
                </p>
              </div>
            </div>
          </div>

          {/* Right side: New Chat button + Tier status indicators */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleNewChat}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                         bg-primary text-primary-foreground text-sm font-medium
                         hover:bg-primary/90 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span className="hidden sm:inline">New Chat</span>
            </button>
            <div className="hidden md:block h-6 w-px bg-border" />
            <TierStatus />
          </div>
        </div>
      </header>

      {/**
       * Main content area with sidebar
       */}
      <div className="flex-1 flex overflow-hidden">
        {/**
         * Mobile sidebar overlay
         *
         * On mobile, the sidebar appears as an overlay.
         * The backdrop dims the content and closes sidebar when clicked.
         */}
        {sidebarOpen && (
          <div
            className="md:hidden fixed inset-0 bg-black/50 z-30"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/**
         * Sidebar Container (Desktop)
         *
         * FEATURES:
         * - Collapsible: Toggle button to hide/show
         * - Resizable: Drag handle to change width
         *
         * RESPONSIVE BEHAVIOR:
         * - Mobile (< md): Hidden by default, slides in from left when open
         * - Desktop (>= md): Collapsible and resizable
         */}
        <aside
          className={`
            hidden md:flex flex-col relative
            border-r-2 border-border/50 bg-background
            transition-all duration-300 ease-in-out
            ${sidebarCollapsed ? 'w-0 overflow-hidden border-r-0' : ''}
          `}
          style={{ width: sidebarCollapsed ? 0 : sidebarWidth }}
        >
          {/**
           * Sidebar Header with Collapse Button
           */}
          <div className="flex items-center justify-between h-12 px-3 border-b shrink-0">
            <span className="text-sm font-medium text-muted-foreground">Menu</span>
            <button
              onClick={() => setSidebarCollapsed(true)}
              className="p-1.5 rounded-md hover:bg-muted transition-colors"
              aria-label="Collapse sidebar"
            >
              <PanelLeftClose className="w-4 h-4 text-muted-foreground" />
            </button>
          </div>

          {/**
           * Sidebar Content
           */}
          <div className="flex-1 overflow-hidden">
            <Sidebar
              onClose={() => setSidebarOpen(false)}
              isMobile={false}
              onExampleQuery={setPendingQuery}
            />
          </div>

          {/**
           * Resize Handle
           * Drag to change sidebar width
           */}
          <div
            onMouseDown={handleMouseDown}
            className="absolute top-0 right-0 w-1 h-full cursor-col-resize
                       hover:bg-primary/30 active:bg-primary/50 transition-colors
                       group"
          >
            <div className="absolute -right-1 w-3 h-full" />
          </div>
        </aside>

        {/**
         * Expand Button (shown when sidebar is collapsed)
         */}
        {sidebarCollapsed && (
          <div className="hidden md:flex flex-col items-center py-3 px-1 border-r bg-background">
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="p-2 rounded-lg hover:bg-muted transition-colors group"
              aria-label="Expand sidebar"
            >
              <PanelLeft className="w-5 h-5 text-muted-foreground group-hover:text-foreground transition-colors" />
            </button>
          </div>
        )}

        {/**
         * Mobile Sidebar (slides in as overlay)
         */}
        <aside
          className={`
            md:hidden fixed inset-y-0 left-0 z-40
            w-64 border-r-2 border-border/50 bg-background
            transform transition-transform duration-200 ease-in-out
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
            top-14
          `}
        >
          <Sidebar
            onClose={() => setSidebarOpen(false)}
            isMobile={sidebarOpen}
            onExampleQuery={setPendingQuery}
          />
        </aside>

        {/**
         * Chat area
         *
         * Takes remaining space after sidebar.
         * flex-1 means "grow to fill available space".
         */}
        <main className="flex-1 overflow-hidden">
          <ChatContainer />
        </main>
      </div>
    </div>
  )
}
