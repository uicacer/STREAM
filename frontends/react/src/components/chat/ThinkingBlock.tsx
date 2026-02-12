/**
 * ThinkingBlock.tsx - Collapsible Display for AI Reasoning Process
 * =================================================================
 *
 * Some AI models (Claude Sonnet 4, OpenAI o1, etc.) have a "thinking" phase
 * where they reason through the problem before producing output.
 *
 * This component displays that thinking in a collapsible block, similar to
 * how Claude.ai and ChatGPT show reasoning for their advanced models.
 *
 * WHY SHOW THINKING?
 * ------------------
 * 1. Transparency - Users can see HOW the AI reached its answer
 * 2. Trust - Visible reasoning builds confidence in the response
 * 3. Education - Users learn to think through problems similarly
 * 4. Debugging - If the answer is wrong, you can see where reasoning failed
 *
 * WHY COLLAPSIBLE?
 * ----------------
 * Thinking content is often LONG (hundreds of words). Showing it all
 * would push the actual answer way down the screen. Collapsing it
 * keeps the UI clean while making it accessible when wanted.
 */

import { useState } from 'react'
import { ChevronRight, Brain } from 'lucide-react'
import { cn } from '../../lib/utils'

/**
 * useState HOOK
 * -------------
 * React hooks let function components have state (data that changes).
 *
 * const [value, setValue] = useState(initialValue)
 * - value: Current state value
 * - setValue: Function to update it (triggers re-render)
 * - initialValue: What value starts as
 *
 * When setValue is called, React re-renders the component with new value.
 */

interface ThinkingBlockProps {
  /**
   * The thinking/reasoning content to display
   * This comes from the AI's thinking process
   */
  thinking: string

  /**
   * Is the thinking content still streaming in?
   * Shows a pulsing indicator if true
   */
  isStreaming?: boolean
}

export function ThinkingBlock({ thinking, isStreaming = false }: ThinkingBlockProps) {
  /**
   * State: Is the thinking block expanded (showing content)?
   *
   * Initially collapsed (false) to keep UI clean.
   * User can click to expand and see the thinking.
   */
  const [isExpanded, setIsExpanded] = useState(false)

  /**
   * EARLY RETURN
   * ------------
   * If there's no thinking content, don't render anything.
   * "return null" in React means "render nothing".
   *
   * This is a common pattern to avoid rendering empty containers.
   */
  if (!thinking) return null

  return (
    /**
     * Container with left border accent
     *
     * - border-l-2: 2px left border
     * - border-purple-400: Purple color (associated with "thinking")
     * - pl-3: Left padding to space content from border
     * - mb-3: Bottom margin before the main response
     */
    <div className="mb-3 border-l-2 border-purple-400 pl-3">
      {/**
       * Clickable header to toggle expand/collapse
       *
       * Using a <button> is important for accessibility:
       * - Keyboard users can Tab to it and press Enter
       * - Screen readers announce it as interactive
       * - <div onClick> would not be accessible!
       */}
      <button
        onClick={() => setIsExpanded(!isExpanded)} // Toggle: true→false, false→true
        className="flex items-center gap-2 text-sm text-purple-600 hover:text-purple-800 transition-colors"
      >
        {/**
         * Chevron arrow that rotates when expanded
         *
         * cn() combines classes conditionally:
         * - Always: "w-4 h-4 transition-transform duration-200"
         * - When expanded: add "rotate-90"
         *
         * transition-transform: Animate the rotation smoothly
         * duration-200: Animation takes 200ms
         */}
        <ChevronRight
          className={cn(
            "w-4 h-4 transition-transform duration-200",
            isExpanded && "rotate-90"
          )}
        />

        {/* Brain icon - represents thinking/reasoning */}
        <Brain className="w-4 h-4" />

        {/**
         * Label text
         *
         * Changes based on whether still streaming:
         * - Streaming: "Thinking..." (activity in progress)
         * - Done: "Thought process" (complete, past tense)
         */}
        <span>
          {isStreaming ? 'Thinking...' : 'Thought process'}
        </span>

        {/**
         * Pulsing dot while streaming
         *
         * animate-pulse: Fades in and out continuously
         * Only shown when isStreaming is true (&&)
         */}
        {isStreaming && (
          <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
        )}
      </button>

      {/**
       * Expandable content area
       *
       * CONDITIONAL RENDERING:
       * {isExpanded && (...)} means:
       * - If isExpanded is true, render the content
       * - If isExpanded is false, render nothing
       *
       * This is called "short-circuit evaluation" in JavaScript.
       */}
      {isExpanded && (
        <div className="mt-2 text-sm text-muted-foreground bg-purple-50 dark:bg-purple-950/20 p-3 rounded-md">
          {/**
           * Pre-formatted text
           *
           * <pre> preserves whitespace and newlines.
           * - whitespace-pre-wrap: Wraps long lines (pre doesn't by default)
           * - font-sans: Uses normal font (pre defaults to monospace)
           */}
          <pre className="whitespace-pre-wrap font-sans">
            {thinking}
          </pre>
        </div>
      )}
    </div>
  )
}
