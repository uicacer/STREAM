/**
 * TypingIndicator.tsx - Multi-Phase Animated Indicator
 * =====================================================
 *
 * Shows different phases based on the request state:
 *
 * FOR AUTO MODE (3 phases with staged transitions):
 * 1. Analyzing - LLM judge determining complexity (purple/gradient)
 * 2. Routing - Shows complexity result, selecting tier (tier colors)
 * 3. Generating - Full tier colors, generating response
 *
 * Each phase has a minimum display time so users can see the pipeline.
 * First message has longer pauses to help users understand how STREAM works.
 *
 * FOR EXPLICIT TIER (1 phase):
 * - Generating - Direct to tier colors
 */

import { useState, useEffect, useMemo, useRef } from 'react'
import { Laptop, Building2, Cloud, Bot, Sparkles, AlertTriangle, Brain, Router, Zap } from 'lucide-react'

// Tier icon mapping
const tierIcons = {
  local: Laptop,
  lakeshore: Building2,
  cloud: Cloud,
  auto: Bot,
} as const

// Tier color configurations
const tierConfig = {
  local: {
    text: 'text-orange-500',
    bg: 'bg-orange-500/10',
    progress: 'bg-orange-500',
    dot: 'bg-orange-500',
    border: 'border-orange-500/30',
  },
  lakeshore: {
    text: 'text-green-500',
    bg: 'bg-green-500/10',
    progress: 'bg-green-500',
    dot: 'bg-green-500',
    border: 'border-green-500/30',
  },
  cloud: {
    text: 'text-blue-500',
    bg: 'bg-blue-500/10',
    progress: 'bg-blue-500',
    dot: 'bg-blue-500',
    border: 'border-blue-500/30',
  },
  auto: {
    text: 'text-purple-500',
    bg: 'bg-purple-500/10',
    progress: 'bg-gradient-to-r from-purple-500 to-pink-500',
    dot: 'bg-purple-500',
    border: 'border-purple-500/30',
  },
} as const

// Complexity badge colors
const complexityConfig = {
  low: { text: 'text-orange-600 dark:text-orange-400', bg: 'bg-orange-500/10', label: 'LOW' },
  medium: { text: 'text-green-600 dark:text-green-400', bg: 'bg-green-500/10', label: 'MEDIUM' },
  high: { text: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-500/10', label: 'HIGH' },
} as const

// Tier display names
const tierNames: Record<string, string> = {
  local: 'Local',
  lakeshore: 'Lakeshore',
  cloud: 'Cloud',
  auto: 'Auto',
}

// Display phases - routing is now a visible phase with its own highlight duration
type Phase = 'analyzing' | 'routing' | 'generating'

interface TypingIndicatorProps {
  /** The actual tier being used (from metadata) */
  tier?: string
  /** What the user selected in the UI */
  userSelectedTier?: string
  /** Query complexity from metadata */
  complexity?: string
  /** Is this a reasoning model that shows thinking */
  isThinking?: boolean
  /** Whether a fallback is occurring */
  fallback?: boolean
  /** The original tier that was requested but failed */
  originalTier?: string
  /** All tiers that were unavailable (for fallback message) */
  unavailableTiers?: string[]
  /** Number of user messages sent so far (controls pause duration - longer at start, faster over time) */
  userMessageCount?: number
}

export function TypingIndicator({
  tier,
  userSelectedTier = 'auto',
  complexity,
  isThinking = false,
  fallback = false,
  originalTier,
  unavailableTiers = [],
  userMessageCount = 1,
}: TypingIndicatorProps) {
  const [elapsedMs, setElapsedMs] = useState(0)
  const isAutoMode = userSelectedTier === 'auto'

  // === STAGED PHASE DISPLAY ===
  // displayPhase transitions with delays so each stage is visible to the user
  const [displayPhase, setDisplayPhase] = useState<Phase>(
    isAutoMode ? 'analyzing' : 'generating'
  )
  const [analyzeComplete, setAnalyzeComplete] = useState(!isAutoMode)
  const [routeComplete, setRouteComplete] = useState(!isAutoMode)
  const mountTimeRef = useRef(Date.now())
  const transitionsStartedRef = useRef(false)

  // First message: full educational pauses so the user sees the entire pipeline.
  // Every message after: near-instant transitions (just a brief flash so the
  // badges still animate, but no artificial waiting). The user already
  // understands Analyze → Route → Generate from the first message.
  const FIRST_MSG_TIMING: [number, number, number] = [2200, 1600, 1200]
  const FAST_TIMING: [number, number, number] = [400, 250, 0]

  const STAGE_TIMING: Record<number, [number, number, number]> = {
    1: FIRST_MSG_TIMING,
  }
  const DEFAULT_TIMING = FAST_TIMING

  const [ANALYZE_MIN_DISPLAY, ROUTE_DISPLAY_TIME, GENERATE_DISPLAY_TIME] =
    STAGE_TIMING[userMessageCount] ?? DEFAULT_TIMING

  // Update timer every 100ms
  useEffect(() => {
    const startTime = Date.now()
    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTime)
    }, 100)
    return () => clearInterval(interval)
  }, [])

  // Detect when backend analysis + routing is complete (metadata arrived)
  const metadataArrived = !!(tier && tier !== 'auto' && complexity)

  // Run staged transitions when metadata arrives
  useEffect(() => {
    if (!metadataArrived || !isAutoMode || transitionsStartedRef.current) return
    transitionsStartedRef.current = true

    const timeSinceMount = Date.now() - mountTimeRef.current
    const analyzeDelay = Math.max(0, ANALYZE_MIN_DISPLAY - timeSinceMount)

    let cancelled = false

    const runTransitions = async () => {
      // Hold Analyze phase for minimum display time
      await new Promise(r => setTimeout(r, analyzeDelay))
      if (cancelled) return

      // Complete Analyze → transition to Route
      setAnalyzeComplete(true)
      setDisplayPhase('routing')

      // Hold Route phase for its display time
      await new Promise(r => setTimeout(r, ROUTE_DISPLAY_TIME))
      if (cancelled) return

      // Complete Route
      setRouteComplete(true)

      // Brief pause for route checkmark to be visible
      await new Promise(r => setTimeout(r, 300))
      if (cancelled) return

      // Transition to Generate (holds for GENERATE_DISPLAY_TIME on first few messages)
      setDisplayPhase('generating')

      // On first few messages, keep Generate highlighted so user sees the full pipeline
      if (GENERATE_DISPLAY_TIME > 0) {
        await new Promise(r => setTimeout(r, GENERATE_DISPLAY_TIME))
      }
    }

    runTransitions()
    return () => { cancelled = true }
  }, [metadataArrived, isAutoMode, ANALYZE_MIN_DISPLAY, ROUTE_DISPLAY_TIME])

  // Get the display tier (use actual tier if available, otherwise user selection)
  const displayTier = tier && tier !== 'auto' ? tier : userSelectedTier

  // Get colors based on display phase
  const colors = useMemo(() => {
    if (displayPhase === 'analyzing') {
      return tierConfig.auto
    }
    return tierConfig[displayTier as keyof typeof tierConfig] || tierConfig.auto
  }, [displayPhase, displayTier])

  // Get the appropriate icon based on display phase
  const Icon = useMemo(() => {
    if (displayPhase === 'analyzing') return Brain
    if (displayPhase === 'routing') return Router
    return tierIcons[displayTier as keyof typeof tierIcons] || Bot
  }, [displayPhase, displayTier])

  // Get complexity label for display
  const complexityLabel = useMemo(() => {
    if (!complexity) return null
    const labels: Record<string, string> = {
      low: 'Low',
      medium: 'Medium',
      high: 'High',
    }
    return labels[complexity] || complexity
  }, [complexity])

  // Get status text and message based on display phase
  const { statusText, statusMessage } = useMemo(() => {
    if (displayPhase === 'analyzing') {
      return {
        statusText: 'Analyzing...',
        statusMessage: 'Evaluating query complexity',
      }
    }

    if (displayPhase === 'routing') {
      return {
        statusText: 'Routing...',
        statusMessage: complexityLabel
          ? `${complexityLabel} complexity → selecting optimal tier`
          : 'Selecting optimal tier',
      }
    }

    // Generating phase - tier-specific messages
    let message = 'Processing...'

    if (displayTier === 'local') {
      message = 'Local model processing...'
    } else if (displayTier === 'lakeshore') {
      message = 'Lakeshore HPC processing...'
    } else if (displayTier === 'cloud') {
      message = 'Cloud API processing...'
    }

    return {
      statusText: isThinking ? 'Thinking...' : 'Generating...',
      statusMessage: message,
    }
  }, [displayPhase, displayTier, isThinking, complexityLabel])

  // Calculate progress
  const expectedMs = displayTier === 'lakeshore' ? 8000 : displayTier === 'local' ? 3000 : 4000
  const ratio = elapsedMs / expectedMs
  const progress = Math.min(90 * (1 - Math.exp(-ratio * 1.5)), 90)

  const elapsedSeconds = (elapsedMs / 1000).toFixed(1)

  return (
    <div className="flex flex-col gap-3 py-4">
      {/* Fallback warning banner */}
      {fallback && originalTier && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-600 dark:text-yellow-400">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm font-medium">
            {unavailableTiers.length > 0
              ? `${unavailableTiers.map(t => tierNames[t] || t).join(' and ')} unavailable — using ${tierNames[displayTier] || displayTier} instead`
              : `${tierNames[originalTier] || originalTier} unavailable — using ${tierNames[displayTier] || displayTier} instead`
            }
          </span>
        </div>
      )}

      {/* Phase indicator badges (for auto mode) */}
      {isAutoMode && (
        <div className="flex items-center gap-2 ml-[60px]">
          {/* Phase 1: Analyzing */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors duration-300
              ${displayPhase === 'analyzing'
                ? 'bg-purple-500/20 text-purple-600 dark:text-purple-400 ring-2 ring-purple-500/30'
                : 'bg-purple-500/10 text-purple-500/60'
              }
              ${''}
            `}
          >
            <Brain className="w-3 h-3" />
            <span>Analyze</span>
            {analyzeComplete && <span className="text-green-500">✓</span>}
          </div>

          {/* Arrow - lights up when analyze completes */}
          <Zap className={`w-3 h-3 transition-colors duration-300 ${
            analyzeComplete ? colors.text : 'text-muted-foreground'
          }`} />

          {/* Phase 2: Routing */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors duration-300
              ${displayPhase === 'routing'
                ? `${colors.bg} ${colors.text} ring-2 ${colors.border}`
                : routeComplete
                  ? `${colors.bg} ${colors.text} opacity-60`
                  : 'bg-muted text-muted-foreground'
              }
              ${''}
            `}
          >
            <Router className="w-3 h-3" />
            <span>Route</span>
            {routeComplete && <span className="text-green-500">✓</span>}
          </div>

          {/* Arrow - lights up when route completes */}
          <Zap className={`w-3 h-3 transition-colors duration-300 ${
            routeComplete ? colors.text : 'text-muted-foreground'
          }`} />

          {/* Phase 3: Generating */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors duration-300
              ${displayPhase === 'generating'
                ? `${colors.bg} ${colors.text} ring-2 ${colors.border}`
                : 'bg-muted text-muted-foreground'
              }`}
          >
            <Sparkles className="w-3 h-3" />
            <span>Generate</span>
          </div>

          {/* Complexity badge (shown after routing complete) */}
          {routeComplete && complexityLabel && (
            <div
              className={`ml-2 px-2.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5
                ${complexityConfig[complexity as keyof typeof complexityConfig]?.bg || 'bg-muted'}
                ${complexityConfig[complexity as keyof typeof complexityConfig]?.text || 'text-muted-foreground'}
              `}
            >
              <span className="opacity-70">Query complexity level:</span>
              <span>{complexityLabel}</span>
            </div>
          )}
        </div>
      )}

      {/* Main indicator row */}
      <div className="flex items-center gap-3">
        {/* Icon with pulse animation */}
        <div className={`relative flex items-center justify-center w-12 h-12 rounded-full transition-all duration-500 ${colors.bg}`}>
          <Icon className={`w-6 h-6 transition-all duration-300 ${colors.text}`} />
          <span
            className={`absolute inset-0 rounded-full ${colors.bg} animate-ping`}
            style={{ animationDuration: '1.5s' }}
          />
        </div>

        <div className="flex flex-col gap-1">
          {/* Status with bouncing dots and timer */}
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <span
                className={`w-2 h-2 rounded-full ${colors.dot}`}
                style={{ animation: 'bounce 1.4s ease-in-out infinite', animationDelay: '0ms' }}
              />
              <span
                className={`w-2 h-2 rounded-full ${colors.dot}`}
                style={{ animation: 'bounce 1.4s ease-in-out infinite', animationDelay: '200ms' }}
              />
              <span
                className={`w-2 h-2 rounded-full ${colors.dot}`}
                style={{ animation: 'bounce 1.4s ease-in-out infinite', animationDelay: '400ms' }}
              />
            </div>
            <span className="text-base font-medium text-foreground">{statusText}</span>
            <span className="px-2.5 py-1 text-sm font-mono bg-muted rounded-full text-muted-foreground">
              {elapsedSeconds}s
            </span>
          </div>

          {/* Current status message */}
          <div className="flex items-center gap-1.5">
            <Sparkles className={`w-3 h-3 ${colors.text} animate-pulse`} />
            <span className="text-sm text-muted-foreground">
              {statusMessage}
            </span>
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="ml-[60px] w-64">
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ease-out ${colors.progress}`}
            style={{ width: `${Math.max(progress, 5)}%` }}
          />
        </div>
      </div>
    </div>
  )
}
