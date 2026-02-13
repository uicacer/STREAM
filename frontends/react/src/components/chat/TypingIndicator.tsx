/**
 * TypingIndicator.tsx - Multi-Phase Animated Indicator
 * =====================================================
 *
 * Shows different phases based on the request state:
 *
 * FOR AUTO MODE (3 phases):
 * 1. Analyzing - LLM judge determining complexity (purple/gradient)
 * 2. Routing - Shows complexity result, transitioning to tier (tier colors)
 * 3. Generating - Full tier colors, generating response
 *
 * FOR EXPLICIT TIER (1 phase):
 * - Generating - Direct to tier colors
 */

import { useState, useEffect, useMemo } from 'react'
import { Home, Building2, Cloud, Bot, Sparkles, AlertTriangle, Brain, Router, Zap } from 'lucide-react'

// Tier icon mapping
const tierIcons = {
  local: Home,
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

// Phase types (routing is instant, not a visible phase)
type Phase = 'analyzing' | 'generating'

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
}

export function TypingIndicator({
  tier,
  userSelectedTier = 'auto',
  complexity,
  isThinking = false,
  fallback = false,
  originalTier,
  unavailableTiers = [],
}: TypingIndicatorProps) {
  const [elapsedMs, setElapsedMs] = useState(0)

  // Update timer every 100ms
  useEffect(() => {
    const startTime = Date.now()
    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTime)
    }, 100)
    return () => clearInterval(interval)
  }, [])

  // Determine the current phase based on ACTUAL DATA received from backend
  // The backend sends metadata (tier + complexity) together after LLM judge completes
  // - Phase 1 (analyzing): No metadata yet - backend is running LLM judge
  // - Phase 2 (generating): Have metadata - routing complete, waiting for tokens
  // Note: "routing" is not a separate phase - it completes instantly with analysis
  const phase: Phase = useMemo(() => {
    // If user explicitly selected a tier (not auto), skip to generating
    // The backend won't run the LLM judge in this case
    if (userSelectedTier !== 'auto') {
      return 'generating'
    }

    // Auto mode: determine phase based on ACTUAL metadata received
    // No tier info yet = backend is still analyzing with LLM judge
    if (!tier || tier === 'auto') {
      return 'analyzing'
    }

    // Have metadata = analysis AND routing are complete, now generating
    return 'generating'
  }, [userSelectedTier, tier])

  // Track if routing decision has been made (for badge display)
  const routingComplete = !!(tier && tier !== 'auto' && complexity)

  // Get the display tier (use actual tier if available, otherwise user selection)
  const displayTier = tier && tier !== 'auto' ? tier : userSelectedTier

  // Get colors based on phase
  const colors = useMemo(() => {
    if (phase === 'analyzing') {
      return tierConfig.auto
    }
    return tierConfig[displayTier as keyof typeof tierConfig] || tierConfig.auto
  }, [phase, displayTier])

  // Get the appropriate icon
  const Icon = useMemo(() => {
    if (phase === 'analyzing') return Brain
    return tierIcons[displayTier as keyof typeof tierIcons] || Bot
  }, [phase, displayTier])

  // Get complexity label for display (capitalized)
  const complexityLabel = useMemo(() => {
    if (!complexity) return null
    const labels: Record<string, string> = {
      low: 'Low',
      medium: 'Medium',
      high: 'High',
    }
    return labels[complexity] || complexity
  }, [complexity])

  // Get status text and message
  const { statusText, statusMessage } = useMemo(() => {
    if (phase === 'analyzing') {
      return {
        statusText: 'Analyzing...',
        statusMessage: 'Evaluating query complexity',
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
  }, [phase, displayTier, isThinking])

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
            {/* Show all unavailable tiers if available, otherwise just original_tier */}
            {unavailableTiers.length > 0
              ? `${unavailableTiers.map(t => tierNames[t] || t).join(' and ')} unavailable — using ${tierNames[displayTier] || displayTier} instead`
              : `${tierNames[originalTier] || originalTier} unavailable — using ${tierNames[displayTier] || displayTier} instead`
            }
          </span>
        </div>
      )}

      {/* Phase indicator badges (for auto mode) */}
      {userSelectedTier === 'auto' && (
        <div className="flex items-center gap-2 ml-[60px]">
          {/* Phase 1: Analyzing */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all duration-300
              ${phase === 'analyzing'
                ? 'bg-purple-500/20 text-purple-600 dark:text-purple-400 ring-2 ring-purple-500/30'
                : 'bg-purple-500/10 text-purple-500/60'
              }`}
          >
            <Brain className="w-3 h-3" />
            <span>Analyze</span>
            {routingComplete && <span className="text-green-500">✓</span>}
          </div>

          {/* Arrow */}
          <Zap className="w-3 h-3 text-muted-foreground" />

          {/* Phase 2: Routing (completes instantly with analysis) */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all duration-300
              ${routingComplete
                ? `${colors.bg} ${colors.text}/60`
                : 'bg-muted text-muted-foreground'
              }`}
          >
            <Router className="w-3 h-3" />
            <span>Route</span>
            {routingComplete && <span className="text-green-500">✓</span>}
          </div>

          {/* Arrow */}
          <Zap className="w-3 h-3 text-muted-foreground" />

          {/* Phase 3: Generating */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all duration-300
              ${phase === 'generating' && routingComplete
                ? `${colors.bg} ${colors.text} ring-2 ${colors.border}`
                : 'bg-muted text-muted-foreground'
              }`}
          >
            <Sparkles className="w-3 h-3" />
            <span>Generate</span>
          </div>

          {/* Complexity badge (shown after routing complete) */}
          {routingComplete && complexityLabel && (
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
