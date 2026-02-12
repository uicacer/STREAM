/**
 * TypingIndicator.tsx - Animated Indicator with Timer & Status Messages
 * ======================================================================
 *
 * Shows animated dots, elapsed time, and status messages while waiting
 * for the AI response. Messages are based on typical timing patterns
 * for each tier (not real-time status from server).
 */

import { useState, useEffect } from 'react'
import { Home, Building2, Cloud, Bot, Sparkles } from 'lucide-react'

const tierIcons = {
  local: Home,
  lakeshore: Building2,
  cloud: Cloud,
  auto: Bot,
} as const

const tierColors = {
  local: 'text-orange-500',
  lakeshore: 'text-green-500',
  cloud: 'text-blue-500',
  auto: 'text-primary',
} as const

const tierBgColors = {
  local: 'bg-orange-500/10',
  lakeshore: 'bg-green-500/10',
  cloud: 'bg-blue-500/10',
  auto: 'bg-primary/10',
} as const

// Solid colors for progress bar (can't use dynamic class generation with Tailwind)
const tierProgressColors = {
  local: 'bg-orange-500',
  lakeshore: 'bg-green-500',
  cloud: 'bg-blue-500',
  auto: 'bg-primary',
} as const

// Dot colors for bouncing animation
const tierDotColors = {
  local: 'bg-orange-500',
  lakeshore: 'bg-green-500',
  cloud: 'bg-blue-500',
  auto: 'bg-primary',
} as const

/**
 * Get status message based on tier and elapsed time.
 * These reflect typical processing stages for each tier.
 */
function getStatusMessage(tier: string, elapsedMs: number): string {
  const seconds = elapsedMs / 1000

  if (tier === 'local') {
    if (seconds < 1) return 'Sending to Ollama...'
    if (seconds < 3) return 'Model processing...'
    if (seconds < 5) return 'Generating response...'
    return 'Still working...'
  }

  if (tier === 'lakeshore') {
    // Lakeshore typically takes 3-8 seconds due to HPC overhead
    if (seconds < 1) return 'Connecting to UIC HPC cluster...'
    if (seconds < 2) return 'Submitting to Globus Compute...'
    if (seconds < 4) return 'Waiting for GPU worker...'
    if (seconds < 6) return 'vLLM processing request...'
    if (seconds < 10) return 'Generating response...'
    return 'HPC job running (longer queries take more time)...'
  }

  if (tier === 'cloud') {
    if (seconds < 1) return 'Connecting to Anthropic API...'
    if (seconds < 2) return 'Claude processing...'
    if (seconds < 4) return 'Generating response...'
    return 'Still working...'
  }

  // Auto mode - we don't know the tier yet
  if (seconds < 1) return 'Analyzing query complexity...'
  if (seconds < 2) return 'Selecting optimal tier...'
  if (seconds < 3) return 'Routing request...'
  return 'Processing...'
}

/**
 * Asymptotic progress - never reaches 100% until actually done
 * Uses logarithmic curve to slow down as it approaches the end
 */
function getAsymptoticProgress(elapsedMs: number, expectedMs: number): number {
  // This formula approaches 90% asymptotically and never reaches 100%
  // The longer it takes, the slower the progress bar moves
  const ratio = elapsedMs / expectedMs
  // Logarithmic slowdown: fast at first, then slows down
  const progress = 90 * (1 - Math.exp(-ratio * 1.5))
  return Math.min(progress, 90) // Never exceed 90%
}

interface TypingIndicatorProps {
  tier?: string
  isThinking?: boolean
}

export function TypingIndicator({ tier = 'auto', isThinking = false }: TypingIndicatorProps) {
  const [elapsedMs, setElapsedMs] = useState(0)

  // Update timer every 100ms
  useEffect(() => {
    const startTime = Date.now()
    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTime)
    }, 100)

    return () => clearInterval(interval)
  }, [])

  const Icon = tierIcons[tier as keyof typeof tierIcons] || Bot
  const iconColor = tierColors[tier as keyof typeof tierColors] || 'text-primary'
  const bgColor = tierBgColors[tier as keyof typeof tierBgColors] || 'bg-primary/10'
  const progressColor = tierProgressColors[tier as keyof typeof tierProgressColors] || 'bg-primary'
  const dotColor = tierDotColors[tier as keyof typeof tierDotColors] || 'bg-primary'
  const elapsedSeconds = (elapsedMs / 1000).toFixed(1)
  const statusMessage = getStatusMessage(tier, elapsedMs)
  const statusText = isThinking ? 'Thinking...' : 'Generating...'

  // Expected wait time varies by tier
  const expectedMs = tier === 'lakeshore' ? 8000 : tier === 'local' ? 3000 : 4000
  const progress = getAsymptoticProgress(elapsedMs, expectedMs)

  return (
    <div className="flex flex-col gap-3 py-4">
      {/* Main indicator row */}
      <div className="flex items-center gap-3">
        {/* Icon with pulse animation - colored by tier */}
        <div className={`relative flex items-center justify-center w-12 h-12 rounded-full ${bgColor}`}>
          <Icon className={`w-6 h-6 ${iconColor}`} />
          <span
            className={`absolute inset-0 rounded-full ${bgColor} animate-ping`}
            style={{ animationDuration: '1.5s' }}
          />
        </div>

        <div className="flex flex-col gap-1">
          {/* Status with bouncing dots and timer */}
          <div className="flex items-center gap-2">
            <div className="flex gap-1">
              <span className={`w-2 h-2 rounded-full ${dotColor} animate-bounce [animation-delay:0ms]`} />
              <span className={`w-2 h-2 rounded-full ${dotColor} animate-bounce [animation-delay:150ms]`} />
              <span className={`w-2 h-2 rounded-full ${dotColor} animate-bounce [animation-delay:300ms]`} />
            </div>
            <span className="text-base font-medium text-foreground">{statusText}</span>
            <span className="px-2.5 py-1 text-sm font-mono bg-muted rounded-full text-muted-foreground">
              {elapsedSeconds}s
            </span>
          </div>

          {/* Current status message */}
          <div className="flex items-center gap-1.5">
            <Sparkles className={`w-3 h-3 ${iconColor} animate-pulse`} />
            <span className="text-sm text-muted-foreground">
              {statusMessage}
            </span>
          </div>
        </div>
      </div>

      {/* Progress bar - shown for all tiers */}
      <div className="ml-[60px] w-64">
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ease-out ${progressColor}`}
            style={{ width: `${Math.max(progress, 5)}%` }}
          />
        </div>
      </div>
    </div>
  )
}
