/**
 * SettingsPanel.tsx - Settings and Stats Panel
 * =============================================
 *
 * This component displays:
 * - Tier selector (Auto/Local/Lakeshore/Cloud)
 * - Globus authentication (for Lakeshore)
 * - Judge strategy selector (only in Auto mode)
 * - Temperature slider
 * - Session statistics (query counts, costs)
 * - Example queries
 *
 * DESIGN: Matches the Streamlit sidebar functionality
 */

import { useState, useEffect } from 'react'
import {
  Bot,
  Cpu,
  Building2,
  Cloud,
  Zap,
  Target,
  Sparkles,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Lightbulb,
  Lock,
  Unlock,
  Loader2,
  AlertTriangle,
} from 'lucide-react'
import { useSettingsStore } from '../../stores/settingsStore'
import { useChatStore } from '../../stores/chatStore'
import { checkAuthStatus, authenticateGlobus } from '../../api/auth'
import type { Tier, JudgeStrategy } from '../../types'

/**
 * Example queries for quick start
 * These demonstrate different complexity levels:
 * - LOW: Simple "what is" questions → routes to LOCAL
 * - MEDIUM: How-to and explanations → routes to LAKESHORE
 * - HIGH: Analysis and comparisons → routes to CLOUD
 */
const EXAMPLE_QUERIES = [
  "What is Python?",                                    // LOW → Local
  "How do I submit a GPU job?",                         // MEDIUM → Lakeshore
  "Explain quantum computing",                          // MEDIUM → Lakeshore
  "Design a microservices architecture for a real-time collaborative document editing system with conflict resolution, version control, and offline support. Include security considerations and scalability patterns.",  // HIGH → Cloud
]

/**
 * Tier configuration with icons and descriptions
 */
const TIER_CONFIG: Record<Tier, { icon: typeof Bot; label: string; description: string }> = {
  auto: {
    icon: Bot,
    label: 'Auto (Smart Routing)',
    description: 'Let STREAM choose based on complexity',
  },
  local: {
    icon: Cpu,
    label: 'Local (Ollama)',
    description: 'Free, runs on your machine',
  },
  lakeshore: {
    icon: Building2,
    label: 'Lakeshore (Campus GPU)',
    description: 'UIC HPC cluster',
  },
  cloud: {
    icon: Cloud,
    label: 'Cloud (Claude/GPT)',
    description: 'Most capable, paid',
  },
}

/**
 * Judge strategy configuration
 */
const JUDGE_CONFIG: Record<JudgeStrategy, { icon: typeof Zap; label: string; description: string }> = {
  'ollama-1b': {
    icon: Zap,
    label: 'Ollama 1B',
    description: 'Fastest, less accurate, free',
  },
  'ollama-3b': {
    icon: Target,
    label: 'Ollama 3B',
    description: 'Balanced, free',
  },
  haiku: {
    icon: Sparkles,
    label: 'Claude Haiku',
    description: '~$1/5K judgments, most accurate',
  },
}

interface SettingsPanelProps {
  onExampleQuery?: (query: string) => void
}

export function SettingsPanel({ onExampleQuery }: SettingsPanelProps) {
  /**
   * Get settings from store
   */
  const tier = useSettingsStore((state) => state.tier)
  const judgeStrategy = useSettingsStore((state) => state.judgeStrategy)
  const temperature = useSettingsStore((state) => state.temperature)
  const setTier = useSettingsStore((state) => state.setTier)
  const setJudgeStrategy = useSettingsStore((state) => state.setJudgeStrategy)
  const setTemperature = useSettingsStore((state) => state.setTemperature)

  /**
   * Get messages for stats calculation
   */
  const messages = useChatStore((state) => state.messages)

  /**
   * Local state for expandable sections
   */
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [statsOpen, setStatsOpen] = useState(true)

  /**
   * Globus authentication state
   */
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null)
  const [isAuthenticating, setIsAuthenticating] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)

  /**
   * Check Globus auth status on mount
   */
  useEffect(() => {
    async function checkAuth() {
      console.log('[SettingsPanel] Checking Globus auth status...')
      const status = await checkAuthStatus()
      console.log('[SettingsPanel] Auth status:', status)
      setIsAuthenticated(status.authenticated)
    }
    checkAuth()
  }, [])

  /**
   * Handle Globus authentication
   */
  const handleAuthenticate = async () => {
    setIsAuthenticating(true)
    setAuthError(null)

    const result = await authenticateGlobus()

    setIsAuthenticating(false)

    if (result.success) {
      setIsAuthenticated(true)
      setAuthError(null)
    } else {
      setAuthError(result.message)
    }
  }

  /**
   * Calculate session statistics from messages
   *
   * NOTE: Cost may come as string from backend, so we parse it as float
   */
  const stats = messages.reduce(
    (acc, msg) => {
      if (msg.role === 'assistant' && msg.metadata) {
        acc.queries += 1
        const tierName = msg.metadata.tier
        if (tierName === 'local') acc.localQueries += 1
        else if (tierName === 'lakeshore') acc.lakeshoreQueries += 1
        else if (tierName === 'cloud') acc.cloudQueries += 1
        // Parse cost as number (may come as string from backend)
        const cost = parseFloat(String(msg.metadata.cost)) || 0
        acc.totalCost += cost
      }
      return acc
    },
    { queries: 0, localQueries: 0, lakeshoreQueries: 0, cloudQueries: 0, totalCost: 0 }
  )

  /**
   * Is auth status still loading?
   */
  const isAuthLoading = isAuthenticated === null

  /**
   * Should we show the Lakeshore auth prompt?
   * Only show when:
   * - Tier is lakeshore or auto (needs Lakeshore access)
   * - Auth check is complete (not loading)
   * - User is NOT authenticated
   */
  const showLakeshoreAuth = (tier === 'lakeshore' || tier === 'auto') && !isAuthLoading && !isAuthenticated

  /**
   * Should we show the "authenticated" success message?
   */
  const showAuthSuccess = (tier === 'lakeshore' || tier === 'auto') && isAuthenticated === true

  return (
    <div className="space-y-4">
      {/**
       * Tier Selector
       */}
      <div>
        <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
          <Bot className="w-4 h-4" />
          Model Tier
        </h3>
        <div className="space-y-1">
          {(Object.entries(TIER_CONFIG) as [Tier, typeof TIER_CONFIG['auto']][]).map(
            ([tierKey, config]) => {
              const Icon = config.icon
              const isSelected = tier === tierKey
              const isLakeshoreUnavailable = tierKey === 'lakeshore' && !isAuthenticated

              return (
                <button
                  key={tierKey}
                  onClick={() => setTier(tierKey)}
                  className={`
                    w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm
                    transition-colors
                    ${isSelected
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-muted text-foreground'
                    }
                  `}
                >
                  <Icon className="w-4 h-4 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="font-medium truncate flex items-center gap-1.5">
                      {config.label}
                      {isLakeshoreUnavailable && (
                        <Lock className="w-3 h-3 text-yellow-500" />
                      )}
                    </div>
                    <div className={`text-xs truncate ${isSelected ? 'text-primary-foreground/70' : 'text-muted-foreground'}`}>
                      {config.description}
                    </div>
                  </div>
                </button>
              )
            }
          )}
        </div>
      </div>

      {/**
       * Lakeshore Authentication Panel
       */}
      {showLakeshoreAuth && (
        <div className="border rounded-lg p-3 bg-yellow-500/10 border-yellow-500/30">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-medium text-yellow-600 dark:text-yellow-400">
                Lakeshore Authentication Required
              </h4>
              <p className="text-xs text-muted-foreground mt-1">
                To use the UIC HPC cluster, you need to authenticate with Globus Compute.
              </p>

              {authError && (
                <p className="text-xs text-red-500 mt-2">{authError}</p>
              )}

              <button
                onClick={handleAuthenticate}
                disabled={isAuthenticating}
                className="mt-2 w-full flex items-center justify-center gap-2 px-3 py-2
                           bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg text-sm
                           disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isAuthenticating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Authenticating...
                  </>
                ) : (
                  <>
                    <Unlock className="w-4 h-4" />
                    Authenticate with Globus
                  </>
                )}
              </button>

              <p className="text-xs text-muted-foreground mt-2">
                A browser window will open for authentication.
              </p>
            </div>
          </div>
        </div>
      )}

      {/**
       * Authentication Loading State
       */}
      {isAuthLoading && (tier === 'lakeshore' || tier === 'auto') && (
        <div className="border rounded-lg p-3 bg-muted/50">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Checking Lakeshore authentication...</span>
          </div>
        </div>
      )}

      {/**
       * Authentication Success Message
       * Shows for both 'auto' and 'lakeshore' tiers when authenticated
       */}
      {showAuthSuccess && (
        <div className="border rounded-lg p-3 bg-green-500/10 border-green-500/30">
          <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
            <Unlock className="w-4 h-4" />
            <span className="text-sm font-medium">Lakeshore authenticated</span>
          </div>
        </div>
      )}

      {/**
       * Advanced Settings (expandable)
       */}
      <div className="border-t pt-3">
        <button
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className="w-full flex items-center justify-between text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          <span>Advanced Settings</span>
          {advancedOpen ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {advancedOpen && (
          <div className="mt-3 space-y-4">
            {/**
             * Temperature Slider
             */}
            <div>
              <label className="text-sm text-muted-foreground block mb-1">
                Temperature: {temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer
                           [&::-webkit-slider-thumb]:appearance-none
                           [&::-webkit-slider-thumb]:w-4
                           [&::-webkit-slider-thumb]:h-4
                           [&::-webkit-slider-thumb]:rounded-full
                           [&::-webkit-slider-thumb]:bg-primary
                           [&::-webkit-slider-thumb]:cursor-pointer"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>Focused</span>
                <span>Creative</span>
              </div>
            </div>

            {/**
             * Judge Strategy (only enabled in Auto mode)
             */}
            <div>
              <label className="text-sm text-muted-foreground block mb-2">
                Complexity Judge {tier !== 'auto' && '(Auto mode only)'}
              </label>
              <div className="space-y-1">
                {(Object.entries(JUDGE_CONFIG) as [JudgeStrategy, typeof JUDGE_CONFIG['ollama-1b']][]).map(
                  ([strategyKey, config]) => {
                    const Icon = config.icon
                    const isSelected = judgeStrategy === strategyKey
                    const isDisabled = tier !== 'auto'

                    return (
                      <button
                        key={strategyKey}
                        onClick={() => !isDisabled && setJudgeStrategy(strategyKey)}
                        disabled={isDisabled}
                        className={`
                          w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm
                          transition-colors
                          ${isDisabled
                            ? 'opacity-50 cursor-not-allowed'
                            : isSelected
                            ? 'bg-primary/10 text-primary border border-primary/30'
                            : 'hover:bg-muted text-foreground'
                          }
                        `}
                      >
                        <Icon className="w-4 h-4 flex-shrink-0" />
                        <div className="min-w-0">
                          <div className="font-medium">{config.label}</div>
                          <div className="text-xs text-muted-foreground">
                            {config.description}
                          </div>
                        </div>
                      </button>
                    )
                  }
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/**
       * Session Stats
       */}
      <div className="border-t pt-3">
        <button
          onClick={() => setStatsOpen(!statsOpen)}
          className="w-full flex items-center justify-between text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          <span className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Session Stats
          </span>
          {statsOpen ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {statsOpen && (
          <div className="mt-3 grid grid-cols-2 gap-2">
            <div className="bg-muted/50 rounded-lg p-2">
              <div className="text-lg font-semibold">{stats.queries}</div>
              <div className="text-xs text-muted-foreground">Total Queries</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-2">
              <div className="text-lg font-semibold">${stats.totalCost.toFixed(4)}</div>
              <div className="text-xs text-muted-foreground">Total Cost</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-2">
              <div className="text-lg font-semibold">{stats.localQueries}</div>
              <div className="text-xs text-muted-foreground">💻 Local (Free)</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-2">
              <div className="text-lg font-semibold">{stats.lakeshoreQueries}</div>
              <div className="text-xs text-muted-foreground">🏫 Lakeshore</div>
            </div>
            <div className="bg-muted/50 rounded-lg p-2 col-span-2">
              <div className="text-lg font-semibold">{stats.cloudQueries}</div>
              <div className="text-xs text-muted-foreground">☁️ Cloud</div>
            </div>
          </div>
        )}
      </div>

      {/**
       * Example Queries
       */}
      <div className="border-t pt-3">
        <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
          <Lightbulb className="w-4 h-4" />
          Try These
        </h3>
        <div className="space-y-1">
          {EXAMPLE_QUERIES.map((query) => (
            <button
              key={query}
              onClick={() => onExampleQuery?.(query)}
              className="w-full text-left px-3 py-2 text-sm rounded-lg
                         bg-muted/50 hover:bg-muted transition-colors
                         text-foreground"
            >
              {query}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
