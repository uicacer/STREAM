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

import { useRef, useState } from 'react'
import {
  Bot,
  Laptop,
  Building2,
  Cloud,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Lightbulb,
  Lock,
  Unlock,
  Loader2,
  AlertTriangle,
} from 'lucide-react'
import { ModelLogo } from '../icons/ProviderLogos'
import { useSettingsStore } from '../../stores/settingsStore'
import { useChatStore } from '../../stores/chatStore'
import { useHealthStore, getTierDisplayInfo } from '../../stores/healthStore'
import { authenticateGlobus } from '../../api/auth'
import type { Tier, JudgeStrategy, CloudProvider, LocalModel, LakeshoreModel } from '../../types'

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
  "Design a microservices architecture for a real-time chat app",  // HIGH → Cloud
]

/**
 * Tier configuration with icons and descriptions
 */
const TIER_CONFIG: Record<Tier, { icon: typeof Bot; label: string; shortLabel: string; description: string; color: string }> = {
  auto: {
    icon: Bot,
    label: 'Auto (Smart Routing)',
    shortLabel: 'Auto',
    description: 'Let STREAM choose based on query complexity',
    color: 'text-slate-400',
  },
  local: {
    icon: Laptop,
    label: 'Local (Ollama)',
    shortLabel: 'Local',
    description: 'Free, runs on your machine',
    color: 'text-orange-500',
  },
  lakeshore: {
    icon: Building2,
    label: 'Lakeshore (Campus GPU)',
    shortLabel: 'Lakeshore',
    description: 'UIC HPC cluster',
    color: 'text-green-500',
  },
  cloud: {
    icon: Cloud,
    label: 'Cloud (Claude/GPT)',
    shortLabel: 'Cloud',
    description: 'Most capable, paid',
    color: 'text-blue-500',
  },
}

/**
 * Judge strategy configuration
 */
const JUDGE_CONFIG: Record<JudgeStrategy, { model: string; label: string; description: string }> = {
  'ollama-1b': {
    model: 'llama',
    label: 'Llama 1B',
    description: 'Fastest, less accurate, free',
  },
  'ollama-3b': {
    model: 'llama',
    label: 'Llama 3B',
    description: 'Balanced, free',
  },
  haiku: {
    model: 'haiku',
    label: 'Claude Haiku',
    description: '~$1/5K judgments, most accurate',
  },
}

/**
 * Per-tier model configurations
 *
 * Source: stream/gateway/litellm_config.yaml
 * Each tier has its own set of available models.
 */
const LOCAL_MODEL_CONFIG: Record<LocalModel, { label: string; description: string }> = {
  'local-llama-tiny': {
    label: 'Llama 3.2 1B',
    description: 'Fastest, least capable',
  },
  'local-llama': {
    label: 'Llama 3.2 3B',
    description: 'Balanced speed & quality',
  },
  'local-llama-quality': {
    label: 'Llama 3.1 8B',
    description: 'Best local quality, slower',
  },
}

const LAKESHORE_MODEL_CONFIG: Record<LakeshoreModel, { label: string; description: string }> = {
  // Demo config: all keys point to Qwen 1.5B on the backend (single SLURM job).
  // Labels reflect what's actually running. To switch to 32B production models,
  // update these labels AND the LAKESHORE_MODELS dict in config.py.
  'lakeshore-qwen-1.5b': {
    label: 'Qwen 2.5 1.5B',
    description: 'General purpose',
  },
  'lakeshore-qwen-32b': {
    label: 'Qwen 2.5 32B',
    description: 'General purpose (AWQ)',
  },
  'lakeshore-coder-1.5b': {
    label: 'Qwen 2.5 Coder 1.5B',
    description: 'Coding specialist',
  },
  'lakeshore-deepseek-r1': {
    label: 'DeepSeek R1 1.5B',
    description: 'Deep reasoning',
  },
  'lakeshore-qwq': {
    label: 'QwQ 1.5B',
    description: 'Reasoning (o1-style)',
  },
}

const CLOUD_PROVIDER_CONFIG: Record<CloudProvider, { label: string; provider: string }> = {
  'cloud-claude': {
    label: 'Claude Sonnet 4',
    provider: 'Anthropic',
  },
  'cloud-gpt': {
    label: 'GPT-4 Turbo',
    provider: 'OpenAI',
  },
  'cloud-gpt-cheap': {
    label: 'GPT-3.5 Turbo',
    provider: 'OpenAI',
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
  const localModel = useSettingsStore((state) => state.localModel)
  const lakeshoreModel = useSettingsStore((state) => state.lakeshoreModel)
  const cloudProvider = useSettingsStore((state) => state.cloudProvider)
  const setTier = useSettingsStore((state) => state.setTier)
  const setJudgeStrategy = useSettingsStore((state) => state.setJudgeStrategy)
  const setTemperature = useSettingsStore((state) => state.setTemperature)
  const setLocalModel = useSettingsStore((state) => state.setLocalModel)
  const setLakeshoreModel = useSettingsStore((state) => state.setLakeshoreModel)
  const setCloudProvider = useSettingsStore((state) => state.setCloudProvider)

  /**
   * Get messages for stats calculation
   */
  const messages = useChatStore((state) => state.messages)

  /**
   * Get tier health status and fetch function
   */
  const localHealth = useHealthStore((state) => state.local)
  const lakeshoreHealth = useHealthStore((state) => state.lakeshore)
  const cloudHealth = useHealthStore((state) => state.cloud)
  const changingTier = useHealthStore((state) => state.changingTier)
  const fetchHealthForModelChange = useHealthStore((state) => state.fetchHealthForModelChange)

  /**
   * Handle model change for any tier - updates setting AND triggers health re-check.
   * Debounced to prevent rapid clicks from firing multiple health checks.
   */
  const modelChangeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const triggerHealthRecheck = (tier: string) => {
    if (modelChangeTimer.current) {
      clearTimeout(modelChangeTimer.current)
    }
    modelChangeTimer.current = setTimeout(() => {
      fetchHealthForModelChange(tier)
    }, 100)
  }

  const handleLocalModelChange = (model: LocalModel) => {
    console.log('[SettingsPanel] Local model changing to:', model)
    setLocalModel(model)
    triggerHealthRecheck('local')
  }

  const handleLakeshoreModelChange = (model: LakeshoreModel) => {
    console.log('[SettingsPanel] Lakeshore model changing to:', model)
    setLakeshoreModel(model)
    triggerHealthRecheck('lakeshore')
  }

  const handleCloudProviderChange = (provider: CloudProvider) => {
    console.log('[SettingsPanel] Cloud provider changing to:', provider)
    setCloudProvider(provider)
    triggerHealthRecheck('cloud')
  }

  /**
   * Local state for expandable sections
   */
  const [localOpen, setLocalOpen] = useState(false)
  const [lakeshoreOpen, setLakeshoreOpen] = useState(false)
  const [cloudOpen, setCloudOpen] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [statsOpen, setStatsOpen] = useState(true)

  /**
   * Globus authentication state
   * Note: We use lakeshoreHealth?.authenticated from healthStore for display (polled every 30s)
   * These local states are only for the authentication button logic
   */
  const [isAuthenticating, setIsAuthenticating] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)

  // Use healthStore's auth status (polled every 30 seconds) instead of local state
  const isAuthenticated = lakeshoreHealth?.authenticated === true

  /**
   * Handle Globus authentication
   */
  const handleAuthenticate = async () => {
    setIsAuthenticating(true)
    setAuthError(null)

    const result = await authenticateGlobus()

    setIsAuthenticating(false)

    if (result.success) {
      setAuthError(null)
      // Auth status will be updated via healthStore polling
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
   * Check if healthStore has loaded lakeshore data yet
   */
  const isAuthLoading = lakeshoreHealth === null

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
      <div className="mb-2">
        <h3 className="text-sm font-medium mb-2">
          Model Tier
        </h3>
        <div className="grid grid-cols-2 gap-1">
          {(Object.entries(TIER_CONFIG) as [Tier, typeof TIER_CONFIG['auto']][]).map(
            ([tierKey, config]) => {
              const Icon = config.icon
              const isSelected = tier === tierKey
              const isLakeshoreUnavailable = tierKey === 'lakeshore' && !isAuthenticated

              // Get health status for this tier (auto doesn't have health status)
              const healthStatus = tierKey === 'local' ? localHealth
                : tierKey === 'lakeshore' ? lakeshoreHealth
                : tierKey === 'cloud' ? cloudHealth
                : null

              // Get display info from centralized function (single source of truth)
              const displayInfo = tierKey === 'auto' ? null
                : getTierDisplayInfo(tierKey, healthStatus)

              const statusDot = displayInfo?.color ?? null
              const statusTooltip = displayInfo?.tooltip ?? config.description

              return (
                <button
                  key={tierKey}
                  onClick={() => setTier(tierKey)}
                  title={statusTooltip || config.description}
                  className={`
                    flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg text-sm font-medium
                    transition-colors
                    ${isSelected
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-muted text-foreground'
                    }
                  `}
                >
                  <Icon className={`w-6 h-6 flex-shrink-0 ${!isSelected ? config.color : ''}`} />
                  {config.shortLabel}
                  {changingTier === tierKey ? (
                    <Loader2 className="w-2.5 h-2.5 animate-spin flex-shrink-0" />
                  ) : statusDot && (
                    <span className={`w-1.5 h-1.5 rounded-full ${statusDot} flex-shrink-0`} />
                  )}
                  {isLakeshoreUnavailable && (
                    <Lock className="w-2.5 h-2.5 text-yellow-500 flex-shrink-0" />
                  )}
                </button>
              )
            }
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-1.5">{TIER_CONFIG[tier].description}</p>
      </div>

      {/**
       * Per-Tier Model Selection (3 collapsible sections)
       */}

      {/* Local Models */}
      <div className="border rounded-lg bg-muted/30 overflow-hidden">
        <button
          onClick={() => setLocalOpen(!localOpen)}
          className="w-full flex items-center gap-2 px-3 py-2.5 text-sm hover:bg-muted/50 transition-colors"
        >
          <Laptop className="w-4 h-4 flex-shrink-0 text-orange-500" />
          <span className="font-medium">Local Models</span>
          <span className="text-xs text-muted-foreground ml-auto mr-2">
            {LOCAL_MODEL_CONFIG[localModel]?.label}
          </span>
          {localOpen ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
        </button>
        {localOpen && (
          <div className="px-3 pb-3 space-y-1">
            {(Object.entries(LOCAL_MODEL_CONFIG) as [LocalModel, typeof LOCAL_MODEL_CONFIG['local-llama']][]).map(
              ([modelKey, config]) => {
                const isSelected = localModel === modelKey
                return (
                  <button
                    key={modelKey}
                    onClick={() => handleLocalModelChange(modelKey)}
                    className={`
                      w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left text-sm
                      transition-colors
                      ${isSelected
                        ? 'bg-primary/10 text-primary border border-primary/30'
                        : 'hover:bg-muted text-foreground'
                      }
                    `}
                  >
                    <ModelLogo model={modelKey} className="w-4 h-4 flex-shrink-0" />
                    <div className="min-w-0 flex-1">
                      <div className="font-medium">{config.label}</div>
                      <div className="text-xs text-muted-foreground">{config.description}</div>
                    </div>
                  </button>
                )
              }
            )}
          </div>
        )}
      </div>

      {/* Lakeshore Models */}
      <div className="border rounded-lg bg-muted/30 overflow-hidden">
        <button
          onClick={() => setLakeshoreOpen(!lakeshoreOpen)}
          className="w-full flex items-center gap-2 px-3 py-2.5 text-sm hover:bg-muted/50 transition-colors"
        >
          <Building2 className="w-4 h-4 flex-shrink-0 text-green-500" />
          <span className="font-medium">Lakeshore Models</span>
          <span className="text-xs text-muted-foreground ml-auto mr-2">
            {LAKESHORE_MODEL_CONFIG[lakeshoreModel]?.label ?? lakeshoreModel}
          </span>
          {lakeshoreOpen ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
        </button>
        {lakeshoreOpen && (
          <div className="px-3 pb-3 space-y-1">
            {(Object.entries(LAKESHORE_MODEL_CONFIG) as [LakeshoreModel, typeof LAKESHORE_MODEL_CONFIG['lakeshore-qwen-1.5b']][]).map(
              ([modelKey, config]) => {
                const isSelected = lakeshoreModel === modelKey
                return (
                  <button
                    key={modelKey}
                    onClick={() => handleLakeshoreModelChange(modelKey)}
                    className={`
                      w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left text-sm
                      transition-colors
                      ${isSelected
                        ? 'bg-primary/10 text-primary border border-primary/30'
                        : 'hover:bg-muted text-foreground'
                      }
                    `}
                  >
                    <ModelLogo model={modelKey} className="w-4 h-4 flex-shrink-0" />
                    <div className="min-w-0 flex-1">
                      <div className="font-medium">{config.label}</div>
                      <div className="text-xs text-muted-foreground">{config.description}</div>
                    </div>
                  </button>
                )
              }
            )}
          </div>
        )}
      </div>

      {/* Cloud Models */}
      <div className="border rounded-lg bg-muted/30 overflow-hidden">
        <button
          onClick={() => setCloudOpen(!cloudOpen)}
          className="w-full flex items-center gap-2 px-3 py-2.5 text-sm hover:bg-muted/50 transition-colors"
        >
          <Cloud className="w-4 h-4 flex-shrink-0 text-blue-500" />
          <span className="font-medium">Cloud Models</span>
          <span className="text-xs text-muted-foreground ml-auto mr-2">
            {CLOUD_PROVIDER_CONFIG[cloudProvider]?.label}
          </span>
          {cloudOpen ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
        </button>
        {cloudOpen && (
          <div className="px-3 pb-3 space-y-1">
            {(Object.entries(CLOUD_PROVIDER_CONFIG) as [CloudProvider, typeof CLOUD_PROVIDER_CONFIG['cloud-claude']][]).map(
              ([providerKey, config]) => {
                const isSelected = cloudProvider === providerKey
                return (
                  <button
                    key={providerKey}
                    onClick={() => handleCloudProviderChange(providerKey)}
                    className={`
                      w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left text-sm
                      transition-colors
                      ${isSelected
                        ? 'bg-primary/10 text-primary border border-primary/30'
                        : 'hover:bg-muted text-foreground'
                      }
                    `}
                  >
                    <ModelLogo model={providerKey} className="w-4 h-4 flex-shrink-0" />
                    <span className="font-medium flex-1">{config.label}</span>
                    <span className="text-xs text-muted-foreground">{config.provider}</span>
                  </button>
                )
              }
            )}
          </div>
        )}
      </div>

      {/**
       * Lakeshore Authentication Panel
       */}
      {showLakeshoreAuth && (
        <div className="border rounded-lg p-3 bg-yellow-500/10 border-yellow-500/30">
          {/* Centered title with warning icon */}
          <div className="flex items-center justify-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
            <h4 className="text-sm font-medium text-yellow-600 dark:text-yellow-400">
              Lakeshore Authentication Required
            </h4>
          </div>

          {/* Left-aligned content */}
          <p className="text-xs text-muted-foreground">
            To use the UIC HPC cluster, you need to authenticate with Globus Compute.
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            <strong>Disable VPN before authenticating.</strong> You can reconnect after.
          </p>

          {authError && (
            <p className="text-xs text-red-500 mt-2">{authError}</p>
          )}

          <button
            onClick={handleAuthenticate}
            disabled={isAuthenticating}
            className="mt-3 w-full flex items-center justify-center gap-2 px-3 py-2
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

          <p className="text-xs text-muted-foreground mt-2 text-center">
            A browser window will open for authentication.
          </p>
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
          <div className="flex items-center justify-center gap-2 text-green-600 dark:text-green-400">
            <Unlock className="w-4 h-4" />
            <span className="text-sm font-medium">Lakeshore authenticated</span>
          </div>
        </div>
      )}

      {/**
       * Session Stats (expanded by default)
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
          <div className="mt-3 space-y-2">
            {/* Total cost - full width row */}
            <div className="bg-muted/50 rounded-lg p-2 flex items-center justify-between">
              <div className="text-xs text-muted-foreground">Total Cost</div>
              <div className="text-sm font-semibold">${stats.totalCost.toFixed(4)}</div>
            </div>

            {/* Per-tier breakdown - 2x2 grid */}
            <div className="grid grid-cols-2 gap-1">
              <div className="bg-muted/50 rounded-lg p-2 flex items-center justify-center gap-1.5 text-sm font-medium">
                <span>{stats.queries}</span>
                <span className="text-xs text-muted-foreground font-normal">Total</span>
              </div>
              <div className="bg-muted/50 rounded-lg p-2 flex items-center justify-center gap-1.5 text-sm font-medium">
                <Laptop className="w-4 h-4 text-orange-500" />
                <span>{stats.localQueries}</span>
                <span className="text-xs text-muted-foreground font-normal">Local</span>
              </div>
              <div className="bg-muted/50 rounded-lg p-2 flex items-center justify-center gap-1.5 text-sm font-medium">
                <Building2 className="w-4 h-4 text-green-500" />
                <span>{stats.lakeshoreQueries}</span>
                <span className="text-xs text-muted-foreground font-normal">Lakeshore</span>
              </div>
              <div className="bg-muted/50 rounded-lg p-2 flex items-center justify-center gap-1.5 text-sm font-medium">
                <Cloud className="w-4 h-4 text-blue-500" />
                <span>{stats.cloudQueries}</span>
                <span className="text-xs text-muted-foreground font-normal">Cloud</span>
              </div>
            </div>
          </div>
        )}
      </div>

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
                        <ModelLogo model={config.model} className="w-4 h-4 flex-shrink-0" />
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
       * Example Queries
       */}
      <div className="border-t pt-3 mt-4">
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
