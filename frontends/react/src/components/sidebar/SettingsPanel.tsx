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

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Bot,
  Laptop,
  Building2,
  Cloud,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  BarChart3,
  Lightbulb,
  Lock,
  Unlock,
  Loader2,
  AlertTriangle,
  Globe,
  Eye,
  EyeOff,
  Thermometer,
  Scale,
  Key,
  Search,
  Star,
  Check,
  X,
  ExternalLink,
} from 'lucide-react'
import { ModelLogo, OpenRouterLogo, DuckDuckGoLogo, TavilyLogo, GoogleLogo } from '../icons/ProviderLogos'
import { useSettingsStore } from '../../stores/settingsStore'
import { useChatStore } from '../../stores/chatStore'
import { useHealthStore, getTierDisplayInfo } from '../../stores/healthStore'
import { authenticateGlobus } from '../../api/auth'
import { validateApiKey, fetchModelCatalog, type CatalogModel } from '../../api/models'
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
  'ollama-3b': {
    model: 'llama',
    label: 'Llama 3B',
    description: 'Balanced, free',
  },
  'gemma-vision': {
    model: 'gemma',
    label: 'Gemma Vision 4B',
    description: 'Can analyze images, free',
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
  'local-llama': {
    label: 'Llama 3.2 3B',
    description: 'Text-only, balanced speed & quality',
  },
  'local-vision': {
    label: 'Gemma 3 4B',
    description: 'Text + Vision (handles images)',
  },
}

const LAKESHORE_MODEL_CONFIG: Record<LakeshoreModel, { label: string; description: string }> = {
  'lakeshore-qwen-vl-72b': {
    label: 'Qwen 2.5 VL 72B',
    description: 'Text + Vision multimodal (AWQ)',
  },
}

/**
 * Cloud provider configuration — curated models for the "quick pick" section.
 *
 * These are organized into two groups:
 *   1. OpenRouter models (cloud-or-*) — accessible with one OpenRouter key
 *   2. Direct provider models (cloud-*) — require individual provider keys
 *
 * The "Browse All" section (Phase 2) dynamically fetches 500+ models
 * from OpenRouter's catalog API.
 */
const OPENROUTER_MODELS: {
  id: string
  label: string
  description: string
  pricing: string
  tags: string[]
}[] = [
  { id: 'cloud-or-claude', label: 'Claude Sonnet 4', description: 'Best balance of capability & cost', pricing: '$3 / $15', tags: ['multimodal', 'reasoning'] },
  { id: 'cloud-or-gpt4o', label: 'GPT-4o', description: 'OpenAI flagship multimodal', pricing: '$2.50 / $10', tags: ['multimodal'] },
  { id: 'cloud-or-gemini-flash', label: 'Gemini 2.5 Flash', description: 'Very fast, 1M context', pricing: '$0.30 / $2.50', tags: ['multimodal'] },
  { id: 'cloud-or-deepseek-r1', label: 'DeepSeek R1', description: 'Top reasoning at low cost', pricing: '$0.70 / $2.50', tags: ['reasoning'] },
  { id: 'cloud-or-deepseek-v3', label: 'DeepSeek V3', description: 'Powerful & extremely affordable', pricing: '$0.38 / $0.89', tags: [] },
  { id: 'cloud-or-llama-maverick', label: 'Llama 4 Maverick', description: 'Best open-source, 1M context', pricing: '$0.17 / $0.60', tags: ['multimodal'] },
  { id: 'cloud-or-glm5', label: 'GLM-5', description: 'Near-Opus capability, fraction of the cost', pricing: '$0.95 / $2.55', tags: ['reasoning'] },
]

const DIRECT_MODELS: { id: string; label: string; provider: string; keyType: 'anthropic' | 'openai' }[] = [
  { id: 'cloud-claude', label: 'Claude Sonnet 4', provider: 'Anthropic', keyType: 'anthropic' },
  { id: 'cloud-gpt', label: 'GPT-4o', provider: 'OpenAI', keyType: 'openai' },
  { id: 'cloud-gpt-cheap', label: 'GPT-4o Mini', provider: 'OpenAI', keyType: 'openai' },
]

/**
 * Get display name for any cloud provider (curated or dynamic).
 * Used in the collapsed section header to show the current selection.
 */
function getCloudProviderLabel(providerId: string): string {
  const orModel = OPENROUTER_MODELS.find(m => m.id === providerId)
  if (orModel) return orModel.label
  const directModel = DIRECT_MODELS.find(m => m.id === providerId)
  if (directModel) return directModel.label
  if (providerId.startsWith('cloud-or-dynamic-')) {
    const parts = providerId.replace('cloud-or-dynamic-', '').split('/')
    return parts[parts.length - 1] || providerId
  }
  return providerId
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
  const webSearchProvider = useSettingsStore((state) => state.webSearchProvider)
  const setWebSearchProvider = useSettingsStore((state) => state.setWebSearchProvider)
  const tavilyApiKey = useSettingsStore((state) => state.tavilyApiKey)
  const setTavilyApiKey = useSettingsStore((state) => state.setTavilyApiKey)
  const serperApiKey = useSettingsStore((state) => state.serperApiKey)
  const setSerperApiKey = useSettingsStore((state) => state.setSerperApiKey)
  const openrouterApiKey = useSettingsStore((state) => state.openrouterApiKey)
  const setOpenrouterApiKey = useSettingsStore((state) => state.setOpenrouterApiKey)
  const anthropicApiKey = useSettingsStore((state) => state.anthropicApiKey)
  const setAnthropicApiKey = useSettingsStore((state) => state.setAnthropicApiKey)
  const openaiApiKey = useSettingsStore((state) => state.openaiApiKey)
  const setOpenaiApiKey = useSettingsStore((state) => state.setOpenaiApiKey)
  const favoriteModels = useSettingsStore((state) => state.favoriteModels)
  const toggleFavoriteModel = useSettingsStore((state) => state.toggleFavoriteModel)

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
  const [showTavilyKey, setShowTavilyKey] = useState(false)
  const [showSerperKey, setShowSerperKey] = useState(false)
  const [statsOpen, setStatsOpen] = useState(true)

  // Cloud section — API key visibility toggles and validation state
  const [showOpenrouterKey, setShowOpenrouterKey] = useState(false)
  const [showAnthropicKey, setShowAnthropicKey] = useState(false)
  const [showOpenaiKey, setShowOpenaiKey] = useState(false)
  const [directKeysOpen, setDirectKeysOpen] = useState(false)
  const [keyValidation, setKeyValidation] = useState<Record<string, { status: 'idle' | 'checking' | 'valid' | 'invalid'; error?: string }>>({})

  // Model catalog browser state
  const [catalogOpen, setCatalogOpen] = useState(false)
  const [catalogModels, setCatalogModels] = useState<CatalogModel[]>([])
  const [catalogFree, setCatalogFree] = useState<CatalogModel[]>([])
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [catalogSearch, setCatalogSearch] = useState('')
  const [catalogFilter, setCatalogFilter] = useState<'all' | 'free' | 'multimodal' | 'reasoning'>('all')

  // Debounced key validation — validates the key after the user stops typing.
  // Uses a 1-second debounce to avoid validating on every keystroke.
  const validationTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const validateKey = useCallback((provider: string, key: string) => {
    if (validationTimer.current) clearTimeout(validationTimer.current)
    if (!key || key.length < 10) {
      setKeyValidation(prev => ({ ...prev, [provider]: { status: 'idle' } }))
      return
    }
    setKeyValidation(prev => ({ ...prev, [provider]: { status: 'checking' } }))
    validationTimer.current = setTimeout(async () => {
      const result = await validateApiKey(provider, key)
      setKeyValidation(prev => ({
        ...prev,
        [provider]: {
          status: result.valid ? 'valid' : 'invalid',
          error: result.error,
        },
      }))
    }, 1000)
  }, [])

  // Fetch catalog when the "Browse All" section is expanded
  useEffect(() => {
    if (!catalogOpen) return
    setCatalogLoading(true)
    fetchModelCatalog(openrouterApiKey || undefined)
      .then((catalog) => {
        setCatalogModels(catalog.models)
        setCatalogFree(catalog.free)
      })
      .catch((err) => console.error('Failed to fetch model catalog:', err))
      .finally(() => setCatalogLoading(false))
  }, [catalogOpen, openrouterApiKey])

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
          <Laptop className="w-5 h-5 flex-shrink-0 text-orange-500" />
          <span className="font-medium">Local Models</span>
          <span className="text-xs text-muted-foreground ml-auto mr-2">
            {LOCAL_MODEL_CONFIG[localModel]?.label}
          </span>
          {localOpen ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
        </button>
        {localOpen && (
          <div className="px-3 pb-3 space-y-1 max-h-48 overflow-y-auto">
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
          <Building2 className="w-5 h-5 flex-shrink-0 text-green-500" />
          <span className="font-medium">Lakeshore Models</span>
          <span className="text-xs text-muted-foreground ml-auto mr-2">
            {LAKESHORE_MODEL_CONFIG[lakeshoreModel]?.label ?? lakeshoreModel}
          </span>
          {lakeshoreOpen ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
        </button>
        {lakeshoreOpen && (
          <div className="px-3 pb-3 space-y-1 max-h-48 overflow-y-auto">
            {(Object.entries(LAKESHORE_MODEL_CONFIG) as [LakeshoreModel, typeof LAKESHORE_MODEL_CONFIG['lakeshore-qwen-vl-72b']][]).map(
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

      {/* Cloud Models — redesigned with API keys and model catalog */}
      <div className="border rounded-lg bg-muted/30 overflow-hidden">
        <button
          onClick={() => setCloudOpen(!cloudOpen)}
          className="w-full flex items-center gap-2 px-3 py-2.5 text-sm hover:bg-muted/50 transition-colors"
        >
          <Cloud className="w-5 h-5 flex-shrink-0 text-blue-500" />
          <span className="font-medium">Cloud Models</span>
          <span className="text-xs text-muted-foreground ml-auto mr-2">
            {getCloudProviderLabel(cloudProvider)}
          </span>
          {cloudOpen ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
        </button>
        {cloudOpen && (
          <div className="px-3 pb-3 space-y-3">
            {/* ---- API Keys Section ---- */}
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wider pt-1">
                <Key className="w-3 h-3" />
                API Keys
              </div>

              {/* OpenRouter key — primary, recommended */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium">OpenRouter <span className="text-muted-foreground font-normal">(one key for all models)</span></label>
                  {keyValidation.openrouter?.status === 'checking' && <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />}
                  {keyValidation.openrouter?.status === 'valid' && <Check className="w-3 h-3 text-green-500" />}
                  {keyValidation.openrouter?.status === 'invalid' && <X className="w-3 h-3 text-red-500" />}
                </div>
                <div className="relative">
                  <input
                    type={showOpenrouterKey ? 'text' : 'password'}
                    value={openrouterApiKey}
                    onChange={(e) => {
                      setOpenrouterApiKey(e.target.value)
                      validateKey('openrouter', e.target.value)
                    }}
                    placeholder="sk-or-v1-..."
                    className="w-full px-2.5 py-1.5 pr-8 text-xs rounded-md border bg-background focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                  <button
                    onClick={() => setShowOpenrouterKey(!showOpenrouterKey)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showOpenrouterKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                  </button>
                </div>
                {keyValidation.openrouter?.status === 'invalid' && (
                  <p className="text-[10px] text-red-500">{keyValidation.openrouter.error}</p>
                )}
                <a
                  href="https://openrouter.ai/keys"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[10px] text-blue-500 hover:underline flex items-center gap-0.5"
                >
                  Get a free key at openrouter.ai/keys <ExternalLink className="w-2.5 h-2.5" />
                </a>
              </div>

              {/* Direct Provider Keys — collapsible "advanced" section */}
              <button
                onClick={() => setDirectKeysOpen(!directKeysOpen)}
                className="w-full flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors pt-1"
              >
                {directKeysOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Direct Provider Keys
              </button>
              {directKeysOpen && (
                <div className="space-y-2 pl-2 border-l-2 border-muted">
                  {/* Anthropic key */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium">Anthropic</label>
                      {keyValidation.anthropic?.status === 'checking' && <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />}
                      {keyValidation.anthropic?.status === 'valid' && <Check className="w-3 h-3 text-green-500" />}
                      {keyValidation.anthropic?.status === 'invalid' && <X className="w-3 h-3 text-red-500" />}
                    </div>
                    <div className="relative">
                      <input
                        type={showAnthropicKey ? 'text' : 'password'}
                        value={anthropicApiKey}
                        onChange={(e) => {
                          setAnthropicApiKey(e.target.value)
                          validateKey('anthropic', e.target.value)
                        }}
                        placeholder="sk-ant-..."
                        className="w-full px-2.5 py-1.5 pr-8 text-xs rounded-md border bg-background focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                      <button
                        onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        {showAnthropicKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                      </button>
                    </div>
                    {keyValidation.anthropic?.status === 'invalid' && (
                      <p className="text-[10px] text-red-500">{keyValidation.anthropic.error}</p>
                    )}
                    <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer" className="text-[10px] text-blue-500 hover:underline flex items-center gap-0.5">
                      Get key at console.anthropic.com <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  </div>

                  {/* OpenAI key */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium">OpenAI</label>
                      {keyValidation.openai?.status === 'checking' && <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />}
                      {keyValidation.openai?.status === 'valid' && <Check className="w-3 h-3 text-green-500" />}
                      {keyValidation.openai?.status === 'invalid' && <X className="w-3 h-3 text-red-500" />}
                    </div>
                    <div className="relative">
                      <input
                        type={showOpenaiKey ? 'text' : 'password'}
                        value={openaiApiKey}
                        onChange={(e) => {
                          setOpenaiApiKey(e.target.value)
                          validateKey('openai', e.target.value)
                        }}
                        placeholder="sk-..."
                        className="w-full px-2.5 py-1.5 pr-8 text-xs rounded-md border bg-background focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                      <button
                        onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        {showOpenaiKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                      </button>
                    </div>
                    {keyValidation.openai?.status === 'invalid' && (
                      <p className="text-[10px] text-red-500">{keyValidation.openai.error}</p>
                    )}
                    <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-[10px] text-blue-500 hover:underline flex items-center gap-0.5">
                      Get key at platform.openai.com <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  </div>
                </div>
              )}
            </div>

            {/* ---- Divider ---- */}
            <div className="border-t border-border" />

            {/* ---- Model Selection ---- */}
            <div className="space-y-2">
              {/* OpenRouter frontier models */}
              <div className="flex items-center justify-between">
                <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                  <OpenRouterLogo className="w-3 h-3" />
                  Via OpenRouter
                </div>
                <span className="text-[9px] text-muted-foreground/60 italic">in / out per 1M tokens</span>
              </div>
              <div className="space-y-1">
                {OPENROUTER_MODELS.map((model) => {
                  const isSelected = cloudProvider === model.id
                  return (
                    <button
                      key={model.id}
                      onClick={() => handleCloudProviderChange(model.id)}
                      className={`
                        w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left text-xs
                        transition-colors
                        ${isSelected
                          ? 'bg-primary/10 text-primary border border-primary/30'
                          : 'hover:bg-muted text-foreground'
                        }
                      `}
                    >
                      <ModelLogo model={model.id} className="w-3.5 h-3.5 flex-shrink-0" />
                      <span className="font-medium flex-1 truncate">{model.label}</span>
                      <span className="text-[10px] text-muted-foreground flex-shrink-0">{model.pricing}</span>
                    </button>
                  )
                })}
              </div>

              {/* Favorite models from catalog (if any) */}
              {favoriteModels.length > 0 && (
                <>
                  <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
                    <Star className="w-3 h-3" /> Favorites
                  </div>
                  <div className="space-y-1">
                    {favoriteModels.map((modelId) => {
                      const dynamicId = `cloud-or-dynamic-${modelId}`
                      const isSelected = cloudProvider === dynamicId
                      const parts = modelId.split('/')
                      const displayName = parts[parts.length - 1] || modelId
                      return (
                        <button
                          key={dynamicId}
                          onClick={() => handleCloudProviderChange(dynamicId)}
                          className={`
                            w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left text-xs
                            transition-colors
                            ${isSelected
                              ? 'bg-primary/10 text-primary border border-primary/30'
                              : 'hover:bg-muted text-foreground'
                            }
                          `}
                        >
                          <ModelLogo model={modelId} className="w-3.5 h-3.5 flex-shrink-0" />
                          <span className="font-medium flex-1 truncate">{displayName}</span>
                          <Star className="w-3 h-3 flex-shrink-0 text-yellow-500 fill-yellow-500" />
                        </button>
                      )
                    })}
                  </div>
                </>
              )}

              {/* Direct provider models */}
              {(anthropicApiKey || openaiApiKey) && (
                <>
                  <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                    Direct API
                  </div>
                  <div className="space-y-1">
                    {DIRECT_MODELS
                      .filter((m) =>
                        (m.keyType === 'anthropic' && anthropicApiKey) ||
                        (m.keyType === 'openai' && openaiApiKey)
                      )
                      .map((model) => {
                        const isSelected = cloudProvider === model.id
                        return (
                          <button
                            key={model.id}
                            onClick={() => handleCloudProviderChange(model.id)}
                            className={`
                              w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left text-xs
                              transition-colors
                              ${isSelected
                                ? 'bg-primary/10 text-primary border border-primary/30'
                                : 'hover:bg-muted text-foreground'
                              }
                            `}
                          >
                            <ModelLogo model={model.id} className="w-3.5 h-3.5 flex-shrink-0" />
                            <span className="font-medium flex-1 truncate">{model.label}</span>
                            <span className="text-[10px] text-muted-foreground flex-shrink-0">{model.provider} Direct</span>
                          </button>
                        )
                      })}
                  </div>
                </>
              )}

              {/* ---- Browse All Models (catalog) ---- */}
              <button
                onClick={() => setCatalogOpen(!catalogOpen)}
                className="w-full flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors pt-1"
              >
                {catalogOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Browse All Models <span className="text-[10px]">({catalogModels.length > 0 ? `${catalogModels.length}+` : '500+'})</span>
              </button>
              {catalogOpen && (
                <div className="space-y-2">
                  {/* Search bar */}
                  <div className="relative">
                    <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground" />
                    <input
                      type="text"
                      value={catalogSearch}
                      onChange={(e) => setCatalogSearch(e.target.value)}
                      placeholder="Search models..."
                      className="w-full pl-7 pr-2.5 py-1.5 text-xs rounded-md border bg-background focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  {/* Filter pills */}
                  <div className="flex gap-1 flex-wrap">
                    {([
                      { key: 'all', label: 'All' },
                      { key: 'free', label: 'Free' },
                      { key: 'multimodal', label: 'Multimodal' },
                      { key: 'reasoning', label: 'Reasoning' },
                    ] as const).map(({ key, label }) => (
                      <button
                        key={key}
                        onClick={() => setCatalogFilter(key)}
                        className={`px-2 py-0.5 text-[10px] rounded-full transition-colors ${
                          catalogFilter === key
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted text-muted-foreground hover:bg-muted/80'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  {/* Model list */}
                  {catalogLoading ? (
                    <div className="flex items-center justify-center py-4">
                      <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                      <span className="ml-2 text-xs text-muted-foreground">Loading catalog...</span>
                    </div>
                  ) : (
                    <div className="max-h-80 overflow-y-auto space-y-0.5">
                      {(catalogFilter === 'free' ? catalogFree : catalogModels)
                        .filter((m) => {
                          if (catalogFilter === 'multimodal' && !m.supports_vision) return false
                          if (catalogFilter === 'reasoning') {
                            const id = m.id.toLowerCase()
                            const isReasoning = id.includes('o1') || id.includes('o3') || id.includes('o4') || id.includes('deepseek-r1') || id.includes('qwq') || id.includes('glm') || m.name.toLowerCase().includes('reason')
                            if (!isReasoning) return false
                          }
                          if (!catalogSearch) return true
                          const q = catalogSearch.toLowerCase()
                          return m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q) || m.provider.toLowerCase().includes(q)
                        })
                        .slice(0, 200)
                        .map((model) => {
                          const dynamicId = `cloud-or-dynamic-${model.id}`
                          const isSelected = cloudProvider === dynamicId
                          const isFavorited = favoriteModels.includes(model.id)
                          return (
                            <div
                              key={model.id}
                              className={`
                                flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-xs
                                transition-colors cursor-pointer
                                ${isSelected
                                  ? 'bg-primary/10 text-primary border border-primary/30'
                                  : 'hover:bg-muted text-foreground'
                                }
                              `}
                              onClick={() => handleCloudProviderChange(dynamicId)}
                            >
                              <ModelLogo model={model.id} className="w-3.5 h-3.5 flex-shrink-0" />
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-1">
                                  <span className="font-medium truncate">{model.name}</span>
                                  {model.is_free && (
                                    <span className="px-1 py-0 text-[9px] bg-green-500/20 text-green-600 dark:text-green-400 rounded">FREE</span>
                                  )}
                                  {model.supports_vision && (
                                    <span className="px-1 py-0 text-[9px] bg-blue-500/20 text-blue-600 dark:text-blue-400 rounded">Vision</span>
                                  )}
                                </div>
                                <div className="text-[10px] text-muted-foreground flex items-center gap-2">
                                  <span>{model.provider}</span>
                                  <span>{model.pricing.prompt_display}/{model.pricing.completion_display}</span>
                                  {model.context_length > 0 && <span>{Math.round(model.context_length / 1000)}K ctx</span>}
                                </div>
                              </div>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  toggleFavoriteModel(model.id)
                                }}
                                className="p-0.5 hover:bg-muted rounded flex-shrink-0"
                                title={isFavorited ? 'Remove from favorites' : 'Add to favorites'}
                              >
                                <Star className={`w-3 h-3 ${isFavorited ? 'text-yellow-500 fill-yellow-500' : 'text-muted-foreground'}`} />
                              </button>
                            </div>
                          )
                        })}
                      {catalogModels.length === 0 && !catalogLoading && (
                        <p className="text-[10px] text-muted-foreground text-center py-2">
                          {openrouterApiKey ? 'No models found' : 'Enter an OpenRouter API key to browse models'}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
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
            <BarChart3 className="w-5 h-5" />
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
              <label className="text-sm text-muted-foreground block mb-1 flex items-center gap-2">
                <Thermometer className="w-4 h-4" />
                Temperature: {temperature.toFixed(1)}
              </label>
              <p className="text-xs text-muted-foreground mb-2">
                Controls how creative or focused the AI responses are. Lower values give more predictable answers.
              </p>
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
            <div className="border-t border-border/50 pt-4">
              <label className="text-sm text-muted-foreground block mb-1 flex items-center gap-2">
                <Scale className="w-4 h-4" />
                Complexity Judge {tier !== 'auto' && '(Auto mode only)'}
              </label>
              <p className="text-xs text-muted-foreground mb-2">
                In Auto mode, this model classifies your query's complexity to decide which tier handles it.
              </p>
              <div className="space-y-1">
                {(Object.entries(JUDGE_CONFIG) as [JudgeStrategy, typeof JUDGE_CONFIG['ollama-3b']][]).map(
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

            {/**
             * Web Search Provider
             */}
            <div className="border-t border-border/50 pt-4">
              <label className="text-sm text-muted-foreground block mb-2 flex items-center gap-2">
                <Globe className="w-4 h-4" />
                Web Search Provider
              </label>
              <p className="text-xs text-muted-foreground mb-2">
                When web search is enabled (globe icon in chat input), STREAM searches the internet
                for your query and includes results in the AI's context.
              </p>
              <div className="space-y-1">
                {/* DuckDuckGo option */}
                <button
                  onClick={() => setWebSearchProvider('duckduckgo')}
                  className={`
                    w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm
                    transition-colors
                    ${webSearchProvider === 'duckduckgo'
                      ? 'bg-primary/10 text-primary border border-primary/30'
                      : 'hover:bg-muted text-foreground'
                    }
                  `}
                >
                  <DuckDuckGoLogo className="w-4 h-4 flex-shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium">DuckDuckGo</div>
                    <div className="text-xs text-muted-foreground">
                      Free · No API key · Privacy-focused
                    </div>
                  </div>
                </button>

                {/* Tavily option */}
                <button
                  onClick={() => setWebSearchProvider('tavily')}
                  className={`
                    w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm
                    transition-colors
                    ${webSearchProvider === 'tavily'
                      ? 'bg-primary/10 text-primary border border-primary/30'
                      : 'hover:bg-muted text-foreground'
                    }
                  `}
                >
                  <TavilyLogo className="w-4 h-4 flex-shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium">Tavily</div>
                    <div className="text-xs text-muted-foreground">
                      AI-optimized · 1K free/mo · API key required
                    </div>
                  </div>
                </button>

                {/* Tavily config — appears directly below the Tavily button */}
                {webSearchProvider === 'tavily' && (
                  <div className="ml-6 mt-1 mb-2">
                    <label className="text-xs text-muted-foreground block mb-1">
                      API Key
                    </label>
                    <div className="relative">
                      <input
                        type={showTavilyKey ? 'text' : 'password'}
                        value={tavilyApiKey}
                        onChange={(e) => setTavilyApiKey(e.target.value)}
                        placeholder="tvly-..."
                        className="w-full px-3 py-1.5 pr-10 text-xs rounded-lg border border-muted-foreground/30 bg-background focus:outline-none focus:ring-2 focus:ring-ring focus:border-primary"
                      />
                      <button
                        onClick={() => setShowTavilyKey(!showTavilyKey)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        aria-label={showTavilyKey ? 'Hide API key' : 'Show API key'}
                      >
                        {showTavilyKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Get a free key at{' '}
                      <a href="https://tavily.com" target="_blank" rel="noopener noreferrer" className="text-primary underline">
                        tavily.com
                      </a>
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Free: 1,000 credits/month · Paid: from $30/mo for 4,000 credits
                    </p>
                  </div>
                )}

                {/* Google Search option */}
                <button
                  onClick={() => setWebSearchProvider('google')}
                  className={`
                    w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm
                    transition-colors
                    ${webSearchProvider === 'google'
                      ? 'bg-primary/10 text-primary border border-primary/30'
                      : 'hover:bg-muted text-foreground'
                    }
                  `}
                >
                  <GoogleLogo className="w-4 h-4 flex-shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium">Google Search <span className="font-normal text-muted-foreground">via Serper</span></div>
                    <div className="text-xs text-muted-foreground">
                      Google results · 2,500 free queries · API key required
                    </div>
                  </div>
                </button>

                {/* Google (Serper) config — appears directly below the Google button */}
                {webSearchProvider === 'google' && (
                  <div className="ml-6 mt-1 mb-2">
                    <label className="text-xs text-muted-foreground block mb-1">
                      API Key
                    </label>
                    <div className="relative">
                      <input
                        type={showSerperKey ? 'text' : 'password'}
                        value={serperApiKey}
                        onChange={(e) => setSerperApiKey(e.target.value)}
                        placeholder="serper-api-key..."
                        className="w-full px-3 py-1.5 pr-10 text-xs rounded-lg border border-muted-foreground/30 bg-background focus:outline-none focus:ring-2 focus:ring-ring focus:border-primary"
                      />
                      <button
                        onClick={() => setShowSerperKey(!showSerperKey)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        aria-label={showSerperKey ? 'Hide API key' : 'Show API key'}
                      >
                        {showSerperKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Get a free key at{' '}
                      <a href="https://serper.dev" target="_blank" rel="noopener noreferrer" className="text-primary underline">
                        serper.dev
                      </a>
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Free: 2,500 queries (one-time) · Paid: from $50 for 50,000 queries
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/**
       * Example Queries
       */}
      <div className="border-t border-border/50 pt-4 mt-6">
        <h3 className="text-sm font-medium mb-1 flex items-center gap-2 text-muted-foreground">
          <Lightbulb className="w-5 h-5" />
          Try These
        </h3>
        <p className="text-xs text-muted-foreground mb-2">
          Quick examples to get started
        </p>
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
