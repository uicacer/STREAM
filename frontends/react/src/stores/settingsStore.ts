/**
 * settingsStore.ts - Global State for User Settings
 * ==================================================
 *
 * This store manages user preferences like:
 * - Which tier to use (auto, local, lakeshore, cloud)
 * - Which judge model (for auto-routing)
 * - Temperature (creativity level)
 *
 * SINGLE SOURCE OF TRUTH PRINCIPLE:
 * ---------------------------------
 * - DEFAULTS come from the backend (/v1/config)
 * - USER PREFERENCES are stored in localStorage
 * - On first load: fetch defaults from backend
 * - On subsequent loads: use user's saved preferences
 *
 * This prevents frontend/backend config drift.
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Tier, JudgeStrategy, ChatSettings, CloudProvider, LocalModel, LakeshoreModel, WebSearchProvider } from '../types'

/**
 * SettingsState - The shape of our settings store
 */
interface SettingsState {
  // ============= Settings Values =============
  tier: Tier
  judgeStrategy: JudgeStrategy
  temperature: number
  theme: 'system' | 'light' | 'dark'
  localModel: LocalModel
  lakeshoreModel: LakeshoreModel
  cloudProvider: CloudProvider

  // Web Search — internet connectivity for LLM queries
  webSearch: boolean
  webSearchProvider: WebSearchProvider
  tavilyApiKey: string
  serperApiKey: string

  // Cloud API Keys — user-provided keys for cloud model access.
  // Stored in localStorage (never sent to the server for storage).
  // Sent with each chat request so the backend can authenticate
  // with the cloud provider on the user's behalf.
  openrouterApiKey: string
  anthropicApiKey: string
  openaiApiKey: string

  // Favorite models — user's pinned models from the OpenRouter catalog.
  // These appear at the top of the model selector for quick access.
  favoriteModels: string[]

  /**
   * Has the store been initialized with backend defaults?
   * Used to track if we need to fetch config on first load.
   */
  _initialized: boolean

  // ============= Actions =============
  setTier: (tier: Tier) => void
  setJudgeStrategy: (strategy: JudgeStrategy) => void
  setTemperature: (temp: number) => void
  setTheme: (theme: 'system' | 'light' | 'dark') => void
  setLocalModel: (model: LocalModel) => void
  setLakeshoreModel: (model: LakeshoreModel) => void
  setCloudProvider: (provider: CloudProvider) => void
  setWebSearch: (enabled: boolean) => void
  setWebSearchProvider: (provider: WebSearchProvider) => void
  setTavilyApiKey: (key: string) => void
  setSerperApiKey: (key: string) => void
  setOpenrouterApiKey: (key: string) => void
  setAnthropicApiKey: (key: string) => void
  setOpenaiApiKey: (key: string) => void
  toggleFavoriteModel: (modelId: string) => void
  getSettings: () => ChatSettings

  /**
   * Initialize store with backend defaults (called on app startup)
   * Only applies defaults if user hasn't set preferences yet.
   */
  initializeFromBackend: (defaults: {
    tier: string
    judgeStrategy: string
    temperature: number
  }) => void
}

/**
 * useSettingsStore - Zustand store with localStorage persistence
 *
 * HOW IT WORKS:
 * 1. First time user:
 *    - localStorage is empty
 *    - Store uses fallback defaults below
 *    - App calls initializeFromBackend() with backend defaults
 *    - Backend defaults are applied and saved to localStorage
 *
 * 2. Returning user:
 *    - localStorage has their preferences
 *    - persist() middleware loads them automatically
 *    - initializeFromBackend() sees _initialized=true, does nothing
 *    - User's preferences are preserved
 */
export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      // ============= Fallback Defaults =============
      // These are ONLY used if:
      // 1. localStorage is empty AND
      // 2. Backend /v1/config fetch fails
      //
      // In normal operation, backend defaults override these.

      tier: 'auto',
      judgeStrategy: 'ollama-3b',  // Safe fallback (matches backend default)
      temperature: 0.7,
      theme: 'dark',
      localModel: 'local-llama',          // Default local model (3B)
      lakeshoreModel: 'lakeshore-qwen-1.5b',   // Default lakeshore model
      cloudProvider: 'cloud-or-claude',   // Default: Claude via OpenRouter (one key for all)
      webSearch: false,                   // Web search off by default
      webSearchProvider: 'duckduckgo',    // Free, no API key needed
      tavilyApiKey: '',                   // User provides if they choose Tavily
      serperApiKey: '',                   // User provides if they choose Google (via Serper.dev)
      openrouterApiKey: '',              // User's OpenRouter API key (one key for all models)
      anthropicApiKey: '',               // User's direct Anthropic API key
      openaiApiKey: '',                  // User's direct OpenAI API key
      favoriteModels: [],                // User's pinned models from the catalog
      _initialized: false,

      // ============= Actions =============

      setTier: (tier) => set({ tier }),

      setJudgeStrategy: (judgeStrategy) => set({ judgeStrategy }),

      setTemperature: (temperature) => set({ temperature }),

      setLocalModel: (localModel) => set({ localModel }),

      setLakeshoreModel: (lakeshoreModel) => set({ lakeshoreModel }),

      setCloudProvider: (cloudProvider) => set({ cloudProvider }),

      setWebSearch: (webSearch) => set({ webSearch }),

      setWebSearchProvider: (webSearchProvider) => set({ webSearchProvider }),

      setTavilyApiKey: (tavilyApiKey) => set({ tavilyApiKey }),

      setSerperApiKey: (serperApiKey) => set({ serperApiKey }),

      setOpenrouterApiKey: (openrouterApiKey) => set({ openrouterApiKey }),

      setAnthropicApiKey: (anthropicApiKey) => set({ anthropicApiKey }),

      setOpenaiApiKey: (openaiApiKey) => set({ openaiApiKey }),

      toggleFavoriteModel: (modelId) => set((state) => ({
        favoriteModels: state.favoriteModels.includes(modelId)
          ? state.favoriteModels.filter(id => id !== modelId)
          : [...state.favoriteModels, modelId],
      })),

      setTheme: (theme) => {
        set({ theme })

        // Apply theme to document
        const root = document.documentElement
        if (theme === 'dark') {
          root.classList.add('dark')
        } else if (theme === 'light') {
          root.classList.remove('dark')
        } else {
          // 'system' - check OS preference
          const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
          root.classList.toggle('dark', prefersDark)
        }
      },

      getSettings: () => {
        const state = get()
        return {
          tier: state.tier,
          judgeStrategy: state.judgeStrategy,
          temperature: state.temperature,
          localModel: state.localModel,
          lakeshoreModel: state.lakeshoreModel,
          cloudProvider: state.cloudProvider,
          webSearch: state.webSearch,
          webSearchProvider: state.webSearchProvider,
        }
      },

      /**
       * Initialize with backend defaults
       *
       * IMPORTANT: Only applies defaults if _initialized is false.
       * This means:
       * - First time users get backend defaults
       * - Returning users keep their saved preferences
       */
      initializeFromBackend: (defaults) => {
        const state = get()

        // Already initialized (returning user with saved preferences)
        if (state._initialized) {
          console.log('[Settings] Using saved user preferences')
          return
        }

        // First time user - apply backend defaults
        console.log('[Settings] Applying backend defaults:', defaults)
        set({
          tier: defaults.tier as Tier,
          judgeStrategy: defaults.judgeStrategy as JudgeStrategy,
          temperature: defaults.temperature,
          _initialized: true,
        })
      },
    }),
    {
      name: 'stream-settings',
      /**
       * Version for migrations - increment when storage format changes
       */
      version: 10,
      /**
       * Migration: Runs automatically when storage version changes.
       * Each version upgrade fixes a specific issue with persisted state.
       */
      migrate: (persistedState: unknown, version: number) => {
        const state = persistedState as Record<string, unknown>

        if (version < 2) {
          // v1 → v2: Remove persisted tier, reset to use backend default
          // This fixes the bug where users got stuck on "cloud" tier
          console.log('[Settings] Migration v2: removing persisted tier')
          const { tier: _removedTier, ...rest } = state
          return {
            ...rest,
            _initialized: false, // Re-fetch backend defaults
          }
        }

        if (version < 3) {
          // v2 → v3: Switch from system theme to dark theme.
          // The app now defaults to dark mode for a comfortable reading
          // experience. This overrides any previously saved 'system' or
          // 'light' preference so existing users get the new default.
          console.log('[Settings] Migration v3: switching to dark theme')
          return {
            ...state,
            theme: 'dark',
          }
        }

        if (version < 4) {
          // v3 → v4: Add per-tier model selection.
          // Set defaults for existing users who don't have these fields yet.
          console.log('[Settings] Migration v4: adding per-tier model defaults')
          return {
            ...state,
            localModel: 'local-llama',
            lakeshoreModel: 'lakeshore-qwen-1.5b',
          }
        }

        if (version < 5) {
          // v4 → v5: Lakeshore multi-model upgrade.
          // Migrate from legacy "lakeshore-qwen" to new default "lakeshore-qwen-1.5b".
          console.log('[Settings] Migration v5: upgrading lakeshore model default')
          return {
            ...state,
            lakeshoreModel: 'lakeshore-qwen-1.5b',
          }
        }

        if (version < 6) {
          // v5 → v6: Multimodal support.
          // Remove obsolete local models (llama3.2:1b and llama3.1:8b).
          // Remove obsolete judge strategy (ollama-1b).
          // If user had selected a removed model/strategy, reset to default.
          console.log('[Settings] Migration v6: multimodal model cleanup')
          const removedLocalModels = ['local-llama-tiny', 'local-llama-quality']
          const removedJudgeStrategies = ['ollama-1b']
          return {
            ...state,
            localModel: removedLocalModels.includes(state.localModel as string)
              ? 'local-llama'
              : state.localModel,
            judgeStrategy: removedJudgeStrategies.includes(state.judgeStrategy as string)
              ? 'ollama-3b'
              : state.judgeStrategy,
          }
        }

        if (version < 7) {
          // v6 → v7: Web search support.
          // Add web search fields with safe defaults for existing users.
          console.log('[Settings] Migration v7: adding web search defaults')
          return {
            ...state,
            webSearch: false,
            webSearchProvider: 'duckduckgo',
            tavilyApiKey: '',
          }
        }

        if (version < 8) {
          // v7 → v8: Google Search (via Serper.dev) support.
          // Add Serper API key field, remove any old Google Custom Search fields.
          console.log('[Settings] Migration v8: adding Serper API key for Google search')
          const { googleApiKey: _removed1, googleCx: _removed2, ...rest } = state as Record<string, unknown>
          return {
            ...rest,
            serperApiKey: '',
          }
        }

        if (version < 9) {
          // v8 → v9: OpenRouter integration.
          // Add cloud API key fields for user-provided keys.
          // Migrate cloudProvider from old direct-only names to OpenRouter
          // default so new and existing users benefit from aggregator access.
          console.log('[Settings] Migration v9: adding OpenRouter and cloud API keys')

          // Map old direct provider names to their OpenRouter equivalents.
          // Users who had "cloud-claude" selected now get "cloud-or-claude",
          // which routes through OpenRouter instead of requiring a direct
          // Anthropic API key. They can still switch to direct mode if they
          // have their own provider keys.
          const providerMigration: Record<string, string> = {
            'cloud-claude': 'cloud-or-claude',
            'cloud-gpt': 'cloud-or-gpt4o',
            'cloud-gpt-cheap': 'cloud-or-gpt4o-mini',
          }
          const oldProvider = state.cloudProvider as string
          const newProvider = providerMigration[oldProvider] || oldProvider

          return {
            ...state,
            cloudProvider: newProvider,
            openrouterApiKey: '',
            anthropicApiKey: '',
            openaiApiKey: '',
            favoriteModels: [],
          }
        }

        if (version < 10) {
          // v9 → v10: Frontier model refresh.
          // Removed GPT-4o Mini and Llama 3.1 70B from curated list.
          // Map users who had those selected to their frontier replacements.
          console.log('[Settings] Migration v10: updating to frontier curated models')
          const state = persistedState as Record<string, unknown>
          const retiredMigration: Record<string, string> = {
            'cloud-or-gpt4o-mini': 'cloud-or-gemini-flash',
            'cloud-or-llama-70b': 'cloud-or-llama-maverick',
          }
          const currentProvider = state.cloudProvider as string
          const updated = retiredMigration[currentProvider] || currentProvider
          return { ...state, cloudProvider: updated }
        }

        return state
      },
      /**
       * Only persist these fields to localStorage
       *
       * NOTE: We intentionally DO NOT persist `tier` here.
       * The tier should always come from the backend default ("auto")
       * so that auto-routing works correctly. Users can change it
       * during a session, but it resets on refresh.
       *
       * This prevents the bug where users get "stuck" on cloud tier.
       */
      partialize: (state) => ({
        // tier is NOT persisted - always uses backend default
        judgeStrategy: state.judgeStrategy,
        temperature: state.temperature,
        theme: state.theme,
        localModel: state.localModel,           // Persist per-tier model choices
        lakeshoreModel: state.lakeshoreModel,
        cloudProvider: state.cloudProvider,
        // Web search preferences — persisted so users don't have to reconfigure
        // webSearch toggle is NOT persisted (resets to off each session)
        webSearchProvider: state.webSearchProvider,
        tavilyApiKey: state.tavilyApiKey,
        serperApiKey: state.serperApiKey,
        // Cloud API keys — persisted so users don't re-enter keys each session.
        // These are stored in localStorage only, never sent to the server
        // for persistent storage. They're included in chat requests so the
        // backend can authenticate with the provider on the user's behalf.
        openrouterApiKey: state.openrouterApiKey,
        anthropicApiKey: state.anthropicApiKey,
        openaiApiKey: state.openaiApiKey,
        favoriteModels: state.favoriteModels,
        _initialized: state._initialized,
      }),
    }
  )
)

/**
 * Initialize theme on app load
 */
if (typeof window !== 'undefined') {
  const theme = useSettingsStore.getState().theme
  useSettingsStore.getState().setTheme(theme)
}
