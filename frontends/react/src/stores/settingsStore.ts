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
import type { Tier, JudgeStrategy, ChatSettings, CloudProvider, LocalModel, LakeshoreModel } from '../types'

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
      cloudProvider: 'cloud-claude',      // Default cloud provider
      _initialized: false,

      // ============= Actions =============

      setTier: (tier) => set({ tier }),

      setJudgeStrategy: (judgeStrategy) => set({ judgeStrategy }),

      setTemperature: (temperature) => set({ temperature }),

      setLocalModel: (localModel) => set({ localModel }),

      setLakeshoreModel: (lakeshoreModel) => set({ lakeshoreModel }),

      setCloudProvider: (cloudProvider) => set({ cloudProvider }),

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
      version: 5,
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
