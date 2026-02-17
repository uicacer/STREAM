/**
 * healthStore.ts - Tier Health Status Store
 * ==========================================
 *
 * Manages real-time tier availability status with polling.
 * Updates every 30 seconds to show users which tiers are available.
 *
 * POLLING APPROACH:
 * - Polls /health/tiers every 30 seconds
 * - Also refreshes on window focus (when user returns to tab)
 * - Scales better than persistent connections for many users
 *
 * ARCHITECTURE NOTE:
 * In Docker mode, the backend cannot check Globus auth status (tokens are on host),
 * so we fetch auth status from auth_server.py (port 8765) running on the host.
 * In desktop mode, the backend checks auth directly (same machine), so we use the
 * backend's authenticated field when the auth helper is not running.
 */

import { create } from 'zustand'
import { fetchTierHealth, refreshTierHealth, type TierStatus } from '../api/health'
import { checkAuthStatus } from '../api/auth'
import { useSettingsStore } from './settingsStore'

const POLL_INTERVAL = 30000 // 30 seconds

interface HealthState {
  // Tier status
  local: TierStatus | null
  lakeshore: TierStatus | null
  cloud: TierStatus | null

  // Meta
  lastUpdate: string | null
  isLoading: boolean
  isProviderChanging: boolean // true ONLY when user switches cloud provider
  error: string | null

  // Actions
  fetchHealth: () => Promise<void>
  fetchHealthForProviderChange: () => Promise<void>
  forceRefresh: () => Promise<void>
  startPolling: () => void
  stopPolling: () => void
}

let pollInterval: ReturnType<typeof setInterval> | null = null

// Request counter: newer requests override older ones instead of being skipped
let currentRequestId = 0

export const useHealthStore = create<HealthState>((set, get) => ({
  // Initial state
  local: null,
  lakeshore: null,
  cloud: null,
  lastUpdate: null,
  isLoading: false,
  isProviderChanging: false,
  error: null,

  // Fetch current health status (background poll - no spinner)
  fetchHealth: async () => {
    const requestId = ++currentRequestId

    try {
      set({ isLoading: true, error: null })

      const cloudProvider = useSettingsStore.getState().cloudProvider
      const [healthData, authData] = await Promise.all([
        fetchTierHealth(cloudProvider),
        checkAuthStatus().catch(() => null),
      ])

      if (requestId !== currentRequestId) return

      // Determine auth status: prefer auth helper (Docker mode), fall back to
      // backend's value (desktop mode where auth helper doesn't run).
      let authenticated = false
      if (authData !== null) {
        // Auth helper is running (Docker mode) — use its result
        authenticated = authData.authenticated
      } else {
        // Auth helper not running (desktop mode) — use the backend's check,
        // which calls globus_is_authenticated() on ~/.globus_compute/storage.db
        authenticated = healthData.tiers.lakeshore.authenticated === true
      }

      const lakeshoreWithAuth: TierStatus = {
        ...healthData.tiers.lakeshore,
        authenticated: authenticated,
      }

      if (requestId !== currentRequestId) return

      set({
        local: healthData.tiers.local,
        lakeshore: lakeshoreWithAuth,
        cloud: healthData.tiers.cloud,
        lastUpdate: healthData.timestamp,
        isLoading: false,
        isProviderChanging: false,
      })
    } catch (err) {
      if (requestId !== currentRequestId) return
      const message = err instanceof Error ? err.message : 'Failed to fetch health'
      set({ error: message, isLoading: false, isProviderChanging: false })
    }
  },

  // Fetch health after user switches cloud provider (shows spinner)
  fetchHealthForProviderChange: async () => {
    const requestId = ++currentRequestId
    const startTime = Date.now()
    const MIN_LOADING_TIME = 600

    try {
      set({ isLoading: true, isProviderChanging: true, error: null })

      const cloudProvider = useSettingsStore.getState().cloudProvider
      console.log(`[healthStore] Provider change: fetching for ${cloudProvider}`)

      const [healthData, authData] = await Promise.all([
        fetchTierHealth(cloudProvider),
        checkAuthStatus().catch(() => null),
      ])

      if (requestId !== currentRequestId) return

      // Same auth logic as fetchHealth — prefer auth helper, fall back to backend
      let authenticated = false
      if (authData !== null) {
        authenticated = authData.authenticated
      } else {
        authenticated = healthData.tiers.lakeshore.authenticated === true
      }

      const lakeshoreWithAuth: TierStatus = {
        ...healthData.tiers.lakeshore,
        authenticated: authenticated,
      }

      // Ensure minimum loading time so spinner is visible
      const elapsed = Date.now() - startTime
      if (elapsed < MIN_LOADING_TIME) {
        await new Promise(resolve => setTimeout(resolve, MIN_LOADING_TIME - elapsed))
      }

      if (requestId !== currentRequestId) return

      set({
        local: healthData.tiers.local,
        lakeshore: lakeshoreWithAuth,
        cloud: healthData.tiers.cloud,
        lastUpdate: healthData.timestamp,
        isLoading: false,
        isProviderChanging: false,
      })
    } catch (err) {
      if (requestId !== currentRequestId) return
      const message = err instanceof Error ? err.message : 'Failed to fetch health'

      const elapsed = Date.now() - startTime
      if (elapsed < MIN_LOADING_TIME) {
        await new Promise(resolve => setTimeout(resolve, MIN_LOADING_TIME - elapsed))
      }

      if (requestId !== currentRequestId) return
      set({ error: message, isLoading: false, isProviderChanging: false })
    }
  },

  // Force refresh (bypasses backend cache)
  forceRefresh: async () => {
    try {
      set({ isLoading: true, error: null })
      await refreshTierHealth()
      // After refresh, fetch the new status
      await get().fetchHealth()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to refresh'
      set({ error: message, isLoading: false })
    }
  },

  // Start polling
  startPolling: () => {
    const beginPolling = () => {
      // Fetch immediately
      get().fetchHealth()

      // Stop any existing interval
      if (pollInterval) {
        clearInterval(pollInterval)
      }

      // Start new interval
      pollInterval = setInterval(() => {
        get().fetchHealth()
      }, POLL_INTERVAL)

      // Also refresh on window focus
      const handleFocus = () => {
        get().fetchHealth()
      }
      window.addEventListener('focus', handleFocus)

      // Store cleanup function
      ;(window as unknown as { _healthCleanup?: () => void })._healthCleanup = () => {
        window.removeEventListener('focus', handleFocus)
      }
    }

    // Wait for settingsStore to hydrate from localStorage before the first poll.
    // Without this, the first poll uses the default cloudProvider ('cloud-claude')
    // instead of the user's saved selection (e.g., 'cloud-gpt'), causing the
    // backend to test the wrong provider and report a false auth error.
    if (useSettingsStore.persist.hasHydrated()) {
      beginPolling()
    } else {
      useSettingsStore.persist.onFinishHydration(beginPolling)
    }
  },

  // Stop polling
  stopPolling: () => {
    if (pollInterval) {
      clearInterval(pollInterval)
      pollInterval = null
    }
    const cleanup = (window as unknown as { _healthCleanup?: () => void })._healthCleanup
    if (cleanup) {
      cleanup()
    }
  },
}))

/**
 * Helper to check if a tier is available
 */
export function isTierAvailable(tier: 'local' | 'lakeshore' | 'cloud'): boolean {
  const state = useHealthStore.getState()
  const status = state[tier]
  return status?.available ?? false
}

/**
 * Tier display info - single source of truth for status colors and tooltips
 */
export interface TierDisplayInfo {
  color: string   // Tailwind class e.g. 'bg-green-500'
  tooltip: string // Human-readable status message
}

/**
 * Get tier display info (color, tooltip) - SINGLE SOURCE OF TRUTH
 *
 * Status logic:
 * - Lakeshore: green if authenticated AND HPC available, red otherwise
 *   (auth status is shown separately via the "Lakeshore authenticated" panel)
 * - Other tiers: green if available, red if not
 * - Loading (null status): gray
 */
export function getTierDisplayInfo(
  tier: 'local' | 'lakeshore' | 'cloud',
  tierStatus: TierStatus | null
): TierDisplayInfo {
  // Loading state
  if (!tierStatus) {
    return { color: 'bg-gray-400', tooltip: 'Loading...' }
  }

  // Lakeshore: green only when fully ready (authenticated + HPC available)
  if (tier === 'lakeshore') {
    if (tierStatus.authenticated === true && tierStatus.available) {
      return { color: 'bg-green-500', tooltip: 'Available' }
    }
    const reason = tierStatus.authenticated !== true ? 'Not authenticated' : 'HPC unavailable'
    return { color: 'bg-red-500', tooltip: reason }
  }

  // Other tiers: simple available/unavailable
  if (tierStatus.available) {
    return { color: 'bg-green-500', tooltip: 'Available' }
  }
  return { color: 'bg-red-500', tooltip: 'Unavailable' }
}
