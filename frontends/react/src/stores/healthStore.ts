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
 * The backend runs in Docker and cannot check Globus auth status (tokens are on host).
 * So we also fetch auth status from the local auth_server.py (port 8765) and merge it
 * with the health data for Lakeshore. This gives us accurate auth + availability info.
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
  error: string | null

  // Actions
  fetchHealth: () => Promise<void>
  forceRefresh: () => Promise<void>
  startPolling: () => void
  stopPolling: () => void
}

let pollInterval: ReturnType<typeof setInterval> | null = null

export const useHealthStore = create<HealthState>((set, get) => ({
  // Initial state
  local: null,
  lakeshore: null,
  cloud: null,
  lastUpdate: null,
  isLoading: false,
  error: null,

  // Fetch current health status
  fetchHealth: async () => {
    // Prevent concurrent fetches - if already loading, skip this call
    // This avoids race conditions where multiple fetches complete out of order
    if (get().isLoading) {
      console.log('[healthStore] fetchHealth skipped - already loading')
      return
    }

    try {
      set({ isLoading: true, error: null })
      console.log('[healthStore] fetchHealth started, isLoading=true')

      // Get user's selected cloud provider to check the RIGHT provider's health
      // This ensures the Cloud indicator shows green if GPT works, even if Claude has billing issues
      const cloudProvider = useSettingsStore.getState().cloudProvider
      console.log('[healthStore] Fetching health for cloudProvider:', cloudProvider)

      // Fetch health from backend (Docker) and auth from local auth server in parallel
      const [healthData, authData] = await Promise.all([
        fetchTierHealth(cloudProvider),
        checkAuthStatus().catch(() => ({ authenticated: false })), // Don't fail if auth server is down
      ])

      // Merge auth status into Lakeshore data
      // The backend (Docker) can't check Globus auth, but local auth_server.py can
      const lakeshoreWithAuth: TierStatus = {
        ...healthData.tiers.lakeshore,
        authenticated: authData.authenticated, // Use auth from local server, not Docker
      }

      console.log('[healthStore] fetchHealth completed, cloud available:', healthData.tiers.cloud.available)
      set({
        local: healthData.tiers.local,
        lakeshore: lakeshoreWithAuth,
        cloud: healthData.tiers.cloud,
        lastUpdate: healthData.timestamp,
        isLoading: false,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch health'
      console.log('[healthStore] fetchHealth error:', message)
      set({ error: message, isLoading: false })
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
 * - Lakeshore: red if not authenticated, yellow if authenticated but HPC down, green if ready
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

  // Lakeshore has special authentication logic
  if (tier === 'lakeshore') {
    if (tierStatus.authenticated !== true) {
      return { color: 'bg-red-500', tooltip: 'Not authenticated' }
    }
    if (!tierStatus.available) {
      return { color: 'bg-yellow-500', tooltip: 'HPC unavailable' }
    }
    return { color: 'bg-green-500', tooltip: 'Available' }
  }

  // Other tiers: simple available/unavailable
  if (tierStatus.available) {
    return { color: 'bg-green-500', tooltip: 'Available' }
  }
  return { color: 'bg-red-500', tooltip: 'Unavailable' }
}
