/**
 * healthStore.ts - Tier Health Status Store
 * ==========================================
 *
 * Manages tier availability status with ON-DEMAND checks only.
 *
 * NO PERIODIC POLLING:
 * Health checks are expensive — especially Lakeshore, which submits a real
 * 1-token inference job through Globus Compute to verify the GPU is running.
 * With thousands of users, polling every 30 seconds would overwhelm Lakeshore
 * with health check jobs that burn GPU time and Globus Compute quota.
 *
 * Instead, health is checked only when the user takes an action:
 *   - App startup: checks Local and Cloud with the user's selected models
 *     (e.g., verifies "gemma3:4b" is in Ollama, not just the default 3B).
 *     Lakeshore stays Level 1 (auth check only — no GPU job).
 *   - User changes tier: Level 2 check (verifies the specific model works)
 *   - User changes model: Level 2 check for the new model
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

interface HealthState {
  // Tier status
  local: TierStatus | null
  lakeshore: TierStatus | null
  cloud: TierStatus | null

  // Meta
  lastUpdate: string | null
  isLoading: boolean
  changingTier: string | null // which tier is being re-checked (null = none)
  error: string | null

  // Actions
  fetchHealth: () => Promise<void>
  fetchHealthForModelChange: (tier: string) => Promise<void>
  forceRefresh: () => Promise<void>
  markTierFailed: (tier: string) => void
}

// Request counter: newer requests override older ones instead of being skipped
let currentRequestId = 0

export const useHealthStore = create<HealthState>((set, get) => ({
  // Initial state
  local: null,
  lakeshore: null,
  cloud: null,
  lastUpdate: null,
  isLoading: false,
  changingTier: null,
  error: null,

  // Fetch health status on app startup.
  // Sends local model and cloud provider so the backend checks the actual
  // models the user has selected (e.g., verifies "llama3.1:8b" is in Ollama,
  // not just the default 3B). Lakeshore model is NOT sent — that would
  // trigger an expensive Level 2 check (1-token GPU job via Globus Compute).
  // Local and Cloud Level 2 checks are fast (~100ms and ~1s respectively).
  fetchHealth: async () => {
    const requestId = ++currentRequestId

    try {
      set({ isLoading: true, error: null })

      // Send local model + cloud provider so the backend verifies the
      // user's actual selections. Do NOT send lakeshore model — that
      // triggers a 10-30s GPU job via Globus Compute.
      const { cloudProvider, localModel } = useSettingsStore.getState()

      const [healthData, authData] = await Promise.all([
        fetchTierHealth(cloudProvider, localModel, undefined),
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
        changingTier: null,
      })
    } catch (err) {
      if (requestId !== currentRequestId) return
      const message = err instanceof Error ? err.message : 'Failed to fetch health'
      set({ error: message, isLoading: false, changingTier: null })
    }
  },

  // Fetch health after user switches a specific tier's model (shows spinner on that tier)
  fetchHealthForModelChange: async (tier: string) => {
    const requestId = ++currentRequestId
    const startTime = Date.now()
    const MIN_LOADING_TIME = 600

    try {
      set({ isLoading: true, changingTier: tier, error: null })

      // Only send the model param for the tier that changed.
      // Sending ALL model params triggers expensive Level 2 checks for every tier
      // (e.g., changing the local model would trigger a 20s Lakeshore GPU job).
      const { cloudProvider, localModel, lakeshoreModel } = useSettingsStore.getState()
      console.log(`[healthStore] Model change on ${tier}: checking health`)

      const [healthData, authData] = await Promise.all([
        fetchTierHealth(
          tier === 'cloud' ? cloudProvider : undefined,
          tier === 'local' ? localModel : undefined,
          tier === 'lakeshore' ? lakeshoreModel : undefined,
        ),
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
        changingTier: null,
      })
    } catch (err) {
      if (requestId !== currentRequestId) return
      const message = err instanceof Error ? err.message : 'Failed to fetch health'

      const elapsed = Date.now() - startTime
      if (elapsed < MIN_LOADING_TIME) {
        await new Promise(resolve => setTimeout(resolve, MIN_LOADING_TIME - elapsed))
      }

      if (requestId !== currentRequestId) return
      set({ error: message, isLoading: false, changingTier: null })
    }
  },

  // Force refresh (bypasses backend cache, then re-fetches)
  // Used when user explicitly wants to re-check all tiers (e.g., after
  // authenticating with Globus or starting Ollama).
  forceRefresh: async () => {
    try {
      set({ isLoading: true, error: null })
      await refreshTierHealth()
      // After cache invalidation, fetch fresh Level 1 status
      await get().fetchHealth()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to refresh'
      set({ error: message, isLoading: false })
    }
  },

  // Mark a tier as unavailable immediately (no network request).
  // Called when the SSE stream reports a runtime fallback — meaning the
  // backend tried this tier, inference failed, and it fell back to another.
  // The backend already called mark_tier_unavailable() in streaming.py;
  // this mirrors that on the frontend so the health dot turns red instantly
  // without waiting for the next health poll.
  markTierFailed: (tier: string) => {
    const key = tier as 'local' | 'lakeshore' | 'cloud'
    const current = get()[key]
    if (current) {
      set({ [key]: { ...current, available: false, error: 'Inference failed' } })
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
