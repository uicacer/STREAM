/**
 * health.ts - Tier Health API Client
 * ===================================
 *
 * Fetches tier availability status from the backend.
 * Used by healthStore to poll for real-time updates.
 */

export interface TierStatus {
  available: boolean
  error: string | null
  error_type: 'auth' | 'connection' | 'timeout' | 'unknown' | null  // Why is it unavailable?
  last_check: string | null
  authenticated?: boolean // Lakeshore only
}

export interface TierHealthResponse {
  tiers: {
    local: TierStatus
    lakeshore: TierStatus
    cloud: TierStatus
  }
  available_tiers: string[]
  timestamp: string
}

/**
 * Fetch current health status of all tiers
 *
 * @param cloudProvider - Optional. If provided, checks health for this specific
 *                       cloud provider instead of the default (Claude).
 *                       Pass the user's selected provider (e.g., "cloud-gpt")
 *                       so the Cloud indicator shows the correct status.
 */
export async function fetchTierHealth(cloudProvider?: string): Promise<TierHealthResponse> {
  // Build URL with optional cloud_provider query param
  const url = cloudProvider
    ? `/health/tiers?cloud_provider=${encodeURIComponent(cloudProvider)}`
    : '/health/tiers'

  const response = await fetch(url)

  if (!response.ok) {
    throw new Error(`Failed to fetch tier health: ${response.status}`)
  }

  return response.json()
}

/**
 * Force refresh tier health checks
 *
 * Use this when tiers might have changed status and you don't
 * want to wait for the next poll interval.
 */
export async function refreshTierHealth(): Promise<{
  status: string
  available_tiers: string[]
  timestamp: string
}> {
  const response = await fetch('/health/tiers/refresh', {
    method: 'POST',
  })

  if (!response.ok) {
    throw new Error(`Failed to refresh tier health: ${response.status}`)
  }

  return response.json()
}
