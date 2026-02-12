/**
 * config.ts - Fetch Application Configuration from Backend
 * =========================================================
 *
 * PRINCIPLE: Single Source of Truth
 * ---------------------------------
 * All model/routing configuration is defined in the Python backend.
 * The frontend fetches this configuration on startup instead of
 * defining its own defaults.
 *
 * WHY?
 * - Prevents frontend/backend config drift (like ollama-1b vs ollama-3b bug)
 * - Backend is the authority on model logic
 * - Frontend just displays what backend tells it
 * - Easier maintenance - change config in one place
 */

/**
 * Backend configuration response shape
 */
export interface AppConfig {
  version: string

  tiers: {
    available: string[]
    all: string[]
    default: string
    info: Record<string, {
      name: string
      description: string
    }>
  }

  judge: {
    strategies: string[]
    default: string
    info: Record<string, {
      description: string
      timeout: number
    }>
  }

  models: {
    default_by_tier: Record<string, string>
  }

  defaults: {
    tier: string
    judgeStrategy: string
    temperature: number
  }
}

/**
 * Cached config to avoid repeated fetches
 */
let cachedConfig: AppConfig | null = null

/**
 * Fetch application configuration from the backend
 *
 * This should be called once on app startup.
 * The response is cached to avoid repeated network calls.
 *
 * @returns AppConfig from the backend
 * @throws Error if fetch fails
 */
export async function fetchConfig(): Promise<AppConfig> {
  // Return cached config if available
  if (cachedConfig) {
    return cachedConfig
  }

  const response = await fetch('/v1/config')

  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.status}`)
  }

  cachedConfig = await response.json()
  return cachedConfig!
}

/**
 * Get cached config (or null if not yet fetched)
 *
 * Use this in components that render before config is loaded.
 */
export function getCachedConfig(): AppConfig | null {
  return cachedConfig
}

/**
 * Clear cached config (useful for testing or forced refresh)
 */
export function clearConfigCache(): void {
  cachedConfig = null
}
