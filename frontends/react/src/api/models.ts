/**
 * models.ts - Cloud Model Catalog API Client
 * ===========================================
 *
 * This module provides functions for interacting with the backend model
 * catalog endpoint and API key validation endpoint.
 *
 * TWO KEY CAPABILITIES:
 *
 * 1. API Key Validation:
 *    Before the user starts chatting, we validate their API key to catch
 *    errors early. This makes a minimal test call (1 token) to the provider.
 *
 * 2. Dynamic Model Catalog:
 *    OpenRouter provides 500+ models. The backend proxies their catalog API
 *    and caches it for 1 hour. This module fetches and types that data.
 *
 * The frontend adds its own 5-minute cache on top of the backend's 1-hour
 * cache. This means:
 *   - First settings panel open → backend fetches from OpenRouter (~500ms)
 *   - Next 5 minutes → instant from frontend cache
 *   - After 5 minutes → backend serves its 1-hour cache (~50ms)
 *   - After 1 hour → backend refetches from OpenRouter
 */

// =============================================================================
// TYPES
// =============================================================================

/**
 * A single model from the OpenRouter catalog.
 *
 * This is the structured data returned by our backend's /v1/models/catalog
 * endpoint, which processes the raw OpenRouter API response into a
 * frontend-friendly format.
 */
export interface CatalogModel {
  id: string
  name: string
  provider: string
  description: string
  context_length: number
  pricing: {
    prompt: number
    completion: number
    prompt_display: string
    completion_display: string
  }
  is_free: boolean
  supports_vision: boolean
  modality_input: string[]
  modality_output: string[]
  top_provider: Record<string, unknown>
}

export interface ModelCatalog {
  models: CatalogModel[]
  recommended: CatalogModel[]
  free: CatalogModel[]
  categories: Record<string, CatalogModel[]>
  total_count: number
  free_count: number
}

export interface ValidateKeyResult {
  valid: boolean
  error?: string
  warning?: string
}

// =============================================================================
// FRONTEND CACHE
// =============================================================================

let _cachedCatalog: ModelCatalog | null = null
let _cachedAt = 0
const FRONTEND_CACHE_TTL_MS = 5 * 60 * 1000 // 5 minutes

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Validate a user-provided API key.
 *
 * Makes a POST to /v1/health/validate-key which does a minimal 1-token
 * test call to the provider. Returns whether the key is valid.
 *
 * @param provider - "openrouter", "anthropic", or "openai"
 * @param apiKey - The API key to validate
 * @returns {valid: boolean, error?: string, warning?: string}
 */
export async function validateApiKey(
  provider: string,
  apiKey: string,
): Promise<ValidateKeyResult> {
  try {
    const response = await fetch('/v1/health/validate-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, api_key: apiKey }),
    })

    if (!response.ok) {
      return { valid: false, error: `Server error: ${response.status}` }
    }

    return await response.json()
  } catch (error) {
    return { valid: false, error: `Network error: ${error}` }
  }
}

/**
 * Fetch the OpenRouter model catalog.
 *
 * Uses a two-level cache:
 *   1. Frontend cache (5 min) — avoids network requests entirely
 *   2. Backend cache (1 hour) — avoids calling OpenRouter's API
 *
 * The catalog contains 500+ models with pricing, context length,
 * and capability information. The backend categorizes them into
 * "recommended", "free", and provider-grouped lists.
 *
 * @param openrouterApiKey - Optional. Passed to backend for user-specific access
 * @param forceRefresh - Skip frontend cache (still uses backend cache)
 */
export async function fetchModelCatalog(
  openrouterApiKey?: string,
  forceRefresh = false,
): Promise<ModelCatalog> {
  // Check frontend cache first
  const now = Date.now()
  if (!forceRefresh && _cachedCatalog && (now - _cachedAt) < FRONTEND_CACHE_TTL_MS) {
    return _cachedCatalog
  }

  // Build URL with optional API key query parameter
  const params = new URLSearchParams()
  if (openrouterApiKey) {
    params.set('openrouter_api_key', openrouterApiKey)
  }
  const url = `/v1/models/catalog${params.toString() ? `?${params}` : ''}`

  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch model catalog: ${response.status}`)
  }

  const catalog: ModelCatalog = await response.json()

  // Update frontend cache
  _cachedCatalog = catalog
  _cachedAt = now

  return catalog
}

/**
 * Get the display name for a cloud provider model ID.
 *
 * Maps internal model IDs to human-readable names for the UI.
 * Falls back to the raw ID if not found in the known models.
 */
export function getCloudProviderDisplayName(providerId: string): string {
  const known: Record<string, string> = {
    'cloud-or-claude': 'Claude Sonnet 4',
    'cloud-or-gpt4o': 'GPT-4o',
    'cloud-or-gemini-pro': 'Gemini 2.5 Pro',
    'cloud-or-gemini-flash': 'Gemini 2.5 Flash',
    'cloud-or-o3-mini': 'o3-mini',
    'cloud-or-deepseek-r1': 'DeepSeek R1',
    'cloud-or-llama-maverick': 'Llama 4 Maverick',
    'cloud-or-deepseek-v3': 'DeepSeek V3',
    'cloud-or-glm5': 'GLM-5',
    'cloud-claude': 'Claude Sonnet 4',
    'cloud-gpt': 'GPT-4o',
    'cloud-gpt-cheap': 'GPT-4o Mini',
  }
  if (known[providerId]) return known[providerId]

  // Dynamic OpenRouter model — extract the model name from the ID
  if (providerId.startsWith('cloud-or-dynamic-')) {
    const parts = providerId.replace('cloud-or-dynamic-', '').split('/')
    return parts[parts.length - 1] || providerId
  }

  return providerId
}

/**
 * Determine which API key type a cloud provider needs.
 *
 * This helps the UI show appropriate warnings when a key is missing.
 */
export function getRequiredKeyType(
  providerId: string,
): 'openrouter' | 'anthropic' | 'openai' | null {
  if (providerId.startsWith('cloud-or')) return 'openrouter'
  if (providerId === 'cloud-claude') return 'anthropic'
  if (providerId === 'cloud-gpt' || providerId === 'cloud-gpt-cheap') return 'openai'
  return null
}
