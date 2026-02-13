/**
 * costs.ts - Model Pricing API Client
 * ====================================
 *
 * Fetches model pricing from the backend, which reads from litellm_config.yaml
 * (the single source of truth for model costs).
 *
 * Used by:
 * - chatStore.ts to estimate cost when streaming is interrupted
 * - SettingsPanel.tsx to display pricing info (future)
 */

// Cache for model pricing (avoid fetching on every stop)
let cachedPricing: Record<string, { input: number; output: number }> | null = null
let cacheTimestamp = 0
const CACHE_TTL = 5 * 60 * 1000 // 5 minutes

/**
 * Fetch model pricing from the backend.
 * Reads from litellm_config.yaml (single source of truth).
 */
export async function fetchModelPricing(): Promise<Record<string, { input: number; output: number }>> {
  // Use cache if fresh
  if (cachedPricing && Date.now() - cacheTimestamp < CACHE_TTL) {
    return cachedPricing
  }

  try {
    const response = await fetch('/v1/costs/models')
    if (!response.ok) {
      console.warn('[costs] Failed to fetch model pricing:', response.status)
      return {}
    }

    const data = await response.json()
    const pricing = data.costs || {}
    cachedPricing = pricing
    cacheTimestamp = Date.now()
    return pricing
  } catch (error) {
    console.warn('[costs] Error fetching model pricing:', error)
    return {}
  }
}

/**
 * Estimate cost for a partial response when streaming is interrupted.
 *
 * IMPORTANT: This is an ESTIMATION because:
 * 1. Streaming was interrupted before the backend could calculate actual cost
 * 2. Token count is estimated at ~4 characters per token (rough approximation)
 * 3. Actual token count may differ based on model's tokenizer
 *
 * @param text - The partial response text
 * @param model - The model name (e.g., "claude-sonnet-4-20250514")
 * @param pricing - Model pricing data from litellm_config.yaml
 * @returns Estimated cost in USD, or 0 if pricing not available
 */
export function estimatePartialCost(
  text: string,
  model: string | undefined,
  pricing: Record<string, { input: number; output: number }>
): number {
  if (!model || !pricing[model]) {
    return 0
  }

  // Estimate tokens: ~4 characters per token (rough approximation)
  // This is a common rule of thumb for English text with GPT/Claude tokenizers
  const estimatedOutputTokens = Math.ceil(text.length / 4)

  // Get output price per token (pricing is per token, not per million)
  const outputPricePerToken = pricing[model].output

  // Calculate estimated cost
  return estimatedOutputTokens * outputPricePerToken
}
