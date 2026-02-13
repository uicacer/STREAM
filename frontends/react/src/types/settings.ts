/**
 * settings.ts - Type Definitions for User Settings
 * =================================================
 *
 * These types define the configurable options in STREAM.
 * Users can choose which tier to use, which judge model, etc.
 */

/**
 * Tier - The compute tier for processing requests
 *
 * STREAM can route requests to different compute resources:
 *
 * - "auto"      → Let STREAM decide based on query complexity
 *                 Uses a "judge" model to classify the query
 *
 * - "local"     → Use the user's local machine (Ollama)
 *                 Free, private, but limited to smaller models
 *
 * - "lakeshore" → Use Lakeshore HPC cluster
 *                 Powerful, good for complex queries, academic use
 *
 * - "cloud"     → Use cloud APIs (OpenAI, Anthropic, etc.)
 *                 Most capable models, but costs money
 *
 * WHY USE A TYPE INSTEAD OF JUST "string"?
 * TypeScript will catch typos:
 *   tier: Tier = "clod"  // Error! "clod" is not a valid Tier
 *   tier: string = "clod" // No error, bug goes unnoticed
 */
export type Tier = 'auto' | 'local' | 'lakeshore' | 'cloud'

/**
 * JudgeStrategy - Which model classifies query complexity
 *
 * When tier is "auto", STREAM needs to decide where to route.
 * A "judge" model quickly analyzes the query to determine complexity.
 *
 * - "ollama-1b" → Fastest, runs locally, good enough for most cases
 * - "ollama-3b" → Slower but more accurate classification
 * - "haiku"     → Claude Haiku via API (most accurate, but costs money)
 *
 * Trade-off: Faster judges add less latency but may misclassify.
 */
export type JudgeStrategy = 'ollama-1b' | 'ollama-3b' | 'haiku'

/**
 * CloudProvider - Available cloud model providers
 *
 * When using the "cloud" tier, users can choose which provider to use.
 * Each has different capabilities and pricing.
 *
 * - "cloud-claude" → Claude Sonnet (Anthropic) - Best for reasoning/coding
 * - "cloud-gpt"    → GPT-4 Turbo (OpenAI) - Strong general-purpose
 * - "cloud-gpt-cheap" → GPT-3.5 Turbo (OpenAI) - Fast and affordable
 */
export type CloudProvider = 'cloud-claude' | 'cloud-gpt' | 'cloud-gpt-cheap'

/**
 * CloudProviderInfo - Metadata about a cloud provider
 */
export interface CloudProviderInfo {
  name: string
  provider: string
  description: string
}

/**
 * ChatSettings - All user-configurable options for chat
 *
 * These settings affect how requests are processed.
 */
export interface ChatSettings {
  /**
   * Which compute tier to use
   * See Tier type above for options
   */
  tier: Tier

  /**
   * Which model should classify query complexity
   * Only used when tier is "auto"
   */
  judgeStrategy: JudgeStrategy

  /**
   * Model temperature (0.0 to 1.0)
   *
   * Controls randomness/creativity of responses:
   * - 0.0 = Deterministic (same input → same output)
   * - 0.7 = Balanced (default for most tasks)
   * - 1.0 = Creative (more varied, potentially less accurate)
   *
   * Lower for factual tasks, higher for creative writing.
   */
  temperature: number

  /**
   * Which cloud provider to use when tier is "cloud"
   * Users can switch providers if one is unavailable
   */
  cloudProvider?: CloudProvider
}
