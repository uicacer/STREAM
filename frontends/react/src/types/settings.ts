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
 * - "ollama-3b"     → Balanced accuracy, free (default)
 * - "gemma-vision"  → Vision-capable judge, can analyze images, free
 * - "haiku"         → Claude Haiku via API (most accurate, but costs money)
 *
 * Trade-off: Faster judges add less latency but may misclassify.
 * The gemma-vision judge can see images but adds more latency (~2-3s).
 */
export type JudgeStrategy = 'ollama-3b' | 'gemma-vision' | 'haiku'

/**
 * LocalModel - Available models for the Local tier (Ollama)
 *
 * - "local-llama"  → Llama 3.2 3B - Balanced text-only model (default)
 * - "local-vision" → Gemma 3 4B - Multimodal (text + images)
 */
export type LocalModel = 'local-llama' | 'local-vision'

/**
 * LakeshoreModel - Available models for the Lakeshore tier (Campus GPU)
 *
 * Each model runs as a separate vLLM instance on a different GPU (3g.40gb MIG slice).
 * All 32B models use AWQ 4-bit quantization for quality + efficiency.
 */
export type LakeshoreModel =
  | 'lakeshore-qwen-1.5b'
  | 'lakeshore-qwen-32b-fp16'
  | 'lakeshore-qwen-72b'
  | 'lakeshore-qwen-vl-72b'
  | 'lakeshore-qwen-32b'
  | 'lakeshore-coder-1.5b'
  | 'lakeshore-deepseek-r1'
  | 'lakeshore-qwq'

/**
 * CloudProvider - Available cloud model providers
 *
 * When using the "cloud" tier, users can choose which provider to use.
 * Each has different capabilities and pricing.
 *
 * - "cloud-claude" → Claude Sonnet (Anthropic) - Best for reasoning/coding
 * - "cloud-gpt"    → GPT-4o (OpenAI) - Strong general-purpose with vision
 * - "cloud-gpt-cheap" → GPT-4o Mini (OpenAI) - Fast and affordable with vision
 */
export type CloudProvider = 'cloud-claude' | 'cloud-gpt' | 'cloud-gpt-cheap'

/**
 * WebSearchProvider - Available web search providers
 *
 * When web search is enabled (the globe toggle in the chat input), STREAM
 * searches the internet for the user's query before sending it to the LLM.
 *
 * - "duckduckgo" → Free, no API key needed, good privacy (default)
 *                   Uses the duckduckgo-search Python library.
 *                   Best for desktop mode and campus use.
 *
 * - "tavily"     → AI-optimized results, requires API key
 *                   Returns pre-extracted content designed for LLMs.
 *                   Free tier: 1,000 searches/month.
 *                   Better quality results for complex research queries.
 */
export type WebSearchProvider = 'duckduckgo' | 'tavily'

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
   * Which model to use for the Local tier
   */
  localModel?: LocalModel

  /**
   * Which model to use for the Lakeshore tier
   */
  lakeshoreModel?: LakeshoreModel

  /**
   * Which cloud provider to use when tier is "cloud"
   * Users can switch providers if one is unavailable
   */
  cloudProvider?: CloudProvider

  /**
   * Whether web search is enabled for the current message.
   *
   * When true, the backend searches the internet for the user's query
   * and injects results as context before sending to the LLM. This
   * lets the LLM answer with current information from the web.
   *
   * Controlled by the globe toggle icon in the chat input area.
   */
  webSearch?: boolean

  /**
   * Which web search provider to use.
   * Configured in the Advanced Settings section of the sidebar.
   */
  webSearchProvider?: WebSearchProvider
}
