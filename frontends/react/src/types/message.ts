/**
 * message.ts - Type Definitions for Chat Messages
 * ================================================
 *
 * WHAT ARE TYPESCRIPT TYPES?
 * Types define the "shape" of data - what fields exist and what type each is.
 * They help catch errors BEFORE running the code (at compile time).
 *
 * EXAMPLE WITHOUT TYPES:
 *   const msg = { role: "user", content: "Hello" }
 *   msg.contnet  // Typo! JavaScript won't catch this until runtime
 *
 * EXAMPLE WITH TYPES:
 *   const msg: Message = { role: "user", content: "Hello" }
 *   msg.contnet  // TypeScript error! "contnet" doesn't exist on Message
 *
 * WHY USE INTERFACES?
 * - "interface" defines a contract for object shapes
 * - Any object claiming to be a "Message" MUST have these exact fields
 * - Makes code self-documenting
 */

/**
 * ContentBlock - A single block in a multimodal message
 *
 * When a message contains images, the content is an ARRAY of these blocks
 * instead of a plain string. This follows the OpenAI Vision format:
 *
 *   content: [
 *     { type: "text", text: "What is in this image?" },
 *     { type: "image_url", image_url: { url: "data:image/jpeg;base64,..." } }
 *   ]
 *
 * Each block has a "type" that determines which other field is present:
 *   - "text" → has a "text" field with the text content
 *   - "image_url" → has an "image_url" field with a nested "url" field
 */
export interface TextBlock {
  type: 'text'
  text: string
}

export interface ImageBlock {
  type: 'image_url'
  image_url: { url: string }
}

export type ContentBlock = TextBlock | ImageBlock

/**
 * DocumentContentPart - A single piece of extracted document content
 *
 * Documents are extracted into an ordered list of these parts, preserving
 * the interleaving of text and images as they appear in the original document.
 *
 * For example, a PDF page might produce:
 *   [TextPart("Figure 3 shows..."), ImagePart(chart), TextPart("Table 2:...")]
 *
 * This directly maps to the OpenAI multimodal content format, so extracted
 * documents flow through the existing chat pipeline with zero changes.
 */
export interface DocumentContentPart {
  type: 'text' | 'image'
  text: string | null
  image_base64: string | null
  image_mime: string | null
}

/**
 * DocumentAttachment - A document attached to a chat message
 *
 * When a user uploads a document (PDF, DOCX, etc.), the frontend sends it
 * to the backend for extraction. The result is stored as a DocumentAttachment
 * containing the structured content parts (text + images in document order).
 *
 * These are displayed as compact attachment chips in the chat input, with
 * collapsible previews in chat messages — similar to Claude's file display.
 */
export interface DocumentAttachment {
  id: string
  filename: string
  fileType: string
  fileSize: number
  contentParts: DocumentContentPart[]
  textPreview: string
  totalTextLength: number
  imageCount: number
  pageCount: number
  warnings: string[]
  status: 'uploading' | 'ready' | 'error'
  error?: string
}

/**
 * Message - A single chat message (user or assistant)
 *
 * This represents one message in the conversation, whether it's
 * from the user typing or the AI responding.
 */
export interface Message {
  /**
   * Unique identifier for this message
   * Using crypto.randomUUID() generates these (e.g., "550e8400-e29b-41d4-a716-446655440000")
   */
  id: string

  /**
   * Who sent this message
   * - "user" = The human typed this
   * - "assistant" = The AI generated this
   */
  role: 'user' | 'assistant'

  /**
   * The actual text content of the message
   */
  content: string

  /**
   * Optional: Base64-encoded images attached to this message.
   *
   * These are stored separately from "content" for two reasons:
   *   1. DISPLAY: The UI renders them as image thumbnails below the text
   *   2. API: When sending to the backend, we build the OpenAI vision format
   *      (content: ContentBlock[]) from this array + the text content
   *
   * Each string is a full data URL: "data:image/jpeg;base64,/9j/4AAQ..."
   * The frontend compresses images before storing (max 1024px, JPEG 85%).
   */
  images?: string[]

  /**
   * Optional: Documents attached to this message.
   *
   * Each document has been extracted on the backend into structured
   * content parts (text + images). When sending to the API, these
   * parts are assembled into the OpenAI multimodal content format
   * alongside any pasted images and the user's text.
   */
  documents?: DocumentAttachment[]

  /**
   * Optional: The "thinking" process for reasoning models
   * Only present when using Claude Sonnet 4, o1, etc.
   * Example: "Let me break this down step by step..."
   *
   * The "?" means this field is OPTIONAL - it may or may not exist
   */
  thinking?: string

  /**
   * Optional: Metadata about how the message was generated
   * Only present for assistant messages
   */
  metadata?: MessageMetadata

  /**
   * When this message was created (ISO 8601 format)
   * Example: "2024-01-15T10:30:00.000Z"
   */
  createdAt: string

  /**
   * Whether this message has been summarized (not sent to API)
   * Messages marked as summarized are shown in UI but excluded from API calls
   */
  summarized?: boolean

  /**
   * Whether this message is a summary marker (contains the conversation summary)
   * Used to identify the summary point in the conversation
   */
  isSummaryMarker?: boolean
}

/**
 * MessageMetadata - Information about how an AI response was generated
 *
 * This helps users understand:
 * - Which tier handled their request (local, cloud, etc.)
 * - How long it took
 * - How much it cost
 * - Whether a fallback occurred
 */
export interface MessageMetadata {
  /**
   * Which tier processed this request
   * Examples: "local", "lakeshore", "cloud"
   */
  tier: string

  /**
   * The specific model that generated the response
   * Examples: "llama-3.1-8b", "claude-3-haiku", "gpt-4o-mini"
   */
  model: string

  /**
   * The actual model confirmed by the provider's response metadata.
   * For OpenRouter: "deepseek/deepseek-chat", for Anthropic: "claude-sonnet-4-20250514"
   */
  verified_model?: string

  /**
   * Optional: How complex the query was judged to be
   * Examples: "simple", "moderate", "complex"
   */
  complexity?: string

  /**
   * Optional: How long the response took in milliseconds
   */
  duration?: number

  /**
   * Optional: Cost in dollars (for cloud models)
   * Example: 0.0015 (meaning $0.0015 or 0.15 cents)
   */
  cost?: number

  /**
   * Whether the cost is an estimate (true when streaming was interrupted).
   * Estimated at ~4 characters per token using pricing from litellm_config.yaml.
   */
  cost_estimated?: boolean

  /**
   * Optional: Whether a fallback was used (tier was unavailable)
   * True if the original tier failed and we switched to another
   */
  fallback_used?: boolean

  /**
   * Optional: The original tier that was requested but failed
   * Example: "lakeshore" if Lakeshore was unavailable
   */
  original_tier?: string

  /**
   * Optional: List of tiers that were tried before succeeding
   * Example: ["lakeshore", "cloud"] if Lakeshore failed, Cloud succeeded
   */
  tiers_tried?: string[]
}

/**
 * StreamMetadata - Metadata received DURING streaming
 *
 * This arrives before/during the response, telling us which
 * tier was selected and whether it's a reasoning model.
 *
 * WHY SEPARATE FROM MessageMetadata?
 * - StreamMetadata arrives WHILE streaming (partial info)
 * - MessageMetadata is complete info AFTER the message is done
 */
export interface StreamMetadata {
  tier: string
  model: string
  /**
   * The actual model that served the request, confirmed by the provider's
   * response metadata (e.g., OpenRouter returns "deepseek/deepseek-chat").
   * Unlike `model` (what STREAM requested), this is what actually responded.
   */
  verified_model?: string
  complexity?: string
  cost?: number
  /**
   * Whether the cost is an estimate (true when streaming was interrupted).
   * Estimated at ~4 characters per token using pricing from litellm_config.yaml.
   */
  cost_estimated?: boolean
  /**
   * How long the response took in seconds
   * Sent by backend at end of stream
   */
  duration?: number

  /**
   * Is this a reasoning model (Claude Sonnet 4, o1, etc.)?
   * If true, we show "Thinking..." instead of "Generating..."
   */
  isReasoning?: boolean

  /**
   * Whether a fallback is occurring RIGHT NOW (during streaming)
   * Sent when a tier fails and we're switching to another
   */
  fallback?: boolean

  /**
   * The original tier that was requested but failed
   * Sent with fallback=true event
   */
  original_tier?: string

  /**
   * Reason for the fallback (e.g., "Connection refused", "Timeout")
   */
  reason?: string

  /**
   * Whether a fallback was used (final metadata at end of stream)
   */
  fallback_used?: boolean

  /**
   * List of tiers that were tried before succeeding
   */
  tiers_tried?: string[]

  /**
   * List of tiers that were unavailable (used for fallback message)
   * Example: ["cloud", "lakeshore"] means both were tried and unavailable
   */
  unavailable_tiers?: string[]

  /**
   * Status event from the backend's streaming pipeline.
   * Used for live UX feedback during multi-step operations:
   *   - "summarizing_context": Rolling summarization is in progress
   *     (compressing older messages to fit tier's context window)
   *   - "summarization_complete": Summarization finished, normal
   *     streaming will begin
   */
  status?: string

  /**
   * Whether the conversation context was compressed before inference.
   * Sent with status="summarization_complete" after rolling
   * summarization reduces the message history.
   */
  context_compressed?: boolean
}
