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
}

/**
 * MessageMetadata - Information about how an AI response was generated
 *
 * This helps users understand:
 * - Which tier handled their request (local, cloud, etc.)
 * - How long it took
 * - How much it cost
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
  complexity?: string
  cost?: number
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
}
