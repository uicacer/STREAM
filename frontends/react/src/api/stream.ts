/**
 * stream.ts - Server-Sent Events (SSE) Streaming Client
 * ======================================================
 *
 * WHAT IS SSE (Server-Sent Events)?
 * ---------------------------------
 * SSE is a way for the server to push data to the browser in real-time.
 *
 * Normal HTTP request:
 *   Browser: "Give me data" → Server: "Here's ALL the data" (one response)
 *
 * SSE request:
 *   Browser: "Give me data" → Server: "Here's token 1..."
 *                                    "Here's token 2..."
 *                                    "Here's token 3..."
 *                                    "Done!"
 *
 * This is how ChatGPT/Claude show text appearing word-by-word.
 * The server sends each token as it's generated, and we display it immediately.
 *
 * WHY NOT WEBSOCKETS?
 * - SSE is simpler (one-way: server → browser)
 * - Built into browsers (no library needed)
 * - Works over regular HTTP (easier to proxy/load-balance)
 * - Perfect for streaming text (we don't need to send data back mid-stream)
 *
 * SSE DATA FORMAT:
 * Each "event" from the server looks like this:
 *   data: {"choices":[{"delta":{"content":"Hello"}}]}
 *   data: {"choices":[{"delta":{"content":" world"}}]}
 *   data: [DONE]
 *
 * The "data: " prefix is part of the SSE protocol.
 */

import type { Message, ChatSettings, StreamMetadata } from '../types'

/**
 * StreamCallbacks - Functions called as streaming events occur
 *
 * Instead of returning data, we call these functions with data as it arrives.
 * This is called the "callback pattern" - common in async/streaming code.
 *
 * ALTERNATIVE: We could use Promises or async iterators, but callbacks
 * are simpler here since we need to update UI state incrementally.
 */
export interface StreamCallbacks {
  /**
   * Called for each text token received
   * @param token - A piece of text (word, part of word, punctuation)
   *
   * Example: If the AI says "Hello world", you might get:
   *   onToken("Hello")
   *   onToken(" world")
   */
  onToken: (token: string) => void

  /**
   * Called when we receive routing/model information
   * @param meta - Info about which tier/model is being used
   *
   * This arrives early in the stream, before tokens.
   */
  onMetadata: (meta: StreamMetadata) => void

  /**
   * Called for reasoning model "thinking" content
   * @param thought - Part of the model's reasoning process
   *
   * Only called for Claude Sonnet 4, o1, etc.
   * Regular models don't produce thinking content.
   */
  onThinking: (thought: string) => void

  /**
   * Called when streaming is complete
   * Use this to finalize the message and update UI state.
   */
  onComplete: () => void

  /**
   * Called if an error occurs
   * @param error - Human-readable error message
   */
  onError: (error: string) => void
}

/**
 * streamChat - Streams a chat response from the STREAM middleware
 *
 * This function:
 * 1. Sends the conversation to the server
 * 2. Receives tokens one-by-one via SSE
 * 3. Calls the appropriate callback for each event
 *
 * @param messages - The conversation history
 * @param settings - User's tier/judge/temperature preferences
 * @param callbacks - Functions to call with streaming data
 *
 * USAGE EXAMPLE:
 *   await streamChat(messages, settings, {
 *     onToken: (token) => appendToResponse(token),
 *     onMetadata: (meta) => setMetadata(meta),
 *     onThinking: (thought) => appendToThinking(thought),
 *     onComplete: () => finishMessage(),
 *     onError: (err) => showError(err),
 *   })
 */
export async function streamChat(
  messages: Message[],
  settings: ChatSettings,
  callbacks: StreamCallbacks,
  abortSignal?: AbortSignal
): Promise<void> {
  /**
   * STEP 1: Make the HTTP request
   *
   * We're using the native fetch() API - no libraries needed!
   * The response will be a stream of SSE events.
   */
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      // The model field is used for tier selection in STREAM
      model: settings.tier,

      // Send only role and content (what the API expects)
      // We strip out id, createdAt, etc. which are frontend-only
      messages: messages.map(m => ({ role: m.role, content: m.content })),

      // Request streaming response
      stream: true,

      // User's preferences
      temperature: settings.temperature,
      judge_strategy: settings.judgeStrategy,
    }),
    signal: abortSignal,
  })

  /**
   * STEP 2: Check for HTTP errors
   *
   * A non-OK response means something went wrong before streaming started.
   * Common causes: server down, invalid request, auth failure.
   */
  if (!response.ok) {
    callbacks.onError(`HTTP ${response.status}: ${response.statusText}`)
    return
  }

  /**
   * STEP 3: Set up stream reading
   *
   * response.body is a ReadableStream - we need to read it piece by piece.
   *
   * getReader() - Gets a reader to consume the stream
   * TextDecoder - Converts raw bytes to text
   * buffer - Holds incomplete data between reads (SSE events can be split)
   */
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  /**
   * STEP 4: Read the stream in a loop
   *
   * Each read() call returns:
   * - done: true if stream ended
   * - value: raw bytes of data
   *
   * We keep reading until done is true.
   */
  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    /**
     * STEP 5: Decode bytes to text and handle buffering
     *
     * { stream: true } tells the decoder to handle partial characters.
     * We add to buffer because SSE events might be split across reads.
     */
    buffer += decoder.decode(value, { stream: true })

    /**
     * STEP 6: Split into lines and process
     *
     * SSE events are newline-separated.
     * The last element (pop()) might be incomplete - keep it in buffer.
     */
    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // Keep incomplete last line in buffer

    /**
     * STEP 7: Process each complete line
     */
    for (const line of lines) {
      // SSE data lines start with "data: "
      // Skip empty lines or other SSE fields (event:, id:, retry:)
      if (!line.startsWith('data: ')) continue

      // Extract the JSON data after "data: "
      const data = line.slice(6).trim()

      // "[DONE]" is the OpenAI convention for stream end
      if (data === '[DONE]') {
        callbacks.onComplete()
        return
      }

      /**
       * STEP 8: Parse and handle the JSON data
       *
       * The data follows the OpenAI chat completion format:
       * {
       *   "choices": [{
       *     "delta": { "content": "token text" }
       *   }],
       *   "stream_metadata": { "tier": "local", ... },  // STREAM-specific
       *   "thinking": "reasoning text"                   // For reasoning models
       * }
       */
      try {
        const parsed = JSON.parse(data)

        // Handle STREAM-specific metadata (which tier, model, etc.)
        // Backend sends metadata in TWO events:
        // 1. Initial: {"stream_metadata": {"tier": "local", "model": "..."}}
        // 2. Final:   {"stream_metadata": {"cost": {"total": 0.001}, "duration": 1.23}}
        if (parsed.stream_metadata) {
          const meta = parsed.stream_metadata
          console.log('[stream] Received metadata:', meta)

          // Extract nested cost if present (cost.total structure from backend)
          const normalizedMeta = {
            ...meta,
            // Flatten nested cost structure: cost.total -> cost
            cost: meta.cost?.total ?? meta.cost,
            // Duration comes directly
            duration: meta.duration,
          }

          callbacks.onMetadata(normalizedMeta)
        }

        // Handle thinking content (reasoning models only)
        if (parsed.thinking) {
          callbacks.onThinking(parsed.thinking)
        }

        // Handle regular content tokens
        const content = parsed.choices?.[0]?.delta?.content
        if (content) {
          callbacks.onToken(content)
        }

        // Check for errors in the response
        if (parsed.error) {
          console.error('[stream] Server error:', parsed.error)
          callbacks.onError(parsed.error.message || 'Server error')
        }
      } catch (e) {
        // Skip malformed JSON - sometimes happens with partial data
        console.warn('[stream] Failed to parse SSE data:', data, e)
      }
    }
  }
}

/**
 * REASONING MODEL DETECTION
 * =========================
 *
 * These models produce "thinking" content that should be displayed
 * in a collapsible section. Regular models don't think - they just generate.
 */
const REASONING_MODELS = [
  'claude-sonnet-4',
  'claude-opus-4',
  'o1',
  'o1-mini',
  'o3',
  'deepseek-r1',
]

/**
 * Check if a model name indicates a reasoning model
 *
 * @param model - The model name from metadata
 * @returns true if this model produces thinking content
 *
 * USAGE:
 *   if (isReasoningModel(metadata.model)) {
 *     showThinkingIndicator()
 *   } else {
 *     showGeneratingIndicator()
 *   }
 */
export function isReasoningModel(model: string): boolean {
  return REASONING_MODELS.some(rm =>
    model.toLowerCase().includes(rm.toLowerCase())
  )
}
