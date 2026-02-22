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

import type { Message, ChatSettings, StreamMetadata, ContentBlock } from '../types'
import { documentsToContentBlocks, formatFileSize } from './documents'
import { useSettingsStore } from '../stores/settingsStore'

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

      // Send only role and content (what the API expects).
      // We strip out id, createdAt, etc. which are frontend-only.
      //
      // MULTIMODAL: If a message has images, we build the OpenAI vision
      // format — content becomes an array of ContentBlocks instead of
      // a plain string. This is the format that the backend, Ollama,
      // vLLM, and LiteLLM all understand natively.
      //
      // IMPORTANT — STRIPPING OLD DOCUMENT/IMAGE CONTENT:
      // Only the LAST user message gets full document and image content.
      // Older messages get a brief reference instead (e.g., "[Attached:
      // report.pdf — 5 pages, 11.6K chars]"). Without this, every old
      // message's extracted documents would be re-sent, causing:
      //   - Massive input token counts ($0.12+ per message)
      //   - 30+ second latency from processing old content
      //   - Potential context window overflow
      // The LLM's previous responses about those documents remain in
      // history, so it still has context from the conversation.
      messages: (() => {
        // Find the index of the last user message — only this one gets full content
        const lastUserIndex = messages.reduce(
          (last, m, i) => m.role === 'user' ? i : last, -1
        )

        return messages.map((m, index) => {
          const isLatestUser = index === lastUserIndex
          const hasImages = m.images && m.images.length > 0
          const hasDocs = m.documents && m.documents.length > 0 &&
            m.documents.some(d => d.status === 'ready')

          // For OLDER user messages: replace full document/image content with
          // a short text reference. This dramatically reduces payload size while
          // preserving the conversational context (the LLM already responded to
          // those documents, so the response text carries the context forward).
          if (!isLatestUser && (hasImages || hasDocs)) {
            const refs: string[] = []

            if (hasDocs) {
              for (const doc of m.documents!) {
                if (doc.status !== 'ready') continue
                const meta = [formatFileSize(doc.fileSize)]
                if (doc.pageCount > 0) meta.push(`${doc.pageCount} pages`)
                meta.push(`${(doc.totalTextLength / 1000).toFixed(1)}K chars`)
                refs.push(`[Attached: ${doc.filename} — ${meta.join(', ')}]`)
              }
            }
            if (hasImages) {
              refs.push(`[Attached: ${m.images!.length} image(s)]`)
            }

            const textWithRefs = refs.length > 0
              ? `${refs.join('\n')}\n\n${m.content}`
              : m.content

            return { role: m.role, content: textWithRefs }
          }

          // For the LATEST user message: include full document and image content
          if (hasImages || hasDocs) {
            const contentBlocks: ContentBlock[] = []

            if (hasDocs) {
              const docBlocks = documentsToContentBlocks(m.documents!)
              contentBlocks.push(...docBlocks)
            }

            contentBlocks.push({ type: 'text' as const, text: m.content })

            if (hasImages) {
              contentBlocks.push(
                ...m.images!.map(dataUrl => ({
                  type: 'image_url' as const,
                  image_url: { url: dataUrl },
                }))
              )
            }

            return { role: m.role, content: contentBlocks }
          }

          // No images or documents — send plain string content (backwards compatible)
          return { role: m.role, content: m.content }
        })
      })(),

      // Request streaming response
      stream: true,

      // User's preferences
      temperature: settings.temperature,
      judge_strategy: settings.judgeStrategy,
      local_model: settings.localModel,
      lakeshore_model: settings.lakeshoreModel,
      cloud_provider: settings.cloudProvider,

      // Web search — when enabled, the backend searches the internet for
      // the user's query and injects results as context before the LLM call.
      web_search: settings.webSearch || false,
      web_search_provider: settings.webSearchProvider || 'duckduckgo',
      // API keys are read from the store directly (not included in
      // getSettings() to avoid exposing them in the settings object).
      // Only sent when the corresponding provider is selected.
      ...(settings.webSearch && settings.webSearchProvider === 'tavily'
        ? { tavily_api_key: useSettingsStore.getState().tavilyApiKey }
        : {}
      ),
      ...(settings.webSearch && settings.webSearchProvider === 'google'
        ? { serper_api_key: useSettingsStore.getState().serperApiKey }
        : {}
      ),
    }),
    signal: abortSignal,
  })

  /**
   * STEP 2: Check for HTTP errors
   *
   * A non-OK response means something went wrong before streaming started.
   * Common causes: server down, invalid request, auth failure.
   *
   * We parse the response body to get detailed error info from the backend.
   */
  if (!response.ok) {
    try {
      const errorBody = await response.json()
      // Backend sends structured errors in detail field
      const detail = errorBody.detail || errorBody

      if (detail.error === 'context_too_long') {
        // Context window exceeded - pass structured error for dialog
        callbacks.onError(JSON.stringify({
          type: 'context_exceeded',
          message: detail.message,
          estimated_tokens: detail.estimated_tokens,
          model_limit: detail.model_limit,
        }))
      } else if (detail.error_type === 'model_not_multimodal') {
        // User sent images to a text-only model
        callbacks.onError(JSON.stringify({
          type: 'model_not_multimodal',
          message: detail.message,
        }))
      } else if (detail.error_type === 'auth_subscription') {
        // API key invalid or subscription expired
        callbacks.onError(JSON.stringify({
          type: 'auth_subscription',
          message: detail.message,
          raw_error: detail.raw_error,
          provider: detail.provider,
        }))
      } else if (detail.error_type === 'rate_limit') {
        // Rate limit exceeded
        callbacks.onError(JSON.stringify({
          type: 'rate_limit',
          message: detail.message,
        }))
      } else if (detail.error_type) {
        // Other structured errors from backend
        callbacks.onError(JSON.stringify({
          type: detail.error_type,
          message: detail.message,
        }))
      } else {
        // Other errors - show the message
        callbacks.onError(detail.message || `HTTP ${response.status}: ${response.statusText}`)
      }
    } catch {
      // Couldn't parse JSON - fall back to status text
      callbacks.onError(`HTTP ${response.status}: ${response.statusText}`)
    }
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
   * - value: raw bytes of data (may be present even when done=true!)
   *
   * We keep reading until done is true.
   */
  while (true) {
    const { done, value } = await reader.read()

    /**
     * STEP 5: Decode bytes to text and handle buffering
     *
     * IMPORTANT: Process value BEFORE checking done!
     * The final read may return { done: true, value: <final bytes> }
     * and we need to process those bytes (which may contain [DONE]).
     *
     * { stream: true } tells the decoder to handle partial characters.
     * We add to buffer because SSE events might be split across reads.
     */
    if (value) {
      buffer += decoder.decode(value, { stream: true })
    }

    /**
     * STEP 6: Split into lines and process
     *
     * SSE events are newline-separated.
     * The last element (pop()) might be incomplete - keep it in buffer.
     *
     * IMPORTANT: Process lines BEFORE checking done!
     * The final read may have { done: true, value: "[DONE]" }
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
        // Backend sends metadata in THREE types of events:
        // 1. Initial: {"stream_metadata": {"tier": "local", "model": "..."}}
        // 2. Fallback: {"stream_metadata": {"fallback": true, "original_tier": "lakeshore", "current_tier": "cloud", ...}}
        // 3. Final:   {"stream_metadata": {"cost": {"total": 0.001}, "duration": 1.23, "fallback_used": true, "tiers_tried": [...]}}
        if (parsed.stream_metadata) {
          const meta = parsed.stream_metadata
          console.log('[stream] Received metadata:', meta)

          // Derive original_tier from tiers_tried if not explicitly set
          // tiers_tried[0] is always the originally requested tier
          const derivedOriginalTier = meta.original_tier ??
            (meta.tiers_tried && meta.tiers_tried.length > 1 ? meta.tiers_tried[0] : undefined)

          // Normalize the metadata structure
          const normalizedMeta = {
            ...meta,
            // Backend sends "current_tier" in fallback events, normalize to "tier"
            tier: meta.tier ?? meta.current_tier,
            // Ensure original_tier is set for fallback scenarios
            original_tier: derivedOriginalTier,
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

        // Check for errors in the response (including mid-stream errors from LLM)
        if (parsed.error) {
          console.error('[stream] Server error:', parsed.error)
          const errorMsg = typeof parsed.error === 'string' ? parsed.error : (parsed.error.message || 'Server error')

          // Check if this is a context length error from the LLM
          const isContextError = errorMsg.toLowerCase().includes('context') ||
                                 errorMsg.toLowerCase().includes('token') ||
                                 errorMsg.toLowerCase().includes('too long') ||
                                 errorMsg.toLowerCase().includes('maximum')

          if (isContextError) {
            // Format as structured error for the dialog
            callbacks.onError(JSON.stringify({
              type: 'context_exceeded',
              message: errorMsg,
              estimated_tokens: 0, // Unknown mid-stream
              model_limit: 0,      // Unknown mid-stream
            }))
          } else {
            callbacks.onError(errorMsg)
          }
        }
      } catch (e) {
        // Skip malformed JSON - sometimes happens with partial data
        console.warn('[stream] Failed to parse SSE data:', data, e)
      }
    }

    // Check done AFTER processing all lines from this read
    // This ensures we process [DONE] even if it arrives with done=true
    if (done) break
  }

  // Process any remaining buffer content after stream ends
  // This handles the case where [DONE] is in the final chunk
  if (buffer.trim()) {
    const remainingLines = buffer.split('\n')
    for (const line of remainingLines) {
      if (!line.startsWith('data: ')) continue
      const data = line.slice(6).trim()
      if (data === '[DONE]') {
        callbacks.onComplete()
        return
      }
    }
  }

  // If we reach here without seeing [DONE], still complete the stream
  // This handles edge cases where the server closes without [DONE]
  console.warn('[stream] Stream ended without [DONE] marker')
  callbacks.onComplete()
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
