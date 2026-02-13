/**
 * summarize.ts - Conversation Summarization
 *
 * Uses the Cloud tier (Claude) to summarize a conversation history
 * into a compact format that preserves key context.
 *
 * Used when context limit is exceeded to compress history
 * while maintaining conversational continuity.
 */

import type { Message } from '../types'

const SUMMARIZE_PROMPT = `You are creating a context summary to help an AI assistant continue a conversation.

IMPORTANT: Capture ALL topics discussed, not just the most recent one.

Create a structured summary that includes:

1. **All Topics Covered** - List every distinct subject/question discussed
2. **Key Information Shared** - Important facts, code, decisions, or conclusions from each topic
3. **User's Background/Goals** - What the user is trying to accomplish overall
4. **Unresolved Questions** - Anything left unanswered or in progress

Format as a clear, scannable summary. Be thorough - the AI will use this to continue helping.

Conversation to summarize:
`

/**
 * Summarize a conversation using the Cloud tier
 *
 * This uses SSE streaming internally but collects the full response.
 *
 * @param messages - The conversation history to summarize
 * @returns The summary text, or null if summarization failed
 */
export async function summarizeConversation(messages: Message[]): Promise<string | null> {
  // Build conversation text
  const conversationText = messages
    .map(m => `${m.role.toUpperCase()}: ${m.content}`)
    .join('\n\n')

  try {
    // Call the Cloud tier for summarization
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'cloud', // Force Cloud tier for summarization
        messages: [
          {
            role: 'user',
            content: SUMMARIZE_PROMPT + conversationText,
          },
        ],
        temperature: 0.3, // Lower temperature for more consistent summaries
        stream: true,
      }),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      console.error('[summarize] Failed to summarize:', response.status, error)
      return null
    }

    // Read the SSE stream and collect the full response
    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let summary = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue

        const data = line.slice(6).trim()
        if (data === '[DONE]') continue
        if (!data) continue

        try {
          const parsed = JSON.parse(data)
          const content = parsed.choices?.[0]?.delta?.content
          if (content) {
            summary += content
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }

    if (!summary) {
      console.error('[summarize] No content collected from stream')
      return null
    }

    console.log('[summarize] Successfully summarized conversation:', summary.length, 'chars')
    return summary
  } catch (error) {
    console.error('[summarize] Error summarizing conversation:', error)
    return null
  }
}
