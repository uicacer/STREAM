/**
 * Message.tsx - Single Chat Message
 * ==================================
 *
 * This component renders one message in the chat - either from the user
 * or from the AI assistant. It handles:
 *
 * - Different styling for user vs assistant messages
 * - Thinking block display (for reasoning models)
 * - Routing details display (tier, model, duration, cost)
 * - Copy to clipboard button
 * - Responsive layout
 *
 * DESIGN PRINCIPLES:
 * - User messages: Right-aligned, primary color background
 * - Assistant messages: Left-aligned, no background (cleaner look)
 * - Metadata: Small, unobtrusive, but accessible
 */

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Copy, Check, Laptop, Building2, Cloud, Clock, AlertTriangle } from 'lucide-react'
import { ModelLogo } from '../icons/ProviderLogos'
import { cn } from '../../lib/utils'
import { ThinkingBlock } from './ThinkingBlock'
import type { Message as MessageType } from '../../types'

/**
 * Tier configuration for display with colors
 */
const TIER_CONFIG = {
  local: {
    icon: Laptop,
    label: 'LOCAL',
    color: 'text-orange-600 dark:text-orange-400',
    bgColor: 'bg-orange-500/10',
  },
  lakeshore: {
    icon: Building2,
    label: 'LAKESHORE',
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-500/10',
  },
  cloud: {
    icon: Cloud,
    label: 'CLOUD',
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-500/10',
  },
} as const

interface MessageProps {
  message: MessageType
  isStreaming?: boolean
}

export function Message({ message, isStreaming = false }: MessageProps) {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  /**
   * Get tier configuration for this message
   */
  const tierKey = message.metadata?.tier as keyof typeof TIER_CONFIG | undefined
  const tierConfig = tierKey ? TIER_CONFIG[tierKey] : null
  const TierIcon = tierConfig?.icon

  /**
   * Format model name for display
   */
  const formatModelName = (model: string): string => {
    if (model === 'local-vision') return 'Gemma 3 4B (Vision)'
    if (model.includes('llama')) return 'Llama 3.2 3B'
    if (model.includes('claude')) return 'Claude Sonnet 4'
    if (model === 'cloud-gpt-cheap' || model.includes('4o-mini')) return 'GPT-4o Mini'
    if (model === 'cloud-gpt' || model.includes('4o')) return 'GPT-4o'
    if (model.includes('deepseek')) return 'DeepSeek R1 1.5B'
    if (model.includes('qwq')) return 'QwQ 1.5B'
    if (model.includes('coder') && model.includes('1.5b')) return 'Qwen 2.5 Coder 1.5B'
    if (model.includes('qwen') && model.includes('32b')) return 'Qwen 2.5 32B'
    if (model.includes('qwen') && model.includes('1.5b')) return 'Qwen 2.5 1.5B'
    if (model.includes('qwen')) return 'Qwen 2.5'
    return model
  }

  // Check if this is a summarized message (shown for reference only)
  const isSummarized = message.summarized && !message.isSummaryMarker

  return (
    <div
      className={cn(
        "flex flex-col gap-2",
        isUser ? "items-end" : "items-start",
        isSummarized && "opacity-50" // Dim summarized messages
      )}
    >
      {/* Summarized indicator */}
      {isSummarized && (
        <div className="text-xs text-muted-foreground italic px-1">
          (summarized - for reference only)
        </div>
      )}

      {/* Message content */}
      <div
        className={cn(
          "max-w-[95%] md:max-w-[90%]",
          isUser
            ? "rounded-2xl px-5 py-3 bg-primary text-primary-foreground"
            : "px-1", // No background for assistant - cleaner look
          message.isSummaryMarker && "bg-purple-500/10 border border-purple-500/30 rounded-lg px-4 py-3" // Summary marker styling
        )}
      >
        {/* Thinking block for reasoning models */}
        {!isUser && message.thinking && (
          <ThinkingBlock
            thinking={message.thinking}
            isStreaming={isStreaming}
          />
        )}

        {/* Image thumbnails for user messages with images */}
        {isUser && message.images && message.images.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {message.images.map((dataUrl, index) => (
              <img
                key={index}
                src={dataUrl}
                alt={`Attached image ${index + 1}`}
                className="max-w-48 max-h-48 rounded-lg object-cover cursor-pointer
                           hover:opacity-90 transition-opacity border border-white/20"
                onClick={() => window.open(dataUrl, '_blank')}
                title="Click to view full size"
              />
            ))}
          </div>
        )}

        {/* Message content */}
        {isUser ? (
          <p className="whitespace-pre-wrap text-base">{message.content}</p>
        ) : (
          <div className="prose prose-lg dark:prose-invert max-w-none
                          prose-p:my-3 prose-p:leading-8 prose-p:text-base
                          prose-headings:my-4 prose-headings:font-semibold prose-headings:text-foreground
                          prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg
                          prose-pre:bg-slate-800 prose-pre:text-slate-200 prose-pre:rounded-xl prose-pre:p-4 prose-pre:my-4
                          prose-pre:text-base prose-pre:leading-7 prose-pre:overflow-x-auto
                          prose-code:bg-slate-100 dark:prose-code:bg-slate-700/50 prose-code:rounded-md prose-code:px-1.5 prose-code:py-0.5
                          prose-code:text-base prose-code:font-medium
                          prose-code:before:content-none prose-code:after:content-none
                          prose-ul:my-3 prose-ol:my-3 prose-li:my-1 prose-li:text-base prose-li:leading-8
                          prose-strong:font-semibold prose-strong:text-foreground
                          prose-a:text-primary prose-a:no-underline hover:prose-a:underline">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                a: ({ href, children, ...props }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                    {children}
                  </a>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {/**
       * Routing details row (assistant messages only)
       */}
      {!isUser && message.metadata && (
        <div className="flex items-center gap-3 text-xs text-muted-foreground px-1 flex-wrap">
          {/* Tier badge with color */}
          <div className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 rounded-full",
            tierConfig?.bgColor || 'bg-muted'
          )}>
            {TierIcon && <TierIcon className={cn("w-3.5 h-3.5", tierConfig?.color)} />}
            <span className={cn("font-medium text-xs", tierConfig?.color)}>
              {tierConfig?.label}
            </span>
          </div>

          {/* Model name with provider logo */}
          <span className="text-muted-foreground flex items-center gap-1">
            <ModelLogo model={message.metadata.model || ''} className="w-3.5 h-3.5" />
            {formatModelName(message.metadata.model || 'unknown')}
          </span>

          {/* Duration (if available and valid number) */}
          {(() => {
            const duration = parseFloat(String(message.metadata.duration))
            return !isNaN(duration) && duration > 0 ? (
              <div className="flex items-center gap-1 text-muted-foreground">
                <Clock className="w-3.5 h-3.5" />
                <span>{duration.toFixed(2)}s</span>
              </div>
            ) : null
          })()}

          {/* Cost */}
          <div className="flex items-center gap-1">
            <span>💰</span>
            {tierKey === 'local' ? (
              <span className="text-green-600 dark:text-green-400 font-medium">FREE</span>
            ) : tierKey === 'lakeshore' ? (
              <span className="text-green-600 dark:text-green-400 font-medium">FREE (UIC)</span>
            ) : (() => {
              const cost = parseFloat(String(message.metadata.cost))
              const wasStopped = message.content.includes('[Generation stopped]')
              const isEstimated = message.metadata.cost_estimated === true
              const hasCost = !isNaN(cost) && cost > 0

              // If we have cost data, show it
              if (hasCost) {
                // Show "estimated" suffix if cost was estimated due to interrupted streaming
                // Estimation is based on ~4 characters per token using pricing from litellm_config.yaml
                const suffix = isEstimated ? ' (estimated)' : ''
                return <span>~${cost.toFixed(6)}{suffix}</span>
              }

              // No cost data
              if (wasStopped) {
                // Stopped before cost could be calculated
                return <span className="text-muted-foreground">--</span>
              }

              return <span className="text-muted-foreground">calculating...</span>
            })()}
          </div>

          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="p-1.5 hover:bg-muted rounded-lg transition-colors ml-auto"
            aria-label="Copy message"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </div>
      )}

      {/**
       * Fallback warning (shown when tier was unavailable and we switched)
       */}
      {!isUser && !isStreaming && message.metadata?.fallback_used && message.metadata?.original_tier && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-600 dark:text-yellow-400 mt-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm">
            <span className="font-medium">Tier Fallback:</span>{' '}
            {message.metadata.original_tier.charAt(0).toUpperCase() + message.metadata.original_tier.slice(1)} was unavailable, automatically switched to {message.metadata.tier.charAt(0).toUpperCase() + message.metadata.tier.slice(1)}.
          </span>
        </div>
      )}
    </div>
  )
}
