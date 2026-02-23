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
import { Copy, Check, Laptop, Building2, Cloud, Clock, AlertTriangle, FileText, ChevronDown, ChevronRight, ShieldCheck, ShieldAlert } from 'lucide-react'
import { ModelLogo } from '../icons/ProviderLogos'
import { cn } from '../../lib/utils'
import { ThinkingBlock } from './ThinkingBlock'
import { formatFileSize } from '../../api/documents'
import type { Message as MessageType, DocumentAttachment } from '../../types'

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
    // Dynamic OpenRouter models — extract human-readable name from the ID
    // e.g., "cloud-or-dynamic-anthropic/claude-opus-4.6" → "claude-opus-4.6"
    if (model.startsWith('cloud-or-dynamic-')) {
      const parts = model.replace('cloud-or-dynamic-', '').split('/')
      const rawName = parts[parts.length - 1] || model
      return rawName
        .replace(/-/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
    }

    // Curated OpenRouter models
    const curatedNames: Record<string, string> = {
      'cloud-or-claude': 'Claude Sonnet 4',
      'cloud-or-gpt4o': 'GPT-4o',
      'cloud-or-gemini-pro': 'Gemini 2.5 Pro',
      'cloud-or-gemini-flash': 'Gemini 2.5 Flash',
      'cloud-or-o3-mini': 'o3-mini',
      'cloud-or-deepseek-r1': 'DeepSeek R1',
      'cloud-or-llama-maverick': 'Llama 4 Maverick',
      'cloud-or-deepseek-v3': 'DeepSeek V3',
      'cloud-or-glm5': 'GLM-5',
    }
    if (curatedNames[model]) return curatedNames[model]

    // Direct provider models
    if (model === 'cloud-claude') return 'Claude Sonnet 4'
    if (model === 'cloud-gpt') return 'GPT-4o'
    if (model === 'cloud-gpt-cheap') return 'GPT-4o Mini'

    // Local and Lakeshore models
    if (model === 'local-vision') return 'Gemma 3 4B (Text + Vision)'
    if (model.includes('llama')) return 'Llama 3.2 3B'
    if (model.includes('qwq')) return 'QwQ 1.5B'
    if (model.includes('coder') && model.includes('1.5b')) return 'Qwen 2.5 Coder 1.5B'
    if (model.includes('deepseek')) return 'DeepSeek R1 1.5B'
    if (model.includes('qwen') && model.includes('72b')) return 'Qwen 2.5 72B'
    if (model.includes('qwen') && model.includes('32b')) return 'Qwen 2.5 32B'
    if (model.includes('qwen') && model.includes('1.5b')) return 'Qwen 2.5 1.5B'
    if (model.includes('qwen')) return 'Qwen 2.5'
    return model
  }

  /**
   * Check if the verified model matches what was requested.
   * Maps STREAM's internal model names to expected provider model IDs
   * so we can confirm the right model actually responded.
   */
  const checkModelVerification = (requested: string, verified: string | undefined): 'verified' | 'mismatch' | 'unknown' => {
    if (!verified) return 'unknown'

    const v = verified.toLowerCase()

    // Map STREAM model names to substrings expected in the provider's response
    const expectedPatterns: Record<string, string[]> = {
      // OpenRouter curated models
      'cloud-or-claude': ['claude-sonnet', 'claude-4'],
      'cloud-or-gpt4o': ['gpt-4o'],
      'cloud-or-gemini-pro': ['gemini-2.5-pro', 'gemini-2'],
      'cloud-or-gemini-flash': ['gemini-2.5-flash', 'gemini-flash'],
      'cloud-or-o3-mini': ['o3-mini'],
      'cloud-or-deepseek-r1': ['deepseek-r1', 'deepseek/deepseek-r1'],
      'cloud-or-llama-maverick': ['llama-4-maverick', 'maverick'],
      'cloud-or-deepseek-v3': ['deepseek-chat', 'deepseek-v3'],
      'cloud-or-glm5': ['glm-5', 'glm5'],
      // Direct provider models
      'cloud-claude': ['claude-sonnet', 'claude-4'],
      'cloud-gpt': ['gpt-4o'],
      'cloud-gpt-cheap': ['gpt-4o-mini'],
      // Local
      'local-llama': ['llama'],
      'local-vision': ['gemma'],
    }

    // For dynamic OpenRouter models, extract the model ID and check directly
    if (requested.startsWith('cloud-or-dynamic-')) {
      const requestedId = requested.replace('cloud-or-dynamic-', '').toLowerCase()
      return v.includes(requestedId) ? 'verified' : 'mismatch'
    }

    const patterns = expectedPatterns[requested]
    if (!patterns) return 'unknown'

    return patterns.some(p => v.includes(p)) ? 'verified' : 'mismatch'
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

        {/* Document attachment previews (collapsible, like Claude's file display) */}
        {isUser && message.documents && message.documents.length > 0 && (
          <div className="flex flex-col gap-1.5 mb-2">
            {message.documents.map((doc) => (
              <CollapsibleDocumentPreview key={doc.id} document={doc} isUserMessage />
            ))}
          </div>
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

          {/* Model name with provider logo and verification */}
          {(() => {
            const model = message.metadata!.model || 'unknown'
            const verified = message.metadata!.verified_model
            const status = checkModelVerification(model, verified)

            return (
              <span className="text-muted-foreground flex items-center gap-1">
                <ModelLogo model={model} className="w-3.5 h-3.5" />
                {formatModelName(model)}
                {status === 'verified' && (
                  <span title={`Verified by provider: ${verified}`}>
                    <ShieldCheck className="w-3.5 h-3.5 text-green-500" />
                  </span>
                )}
                {status === 'mismatch' && (
                  <span title={`Expected ${formatModelName(model)} but provider returned: ${verified}`} className="flex items-center gap-0.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-amber-500" />
                    <span className="text-amber-500 text-[10px]">({verified})</span>
                  </span>
                )}
              </span>
            )
          })()}

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

              if (hasCost) {
                const suffix = isEstimated ? ' (estimated)' : ''
                return <span>~${cost.toFixed(6)}{suffix}</span>
              }

              if (wasStopped) {
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


// =============================================================================
// CollapsibleDocumentPreview — Compact document display in chat messages
// =============================================================================

/**
 * CollapsibleDocumentPreview shows an attached document as a compact chip
 * that can be expanded to reveal a text preview, similar to Claude's file
 * attachment display.
 *
 * Collapsed state: [📄 report.pdf — 12 pages · 3 images · 15.2K chars]
 * Expanded state:  Shows first ~500 characters of extracted text content.
 *
 * This keeps the chat clean while allowing users to verify what was extracted.
 */
function CollapsibleDocumentPreview({
  document: doc,
  isUserMessage = false,
}: {
  document: DocumentAttachment
  isUserMessage?: boolean
}) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div
      className={cn(
        "rounded-lg border transition-colors text-sm",
        isUserMessage
          ? "border-white/20 bg-white/10"
          : "border-border bg-muted/50"
      )}
    >
      {/* Collapsed header — always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-black/5 dark:hover:bg-white/5 rounded-lg transition-colors"
      >
        {isExpanded
          ? <ChevronDown className="w-3.5 h-3.5 flex-shrink-0 opacity-60" />
          : <ChevronRight className="w-3.5 h-3.5 flex-shrink-0 opacity-60" />
        }
        <FileText className="w-4 h-4 flex-shrink-0 opacity-70" />
        <span className="font-medium truncate">{doc.filename}</span>
        <span className="text-xs opacity-60 flex-shrink-0">
          {formatFileSize(doc.fileSize)}
          {doc.pageCount > 0 && ` · ${doc.pageCount} pages`}
          {doc.imageCount > 0 && ` · ${doc.imageCount} images`}
          {` · ${(doc.totalTextLength / 1000).toFixed(1)}K chars`}
        </span>
      </button>

      {/* Expanded preview — shows first ~500 chars of extracted text */}
      {isExpanded && (
        <div className={cn(
          "px-3 pb-3 border-t",
          isUserMessage ? "border-white/10" : "border-border"
        )}>
          <pre className="mt-2 text-xs whitespace-pre-wrap break-words opacity-80 max-h-48 overflow-y-auto font-mono leading-relaxed">
            {doc.textPreview || "(no text extracted)"}
            {doc.totalTextLength > 500 && (
              <span className="text-muted-foreground italic">
                {`\n\n... ${(doc.totalTextLength - 500).toLocaleString()} more characters`}
              </span>
            )}
          </pre>
          {doc.warnings.length > 0 && (
            <div className="mt-2 flex flex-col gap-1">
              {doc.warnings.map((warning, i) => (
                <div key={i} className="flex items-start gap-1.5 text-xs text-yellow-600 dark:text-yellow-400">
                  <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>{warning}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
