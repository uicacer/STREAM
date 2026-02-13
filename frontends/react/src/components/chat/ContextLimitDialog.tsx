/**
 * ContextLimitDialog - Shows when conversation exceeds context window
 *
 * Gives users 4 options:
 * 1. New Chat - Start fresh conversation
 * 2. Trim & Retry - Remove history, keep last question
 * 3. Summarize - Compress history using Cloud, then retry
 * 4. Use Cloud - Switch to Cloud tier (200K context)
 */

import { AlertTriangle, Trash2, Cloud, MessageSquarePlus, FileText, Loader2 } from 'lucide-react'

export interface ContextErrorInfo {
  type: 'context_exceeded'
  message: string
  estimated_tokens: number
  model_limit: number
}

interface ContextLimitDialogProps {
  error: ContextErrorInfo
  onNewChat: () => void
  onTrimHistory: () => void
  onSummarize: () => void
  onUseCloud: () => void
  onClose: () => void
  cloudAvailable: boolean
  isSummarizing: boolean
}

export function ContextLimitDialog({
  error,
  onNewChat,
  onTrimHistory,
  onSummarize,
  onUseCloud,
  onClose,
  cloudAvailable,
  isSummarizing,
}: ContextLimitDialogProps) {
  // Calculate overage if we have the numbers (pre-flight check has them, mid-stream doesn't)
  const hasTokenInfo = error.estimated_tokens > 0 && error.model_limit > 0
  const overagePercent = hasTokenInfo
    ? Math.round(((error.estimated_tokens - error.model_limit) / error.model_limit) * 100)
    : 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-card border rounded-xl shadow-2xl max-w-md w-full mx-4 overflow-hidden">
        {/* Header */}
        <div className="bg-yellow-500/10 border-b border-yellow-500/30 px-6 py-4 flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-yellow-500" />
          <div>
            <h2 className="text-lg font-semibold text-foreground">Context Limit Exceeded</h2>
            {hasTokenInfo && (
              <p className="text-sm text-muted-foreground">
                Conversation is {overagePercent}% over the limit
              </p>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4">
          <p className="text-sm text-muted-foreground">
            {hasTokenInfo
              ? `Your conversation (${error.estimated_tokens.toLocaleString()} tokens) exceeds the model's limit (${error.model_limit.toLocaleString()} tokens).`
              : error.message || 'Your conversation is too long for this model.'}
          </p>

          <p className="text-sm font-medium">Choose how to proceed:</p>

          {/* Options */}
          <div className="space-y-2">
            {/* Option 1: New Chat */}
            <button
              onClick={onNewChat}
              disabled={isSummarizing}
              className="w-full flex items-center gap-3 p-3 rounded-lg border border-border
                         hover:bg-accent hover:border-accent-foreground/20 transition-colors text-left
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="p-2 rounded-lg bg-blue-500/10">
                <MessageSquarePlus className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <div className="font-medium">Start New Chat</div>
                <div className="text-sm text-muted-foreground">Clear conversation and start fresh</div>
              </div>
            </button>

            {/* Option 2: Trim History */}
            <button
              onClick={onTrimHistory}
              disabled={isSummarizing}
              className="w-full flex items-center gap-3 p-3 rounded-lg border border-border
                         hover:bg-accent hover:border-accent-foreground/20 transition-colors text-left
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="p-2 rounded-lg bg-orange-500/10">
                <Trash2 className="w-5 h-5 text-orange-500" />
              </div>
              <div>
                <div className="font-medium">Trim & Retry</div>
                <div className="text-sm text-muted-foreground">Remove history, retry your last question</div>
              </div>
            </button>

            {/* Option 3: Summarize (requires Cloud) */}
            <button
              onClick={onSummarize}
              disabled={!cloudAvailable || isSummarizing}
              className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-colors text-left
                         ${cloudAvailable && !isSummarizing
                           ? 'border-border hover:bg-accent hover:border-accent-foreground/20'
                           : 'border-border/50 opacity-60 cursor-not-allowed'}`}
            >
              <div className="p-2 rounded-lg bg-purple-500/10">
                {isSummarizing ? (
                  <Loader2 className="w-5 h-5 text-purple-500 animate-spin" />
                ) : (
                  <FileText className="w-5 h-5 text-purple-500" />
                )}
              </div>
              <div className="flex-1">
                <div className="font-medium flex items-center gap-2">
                  {isSummarizing ? 'Summarizing...' : 'Summarize & Continue'}
                  {!cloudAvailable && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-600">
                      Requires Cloud
                    </span>
                  )}
                </div>
                <div className="text-sm text-muted-foreground">
                  {isSummarizing
                    ? 'Using Claude to compress conversation history...'
                    : 'Compress history with Claude, keep context'}
                </div>
              </div>
            </button>

            {/* Option 4: Use Cloud */}
            <button
              onClick={onUseCloud}
              disabled={isSummarizing}
              className="w-full flex items-center gap-3 p-3 rounded-lg border border-border
                         hover:bg-accent hover:border-accent-foreground/20 transition-colors text-left
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="p-2 rounded-lg bg-green-500/10">
                <Cloud className="w-5 h-5 text-green-500" />
              </div>
              <div>
                <div className="font-medium">Switch to Cloud</div>
                <div className="text-sm text-muted-foreground">Claude has 200K context - fits this conversation</div>
              </div>
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t bg-muted/30">
          <button
            onClick={onClose}
            disabled={isSummarizing}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}

/**
 * Parse error string to check if it's a context exceeded error
 */
export function parseContextError(error: string): ContextErrorInfo | null {
  try {
    const parsed = JSON.parse(error)
    if (parsed.type === 'context_exceeded') {
      return parsed as ContextErrorInfo
    }
  } catch {
    // Not a JSON error
  }
  return null
}
