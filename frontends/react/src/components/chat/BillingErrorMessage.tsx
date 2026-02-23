/**
 * BillingErrorMessage - Inline billing error shown in chat
 *
 * Shows when the user's OpenRouter account doesn't have enough credits.
 * Common scenario: user set a key spending limit (e.g., $5) but never
 * added actual credits to their account — those are two different things.
 *
 * Links directly to the Credits page where they can add funds.
 */

import { CreditCard, ExternalLink, X } from 'lucide-react'

export interface BillingErrorInfo {
  type: 'billing_limit'
  message: string
}

interface BillingErrorMessageProps {
  error: BillingErrorInfo
  onDismiss: () => void
}

export function BillingErrorMessage({
  error,
  onDismiss,
}: BillingErrorMessageProps) {
  return (
    <div className="flex flex-col gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 max-w-2xl">
      {/* Header */}
      <div className="flex items-start gap-3">
        <CreditCard className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h3 className="font-semibold text-amber-600 dark:text-amber-400">
              Insufficient OpenRouter Credits
            </h3>
            <button
              onClick={onDismiss}
              className="p-1 rounded hover:bg-amber-500/20 transition-colors flex-shrink-0"
              aria-label="Dismiss"
            >
              <X className="w-4 h-4 text-amber-500/70" />
            </button>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            {error.message}
          </p>
        </div>
      </div>

      {/* Action */}
      <div className="flex flex-wrap gap-2 ml-8">
        <a
          href="https://openrouter.ai/settings/credits"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg
                     bg-amber-500/10 text-amber-600 dark:text-amber-400
                     hover:bg-amber-500/20 transition-colors border border-amber-500/30"
        >
          <ExternalLink className="w-4 h-4" />
          Add Credits
        </a>

        <button
          onClick={onDismiss}
          className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg
                     bg-muted hover:bg-muted/80 transition-colors border border-border"
        >
          Dismiss
        </button>
      </div>
    </div>
  )
}

/**
 * Parse error string to check if it's a billing/credit limit error
 */
export function parseBillingError(error: string): BillingErrorInfo | null {
  try {
    const parsed = JSON.parse(error)
    if (parsed.type === 'billing_limit') {
      return {
        type: 'billing_limit',
        message: parsed.message,
      }
    }
  } catch {
    // Not a JSON error
  }
  return null
}
