/**
 * AuthErrorMessage - Inline auth error shown in chat
 *
 * Shows when cloud provider auth fails, displayed inline in the chat
 * instead of as a blocking modal. More modern and less disruptive.
 *
 * Offers quick actions:
 * - Use Local tier (if available)
 * - Use Lakeshore tier (if available)
 * - Go to Settings for more options
 */

import { AlertTriangle, Laptop, Building2, Settings, X } from 'lucide-react'
import { useHealthStore } from '../../stores/healthStore'

export interface AuthErrorInfo {
  type: 'auth_subscription' | 'rate_limit'
  message: string
  raw_error?: string
  provider?: string
}

interface AuthErrorMessageProps {
  error: AuthErrorInfo
  onSwitchTier: (tier: 'local' | 'lakeshore') => void
  onDismiss: () => void
}

export function AuthErrorMessage({
  error,
  onSwitchTier,
  onDismiss,
}: AuthErrorMessageProps) {
  // Get tier availability from health store
  const localStatus = useHealthStore(state => state.local)
  const lakeshoreStatus = useHealthStore(state => state.lakeshore)

  const localAvailable = localStatus?.available ?? false
  const lakeshoreAvailable = lakeshoreStatus?.available ?? false

  const isRateLimit = error.type === 'rate_limit'

  return (
    <div className="flex flex-col gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30 max-w-2xl">
      {/* Header */}
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h3 className="font-semibold text-red-600 dark:text-red-400">
              {isRateLimit ? 'Rate Limit Exceeded' : 'Cloud Authentication Failed'}
            </h3>
            <button
              onClick={onDismiss}
              className="p-1 rounded hover:bg-red-500/20 transition-colors flex-shrink-0"
              aria-label="Dismiss"
            >
              <X className="w-4 h-4 text-red-500/70" />
            </button>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Your cloud API key may be invalid or your subscription may have expired.
            {error.provider && (
              <span className="text-xs opacity-70"> (Provider: {error.provider})</span>
            )}
          </p>
        </div>
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap gap-2 ml-8">
        {localAvailable && (
          <button
            onClick={() => onSwitchTier('local')}
            className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg
                       bg-orange-500/10 text-orange-600 dark:text-orange-400
                       hover:bg-orange-500/20 transition-colors border border-orange-500/30"
          >
            <Laptop className="w-4 h-4" />
            Use Local
          </button>
        )}

        {lakeshoreAvailable && (
          <button
            onClick={() => onSwitchTier('lakeshore')}
            className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg
                       bg-green-500/10 text-green-600 dark:text-green-400
                       hover:bg-green-500/20 transition-colors border border-green-500/30"
          >
            <Building2 className="w-4 h-4" />
            Use Lakeshore
          </button>
        )}

        <button
          onClick={onDismiss}
          className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg
                     bg-muted hover:bg-muted/80 transition-colors border border-border"
        >
          <Settings className="w-4 h-4" />
          Open Settings
        </button>
      </div>

      {/* Technical details (collapsed) */}
      {error.raw_error && (
        <details className="ml-8 text-xs">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            Technical details
          </summary>
          <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-auto max-h-20">
            {error.raw_error}
          </pre>
        </details>
      )}
    </div>
  )
}

/**
 * Parse error string to check if it's an auth/subscription error
 * Handles both 'type' and 'error_type' keys for compatibility
 */
export function parseAuthError(error: string): AuthErrorInfo | null {
  try {
    const parsed = JSON.parse(error)
    // Check for both 'type' and 'error_type' keys
    const errorType = parsed.type || parsed.error_type
    if (errorType === 'auth_subscription' || errorType === 'rate_limit') {
      return {
        type: errorType,
        message: parsed.message,
        raw_error: parsed.raw_error,
        provider: parsed.provider,
      }
    }
  } catch {
    // Not a JSON error
  }
  return null
}
