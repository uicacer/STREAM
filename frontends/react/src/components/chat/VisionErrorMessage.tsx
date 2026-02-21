/**
 * VisionErrorMessage - Inline error for images sent to text-only models
 *
 * Shown when a user attaches an image but the selected model can't process
 * images. Offers quick actions to switch to a vision-capable model or Auto mode.
 */

import { ImageOff, Eye, Bot, X } from 'lucide-react'

export interface VisionErrorInfo {
  type: 'model_not_multimodal'
  message: string
  model?: string
}

interface VisionErrorMessageProps {
  error: VisionErrorInfo
  onSwitchToVision: () => void
  onSwitchToAuto: () => void
  onDismiss: () => void
}

export function VisionErrorMessage({
  error,
  onSwitchToVision,
  onSwitchToAuto,
  onDismiss,
}: VisionErrorMessageProps) {
  return (
    <div className="flex flex-col gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 max-w-2xl">
      {/* Header */}
      <div className="flex items-start gap-3">
        <ImageOff className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h3 className="font-semibold text-amber-600 dark:text-amber-400">
              This Model Can&apos;t Process Images
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

      {/* Quick actions */}
      <div className="flex flex-wrap gap-2 ml-8">
        <button
          onClick={onSwitchToVision}
          className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg
                     bg-purple-500/10 text-purple-600 dark:text-purple-400
                     hover:bg-purple-500/20 transition-colors border border-purple-500/30"
        >
          <Eye className="w-4 h-4" />
          Switch to Vision Model
        </button>

        <button
          onClick={onSwitchToAuto}
          className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg
                     bg-blue-500/10 text-blue-600 dark:text-blue-400
                     hover:bg-blue-500/20 transition-colors border border-blue-500/30"
        >
          <Bot className="w-4 h-4" />
          Use Auto Mode
        </button>
      </div>
    </div>
  )
}

/**
 * Parse error string to check if it's a vision/multimodal error
 */
export function parseVisionError(error: string): VisionErrorInfo | null {
  try {
    const parsed = JSON.parse(error)
    const errorType = parsed.type || parsed.error_type
    if (errorType === 'model_not_multimodal') {
      return {
        type: 'model_not_multimodal',
        message: parsed.message,
        model: parsed.model,
      }
    }
  } catch {
    // Not a JSON error
  }
  return null
}
