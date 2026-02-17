/**
 * TierStatus.tsx - Tier Availability Indicator
 * =============================================
 *
 * Shows real-time availability status of all AI tiers in the header.
 *
 * Status indicators:
 * - Green dot = Available (for Lakeshore: authenticated AND HPC available)
 * - Red dot = Unavailable (for Lakeshore: not authenticated or HPC down)
 *
 * Lakeshore authentication note:
 * Users must disable VPN before authenticating with Globus.
 * VPN can be reconnected after authentication is complete.
 *
 * Polls backend every 30 seconds and refreshes on window focus.
 */

import { useEffect } from 'react'
import { Loader2 } from 'lucide-react'
import { useHealthStore, getTierDisplayInfo } from '../../stores/healthStore'

// Tier configuration
const tiers = [
  { key: 'local', label: 'Local' },
  { key: 'lakeshore', label: 'Lakeshore' },
  { key: 'cloud', label: 'Cloud' },
] as const

type TierKey = 'local' | 'lakeshore' | 'cloud'

export function TierStatus() {
  const local = useHealthStore(state => state.local)
  const lakeshore = useHealthStore(state => state.lakeshore)
  const cloud = useHealthStore(state => state.cloud)
  const error = useHealthStore(state => state.error)
  const isProviderChanging = useHealthStore(state => state.isProviderChanging)
  const startPolling = useHealthStore(state => state.startPolling)
  const stopPolling = useHealthStore(state => state.stopPolling)

  // Start polling on mount
  useEffect(() => {
    startPolling()
    return () => stopPolling()
  }, [startPolling, stopPolling])

  // Get status for each tier using centralized logic
  const getStatus = (tierKey: TierKey) => {
    const tierData = { local, lakeshore, cloud }[tierKey]

    // Handle loading/error states before delegating to centralized logic
    if (!tierData && error) {
      return { color: 'bg-gray-400', tooltip: 'Status unavailable' }
    }

    return getTierDisplayInfo(tierKey, tierData)
  }

  // Debug: log when isProviderChanging changes
  useEffect(() => {
    console.log('[TierStatus] isProviderChanging changed to:', isProviderChanging)
  }, [isProviderChanging])

  return (
    <div className="flex items-center gap-3">
      {tiers.map(({ key, label }) => {
        const { color, tooltip } = getStatus(key)
        // Show pulsing animation on Cloud indicator while checking health
        const showPulse = isProviderChanging && key === 'cloud'

        return (
          <div
            key={key}
            className="relative group"
          >
            {/* Tier label with status dot */}
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors cursor-default ${
              showPulse ? 'bg-blue-500/10' : 'hover:bg-muted/50'
            }`}>
              {showPulse ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-white" />
              ) : (
                <span className={`w-2 h-2 rounded-full ${color}`} />
              )}
              <span className="text-sm text-muted-foreground">{label}</span>
            </div>

            {/* Tooltip on hover */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-2.5 py-1.5
                          bg-popover border rounded-md shadow-md text-xs whitespace-nowrap
                          opacity-0 invisible group-hover:opacity-100 group-hover:visible
                          transition-all duration-150 z-50">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${showPulse ? 'bg-blue-500 animate-pulse' : color} shrink-0`} />
                <span>{showPulse ? 'Checking...' : tooltip}</span>
              </div>
              {/* Tooltip arrow */}
              <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2
                            bg-popover border-l border-t rotate-45" />
            </div>
          </div>
        )
      })}
    </div>
  )
}
