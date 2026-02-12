/**
 * ErrorBoundary.tsx - Catches React Errors
 * =========================================
 *
 * React Error Boundaries catch JavaScript errors in their child component tree,
 * log those errors, and display a fallback UI instead of crashing the whole app.
 *
 * WHY IS THIS NEEDED?
 * Without an error boundary, if ANY component throws an error during rendering,
 * the ENTIRE React app unmounts and shows a blank page. Error boundaries prevent this.
 */

import { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render shows the fallback UI
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log the error to console for debugging
    console.error('[ErrorBoundary] Caught error:', error)
    console.error('[ErrorBoundary] Component stack:', errorInfo.componentStack)
    this.setState({ errorInfo })
  }

  handleReload = () => {
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen flex flex-col items-center justify-center bg-background p-8">
          <div className="max-w-md text-center">
            <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <h1 className="text-2xl font-bold mb-2">Something went wrong</h1>
            <p className="text-muted-foreground mb-4">
              The application encountered an unexpected error.
            </p>

            {/* Error details */}
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-4 text-left">
              <p className="text-sm font-mono text-red-600 dark:text-red-400 break-all">
                {this.state.error?.message || 'Unknown error'}
              </p>
            </div>

            <button
              onClick={this.handleReload}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Reload Page
            </button>

            {/* Debug info for developers */}
            <details className="mt-6 text-left">
              <summary className="text-sm text-muted-foreground cursor-pointer hover:text-foreground">
                Technical Details
              </summary>
              <pre className="mt-2 p-4 bg-muted rounded-lg text-xs overflow-auto max-h-48">
                {this.state.error?.stack}
                {'\n\nComponent Stack:'}
                {this.state.errorInfo?.componentStack}
              </pre>
            </details>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
