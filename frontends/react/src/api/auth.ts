/**
 * auth.ts - Globus Compute Authentication API Client
 * ===================================================
 *
 * This module handles authentication for the Lakeshore HPC tier.
 *
 * WHAT IS GLOBUS COMPUTE?
 * -----------------------
 * Globus Compute (formerly funcX) is a service that allows you to
 * run Python functions on remote HPC clusters. To use it, you need
 * to authenticate with Globus (similar to logging into Google).
 *
 * WHY AUTHENTICATION?
 * - Lakeshore tier uses UIC's HPC cluster
 * - HPC resources require authentication (who's using the compute?)
 * - Globus provides secure, federated authentication
 *
 * ARCHITECTURE NOTE:
 * ------------------
 * Two modes are supported:
 *
 * DESKTOP MODE:
 * - The backend runs on the same machine as the user's browser
 * - We call /v1/auth/status and /v1/auth/globus on the backend directly
 * - The backend can open the browser for OAuth (same machine)
 *
 * DOCKER (SERVER) MODE:
 * - The middleware runs in Docker, which CANNOT open browsers for OAuth
 * - We use a separate "auth helper" server that runs on the HOST machine
 * - This server CAN open browsers since it runs outside Docker
 *
 * AUTH HELPER SERVER (Docker mode only):
 * - Runs on localhost:8765 (start with: python frontends/react/auth_server.py)
 * - GET  /status - Check if authenticated
 * - POST /auth   - Trigger browser-based authentication
 *
 * AUTHENTICATION FLOW (Docker mode):
 * 1. React calls auth helper (localhost:8765/status)
 * 2. If not authenticated, user clicks "Authenticate"
 * 3. React calls auth helper (localhost:8765/auth)
 * 4. Auth helper opens browser for Globus login (works because it's on host!)
 * 5. After login, credentials saved to ~/.globus_compute
 * 6. Auth helper tells Docker middleware to reload credentials
 * 7. Lakeshore tier becomes available!
 *
 * AUTHENTICATION FLOW (Desktop mode):
 * 1. React calls backend (/v1/auth/status)
 * 2. If not authenticated, user clicks "Authenticate"
 * 3. React calls backend (/v1/auth/globus)
 * 4. Backend opens browser for Globus login (same machine)
 * 5. After login, credentials saved to ~/.globus_compute
 * 6. Lakeshore tier becomes available!
 */

/**
 * AUTH HELPER SERVER URL
 *
 * This server runs on the HOST machine (not Docker) so it can open browsers.
 * It starts automatically when you run `npm run dev` (via concurrently).
 */
const AUTH_HELPER_URL = 'http://localhost:8765'

/**
 * AuthStatus - Response from /v1/auth/status
 */
export interface AuthStatus {
  authenticated: boolean
  message: string
}

/**
 * AuthResult - Response from /v1/auth/globus
 */
export interface AuthResult {
  success: boolean
  message: string
}

/**
 * Check if authenticated with Globus Compute
 *
 * Tries the backend endpoint first (/v1/auth/status), which works in both
 * desktop and Docker mode. The backend checks ~/.globus_compute/storage.db
 * directly via globus_is_authenticated().
 *
 * Falls back to the auth helper server (localhost:8765) for Docker mode,
 * where the auth helper runs on the HOST and has access to credential files.
 */
export async function checkAuthStatus(): Promise<AuthStatus> {
  // Try backend endpoint first (works in desktop mode and Docker mode)
  try {
    const response = await fetch('/v1/auth/status')
    if (response.ok) {
      return response.json()
    }
  } catch {
    // Backend endpoint not available, try auth helper below
  }

  // Fall back to auth helper (Docker mode — runs on host)
  try {
    const response = await fetch(`${AUTH_HELPER_URL}/status`)
    if (!response.ok) {
      return {
        authenticated: false,
        message: `Auth helper error: ${response.status}`,
      }
    }
    return response.json()
  } catch {
    return {
      authenticated: false,
      message: 'Could not check authentication status',
    }
  }
}

/**
 * Trigger Globus Compute authentication
 *
 * Tries the backend endpoint first (/v1/auth/globus), which works in desktop
 * mode since the backend runs on the same machine and can open the browser.
 *
 * Falls back to the auth helper server (localhost:8765) for Docker mode,
 * where the backend is inside a container and CANNOT open browsers.
 *
 * DESKTOP MODE FLOW:
 * 1. React calls POST /v1/auth/globus
 * 2. Backend calls authenticate_with_browser_callback()
 * 3. Browser opens with Globus login page
 * 4. User logs in, credentials saved to ~/.globus_compute
 * 5. Success returned to React
 *
 * DOCKER MODE FLOW:
 * 1. React calls POST http://localhost:8765/auth
 * 2. Auth helper imports globus_auth module
 * 3. Auth helper calls authenticate_with_browser_callback()
 * 4. Browser opens with Globus login page
 * 5. User logs in, credentials saved to ~/.globus_compute
 * 6. Auth helper tells Docker middleware to reload credentials
 * 7. Success returned to React
 *
 * PREREQUISITE (Docker only): Start auth helper with: python frontends/react/auth_server.py
 */
export async function authenticateGlobus(): Promise<AuthResult> {
  // Try backend endpoint first (desktop mode — backend can open browser)
  try {
    const response = await fetch('/v1/auth/globus', { method: 'POST' })
    if (response.ok) {
      return response.json()
    }
  } catch {
    // Backend endpoint not available, try auth helper below
  }

  // Fall back to auth helper (Docker mode — runs on host, can open browser)
  try {
    const response = await fetch(`${AUTH_HELPER_URL}/auth`, {
      method: 'POST',
    })
    if (!response.ok) {
      return {
        success: false,
        message: `Auth helper error: ${response.status}`,
      }
    }
    return response.json()
  } catch {
    return {
      success: false,
      message: 'Could not reach authentication service',
    }
  }
}

/**
 * Reload proxy credentials in Docker middleware
 *
 * This is called AFTER authentication to tell the Docker middleware
 * to reload the Globus credentials from the mounted volume.
 *
 * WHY NEEDED?
 * - Credentials are saved to ~/.globus_compute on HOST
 * - Docker has this mounted as a volume
 * - But the middleware may have cached "no credentials" state
 * - This endpoint tells it to re-check the credential files
 *
 * NOTE: The auth helper server calls this automatically after
 * successful authentication, so you usually don't need to call
 * this directly from the UI.
 */
export async function reloadProxyCredentials(): Promise<AuthResult> {
  try {
    // This still calls the Docker middleware (where the proxy lives)
    const response = await fetch('/v1/auth/reload-proxy', {
      method: 'POST',
    })
    if (!response.ok) {
      return {
        success: false,
        message: `Server error: ${response.status}`,
      }
    }
    return response.json()
  } catch (error) {
    return {
      success: false,
      message: `Connection error: ${error}`,
    }
  }
}
