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
 * The middleware runs in Docker, which CANNOT open browsers for OAuth.
 * To solve this, we use a separate "auth helper" server that runs on
 * the HOST machine (not Docker). This server CAN open browsers.
 *
 * AUTH HELPER SERVER:
 * - Runs on localhost:8765 (start with: python frontends/react/auth_server.py)
 * - GET  /status - Check if authenticated
 * - POST /auth   - Trigger browser-based authentication
 *
 * AUTHENTICATION FLOW:
 * 1. React calls auth helper (localhost:8765/status)
 * 2. If not authenticated, user clicks "Authenticate"
 * 3. React calls auth helper (localhost:8765/auth)
 * 4. Auth helper opens browser for Globus login (works because it's on host!)
 * 5. After login, credentials saved to ~/.globus_compute
 * 6. Auth helper tells Docker middleware to reload credentials
 * 7. Lakeshore tier becomes available!
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
 * This calls the auth helper server running on the HOST machine.
 * The auth helper checks if Globus credentials exist in ~/.globus_compute.
 *
 * WHY AUTH HELPER?
 * The Docker middleware can't check host credentials directly.
 * The auth helper runs on the host and has access to the credential files.
 */
export async function checkAuthStatus(): Promise<AuthStatus> {
  try {
    // Call the auth helper server (runs on host, not Docker)
    const response = await fetch(`${AUTH_HELPER_URL}/status`)
    if (!response.ok) {
      return {
        authenticated: false,
        message: `Auth helper error: ${response.status}`,
      }
    }
    return response.json()
  } catch (error) {
    // Auth helper not running - should start automatically with npm run dev
    return {
      authenticated: false,
      message: `Auth helper not responding. Try restarting with: npm run dev`,
    }
  }
}

/**
 * Trigger Globus Compute authentication
 *
 * This calls the auth helper server which runs on the HOST machine.
 * The auth helper CAN open a browser (Docker cannot).
 *
 * FLOW:
 * 1. React calls POST http://localhost:8765/auth
 * 2. Auth helper imports globus_auth module
 * 3. Auth helper calls authenticate_with_browser_callback()
 * 4. Browser opens with Globus login page
 * 5. User logs in, credentials saved to ~/.globus_compute
 * 6. Auth helper tells Docker middleware to reload credentials
 * 7. Success returned to React
 *
 * PREREQUISITE: Start auth helper with: python frontends/react/auth_server.py
 */
export async function authenticateGlobus(): Promise<AuthResult> {
  try {
    // Call the auth helper server (runs on host, can open browser)
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
  } catch (error) {
    // Auth helper not running - should start automatically with npm run dev
    return {
      success: false,
      message: `Auth helper not responding. Try restarting with: npm run dev`,
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
