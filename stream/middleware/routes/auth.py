"""
STREAM Middleware - Globus Authentication API
==============================================

This endpoint handles Globus Compute authentication for the Lakeshore tier.

AUTHENTICATION FLOW:
1. Frontend calls GET /v1/auth/status to check if authenticated
2. If not authenticated, frontend calls POST /v1/auth/globus to trigger auth
3. Backend opens browser for Globus login
4. After auth completes, backend reloads proxy credentials
5. Frontend polls /v1/auth/status until authenticated

NOTE: This only works when the middleware is running on the same machine
as the user's browser (e.g., localhost development). In Docker or remote
deployments, manual authentication is required.
"""

import logging
import os

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# Lakeshore proxy URL for reloading credentials
LAKESHORE_PROXY_URL = os.getenv("LAKESHORE_PROXY_URL", "http://localhost:8001")


@router.get("/auth/status")
async def get_auth_status():
    """
    Check Globus Compute authentication status.

    Returns:
        - authenticated: bool - Whether valid credentials exist
        - message: str - Human-readable status message
    """
    try:
        from stream.middleware.core.globus_auth import is_authenticated

        if is_authenticated():
            return {
                "authenticated": True,
                "message": "Authenticated with Globus Compute",
            }
        else:
            return {
                "authenticated": False,
                "message": "Not authenticated - Lakeshore tier unavailable",
            }
    except ImportError:
        return {
            "authenticated": False,
            "message": "Globus Compute SDK not installed",
        }
    except Exception as e:
        logger.error(f"Error checking auth status: {e}")
        return {
            "authenticated": False,
            "message": f"Error checking authentication: {str(e)}",
        }


@router.post("/auth/globus")
async def authenticate_globus():
    """
    Trigger Globus Compute authentication.

    This opens a browser window on the host machine for authentication.
    Only works when running locally (not in Docker).

    Returns:
        - success: bool - Whether authentication completed
        - message: str - Result message
    """
    try:
        from stream.middleware.core.globus_auth import (
            authenticate_with_browser_callback,
            is_authenticated,
        )

        # Check if already authenticated
        if is_authenticated():
            return {
                "success": True,
                "message": "Already authenticated with Globus Compute",
            }

        # Trigger browser-based authentication
        logger.info("Starting Globus Compute browser authentication...")
        success, message = authenticate_with_browser_callback()

        if success:
            # Reload proxy credentials
            await _reload_proxy_credentials()

        return {
            "success": success,
            "message": message,
        }

    except ImportError:
        return {
            "success": False,
            "message": "Globus Compute SDK not installed",
        }
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return {
            "success": False,
            "message": f"Authentication failed: {str(e)}",
        }


@router.post("/auth/reload-proxy")
async def reload_proxy():
    """
    Reload Globus credentials in the Lakeshore proxy.

    Call this after authenticating manually to update the proxy.
    """
    success, message = await _reload_proxy_credentials()
    return {"success": success, "message": message}


async def _reload_proxy_credentials() -> tuple[bool, str]:
    """
    Tell the Lakeshore proxy to reload Globus credentials.

    The proxy caches credentials, so after authentication completes,
    we need to tell it to re-read from disk.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{LAKESHORE_PROXY_URL}/reload-auth")

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info(f"Proxy credentials reloaded: {result.get('message')}")
                    return True, result.get("message", "Credentials reloaded")
                else:
                    return False, result.get("message", "Reload failed")
            else:
                return False, f"Proxy returned status {response.status_code}"

    except httpx.ConnectError:
        logger.warning("Could not connect to Lakeshore proxy")
        return False, "Proxy not reachable (may not be running)"
    except Exception as e:
        logger.error(f"Failed to reload proxy credentials: {e}")
        return False, f"Failed to reload: {str(e)}"
