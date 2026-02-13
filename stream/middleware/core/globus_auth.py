"""
Globus Compute Zero-Friction OAuth Authentication

This module implements automatic browser-based OAuth authentication for Globus Compute.
The Globus SDK already has built-in browser authentication that opens a browser window
and automatically captures the OAuth response using a local callback server.

# How OAuth 2.0 Works (Educational Overview)
# ==========================================
#
# OAuth 2.0 is an authorization framework that allows applications to obtain limited
# access to user accounts. Here's the flow that the Globus SDK implements:
#
# 1. User needs to authenticate → We detect this with app.login_required()
# 2. We call app.login() → SDK starts a local callback server on a random port
# 3. SDK generates an authorization URL with the local callback address
# 4. SDK opens browser to Globus login page
# 5. User logs in on Globus website (happens in browser)
# 6. Globus redirects back to the local callback URL with an authorization code
# 7. SDK's local server captures this code automatically
# 8. SDK exchanges the code for access tokens
# 9. SDK saves tokens to disk (~/.globus_compute/storage.db)
# 10. User can now make authenticated requests!
#
# The key benefit: Steps 2-9 happen automatically with a single app.login() call.
# No manual code entry, no custom servers, just pure SDK functionality.

Author: Claude & Nassar
Date: February 2026
"""

import logging
from pathlib import Path

from globus_compute_sdk.sdk.auth.globus_app import get_globus_app
from globus_compute_sdk.sdk.client import Client
from globus_sdk import GlobusError
from globus_sdk.login_flows import LocalServerLoginFlowManager

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN AUTHENTICATION FUNCTION
# =============================================================================


def authenticate_with_browser_callback() -> tuple[bool, str]:
    """
    Perform zero-friction OAuth authentication using the Globus SDK's built-in method.

    This function uses the Globus SDK's app.login() method which automatically:
    1. Checks if already authenticated (returns immediately if so)
    2. Starts a local OAuth callback server on a random high port
    3. Opens your browser to the Globus login page
    4. Captures the OAuth code automatically when Globus redirects back
    5. Exchanges the code for access tokens
    6. Saves tokens to ~/.globus_compute/storage.db

    The entire flow is zero-friction - just call this function and the browser
    opens automatically. No manual code entry required!

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        logger.info("=" * 60)
        logger.info("🔐 Starting Zero-Friction Globus Authentication")
        logger.info("=" * 60)
        logger.info("Opening browser for authentication...")
        logger.info("→ Please authenticate in the browser window that will open")
        logger.info("")

        # Get the base GlobusApp - this is the shared app instance
        app = get_globus_app()

        # Configure it to use LocalServerLoginFlowManager (automatic browser callback)
        # instead of CommandLineLoginFlowManager (manual code entry)
        app._login_flow_manager = LocalServerLoginFlowManager(
            app._login_client,
            request_refresh_tokens=True,  # Enable refresh tokens for persistent auth
        )

        # Create a Client instance with our pre-configured app.
        # IMPORTANT: Pass app=app to use our LocalServerLoginFlowManager config.
        # Without this, Client creates its own app with CommandLineLoginFlowManager.
        # This triggers authentication with ALL required Globus Compute scopes.
        # Note: We don't need to use the client directly - creating it triggers the auth flow
        Client(app=app)

        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Authentication Complete!")
        logger.info("=" * 60)

        return True, "✅ Authentication successful!"

    except GlobusError as e:
        # Globus SDK error (could be auth, network, etc.)
        error_message = str(e)

        # Check if it's a browser/environment issue
        if "browser" in error_message.lower() or "display" in error_message.lower():
            logger.error("=" * 60)
            logger.error("Cannot open browser automatically (SSH/headless environment)")
            logger.error("=" * 60)
            logger.error("Please authenticate manually:")
            logger.error(
                "  1. Run: python -c 'from globus_compute_sdk.sdk.auth.globus_app import get_globus_app; get_globus_app().login()'"
            )
            logger.error("  2. Follow the authentication prompts")
            logger.error("  3. Restart STREAM services")
            logger.error("=" * 60)
            return False, (
                "❌ Automatic browser authentication not available.\n"
                "Please authenticate manually (see logs for instructions)."
            )
        else:
            logger.error(f"Globus authentication error: {e}", exc_info=True)
            return False, f"❌ Authentication failed: {error_message}"

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected authentication error: {e}", exc_info=True)
        return False, f"❌ Authentication failed: {str(e)}"


# =============================================================================
# QUICK CHECK FUNCTION
# =============================================================================


def is_authenticated() -> bool:
    """
    Quick check if already authenticated with Globus Compute.

    First checks if credential files exist on disk (fast), then verifies
    tokens are valid using the SDK. This two-step approach:
    - Detects deleted credentials immediately (file check is cheap)
    - Validates token expiry/scopes when files exist (SDK check)

    Returns:
        True if authenticated with valid tokens, False otherwise
    """
    # Step 1: Quick file existence check (catches deleted credentials)
    # Globus Compute stores credentials in ~/.globus_compute/storage.db
    storage_file = Path.home() / ".globus_compute" / "storage.db"

    if not storage_file.exists():
        return False

    # Step 2: File exists - verify tokens are valid using SDK
    # This checks token expiry and required scopes
    try:
        app = get_globus_app()
        return not app.login_required()
    except Exception:
        return False


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    # This runs only if you execute this file directly (for testing)
    # It demonstrates the authentication flow

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("Globus Compute Zero-Friction Authentication Demo")
    print("=" * 70)
    print()

    if is_authenticated():
        print("✓ Already authenticated!")
    else:
        print("Starting authentication flow...")
        success, message = authenticate_with_browser_callback()
        print(f"\n{message}")
