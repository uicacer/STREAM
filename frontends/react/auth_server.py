#!/usr/bin/env python3
"""
React Auth Helper Server
========================

This small server runs on the HOST machine (not Docker) to handle
Globus Compute authentication. It can open a browser because it
runs locally on your machine.

USAGE:
    python frontends/react/auth_server.py

This starts a server on port 8765 that the React frontend can call
to trigger authentication.

WHY IS THIS NEEDED?
- The middleware runs in Docker (can't open browsers)
- Streamlit runs on host (can open browsers)
- React frontend is JavaScript (can't run Python)
- This script bridges the gap: runs on host, handles auth

ENDPOINTS:
- GET  /status  - Check if authenticated
- POST /auth    - Trigger Globus authentication
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add project root to path so we can import stream modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Lakeshore proxy URL (Docker container)
LAKESHORE_PROXY_URL = os.getenv("LAKESHORE_PROXY_URL", "http://localhost:8001")


class AuthHandler(BaseHTTPRequestHandler):
    """Handle authentication requests from React frontend."""

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response with CORS headers."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/status":
            self._handle_status()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/auth":
            self._handle_auth()
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_status(self):
        """Check if authenticated with Globus Compute."""
        try:
            from stream.middleware.core.globus_auth import is_authenticated

            if is_authenticated():
                self._send_json(
                    {"authenticated": True, "message": "Authenticated with Globus Compute"}
                )
            else:
                self._send_json({"authenticated": False, "message": "Not authenticated"})
        except ImportError as e:
            self._send_json({"authenticated": False, "message": f"Globus SDK not available: {e}"})
        except Exception as e:
            logger.error(f"Error checking auth status: {e}")
            self._send_json({"authenticated": False, "message": f"Error: {e}"})

    def _handle_auth(self):
        """Trigger Globus authentication (opens browser)."""
        try:
            from stream.middleware.core.globus_auth import (
                authenticate_with_browser_callback,
                is_authenticated,
            )

            # Check if already authenticated
            if is_authenticated():
                self._send_json({"success": True, "message": "Already authenticated"})
                return

            logger.info("Starting Globus authentication...")
            logger.info("A browser window will open for you to log in.")

            # This will open a browser!
            success, message = authenticate_with_browser_callback()

            if success:
                logger.info("Authentication successful!")
                # Reload proxy credentials
                self._reload_proxy()

            self._send_json({"success": success, "message": message})

        except ImportError as e:
            self._send_json({"success": False, "message": f"Globus SDK not available: {e}"})
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self._send_json({"success": False, "message": f"Error: {e}"})

    def _reload_proxy(self):
        """Restart the Lakeshore proxy Docker container to pick up new credentials.

        The Globus credentials are volume-mounted into the container. After
        authentication, we need to restart the container for it to load
        the new credentials from the mounted volume.

        Also refreshes the middleware's health cache so the UI updates immediately.
        """
        container_name = "stream-lakeshore-proxy"
        logger.info(f"Restarting {container_name} to load new credentials...")

        try:
            result = subprocess.run(
                ["docker", "restart", container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                logger.info(f"✅ {container_name} restarted successfully")
                # Wait for proxy to be ready, then refresh middleware health cache
                time.sleep(3)
                self._refresh_health_cache()
            else:
                logger.warning(f"Failed to restart container: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout restarting {container_name}")
        except FileNotFoundError:
            logger.warning("Docker command not found. Please restart the proxy manually.")
        except Exception as e:
            logger.warning(f"Could not restart proxy: {e}")

    def _refresh_health_cache(self):
        """Refresh the middleware's tier health cache so UI updates immediately."""
        middleware_url = "http://localhost:5000/health/tiers/refresh"
        try:
            req = urllib.request.Request(middleware_url, method="POST")
            with urllib.request.urlopen(req, timeout=10):
                logger.info("✅ Health cache refreshed - UI should update shortly")
        except Exception as e:
            logger.warning(f"Could not refresh health cache: {e}")

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def main():
    """Start the auth server."""
    port = 8765
    server = HTTPServer(("localhost", port), AuthHandler)

    print("=" * 60)
    print("🔐 STREAM React Auth Helper")
    print("=" * 60)
    print(f"Running on http://localhost:{port}")
    print()
    print("This server handles Globus authentication for the React app.")
    print("It runs on your host machine so it can open a browser.")
    print()
    print("Endpoints:")
    print(f"  GET  http://localhost:{port}/status - Check auth status")
    print(f"  POST http://localhost:{port}/auth   - Trigger authentication")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down auth server")
        server.shutdown()


if __name__ == "__main__":
    main()
