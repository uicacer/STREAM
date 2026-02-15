"""
STREAM Desktop App — Main Entry Point.

This is the file that runs when a user launches the STREAM desktop app.
It orchestrates the entire startup sequence:

    1. Apply desktop config defaults (environment variables)
    2. First-run setup (create ~/.stream/, check Ollama, check models)
    3. Start Ollama (local AI model runner) if not already running
    4. Check for port conflicts (e.g., Docker still running)
    5. Start FastAPI server in a background thread
    6. Wait for the server to be ready
    7. Open a native OS window (PyWebView) pointing to the server
    8. When the user closes the window → clean up and exit

WHY PyWebView:
--------------
PyWebView creates a native window (like any desktop app) that displays
a web page. It uses the OS's built-in web renderer:
  - macOS:   WebKit (same engine as Safari)
  - Windows: Edge WebView2 (same engine as Edge browser)
  - Linux:   WebKitGTK

The user sees a normal app window — no browser chrome (no address bar,
no tabs, no bookmarks). It looks and feels like a native desktop app,
but inside it's rendering our React UI.

Alternative approaches we DIDN'T use:
  - Electron: Bundles an entire Chromium browser (~150MB). Overkill.
  - Tauri: Requires Rust toolchain. Complex build process.
  - PyWebView: Just pip install. Uses existing OS browser engine. ~5MB.

HOW TO RUN:
-----------
    python -m stream.desktop.main

Or after PyInstaller packaging (Phase 8):
    Double-click STREAM.app (macOS) / STREAM.exe (Windows)
"""

import logging
import socket
import sys
import threading
import time

import httpx
import uvicorn

# Try importing pywebview at the top level.
# pywebview is an optional dependency — if it's not installed, we fall back
# to opening the UI in the user's default browser instead of a native window.
# We use a flag (_HAS_WEBVIEW) so we only need to check once.
try:
    import webview

    _HAS_WEBVIEW = True
except ImportError:
    _HAS_WEBVIEW = False

# -------------------------------------------------------------------------
# STEP 1: Apply desktop defaults FIRST, before any middleware imports.
# -------------------------------------------------------------------------
# This MUST happen before we import anything from stream.middleware,
# because stream/middleware/config.py reads os.environ at import time.
# If we import config.py first, it would read the wrong values
# (Docker defaults instead of desktop defaults).
from stream.desktop.config import apply_desktop_defaults

apply_desktop_defaults()

# NOW it's safe to import middleware modules — config.py will see
# STREAM_MODE="desktop" and all the other desktop defaults we just set.
from stream.desktop.first_run import is_first_run, run_first_run_setup
from stream.desktop.ollama_lifecycle import start_ollama, stop_ollama
from stream.middleware.app import app as fastapi_app  # The FastAPI application object
from stream.middleware.config import MIDDLEWARE_HOST, MIDDLEWARE_PORT

logger = logging.getLogger(__name__)


def is_port_in_use(host: str, port: int) -> bool:
    """
    Check if a port is already occupied by another process.

    HOW IT WORKS:
    -------------
    We try to create a TCP socket and bind it to the port. If the bind
    succeeds, the port is free (no one else is using it). If it raises
    OSError, something else (like Docker or a stale server) is already
    listening on that port.

    WHY THIS MATTERS:
    -----------------
    Without this check, if Docker's middleware container is already running
    on port 5000, our desktop server would silently fail to start. The health
    check would find Docker's old server (which returns JSON at /), and
    PyWebView would display JSON instead of the React UI. This function
    catches that problem BEFORE we waste time starting threads.

    Args:
        host: Address to check (e.g., "127.0.0.1")
        port: Port number to check (e.g., 5000)

    Returns:
        True if the port is already in use, False if it's available
    """
    # socket.socket() creates a network connection endpoint.
    # AF_INET = IPv4 address family, SOCK_STREAM = TCP (reliable, ordered).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            # bind() tries to claim the port for this socket.
            # If another process already owns it, the OS raises OSError.
            sock.bind((host, port))
            return False  # Bind succeeded → port is free
        except OSError:
            return True  # Bind failed → port is already taken


def start_fastapi_server():
    """
    Start the FastAPI/uvicorn server in the current thread.

    WHY THIS IS A SEPARATE FUNCTION:
    ---------------------------------
    We need FastAPI to run in a BACKGROUND thread (see main() below).
    threading.Thread(target=...) takes a function to run in the new thread.
    So we wrap uvicorn.run() in this function and pass it as the target.

    WHY WE PASS THE APP OBJECT (not a string path):
    ------------------------------------------------
    uvicorn accepts either a string ("module:app") or an app object directly.
    We pass the object because PyInstaller bundles modules into a frozen archive
    where uvicorn's string-based import resolution doesn't work. Passing the
    object directly bypasses uvicorn's import mechanism entirely.
    """
    uvicorn.run(
        # Pass the FastAPI app object directly (imported at module level above).
        # String paths like "stream.middleware.app:app" break in PyInstaller
        # bundles because the frozen module loader works differently.
        fastapi_app,
        host=MIDDLEWARE_HOST,
        port=MIDDLEWARE_PORT,
        # "warning" = only show errors, not every request.
        # In desktop mode, logs would clutter the terminal for no reason.
        # The React UI shows all the information the user needs.
        log_level="warning",
    )


def wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    """
    Poll the health endpoint until the server is ready.

    We can't open the PyWebView window immediately after starting the server
    thread — uvicorn needs a moment to bind to the port and start accepting
    connections. So we poll /health every 0.5 seconds until it responds.

    This is the same pattern used by Docker's healthcheck and by our own
    warm_ping.py — repeatedly try until success or timeout.

    Args:
        host: Server address (always "127.0.0.1" in desktop mode)
        port: Server port (default 5000)
        timeout: Maximum seconds to wait before giving up

    Returns:
        True if server is ready, False if it didn't start in time
    """
    url = f"http://{host}:{port}/health"
    waited = 0.0

    while waited < timeout:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            # Server not ready yet — keep waiting
            pass

        time.sleep(0.5)
        waited += 0.5

    return False


def main():
    """
    Desktop app startup sequence.

    This function orchestrates the entire launch process, step by step.
    Each step depends on the previous one succeeding.
    """

    print("STREAM Desktop starting...")

    # -------------------------------------------------------------------------
    # STEP 2: First-run setup (only on the very first launch)
    # -------------------------------------------------------------------------
    # On the first launch, ~/.stream/ doesn't exist yet. We create it and
    # check that Ollama and AI models are available. This gives the user
    # a friendly welcome message and clear instructions if anything is missing.
    # On subsequent launches, is_first_run() returns False and we skip this.
    if is_first_run():
        run_first_run_setup()

    # -------------------------------------------------------------------------
    # STEP 3 is handled above (first-run setup).

    # -------------------------------------------------------------------------
    # STEP 4: Start Ollama (local AI models)
    # -------------------------------------------------------------------------
    # Ollama must be running before FastAPI starts, because the startup
    # health checks (lifecycle.py) will try to connect to Ollama.
    # If Ollama isn't running, the LOCAL tier just shows as "unavailable"
    # in the UI — the app still works with cloud and Lakeshore tiers.
    ollama_ok = start_ollama()
    if not ollama_ok:
        print("Warning: Ollama is not available — local models will be disabled")
        print("The app will still work with cloud models (requires API keys)")

    # -------------------------------------------------------------------------
    # STEP 5: Check for port conflicts BEFORE starting the server
    # -------------------------------------------------------------------------
    # If another process (like Docker's middleware container or a stale Python
    # process from a previous run) is already listening on our port, uvicorn
    # would silently fail. We'd then connect to the OLD server and see wrong
    # content (e.g., JSON instead of the React UI). Catching this early gives
    # the user a clear error message instead of confusing behavior.
    if is_port_in_use(MIDDLEWARE_HOST, MIDDLEWARE_PORT):
        print(f"ERROR: Port {MIDDLEWARE_PORT} is already in use on {MIDDLEWARE_HOST}")
        print()
        print("Common causes:")
        print("  1. Docker's stream-middleware container is running")
        print("     Fix: docker compose down")
        print("  2. A previous STREAM desktop session didn't shut down cleanly")
        print(f"     Fix: lsof -i :{MIDDLEWARE_PORT}  (find the PID, then kill it)")
        print()
        print("Stop the other process and try again.")
        stop_ollama()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 6: Start FastAPI in a background thread
    # -------------------------------------------------------------------------
    # WHY a background thread?
    # PyWebView's webview.start() (step 8) BLOCKS the main thread — it runs
    # an event loop that keeps the native window alive. If we ran FastAPI on
    # the main thread, we'd never get to open the window.
    #
    # daemon=True means: "Kill this thread automatically when the main thread
    # exits." So when the user closes the PyWebView window (main thread ends),
    # the FastAPI thread dies too. No need for explicit cleanup.
    server_thread = threading.Thread(
        target=start_fastapi_server,
        daemon=True,  # Auto-dies when main thread exits
    )
    server_thread.start()
    print("FastAPI server starting in background...")

    # -------------------------------------------------------------------------
    # STEP 7: Wait for the server to be ready
    # -------------------------------------------------------------------------
    # The server thread needs time to start up (import modules, bind to port,
    # run lifecycle.startup(), etc.). We poll the /health endpoint until it
    # responds, or give up after 30 seconds.
    print("Waiting for server to be ready...")
    server_ready = wait_for_server(MIDDLEWARE_HOST, MIDDLEWARE_PORT)

    if not server_ready:
        print("ERROR: Server failed to start within 30 seconds")
        print("Check the logs for errors")
        stop_ollama()
        sys.exit(1)

    server_url = f"http://{MIDDLEWARE_HOST}:{MIDDLEWARE_PORT}"
    print(f"Server is ready at {server_url}")

    # -------------------------------------------------------------------------
    # STEP 8: Open the native desktop window
    # -------------------------------------------------------------------------
    # webview.create_window() defines the window (title, URL, size).
    # webview.start() actually OPENS it and BLOCKS until the user closes it.
    #
    # Think of it like:
    #   create_window() = "prepare a window with these settings"
    #   start()         = "show the window and wait for the user to close it"
    #
    # Everything after start() only runs AFTER the window is closed.
    if _HAS_WEBVIEW:
        # PyWebView is installed — open a native OS window (no browser chrome).
        # We add a timestamp query parameter (?_t=...) to the URL to prevent
        # WebKit from serving a stale cached version of the page. This is called
        # "cache busting" — the unique timestamp makes WebKit think it's a new URL,
        # so it fetches fresh content instead of using its cache.
        cache_bust_url = f"{server_url}?_t={int(time.time())}"
        webview.create_window(
            title="STREAM",  # Window title bar text
            url=cache_bust_url,  # Load our React UI (cache-busted)
            width=1200,  # Default window width in pixels
            height=800,  # Default window height in pixels
            min_size=(800, 600),  # Minimum resize dimensions
        )

        # start() BLOCKS HERE until the user closes the window.
        # The app is "running" during this entire time.
        webview.start()

    else:
        # PyWebView not installed — fall back to opening in the system browser.
        # This is useful during development when you haven't installed pywebview.
        print("PyWebView not installed — opening in your default browser instead")
        print(f"Open {server_url} in your browser")
        print("Press Ctrl+C to stop the server")

        try:
            # Keep the main thread alive so the daemon server thread keeps running.
            # Without this, the program would exit immediately (daemon threads die
            # when the main thread ends), and the server would stop.
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    # -------------------------------------------------------------------------
    # STEP 9: Cleanup (runs after the window is closed)
    # -------------------------------------------------------------------------
    # If we started Ollama, stop it now. If the user had Ollama running before
    # our app launched, stop_ollama() is a no-op (it checks _ollama_process).
    print("STREAM Desktop shutting down...")
    stop_ollama()
    print("Goodbye!")


# -------------------------------------------------------------------------
# Standard Python entry point guard.
# -------------------------------------------------------------------------
# This block runs ONLY when the file is executed directly:
#   python -m stream.desktop.main     → runs main()
#   from stream.desktop.main import X → does NOT run main()
if __name__ == "__main__":
    main()
