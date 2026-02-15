"""
Ollama process lifecycle management for desktop mode.

WHY THIS FILE EXISTS:
---------------------
In Docker, Ollama runs as its own container — Docker starts it automatically.
In desktop mode, there's no Docker. The user might have Ollama running already
(e.g., they started it from the terminal), or it might not be running at all.

This module handles three scenarios:
  1. Ollama is already running     → do nothing, just use it
  2. Ollama is installed but not running → start it for the user
  3. Ollama is not installed       → warn the user (local tier won't work)

WHAT IS A SUBPROCESS:
---------------------
A subprocess is a separate program that our app launches. When we run
`ollama serve`, it starts Ollama as a background process. We keep a reference
to it (the Popen object) so we can stop it when the user closes the app.

Think of it like opening another app from within our app:
  Our App (main process) → launches → Ollama (subprocess)
  Our App closes         → we stop  → Ollama (subprocess)
"""

import logging
import shutil
import subprocess
import time

import httpx

logger = logging.getLogger(__name__)

# Module-level reference to the Ollama process we started (if any).
# If Ollama was already running before we launched, this stays None
# and we don't touch it on shutdown (it's not ours to stop).
_ollama_process: subprocess.Popen | None = None


def is_ollama_installed() -> bool:
    """
    Check if the `ollama` command-line tool is installed on this machine.

    shutil.which() searches the system PATH for an executable.
    It's the Python equivalent of running `which ollama` in the terminal.
    Returns the full path if found (truthy), or None if not (falsy).
    """
    return shutil.which("ollama") is not None


def is_ollama_running(host: str = "localhost", port: int = 11434) -> bool:
    """
    Check if Ollama is currently running by pinging its HTTP API.

    Ollama exposes a REST API at http://localhost:11434. If we can reach it,
    Ollama is running. If the connection is refused, it's not running.

    Args:
        host: Where Ollama should be listening (always "localhost" on desktop)
        port: Ollama's default port (11434)

    Returns:
        True if Ollama responded, False if connection failed
    """
    try:
        # Send a quick GET request to Ollama's root endpoint.
        # timeout=2 means give up after 2 seconds (don't hang forever).
        response = httpx.get(f"http://{host}:{port}", timeout=2.0)
        # Any response (even an error page) means Ollama is running
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        # ConnectError  = nobody listening on that port
        # TimeoutException = request took too long (unlikely for localhost)
        return False


def start_ollama() -> bool:
    """
    Start the Ollama server as a background subprocess.

    WHY subprocess.Popen INSTEAD OF subprocess.run:
    ------------------------------------------------
    subprocess.run() waits for the command to finish before continuing.
    But "ollama serve" runs FOREVER (it's a server) — so run() would block
    our app from starting. Popen() launches the process and returns immediately,
    letting our app continue while Ollama runs in the background.

    The `daemon`-like behavior:
    - stdout/stderr are piped to /dev/null (DEVNULL) so Ollama's logs
      don't clutter our app's output.
    - We store the process in _ollama_process so stop_ollama() can
      terminate it when the app closes.

    Returns:
        True if Ollama started successfully, False if it failed
    """
    global _ollama_process

    if not is_ollama_installed():
        logger.warning("Ollama is not installed — local AI models won't be available")
        logger.warning("Install from: https://ollama.com/download")
        return False

    if is_ollama_running():
        # Ollama is already running (user started it themselves).
        # We don't need to start it, and we WON'T stop it on shutdown
        # (it's not ours to manage).
        logger.info("Ollama is already running — using existing instance")
        return True

    try:
        logger.info("Starting Ollama server...")

        # Launch "ollama serve" as a background process.
        # - DEVNULL suppresses Ollama's own log output
        # - We DON'T use shell=True (security best practice — avoids shell injection)
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for Ollama to be ready (it needs a moment to start up).
        # We poll the API every 0.5 seconds, up to 10 seconds total.
        # This is similar to what lifecycle.py does with warm_up_all_tiers().
        max_wait_seconds = 10
        waited = 0.0
        while waited < max_wait_seconds:
            if is_ollama_running():
                logger.info("Ollama server is ready")
                return True
            time.sleep(0.5)
            waited += 0.5

        # If we get here, Ollama started but never responded to our pings
        logger.warning("Ollama started but did not respond within 10 seconds")
        return False

    except FileNotFoundError:
        # This shouldn't happen (we checked is_ollama_installed above),
        # but handle it gracefully just in case
        logger.error("Failed to start Ollama — 'ollama' command not found")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False


def stop_ollama() -> None:
    """
    Stop the Ollama subprocess IF we started it.

    Called once when the app is closing (user closed the PyWebView window).

    IMPORTANT: We only stop Ollama if WE started it (_ollama_process is not None).
    If the user had Ollama running before our app launched, we leave it alone —
    stopping someone else's process would be rude and unexpected.

    process.terminate() sends SIGTERM (a polite "please stop" signal).
    process.wait(timeout=5) gives it 5 seconds to shut down gracefully.
    If it doesn't stop in time, process.kill() sends SIGKILL (force stop).
    """
    global _ollama_process

    if _ollama_process is None:
        # We didn't start Ollama — nothing to stop
        return

    try:
        logger.info("Stopping Ollama server...")

        # SIGTERM = "please shut down gracefully" (Ollama can save state)
        _ollama_process.terminate()

        try:
            # Give Ollama up to 5 seconds to shut down cleanly
            _ollama_process.wait(timeout=5)
            logger.info("Ollama server stopped gracefully")
        except subprocess.TimeoutExpired:
            # Ollama didn't stop in time — force kill it
            # SIGKILL = "stop immediately, no cleanup" (last resort)
            _ollama_process.kill()
            logger.warning("Ollama server force-killed (did not stop in time)")

    except Exception as e:
        logger.error(f"Error stopping Ollama: {e}")
    finally:
        # Clear the reference regardless of what happened
        _ollama_process = None
