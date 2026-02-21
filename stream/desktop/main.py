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

import atexit  # Registers functions to run when the process exits (our safety-net cleanup)
import logging
import os
import signal
import socket
import subprocess
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
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

from stream.desktop.first_run import is_first_run, run_first_run_setup
from stream.desktop.ollama_lifecycle import start_ollama, stop_ollama
from stream.middleware.app import app as fastapi_app  # The FastAPI application object
from stream.middleware.config import MIDDLEWARE_HOST, MIDDLEWARE_PORT, OLLAMA_MODELS
from stream.middleware.core.ollama_manager import OllamaModelManager

logger = logging.getLogger(__name__)
console = Console()


def check_and_download_ollama_models():
    """
    Check all Ollama models and offer to download any that are missing.

    This runs on the MAIN THREAD before the FastAPI server starts, so
    interactive prompts and long downloads don't block the server lifecycle.
    Uses Ollama's HTTP API with streaming to show a real progress bar.

    Shows status for every model so the user can see what's checked:
      ✓ llama3.2:3b (installed)
      ✓ gemma3:4b (installed)
    Then prompts to download any missing models.
    """
    manager = OllamaModelManager()

    # Phase 1: Check all models and show status
    console.print("\nChecking local AI models...")
    missing = []
    for alias, ollama_model in OLLAMA_MODELS.items():
        if manager.is_model_available(ollama_model):
            console.print(f"  [green]✓[/green] {ollama_model}")
        else:
            console.print(f"  [red]✗[/red] {ollama_model} [dim](not installed)[/dim]")
            missing.append((alias, ollama_model))

    if not missing:
        console.print("[green]All local models ready[/green]\n")
        return

    # Phase 2: Prompt to download missing models
    console.print(f"\n[yellow]{len(missing)} model(s) not installed.[/yellow]")
    for alias, ollama_model in missing:
        size_estimate = manager.get_model_size_estimate(ollama_model)
        console.print(f"\nModel [bold]{ollama_model}[/bold] ({alias})")
        console.print(f"  Size: {size_estimate}")
        response = (
            console.input("  Download? ([bold green]y[/bold green]/[bold red]n[/bold red]): ")
            .strip()
            .lower()
        )

        if response not in ["y", "yes"]:
            console.print(f"  Skipping {ollama_model}")
            continue

        # Download using Ollama HTTP API with streaming progress
        success, error_msg = _download_model_with_progress(manager.ollama_url, ollama_model)

        # If download failed due to IPv6, auto-retry with IPv4
        if not success and error_msg and _is_ipv6_error(error_msg):
            console.print("[yellow]IPv6 connection failed. Retrying with IPv4...[/yellow]")
            success = _retry_download_with_ipv4(manager.ollama_url, ollama_model)

        # Verify the download actually worked (don't trust the pull response alone)
        if success and not manager.is_model_available(ollama_model):
            console.print(
                f"[red]Warning: {ollama_model} download reported success but model is not available[/red]"
            )
            console.print(f"[yellow]Try manually: ollama pull {ollama_model}[/yellow]")


def _download_model_with_progress(ollama_url: str, model_name: str) -> tuple[bool, str | None]:
    """
    Download an Ollama model with a Rich progress bar.

    Uses the /api/pull endpoint with streaming, which returns NDJSON lines
    containing 'total' and 'completed' byte counts per layer.

    Returns:
        Tuple of (success, error_message). error_message is None on success.
    """
    import json

    try:
        with httpx.stream(
            "POST",
            f"{ollama_url}/api/pull",
            json={"name": model_name},
            timeout=None,  # No timeout — large models can take a while
        ) as resp:
            if resp.status_code != 200:
                msg = f"Failed to start download: HTTP {resp.status_code}"
                console.print(f"[red]{msg}[/red]")
                return False, msg

            got_success = False

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Pulling {model_name}", total=None)
                current_digest = None

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Check for error responses from Ollama
                    error = data.get("error")
                    if error:
                        progress.update(task, description="[red]Error[/red]")
                        console.print(f"[red]Download failed: {error}[/red]")
                        return False, error

                    status = data.get("status", "")
                    total = data.get("total")
                    completed = data.get("completed", 0)
                    digest = data.get("digest")

                    # When a new layer starts, reset the progress bar
                    if digest and digest != current_digest:
                        current_digest = digest
                        short_digest = digest.split(":")[-1][:12]
                        progress.update(
                            task, description=f"{status} {short_digest}", completed=0, total=total
                        )
                    elif total:
                        progress.update(task, completed=completed, total=total)
                    else:
                        # Status-only lines (e.g., "verifying sha256 digest")
                        progress.update(task, description=status)

                    if status == "success":
                        got_success = True
                        progress.update(task, description=f"[green]Downloaded {model_name}[/green]")

        if got_success:
            console.print(f"[green]✓ {model_name} ready[/green]")
            return True, None
        else:
            msg = f"Download did not complete for {model_name}"
            console.print(f"[red]{msg}[/red]")
            return False, msg

    except httpx.ConnectError as e:
        msg = f"Cannot connect to Ollama at {ollama_url} — is it running?"
        console.print(f"[red]{msg}[/red]")
        return False, str(e)
    except KeyboardInterrupt:
        console.print("\n[yellow]Download cancelled[/yellow]")
        return False, "cancelled"
    except Exception as e:
        msg = str(e)
        console.print(f"[red]Download failed: {msg}[/red]")
        return False, msg


def _is_ipv6_error(error_msg: str) -> bool:
    """Detect IPv6-related network errors in Ollama pull responses."""
    # IPv6 errors contain "socket is not connected" with IPv6 addresses (contain "::")
    # or explicit "network is unreachable" on IPv6 addresses
    return "socket is not connected" in error_msg or (
        "network is unreachable" in error_msg and "::" in error_msg
    )


def _get_active_network_service() -> str | None:
    """
    Get the name of the active macOS network service (e.g., "Wi-Fi", "Ethernet").

    Uses `route get default` to find the default interface (e.g., en0),
    then maps it to a network service name via `networksetup`.
    """
    import platform
    import re

    if platform.system() != "Darwin":
        return None

    try:
        # Get default network interface (e.g., "en0")
        result = subprocess.run(
            ["route", "get", "default"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        match = re.search(r"interface:\s*(\S+)", result.stdout)
        if not match:
            return None
        interface = match.group(1)  # e.g., "en0"

        # Map interface to network service name (e.g., "en0" → "Wi-Fi")
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse output: "Hardware Port: Wi-Fi\nDevice: en0\n..."
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if f"Device: {interface}" in line and i > 0:
                port_match = re.search(r"Hardware Port:\s*(.+)", lines[i - 1])
                if port_match:
                    return port_match.group(1)

    except Exception:
        pass

    return None


def _retry_download_with_ipv4(ollama_url: str, model_name: str) -> bool:
    """
    Retry a failed download after temporarily disabling IPv6 on macOS.

    On some networks, IPv6 routing is broken but IPv4 works fine. Ollama
    (a Go binary) tries IPv6 first per the Happy Eyeballs algorithm, fails,
    and doesn't fall back to IPv4 gracefully. Temporarily disabling IPv6
    on the active network interface forces all connections through IPv4.

    Two approaches depending on terminal availability:

    TERMINAL MODE (dev mode, CLI):
        Uses `sudo networksetup` directly in the terminal. The user sees
        a normal password prompt inline — no pop-up dialogs, no temp files.
        Simpler and more reliable than the GUI approach.

    GUI MODE (bundled .app, no terminal):
        Uses `osascript` to show a macOS admin password dialog. More complex
        (uses temp files for coordination) but works without a terminal.

    Returns:
        True if the download succeeded after the IPv4 fix, False otherwise.
    """
    import platform

    if platform.system() != "Darwin":
        console.print("[yellow]Automatic IPv6 fix is only available on macOS[/yellow]")
        console.print(f"[yellow]Try: ollama pull {model_name}[/yellow]")
        return False

    service = _get_active_network_service()
    if not service:
        console.print("[yellow]Could not detect active network interface[/yellow]")
        return False

    if sys.stdin.isatty():
        return _retry_ipv4_terminal(ollama_url, model_name, service)
    else:
        return _retry_ipv4_osascript(ollama_url, model_name, service)


def _retry_ipv4_terminal(ollama_url: str, model_name: str, service: str) -> bool:
    """
    Terminal-based IPv4 retry using sudo (for dev mode / CLI usage).

    Since we have an interactive terminal, we can use `sudo` directly.
    The password prompt appears inline in the terminal — clean and simple.
    macOS caches sudo credentials for 5 minutes, so re-enabling IPv6
    at the end won't require a second password entry.
    """
    console.print(f"\n  Need to temporarily disable IPv6 on [bold]{service}[/bold]")
    console.print("  (your terminal will ask for your admin password)")
    console.print()

    try:
        result = subprocess.run(
            ["sudo", "networksetup", "-setv6off", service],
            timeout=60,
        )
        if result.returncode != 0:
            console.print("[yellow]Could not disable IPv6[/yellow]")
            return False

        time.sleep(1)

        console.print("  Retrying download over IPv4...")
        success, _ = _download_model_with_progress(ollama_url, model_name)
        return success

    except subprocess.TimeoutExpired:
        console.print("[yellow]Timed out waiting for password[/yellow]")
        return False
    except Exception as e:
        console.print(f"[yellow]IPv4 retry failed: {e}[/yellow]")
        return False
    finally:
        try:
            subprocess.run(
                ["sudo", "networksetup", "-setv6automatic", service],
                timeout=10,
                capture_output=True,
            )
            console.print(f"  IPv6 re-enabled on {service}")
        except Exception:
            console.print(f"[yellow]Warning: Could not re-enable IPv6 on {service}[/yellow]")
            console.print(
                f"[yellow]Run manually: sudo networksetup -setv6automatic '{service}'[/yellow]"
            )


def _retry_ipv4_osascript(ollama_url: str, model_name: str, service: str) -> bool:
    """
    GUI-based IPv4 retry using osascript (for bundled .app without terminal).

    Uses a single osascript call with administrator privileges to disable
    IPv6, wait for the download, then re-enable IPv6. The user only enters
    their admin password once via the macOS system dialog.

    Coordination between Python and the shell script uses flag files:
      1. Shell disables IPv6 → writes "disabled" flag
      2. Python sees flag → does the download
      3. Python writes "done" flag → shell re-enables IPv6
    """
    import tempfile

    console.print(f"  Temporarily disabling IPv6 on [bold]{service}[/bold]...")
    console.print("  (macOS will ask for your password once)")

    tmp_dir = tempfile.mkdtemp(prefix="stream_ipv6_")
    disabled_flag = os.path.join(tmp_dir, "disabled")
    done_flag = os.path.join(tmp_dir, "done")

    shell_script = (
        f"networksetup -setv6off '{service}' && "
        f"touch '{disabled_flag}' && "
        f"for i in $(seq 1 600); do "
        f"  if [ -f '{done_flag}' ]; then break; fi; "
        f"  sleep 1; "
        f"done; "
        f"networksetup -setv6automatic '{service}'"
    )
    apple_script = f'do shell script "{shell_script}" with administrator privileges'

    try:
        proc = subprocess.Popen(
            ["osascript", "-e", apple_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        waited = 0.0
        while waited < 60:
            if os.path.exists(disabled_flag):
                break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
            waited += 0.5

        if not os.path.exists(disabled_flag):
            stderr = proc.stderr.read().decode() if proc.poll() is not None else ""
            console.print(
                f"[yellow]Could not disable IPv6"
                f"{': ' + stderr.strip() if stderr.strip() else ''}[/yellow]"
            )
            return False

        time.sleep(1)

        console.print("  Retrying download over IPv4...")
        success, _ = _download_model_with_progress(ollama_url, model_name)
        return success

    except Exception as e:
        console.print(f"[yellow]IPv4 retry failed: {e}[/yellow]")
        return False
    finally:
        try:
            with open(done_flag, "w") as f:
                f.write("done")

            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

            if proc.returncode == 0:
                console.print(f"  IPv6 re-enabled on {service}")
            else:
                stderr = proc.stderr.read().decode().strip() if proc.stderr else ""
                console.print(
                    f"[yellow]Warning: IPv6 re-enable may have failed on {service}[/yellow]"
                )
                if stderr:
                    console.print(f"[yellow]  {stderr}[/yellow]")
                console.print(
                    f"[yellow]Run manually if needed: "
                    f"sudo networksetup -setv6automatic '{service}'[/yellow]"
                )
        except Exception:
            console.print(f"[yellow]Warning: Could not re-enable IPv6 on {service}[/yellow]")
            console.print(
                f"[yellow]Run manually: sudo networksetup -setv6automatic '{service}'[/yellow]"
            )

        try:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def is_port_in_use(host: str, port: int) -> bool:
    """
    Check if something is ACTIVELY LISTENING on a port.

    HOW IT WORKS:
    -------------
    We try to CONNECT to the port (like a client would). If the connection
    succeeds, a server is actively listening — the port is truly in use.
    If the connection is refused or times out, nothing is listening.

    WHY connect() INSTEAD OF bind():
    ---------------------------------
    The old approach used bind() — which fails if the port is in ANY state,
    including TIME_WAIT. TIME_WAIT is a TCP state where the OS keeps the port
    reserved for ~60 seconds after a socket closes, even though nothing is
    listening. This caused false positives: the app would think the port was
    occupied when it was actually free.

    connect() only succeeds if a server is ACTIVELY ACCEPTING connections.
    A port in TIME_WAIT returns ConnectionRefused — correctly identified as
    "not in use." And uvicorn sets SO_REUSEADDR by default, so it can bind
    through TIME_WAIT without any issues.

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
        True if something is actively listening, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # Don't hang forever if something is weird
        try:
            sock.connect((host, port))
            return True  # Connection accepted → a server is listening
        except (TimeoutError, ConnectionRefusedError, OSError):
            return False  # Nothing listening (TIME_WAIT is fine)


def _free_stale_port(host: str, port: int) -> bool:
    """
    Auto-recover from a stale process blocking our port.

    WHY THIS EXISTS:
    ----------------
    When the desktop app crashes or is force-quit, the old FastAPI/uvicorn
    process can linger as an orphan. On next launch, port 5000 is still
    occupied and the app refuses to start. Making the user manually run
    `lsof` and `kill` is a terrible UX for a desktop app — it should just
    clean up after itself.

    SAFETY:
    -------
    We only kill Python processes (likely our own stale server). If something
    unexpected is on the port (e.g., Docker, a different app), we leave it
    alone and let the caller show the manual error message.

    Returns:
        True if the port was successfully freed, False otherwise.
    """
    try:
        # lsof -ti :5000 → returns just PID(s) listening on the port.
        # -t = terse (PIDs only, no headers), -i = filter by internet address.
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        pids = result.stdout.strip()
        if not pids:
            # Port is in TIME_WAIT state (OS hasn't fully released it yet).
            # Brief wait usually resolves this.
            time.sleep(2)
            return not is_port_in_use(host, port)

        killed_any = False
        for pid_str in pids.splitlines():
            pid = int(pid_str.strip())

            if pid == os.getpid():
                continue

            # Only kill processes that look like our own stale STREAM server.
            # If something else is on this port (e.g., another app), leave it alone.
            try:
                ps_result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "command="],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                cmd = ps_result.stdout.strip().lower()
                is_ours = "stream" in cmd or "uvicorn" in cmd or "python" in cmd
            except Exception:
                is_ours = False

            if not is_ours:
                print(f"  PID {pid} is not a STREAM process — skipping")
                continue

            os.kill(pid, signal.SIGTERM)
            print(f"  Sent SIGTERM to stale process (PID {pid})")
            killed_any = True

        if not killed_any:
            return False

        # Give the process time to shut down gracefully.
        time.sleep(2.0)

        if not is_port_in_use(host, port):
            return True

        # SIGTERM wasn't enough (e.g., stuck process). Escalate to SIGKILL.
        for pid_str in pids.splitlines():
            pid = int(pid_str.strip())
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"  Force-killed stale process (PID {pid})")
            except ProcessLookupError:
                pass  # Already dead from SIGTERM

        time.sleep(1.0)
        return not is_port_in_use(host, port)

    except (subprocess.TimeoutExpired, ValueError, ProcessLookupError, PermissionError) as e:
        # TimeoutExpired: lsof hung (unlikely)
        # ValueError: PID wasn't a number (malformed lsof output)
        # ProcessLookupError: process already died between lsof and kill
        # PermissionError: process belongs to another user (not ours)
        logger.warning(f"Could not free port {port}: {e}")
        return False


# -------------------------------------------------------------------------
# Uvicorn server + thread — module-level so cleanup can reach them.
# -------------------------------------------------------------------------
# We use uvicorn.Server instead of uvicorn.run() because run() gives us
# no way to trigger a graceful shutdown. With Server, we can set
# server.should_exit = True, which tells uvicorn to:
#   1. Stop accepting new connections
#   2. Finish processing in-flight requests
#   3. Close the listening socket (release the port!)
#   4. Exit the server loop
#
# We also keep a reference to the thread so we can JOIN it during cleanup.
# Joining means "wait for this thread to actually finish" — unlike sleep(),
# join() blocks until the thread's function returns, guaranteeing the socket
# is fully released before we move on.
_server: uvicorn.Server | None = None  # The uvicorn server instance
_server_thread: threading.Thread | None = None  # The thread running the server
_cleanup_done = False  # Idempotent guard — prevents running cleanup twice


def _cleanup():
    """
    Shut down the server and Ollama. Safe to call multiple times.

    This function is called from two places:
      1. Normally: from main() after the user closes the window (Step 9)
      2. Safety net: from atexit, in case the process exits unexpectedly

    The _cleanup_done flag ensures we don't run cleanup twice.
    """
    global _cleanup_done, _server

    # If we've already cleaned up (e.g., main() called us, then atexit
    # fires again), skip — everything is already shut down.
    if _cleanup_done:
        return
    _cleanup_done = True

    # Step A: Tell uvicorn to shut down gracefully.
    # This sets an internal flag that uvicorn checks on each event loop
    # iteration. When it sees should_exit=True, it stops accepting new
    # connections and begins its shutdown sequence.
    if _server is not None:
        _server.should_exit = True

    # Step B: Wait for the server thread to actually finish.
    # join() blocks until the thread's target function (start_fastapi_server)
    # returns — which only happens AFTER uvicorn has fully closed its socket
    # and released the port. This is the key difference from our old approach
    # of time.sleep(0.5), which was just a guess and often wasn't enough.
    # The 3-second timeout is a safety bound so we don't hang forever if
    # something goes wrong inside uvicorn's shutdown.
    if _server_thread is not None and _server_thread.is_alive():
        _server_thread.join(timeout=3.0)

    # Clear the reference so garbage collection can clean up.
    _server = None

    # Step C: Stop Ollama (the local AI model runner).
    # We do this AFTER the server is fully stopped to avoid race conditions
    # where a late-arriving request tries to talk to Ollama while it's dying.
    stop_ollama()


# Register cleanup to run on process exit. This catches cases where the
# process exits without going through our normal Step 9 cleanup path:
#   - SIGTERM from macOS "Force Quit"
#   - Unhandled exception in main()
#   - sys.exit() from somewhere unexpected
atexit.register(_cleanup)


def start_fastapi_server():
    """
    Start the FastAPI/uvicorn server in the current thread.

    WHY THIS IS A SEPARATE FUNCTION:
    ---------------------------------
    We need FastAPI to run in a BACKGROUND thread (see main() below).
    threading.Thread(target=...) takes a function to run in the new thread.
    So we wrap the server startup in this function and pass it as the target.

    WHY WE PASS THE APP OBJECT (not a string path):
    ------------------------------------------------
    uvicorn accepts either a string ("module:app") or an app object directly.
    We pass the object because PyInstaller bundles modules into a frozen archive
    where uvicorn's string-based import resolution doesn't work. Passing the
    object directly bypasses uvicorn's import mechanism entirely.
    """
    global _server

    config = uvicorn.Config(
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
    _server = uvicorn.Server(config)
    _server.run()


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

    Flags:
        --dev   Start the backend without opening a native window.
                Use this during development alongside `npm run dev:vite`
                for instant hot-reload in the browser at localhost:3000.
    """

    dev_mode = "--dev" in sys.argv

    if dev_mode:
        print("STREAM Desktop starting (dev mode — no window)...")
    else:
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
    # STEP 4b: Check/download required Ollama models
    # -------------------------------------------------------------------------
    # This runs BEFORE the server starts so interactive download prompts
    # don't block the FastAPI startup lifecycle. The download uses Ollama's
    # HTTP streaming API to show a real progress bar.
    if ollama_ok:
        check_and_download_ollama_models()

    # -------------------------------------------------------------------------
    # STEP 5: Check for port conflicts and auto-recover if possible
    # -------------------------------------------------------------------------
    # If a stale process from a previous session is hogging our port, kill it
    # automatically. Desktop apps should clean up after themselves — making the
    # user manually run lsof/kill is bad UX. If auto-recovery fails (e.g.,
    # Docker is running), show the manual instructions as a fallback.
    if is_port_in_use(MIDDLEWARE_HOST, MIDDLEWARE_PORT):
        print(f"Port {MIDDLEWARE_PORT} is in use — cleaning up stale process...")
        if _free_stale_port(MIDDLEWARE_HOST, MIDDLEWARE_PORT):
            print(f"Port {MIDDLEWARE_PORT} freed successfully")
        else:
            print(f"ERROR: Could not free port {MIDDLEWARE_PORT}")
            print()
            print("Common causes:")
            print("  1. Docker's stream-middleware container is running")
            print("     Fix: docker compose down")
            print("  2. A process you don't want killed is using this port")
            print(f"     Fix: lsof -i :{MIDDLEWARE_PORT}  (find it, then kill manually)")
            print()
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
    # exits." This is a safety net — in normal operation, _cleanup() shuts
    # down the server gracefully and joins this thread (waits for it to
    # finish), so the port is properly released. The daemon flag ensures the
    # thread still dies if _cleanup() somehow fails.
    global _server_thread
    _server_thread = threading.Thread(
        target=start_fastapi_server,
        daemon=True,  # Safety net: auto-dies if graceful shutdown fails
    )
    _server_thread.start()
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
    if dev_mode:
        # Dev mode — no window. The developer uses the Vite dev server
        # (localhost:3000) in their browser for hot-reload.
        print()
        print(f"Backend ready at {server_url}")
        print("Open http://localhost:3000 in your browser (Vite dev server)")
        print("Press Ctrl+C to stop the server")

        try:
            # Keep the main thread alive so the daemon server thread keeps running.
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    elif _HAS_WEBVIEW:
        # PyWebView is installed — open a native OS window (no browser chrome).
        # We add a timestamp query parameter (?_t=...) to the URL to prevent
        # WebKit from serving a stale cached version of the page. This is called
        # "cache busting" — the unique timestamp makes WebKit think it's a new URL,
        # so it fetches fresh content instead of using its cache.
        cache_bust_url = f"{server_url}?_t={int(time.time())}"

        # -----------------------------------------------------------------
        # CAMERA SUPPORT: Enable getUserMedia in Qt/WebEngine renderer
        # -----------------------------------------------------------------
        # STREAM's CameraModal (in ImageUpload.tsx) uses the WebRTC API
        # navigator.mediaDevices.getUserMedia() to open a live webcam preview.
        # This works out-of-the-box in regular browsers, but PyWebView's
        # native window renderers need extra configuration:
        #
        # Qt/WebEngine (used on some Linux distros, optional on macOS/Windows):
        #   By default, Qt WebEngine shows a permission dialog that blocks
        #   getUserMedia. Setting QTWEBENGINE_CHROMIUM_FLAGS with
        #   --use-fake-ui-for-media-stream auto-grants camera permission
        #   without showing the dialog. This is safe because:
        #     1. STREAM only accesses the camera when the user explicitly
        #        clicks the Camera button
        #     2. The camera stream is stopped as soon as the modal closes
        #     3. The app runs locally — no remote page can access the camera
        #
        # macOS (WKWebView):
        #   Requires NSCameraUsageDescription in Info.plist (see stream.spec).
        #   The OS shows a system-level permission dialog on first use.
        #   This env var has no effect on WKWebView.
        #
        # Windows (WebView2):
        #   WebView2 (Edge-based) natively supports getUserMedia and shows
        #   a permission prompt automatically. This env var has no effect.
        #
        # If the camera still doesn't work in a particular renderer, the
        # CameraModal's error handling catches it gracefully and suggests
        # the user try the Upload button or open STREAM in their browser.
        os.environ.setdefault(
            "QTWEBENGINE_CHROMIUM_FLAGS",
            "--use-fake-ui-for-media-stream",
        )

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
    # _cleanup() does three things:
    #   1. Sets server.should_exit = True (tells uvicorn to shut down)
    #   2. Joins the server thread (WAITS for uvicorn to fully release the port)
    #   3. Stops Ollama
    #
    # This is also registered with atexit as a safety net, but calling it
    # explicitly here gives us control over the sequence and logging.
    # _cleanup() is idempotent — safe to call from both here and atexit.
    print("STREAM Desktop shutting down...")
    _cleanup()
    print("Goodbye!")

    # -------------------------------------------------------------------------
    # STEP 10: Force-exit to avoid hanging on lingering library threads
    # -------------------------------------------------------------------------
    # After graceful cleanup is complete, os._exit(0) terminates immediately.
    # Without this, Python's interpreter shutdown calls threading._shutdown(),
    # which tries to join ALL non-daemon threads. The Globus Compute SDK's
    # AMQP connection spawns internal threads that linger after
    # executor.shutdown(wait=False), causing the process to hang until the
    # user presses Ctrl+C two more times.
    #
    # Since we've already shut down uvicorn, Globus, database, and Ollama,
    # there's nothing useful left for those threads to do.
    os._exit(0)


# -------------------------------------------------------------------------
# Standard Python entry point guard.
# -------------------------------------------------------------------------
# This block runs ONLY when the file is executed directly:
#   python -m stream.desktop.main     → runs main()
#   from stream.desktop.main import X → does NOT run main()
if __name__ == "__main__":
    main()
