"""
Test compute node connectivity for WebSocket relay feasibility.

This script submits a diagnostic function to Lakeshore via Globus Compute
to check what the COMPUTE NODE (not login node) can actually do:
  1. What hostname are we on? (confirm it's ga-002, not a login node)
  2. Is the `websockets` library installed?
  3. Can we make outbound HTTPS connections? (e.g., to httpbin.org)
  4. Can we open raw TCP sockets to public servers?
  5. Can we resolve DNS names?
  6. What Python version + installed packages are available?

Usage:
    python scripts/test_compute_node_connectivity.py

Requires:
    - GLOBUS_COMPUTE_ENDPOINT_ID set in .env
    - Valid Globus authentication tokens
"""

import json
import os
import sys
import time

# Add project root to path so we can import from stream.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from globus_compute_sdk import Executor  # noqa: E402
from globus_compute_sdk.serialize import AllCodeStrategies, ComputeSerializer  # noqa: E402

# =============================================================================
# REMOTE TEST FUNCTION (runs on compute node)
# =============================================================================
# Using the same exec() pattern as globus_compute_client.py to ensure clean
# bytecode that works from both normal Python and PyInstaller bundles.

_TEST_FN_SOURCE = """\
def test_compute_node_connectivity(relay_test_url="https://httpbin.org/get"):
    results = {}

    # ----- Basic info -----
    import socket
    import sys
    import platform

    results["hostname"] = socket.gethostname()
    results["python_version"] = sys.version
    results["platform"] = platform.platform()

    # ----- Test 1: Is websockets installed? -----
    try:
        import websockets
        results["websockets_installed"] = True
        results["websockets_version"] = websockets.__version__
    except ImportError:
        results["websockets_installed"] = False
        results["websockets_version"] = None

    # ----- Test 2: Is aiohttp installed? (alternative to websockets) -----
    try:
        import aiohttp
        results["aiohttp_installed"] = True
        results["aiohttp_version"] = aiohttp.__version__
    except ImportError:
        results["aiohttp_installed"] = False
        results["aiohttp_version"] = None

    # ----- Test 3: DNS resolution -----
    try:
        addr = socket.getaddrinfo("httpbin.org", 443)
        results["dns_resolution"] = True
        results["dns_resolved_to"] = str(addr[0][4])
    except Exception as e:
        results["dns_resolution"] = False
        results["dns_error"] = str(e)

    # ----- Test 4: Outbound TCP socket (port 443) -----
    try:
        s = socket.create_connection(("httpbin.org", 443), timeout=10)
        s.close()
        results["outbound_tcp_443"] = True
    except Exception as e:
        results["outbound_tcp_443"] = False
        results["outbound_tcp_443_error"] = str(e)

    # ----- Test 5: Outbound TCP socket (port 8765 - typical WebSocket) -----
    # We test connecting to httpbin on 443 above; this tests a non-standard port.
    # Most relays would run on 443 (wss://) anyway, but let's check.
    try:
        s = socket.create_connection(("httpbin.org", 80), timeout=10)
        s.close()
        results["outbound_tcp_80"] = True
    except Exception as e:
        results["outbound_tcp_80"] = False
        results["outbound_tcp_80_error"] = str(e)

    # ----- Test 6: Outbound HTTPS request -----
    try:
        import urllib.request
        resp = urllib.request.urlopen(relay_test_url, timeout=10)
        results["outbound_https"] = True
        results["outbound_https_status"] = resp.status
    except Exception as e:
        results["outbound_https"] = False
        results["outbound_https_error"] = str(e)

    # ----- Test 7: Can we reach vLLM? (sanity check — should always work) -----
    try:
        import requests
        resp = requests.get("http://ga-002:8000/health", timeout=5)
        results["vllm_reachable"] = True
        results["vllm_status"] = resp.status_code
    except Exception as e:
        results["vllm_reachable"] = False
        results["vllm_error"] = str(e)

    # ----- Test 8: List key installed packages -----
    try:
        import subprocess
        pip_output = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=10
        )
        if pip_output.returncode == 0:
            import json as _json
            packages = _json.loads(pip_output.stdout)
            # Only report packages relevant to WebSocket/networking
            relevant = ["websockets", "aiohttp", "httpx", "requests",
                        "urllib3", "certifi", "ssl", "tornado", "fastapi",
                        "uvicorn", "websocket-client"]
            results["relevant_packages"] = {
                p["name"]: p["version"]
                for p in packages
                if p["name"].lower() in relevant
            }
        else:
            results["relevant_packages"] = f"pip list failed: {pip_output.stderr[:200]}"
    except Exception as e:
        results["relevant_packages"] = f"Error: {str(e)}"

    return results
"""

_ns = {}
exec(compile(_TEST_FN_SOURCE, "<test_compute_node_connectivity>", "exec"), _ns)
test_compute_node_connectivity = _ns["test_compute_node_connectivity"]


# =============================================================================
# MAIN: Submit the test and display results
# =============================================================================


def main():
    endpoint_id = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")
    if not endpoint_id:
        print("ERROR: GLOBUS_COMPUTE_ENDPOINT_ID not set in .env")
        print("       Cannot submit test to Lakeshore without an endpoint ID.")
        sys.exit(1)

    print("=" * 65)
    print("  COMPUTE NODE CONNECTIVITY TEST")
    print("  Submitting diagnostic function via Globus Compute...")
    print(f"  Endpoint: {endpoint_id}")
    print("=" * 65)
    print()

    # Create executor (same config as globus_compute_client.py)
    print("[1/3] Creating Globus Compute Executor...")
    t_start = time.perf_counter()

    gce = Executor(endpoint_id=endpoint_id)
    gce.serializer = ComputeSerializer(strategy_code=AllCodeStrategies())

    t_executor = time.perf_counter()
    print(f"      Executor created in {t_executor - t_start:.1f}s")

    # Submit the test function
    print("[2/3] Submitting test function to compute node...")
    future = gce.submit(test_compute_node_connectivity)

    t_submit = time.perf_counter()
    print(f"      Submitted in {t_submit - t_executor:.1f}s")
    print("      Waiting for result (timeout: 60s)...")

    # Wait for result
    try:
        result = future.result(timeout=60)
    except TimeoutError:
        print("\nERROR: Test timed out after 60 seconds.")
        print("       The compute node may be unavailable or Globus is down.")
        gce.shutdown(wait=False)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        gce.shutdown(wait=False)
        sys.exit(1)

    t_result = time.perf_counter()
    print(f"[3/3] Result received in {t_result - t_submit:.1f}s")
    print()

    # Display results
    print("=" * 65)
    print("  RESULTS FROM COMPUTE NODE")
    print("=" * 65)
    print()

    # Basic info
    print(f"  Hostname:       {result.get('hostname', '???')}")
    print(f"  Python:         {result.get('python_version', '???')}")
    print(f"  Platform:       {result.get('platform', '???')}")
    print()

    # Connectivity tests
    print("  CONNECTIVITY TESTS:")
    print("  " + "-" * 50)

    tests = [
        ("DNS resolution", "dns_resolution", "dns_error"),
        ("Outbound TCP :443", "outbound_tcp_443", "outbound_tcp_443_error"),
        ("Outbound TCP :80", "outbound_tcp_80", "outbound_tcp_80_error"),
        ("Outbound HTTPS", "outbound_https", "outbound_https_error"),
        ("vLLM reachable", "vllm_reachable", "vllm_error"),
    ]

    all_pass = True
    for label, key, err_key in tests:
        passed = result.get(key, False)
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "x"
        error = f" — {result.get(err_key, '')}" if not passed and err_key in result else ""
        print(f"  [{icon}] {label:<25} {status}{error}")
        if not passed and key != "vllm_reachable":
            all_pass = False

    print()

    # Package availability
    print("  PACKAGE AVAILABILITY:")
    print("  " + "-" * 50)

    ws_installed = result.get("websockets_installed", False)
    ws_version = result.get("websockets_version", "N/A")
    aio_installed = result.get("aiohttp_installed", False)
    aio_version = result.get("aiohttp_version", "N/A")

    print(
        f"  [{'+'if ws_installed else 'x'}] websockets            {'v' + ws_version if ws_installed else 'NOT INSTALLED'}"
    )
    print(
        f"  [{'+'if aio_installed else 'x'}] aiohttp               {'v' + aio_version if aio_installed else 'NOT INSTALLED'}"
    )

    relevant = result.get("relevant_packages", {})
    if isinstance(relevant, dict) and relevant:
        print()
        print("  Other networking packages found:")
        for pkg, ver in sorted(relevant.items()):
            print(f"    - {pkg} {ver}")
    print()

    # Verdict
    print("=" * 65)
    print("  VERDICT")
    print("=" * 65)
    print()

    if all_pass and ws_installed:
        print("  ALL CLEAR — WebSocket relay is feasible!")
        print("  The compute node can make outbound connections and has websockets.")
        print()
        print("  Next step: build the relay server and modify the remote function.")
    elif all_pass and not ws_installed:
        print("  PARTIAL — Outbound connectivity works, but websockets is not installed.")
        print()
        print("  Options:")
        print("  1. Install websockets in the endpoint's Python environment:")
        print("     pip install websockets")
        print("  2. Use websocket-client (sync) if available")
        if isinstance(relevant, dict) and "websocket-client" in relevant:
            print(f"     -> websocket-client IS available (v{relevant['websocket-client']})")
        print("  3. Use raw HTTP/urllib for a non-WebSocket streaming approach")
    else:
        print("  BLOCKED — Compute node cannot make outbound connections.")
        print("  The firewall blocks outbound TCP from compute nodes.")
        print()
        print("  Alternative approaches:")
        print("  1. Host the relay on a Lakeshore LOGIN node (compute nodes can")
        print("     reach login nodes on the internal network)")
        print("  2. Use Globus Compute's AMQP channel with chunked results")
        print("     (multiple small task submissions instead of one big one)")
        print("  3. Ask ACER to whitelist a specific relay server IP/port")
    print()

    # Dump full JSON for reference
    print("  Full JSON result:")
    print("  " + json.dumps(result, indent=2).replace("\n", "\n  "))
    print()

    # Cleanup
    gce.shutdown(wait=False)


if __name__ == "__main__":
    main()
