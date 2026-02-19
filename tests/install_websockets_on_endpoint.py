"""
Install websockets on the Globus Compute endpoint.

Submits a function via Globus Compute that runs `pip install websockets`
in the exact Python environment where remote functions execute.

Usage:
    python tests/install_websockets_on_endpoint.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from globus_compute_sdk import Executor  # noqa: E402
from globus_compute_sdk.serialize import AllCodeStrategies, ComputeSerializer  # noqa: E402

# Remote function: install websockets and verify
_INSTALL_FN_SOURCE = """\
def install_and_verify_websockets():
    import subprocess
    import sys

    results = {}
    results["python_executable"] = sys.executable
    results["hostname"] = __import__("socket").gethostname()

    # Step 1: Try to install websockets
    try:
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "websockets"],
            capture_output=True, text=True, timeout=60
        )
        results["install_stdout"] = install.stdout[-500:] if install.stdout else ""
        results["install_stderr"] = install.stderr[-500:] if install.stderr else ""
        results["install_returncode"] = install.returncode
    except Exception as e:
        results["install_error"] = str(e)
        results["install_returncode"] = -1

    # Step 2: Verify it's importable
    try:
        import importlib
        importlib.invalidate_caches()

        # Force fresh import (in case it was cached as missing)
        if "websockets" in sys.modules:
            del sys.modules["websockets"]

        import websockets
        results["verify_success"] = True
        results["websockets_version"] = websockets.__version__
    except ImportError as e:
        results["verify_success"] = False
        results["verify_error"] = str(e)

    return results
"""

_ns = {}
exec(compile(_INSTALL_FN_SOURCE, "<install_and_verify_websockets>", "exec"), _ns)
install_and_verify_websockets = _ns["install_and_verify_websockets"]


def main():
    endpoint_id = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")
    if not endpoint_id:
        print("ERROR: GLOBUS_COMPUTE_ENDPOINT_ID not set in .env")
        sys.exit(1)

    print("=" * 65)
    print("  INSTALL WEBSOCKETS ON COMPUTE NODE")
    print(f"  Endpoint: {endpoint_id}")
    print("=" * 65)
    print()

    print("[1/3] Creating Executor...")
    t_start = time.perf_counter()
    gce = Executor(endpoint_id=endpoint_id)
    gce.serializer = ComputeSerializer(strategy_code=AllCodeStrategies())
    print(f"      Done ({time.perf_counter() - t_start:.1f}s)")

    print("[2/3] Submitting install function...")
    t_submit = time.perf_counter()
    future = gce.submit(install_and_verify_websockets)
    print(f"      Submitted ({time.perf_counter() - t_submit:.1f}s)")
    print("      Waiting for result (timeout: 120s)...")

    try:
        result = future.result(timeout=120)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        gce.shutdown(wait=False)
        sys.exit(1)

    t_done = time.perf_counter()
    print(f"[3/3] Done ({t_done - t_submit:.1f}s)")
    print()

    # Display results
    print(f"  Hostname:   {result.get('hostname')}")
    print(f"  Python:     {result.get('python_executable')}")
    print(f"  Return code: {result.get('install_returncode')}")
    print()

    if result.get("install_stdout"):
        print("  pip output:")
        for line in result["install_stdout"].strip().split("\n"):
            print(f"    {line}")
        print()

    if result.get("install_stderr"):
        print("  pip stderr:")
        for line in result["install_stderr"].strip().split("\n"):
            print(f"    {line}")
        print()

    if result.get("verify_success"):
        ver = result.get("websockets_version", "?")
        print(f"  RESULT: websockets v{ver} installed and verified!")
    else:
        print(
            f"  RESULT: Installation failed — {result.get('verify_error', result.get('install_error', '???'))}"
        )

    print()
    gce.shutdown(wait=False)


if __name__ == "__main__":
    main()
