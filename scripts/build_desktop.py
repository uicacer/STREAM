"""
Build the STREAM desktop app for distribution.

This script orchestrates the entire build pipeline:

    1. Build the React frontend (npm run build → frontends/react/dist/)
    2. Run PyInstaller to bundle Python + frontend into a single app
    3. Report the output location

WHY A BUILD SCRIPT:
-------------------
The build process has multiple steps that must happen in order. Running them
manually is error-prone (easy to forget a step or run them out of order).
This script automates the entire process so you can build with one command.

USAGE:
------
    python scripts/build_desktop.py

OUTPUT:
-------
    macOS:   dist/STREAM.app
    Windows: dist/STREAM/STREAM.exe
    Linux:   dist/STREAM/STREAM

PREREQUISITES:
--------------
    pip install pyinstaller   (or: uv pip install pyinstaller)
    Node.js + npm             (for building React frontend)
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

# =========================================================================
# Path constants — all relative to the project root
# =========================================================================
# Path(__file__) is this script's location (scripts/build_desktop.py).
# .parent.parent gets us to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the React source code lives
REACT_DIR = PROJECT_ROOT / "frontends" / "react"

# Where "npm run build" puts the compiled React files
REACT_DIST = REACT_DIR / "dist"

# The PyInstaller configuration file
SPEC_FILE = PROJECT_ROOT / "stream.spec"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """
    Run a shell command and exit if it fails.

    subprocess.run() executes a command just like typing it in a terminal.
    check=True means: "If the command fails (non-zero exit code), raise an
    exception." This ensures we stop the build immediately on any error
    rather than continuing with broken output.

    Args:
        cmd: Command and arguments as a list (e.g., ["npm", "run", "build"])
        cwd: Directory to run the command in (None = current directory)
    """
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def build_react() -> None:
    """
    Build the React frontend.

    "npm run build" compiles the React source code (JSX, TypeScript, CSS) into
    optimized static files (plain HTML, JS, CSS) that any browser can run.
    The output goes to frontends/react/dist/.

    These compiled files are what gets bundled into the desktop app. The user's
    browser engine (WebKit/Edge) loads them just like any website.
    """
    print("\n[1/2] Building React frontend...")

    if not (REACT_DIR / "package.json").exists():
        print("  ERROR: No package.json found in frontends/react/")
        print("  Make sure the React project is set up.")
        sys.exit(1)

    # Check if npm is installed
    if not shutil.which("npm"):
        print("  ERROR: npm not found. Install Node.js from https://nodejs.org/")
        sys.exit(1)

    # Install dependencies (if needed) and build
    run(["npm", "install"], cwd=REACT_DIR)
    run(["npm", "run", "build"], cwd=REACT_DIR)

    # Verify the build produced output
    if not (REACT_DIST / "index.html").exists():
        print("  ERROR: React build did not produce dist/index.html")
        sys.exit(1)

    print("  React build complete.")


def build_pyinstaller() -> None:
    """
    Run PyInstaller to bundle everything into a distributable app.

    PyInstaller analyzes the Python code, finds all dependencies (imports),
    and packages them together with a Python interpreter into a standalone
    directory (or .app on macOS) that can run without Python installed.

    The stream.spec file tells PyInstaller:
      - Which Python file is the entry point
      - Which data files to include (React build, YAML config)
      - Which modules to exclude (streamlit, pytest, etc.)
      - How to name and configure the output
    """
    print("\n[2/2] Running PyInstaller...")

    if not SPEC_FILE.exists():
        print(f"  ERROR: {SPEC_FILE} not found")
        sys.exit(1)

    # Check if pyinstaller is installed
    if not shutil.which("pyinstaller"):
        print("  ERROR: pyinstaller not found.")
        print("  Install with: uv pip install pyinstaller")
        sys.exit(1)

    # --clean removes the previous build cache to ensure a fresh build.
    # --noconfirm overwrites existing output without asking.
    run(["pyinstaller", "--clean", "--noconfirm", str(SPEC_FILE)], cwd=PROJECT_ROOT)

    print("  PyInstaller build complete.")


def report_output() -> None:
    """Print where the built app is located."""
    dist_dir = PROJECT_ROOT / "dist"
    system = platform.system()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)

    if system == "Darwin":
        app_path = dist_dir / "STREAM.app"
        if app_path.exists():
            print(f"  macOS app: {app_path}")
            print(f"  To run:    open {app_path}")
        else:
            print(f"  Output directory: {dist_dir / 'STREAM'}")
    elif system == "Windows":
        print(f"  Windows app: {dist_dir / 'STREAM' / 'STREAM.exe'}")
    else:
        print(f"  Linux app: {dist_dir / 'STREAM' / 'STREAM'}")

    print()


def main() -> None:
    """Run the full build pipeline."""
    print("=" * 60)
    print("STREAM Desktop — Build Pipeline")
    print("=" * 60)

    build_react()
    build_pyinstaller()
    report_output()


if __name__ == "__main__":
    main()
