"""
First-run setup wizard for the STREAM desktop app.

WHEN THIS RUNS:
---------------
The first time a user launches STREAM, there's no ~/.stream/ directory.
This module detects that and walks the user through initial setup:

    1. Create the ~/.stream/ directory structure
    2. Check if Ollama is installed (needed for free local AI models)
    3. Check if required AI models are downloaded
    4. Optionally prompt for cloud API keys

WHY A FIRST-RUN CHECK:
----------------------
Without Ollama and at least one model, the LOCAL tier won't work. Without
API keys, the CLOUD tier won't work. Rather than showing cryptic errors in
the UI, we catch these issues early and guide the user through fixing them.

The setup runs in the terminal BEFORE the PyWebView window opens, so the
user sees clear text prompts. Future versions could show a setup wizard
inside the React UI instead.

USER DATA DIRECTORY:
--------------------
~/.stream/ is where STREAM stores all user data:
    ~/.stream/
    ├── config.toml        # User config (API keys, preferences)
    ├── data/
    │   └── costs.db       # SQLite cost tracking database
    └── logs/
        └── stream.log     # Application logs

Using ~/.stream/ (a hidden directory in the user's home folder) follows
the Unix convention for application data. The dot prefix makes it hidden
in normal file listings, keeping the home directory tidy.
"""

import os
import shutil
from pathlib import Path

from stream.middleware.config import OLLAMA_MODELS

# =========================================================================
# The user data directory — where STREAM stores all persistent data
# =========================================================================
# Path.home() returns the user's home directory:
#   macOS/Linux: /Users/<name>/  or  /home/<name>/
#   Windows:     C:\Users\<name>\
# We append .stream/ to create a hidden directory for our app's data.
STREAM_HOME = Path.home() / ".stream"


def is_first_run() -> bool:
    """
    Check if this is the user's first time launching STREAM.

    We use the existence of ~/.stream/ as the indicator. If it doesn't exist,
    this is a fresh install and we need to run setup.

    Returns:
        True if this is the first run, False if setup has been done before
    """
    return not STREAM_HOME.exists()


def create_directory_structure() -> None:
    """
    Create the ~/.stream/ directory and subdirectories.

    mkdir(parents=True) creates all intermediate directories. For example,
    if ~/.stream/ doesn't exist, it creates both ~/.stream/ and ~/.stream/data/.

    exist_ok=True means "don't raise an error if the directory already exists."
    This makes the function safe to call multiple times (idempotent).
    """
    directories = [
        STREAM_HOME,
        STREAM_HOME / "data",  # SQLite database files
        STREAM_HOME / "logs",  # Application log files
        STREAM_HOME / "cache",  # Temporary cache (tiktoken, etc.)
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"  Created data directory: {STREAM_HOME}")


def check_ollama_installed() -> bool:
    """
    Check if Ollama is installed on this machine.

    shutil.which() searches the system PATH for an executable — the same way
    your shell finds commands when you type them. If it finds "ollama", the
    program is installed and accessible.

    Returns:
        True if Ollama is installed, False otherwise
    """
    return shutil.which("ollama") is not None


def check_models_available() -> dict[str, bool]:
    """
    Check which required AI models are already downloaded.

    Ollama stores models locally after you download them. This function
    asks Ollama for its list of installed models and checks if our
    required models are present.

    Returns:
        Dict mapping model name to availability (True = downloaded)
    """
    # Import here because OllamaModelManager connects to Ollama on init,
    # and we want to handle the case where Ollama isn't running gracefully.
    from stream.middleware.core.ollama_manager import OllamaModelManager

    try:
        manager = OllamaModelManager()
    except Exception:
        # Ollama might not be running — all models are unavailable
        return dict.fromkeys(OLLAMA_MODELS.values(), False)

    return {model: manager.is_model_available(model) for model in OLLAMA_MODELS.values()}


def run_first_run_setup() -> None:
    """
    Walk the user through first-time setup.

    This is the main function called by main.py on first launch. It prints
    friendly messages to the terminal and creates the necessary directories.
    Model downloading is handled later by the middleware lifecycle (lifecycle.py)
    which prompts the user interactively if models are missing.
    """
    print()
    print("=" * 60)
    print("  Welcome to STREAM!")
    print("  Smart Tiered Routing Engine for AI Models")
    print("=" * 60)
    print()
    print("  This appears to be your first time running STREAM.")
    print("  Let's make sure everything is set up correctly.")
    print()

    # -------------------------------------------------------------------------
    # Step 1: Create directory structure
    # -------------------------------------------------------------------------
    print("[1/3] Setting up data directory...")
    create_directory_structure()

    # -------------------------------------------------------------------------
    # Step 2: Check Ollama
    # -------------------------------------------------------------------------
    print()
    print("[2/3] Checking for Ollama (local AI model runner)...")

    if check_ollama_installed():
        print("  Ollama is installed.")

        # Check which models are available
        print("  Checking installed models...")
        model_status = check_models_available()

        all_available = all(model_status.values())
        if all_available:
            print("  All required models are downloaded.")
        else:
            missing = [name for name, available in model_status.items() if not available]
            print(f"  Missing models: {', '.join(missing)}")
            print("  These will be downloaded automatically when the app starts.")
            print("  (This may take a few minutes on first run)")
    else:
        print("  Ollama is NOT installed.")
        print()
        print("  Ollama runs free AI models on your computer (no internet needed).")
        print("  Without it, only Cloud models (requires API keys) will work.")
        print()
        print("  To install Ollama:")
        print("    macOS:   brew install ollama")
        print("    or visit https://ollama.com/download")
        print()

    # -------------------------------------------------------------------------
    # Step 2.5: Copy .env to user config directory
    # -------------------------------------------------------------------------
    # The project root .env file has API keys and other settings. In the
    # bundled STREAM.app, the project root isn't accessible — the app can
    # only read from ~/.stream/. So we copy .env there on first run.
    #
    # This only happens once (first run). After that, the user edits
    # ~/.stream/.env directly to add or update their API keys.
    user_env = STREAM_HOME / ".env"
    project_env = Path(__file__).resolve().parent.parent.parent / ".env"

    if not user_env.exists() and project_env.exists():
        shutil.copy2(project_env, user_env)
        print(f"  Copied API keys to: {user_env}")
    elif not user_env.exists():
        # No project .env found (fresh install from bundled app).
        # Create a template so the user knows where to put their keys.
        user_env.write_text(
            "# STREAM API Keys\n"
            "# Add your cloud provider API keys here.\n"
            "# The app will load these automatically on startup.\n"
            "#\n"
            "# ANTHROPIC_API_KEY=sk-ant-...\n"
            "# OPENAI_API_KEY=sk-...\n"
        )
        print(f"  Created API key template: {user_env}")
        print("  Edit this file to add your cloud API keys.")

    # -------------------------------------------------------------------------
    # Step 3: Check for API keys
    # -------------------------------------------------------------------------
    print()
    print("[3/3] Checking cloud API keys...")

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if has_anthropic or has_openai:
        providers = []
        if has_anthropic:
            providers.append("Anthropic (Claude)")
        if has_openai:
            providers.append("OpenAI (GPT)")
        print(f"  Found API keys for: {', '.join(providers)}")
    else:
        print("  No cloud API keys found.")
        print("  Cloud models (Claude, GPT) will be unavailable.")
        print("  You can add API keys later in the Settings panel.")

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Setup complete! Starting STREAM...")
    print("=" * 60)
    print()
