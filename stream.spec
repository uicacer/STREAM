# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for the STREAM desktop app.

WHAT IS A .spec FILE:
---------------------
PyInstaller uses this file to know HOW to bundle your Python app. It's a
regular Python script that PyInstaller executes during the build process.
Think of it as a recipe: "Take these ingredients (Python files, data files),
exclude these things (test tools, server-only packages), and produce an app."

HOW PyInstaller WORKS:
----------------------
1. ANALYSIS: PyInstaller reads your entry point (main.py) and follows ALL
   imports to find every Python module your app needs. This is called
   "dependency tracing" — it builds a complete graph of what imports what.

2. HIDDEN IMPORTS: Some imports can't be found by static analysis because
   they happen dynamically at runtime (e.g., litellm.llms.anthropic is
   imported based on which AI provider you're using). We list these
   explicitly so PyInstaller knows to include them.

3. DATA FILES: Non-Python files (YAML config, React build, images) aren't
   found by import tracing. We list them in `datas` so they get copied
   into the bundle.

4. EXCLUDES: Packages we DON'T want in the bundle (test tools, server-only
   dependencies like PostgreSQL drivers). Excluding them reduces bundle size.

5. OUTPUT: PyInstaller produces either:
   - A directory with all files (--onedir, default) — fast startup, larger folder
   - A single executable (--onefile) — slower startup, single file to distribute

HOW TO BUILD:
-------------
    pyinstaller stream.spec

Or use the build script:
    python scripts/build_desktop.py
"""

import platform
from pathlib import Path

# collect_data_files is a PyInstaller utility that finds ALL non-Python files
# (JSON, YAML, etc.) inside a package. This is the nuclear option for packages
# like litellm that scatter data files across dozens of subpackages and load
# them dynamically via importlib.resources — impossible to trace statically.
from PyInstaller.utils.hooks import collect_data_files

# =========================================================================
# PATHS
# =========================================================================
# All paths are relative to this .spec file's location (project root).
project_root = Path(SPECPATH)  # SPECPATH is a PyInstaller built-in variable

# Collect ALL data files from litellm (JSON configs, tokenizer vocab, endpoint
# definitions, model cost maps, etc.). litellm loads these via importlib.resources
# and __file__ paths at runtime — PyInstaller can't trace either pattern.
litellm_data = collect_data_files("litellm")


# =========================================================================
# ANALYSIS — Find all Python modules and dependencies
# =========================================================================
a = Analysis(
    # Entry point — the first Python file that runs when the app starts.
    # PyInstaller traces all imports from here to build the dependency graph.
    [str(project_root / "stream" / "desktop" / "main.py")],

    # Where to look for Python modules (in addition to sys.path)
    pathex=[str(project_root)],

    # =====================================================================
    # DATA FILES — Non-Python files to include in the bundle
    # =====================================================================
    # Format: (source_path, destination_directory_in_bundle)
    #
    # These files end up in the _internal/ directory of the bundled app.
    # Python code accesses them via __file__ paths or sys._MEIPASS.
    datas=[
        # The LiteLLM config defines all available AI models (names, API bases,
        # costs). Without it, the app wouldn't know which models exist.
        (
            str(project_root / "stream" / "gateway" / "litellm_config.yaml"),
            "stream/gateway",
        ),

        # The pre-built React frontend (HTML, JS, CSS).
        # static_files.py searches for this at: Path(__file__).parent.parent / "frontend" / "dist"
        # Since __file__ will be at _internal/stream/desktop/static_files.py,
        # .parent.parent = _internal/stream/, so we place it at stream/frontend/dist/.
        (
            str(project_root / "frontends" / "react" / "dist"),
            "stream/frontend/dist",
        ),

        # All litellm data files — tokenizer JSON, endpoint configs, model cost
        # maps, etc. litellm scatters data files across many subpackages and loads
        # them via importlib.resources and __file__ paths, so we must include them
        # all. collect_data_files() (defined above) finds every non-Python file
        # in the litellm package tree.
    ] + litellm_data,

    # No extra binary files (shared libraries) needed
    binaries=[],

    # =====================================================================
    # HIDDEN IMPORTS — Modules that PyInstaller can't find automatically
    # =====================================================================
    # PyInstaller finds imports by reading your code statically. But some
    # modules are imported DYNAMICALLY at runtime (e.g., litellm picks
    # provider modules based on the model name). These need to be listed
    # explicitly so PyInstaller includes them in the bundle.
    hiddenimports=[
        # --- uvicorn internals ---
        # uvicorn uses entry_points and dynamic imports for its event loop,
        # HTTP protocol implementation, and logging configuration.
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",

        # --- litellm AI provider modules ---
        # litellm dynamically imports provider-specific code based on which
        # AI model you're calling. It checks the model name prefix ("claude-*"
        # → Anthropic, "gpt-*" → OpenAI) and imports the matching module.
        "litellm.llms.anthropic",
        "litellm.llms.anthropic.chat",
        "litellm.llms.openai",
        "litellm.llms.openai.chat",
        "litellm.llms.ollama",
        "litellm.llms.ollama.chat",
        "litellm.llms.ollama_chat",
        "litellm.llms.text_completion_codestral",

        # --- litellm core utilities ---
        # litellm uses importlib.resources to load tokenizer data at runtime.
        # PyInstaller can't trace importlib.resources calls, so we must
        # explicitly tell it this package exists.
        "litellm.litellm_core_utils.tokenizers",

        # --- tiktoken (token counting) ---
        # tiktoken uses a plugin system to load tokenizer data. The plugin
        # module contains the actual BPE (Byte Pair Encoding) vocabularies
        # for different models. Without it, token counting fails silently.
        "tiktoken_ext",
        "tiktoken_ext.openai_public",

        # --- Globus Compute (Lakeshore HPC) ---
        # The Globus SDK has optional components loaded at runtime.
        "globus_sdk",
        "globus_compute_sdk",

        # --- Standard library modules sometimes missed ---
        "email.mime.text",
        "email.mime.multipart",

        # --- Pydantic (FastAPI's data validation library) ---
        # Pydantic v2 uses a Rust extension (_pydantic_core) that PyInstaller
        # sometimes misses because it's loaded as a native C extension.
        "pydantic",
        "pydantic_core",
        "pydantic.deprecated.decorator",
    ],

    # =====================================================================
    # EXCLUDES — Packages we do NOT want in the bundle
    # =====================================================================
    # These save space and reduce build time. They're either:
    #   - Server-only: Not needed in desktop mode (PostgreSQL, Streamlit)
    #   - Dev-only: Testing and development tools
    #   - Unused: Large optional dependencies we don't need
    excludes=[
        # Server-only dependencies (desktop uses SQLite, not PostgreSQL)
        "psycopg2",
        "psycopg2-binary",

        # Streamlit frontend (desktop uses React via PyWebView instead)
        "streamlit",

        # Development/testing tools
        "pytest",
        "pytest_asyncio",
        "ruff",
        "pre_commit",
        "ipykernel",
        "jupyter",
        "jupyter_core",
        "jupyter_client",
        "notebook",

        # Heavy scientific libraries (not needed for chat routing)
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",

        # Other unused optional dependencies
        "tkinter",
        "PIL",
    ],

    # Let PyInstaller handle hooks (plugins that know how to bundle
    # specific packages like PyWebView, Pydantic, etc.)
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # Don't obfuscate the bytecode (makes debugging easier)
    cipher=None,

    # Suppress "missing module" warnings for excluded packages
    noarchive=False,
)


# =========================================================================
# PYZ — Compress Python bytecode into a single archive
# =========================================================================
# PYZ (Python Zip) bundles all .pyc files into one compressed archive.
# This is an optimization — loading one archive is faster than loading
# hundreds of individual .pyc files from disk.
pyz = PYZ(a.pure)


# =========================================================================
# EXE — Create the executable
# =========================================================================
# This defines the actual executable file that users double-click to launch.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Binaries go in the directory, not inside the exe
    name="STREAM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Don't strip debug symbols (helps with crash reports)
    upx=True,  # Compress with UPX if available (reduces binary size)
    console=False,  # No terminal window — we're a GUI app (PyWebView)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Build for the current architecture
    codesign_identity=None,  # No code signing yet (Phase 8 future work)
    entitlements_file=None,
)


# =========================================================================
# COLLECT — Gather everything into an output directory
# =========================================================================
# COLLECT takes all the pieces (executable, Python modules, data files,
# shared libraries) and assembles them into the final output directory.
# On macOS, this becomes the .app bundle's internal structure.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="STREAM",
)


# =========================================================================
# BUNDLE — macOS .app bundle (macOS only)
# =========================================================================
# On macOS, users expect a .app bundle (a special directory that Finder
# treats as a single application). This wraps the COLLECT output in the
# standard macOS app structure:
#   STREAM.app/
#   ├── Contents/
#   │   ├── Info.plist     ← App metadata (name, version, icon)
#   │   ├── MacOS/
#   │   │   └── STREAM     ← The actual executable
#   │   └── Resources/
#   │       └── ...        ← Python modules, data files, etc.
if platform.system() == "Darwin":
    app = BUNDLE(
        coll,
        name="STREAM.app",
        # icon= sets the .app's icon file. macOS uses .icns format, which
        # contains the same icon at multiple resolutions (16x16 to 1024x1024).
        # This icon appears in the Dock, Finder, Spotlight, and the app switcher.
        # Generated from frontends/react/public/favicon.svg using rsvg-convert + iconutil.
        icon=str(project_root / "assets" / "icon.icns"),
        # The bundle identifier is a reverse-DNS name that uniquely identifies
        # your app on macOS. It's used by the OS for preferences, keychain
        # access, and other per-app settings.
        bundle_identifier="edu.uic.stream",
        info_plist={
            "CFBundleName": "STREAM",
            "CFBundleDisplayName": "STREAM",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
            # NSHighResolutionCapable enables Retina display support.
            # Without it, the app would render at 1x resolution and look blurry.
            "NSHighResolutionCapable": True,
            # CFBundleIconFile tells macOS which file in Resources/ is the app icon.
            # PyInstaller copies icon.icns into Contents/Resources/ automatically.
            "CFBundleIconFile": "icon.icns",
        },
    )
