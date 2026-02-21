# STREAM Desktop App — Complete Guide

*This is a living document. We update it as we build.*
*Last updated: 2026-02-16*

## Table of Contents

1. [Why a Desktop App?](#1-why-a-desktop-app)
2. [What is Desktop App Packaging?](#2-what-is-desktop-app-packaging)
3. [Framework Options (and why we chose PyWebView)](#3-framework-options-and-why-we-chose-pywebview)
4. [Our Tech Stack (PyWebView + PyInstaller)](#4-our-tech-stack)
5. [Architecture: Docker vs Desktop](#5-architecture-docker-vs-desktop)
6. [How GPU Access Works](#6-how-gpu-access-works)
7. [Implementation Phases (Step-by-Step)](#7-implementation-phases)
8. [New Files We'll Create](#8-new-files-well-create)
9. [Existing Files We'll Modify](#9-existing-files-well-modify)
10. [How to Build and Test](#10-how-to-build-and-test)
11. [Distribution and Licensing](#11-distribution-and-licensing)
12. [Future Roadmap](#12-future-roadmap)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Why a Desktop App?

**Problem:** STREAM currently runs as 5 Docker containers. This has 3 issues:

1. **NO GPU ON MAC:** Docker on Mac runs a Linux VM. That VM CANNOT access Apple Silicon's GPU via Metal. So Ollama runs on CPU only — the complexity judge takes 3-5 seconds instead of 0.3-0.5 seconds.

2. **COMPLEX SETUP:** Users must install Docker Desktop, clone the repo, configure `.env`, and run `docker-compose`. That's a lot of friction.

3. **NO BUDGET FOR CLOUD:** We can't host STREAM online right now. But we still want to distribute it to users.

**Solution:** Package STREAM as a native desktop app.
- User downloads one file (`.dmg` on Mac, `.exe` on Windows, `.AppImage` on Linux)
- Double-clicks to install
- Double-clicks to run
- Ollama runs natively → full GPU access → 5-10x faster

And here's the best part: we keep the SAME codebase. One environment variable (`STREAM_MODE=desktop` vs `STREAM_MODE=server`) controls which code paths activate. So when we're ready to deploy to the cloud later, we just use the server mode. Zero code changes.

---

## 2. What is Desktop App Packaging?

"Packaging" means turning your development project into something a regular user (non-developer) can install and run.

**Without packaging:**
- User needs: Python 3.12, Node.js, npm, git, `.env` file, `pip install`...
- User runs: `python -m stream.middleware.app` — not user-friendly!

**With packaging:**
- User downloads: `STREAM.dmg` (one file)
- User installs: Drag to Applications folder
- User runs: Double-click STREAM icon

### How It Works (simplified)

```
Your Python code
+ Python interpreter (the engine that runs Python)
+ All pip packages (fastapi, litellm, etc.)
+ React frontend (pre-built HTML/JS/CSS)
+ Config files
──────────────────────────────────
= One standalone executable
```

Think of it like shipping a car:
- **PyInstaller** = the factory that assembles all the parts
- **PyWebView** = the windshield (the window users look through)
- **FastAPI** = the engine (handles all the logic)
- **React** = the dashboard (what the user sees through the windshield)
- **Ollama** = the fuel (provides AI responses)

---

## 3. Framework Options (and why we chose PyWebView)

There are 4 ways to turn a web app into a desktop app. We evaluated all of them for STREAM.

### Option 1: Browser-Based (simplest — no native window)

```
┌────────────────────────────────────────────────┐
│  User's Browser (Chrome, Safari, etc.)          │
│  ┌──────────────────────────────────────────┐  │
│  │  http://localhost:5000                    │  │
│  │  ┌────────────────────────────────────┐  │  │
│  │  │  STREAM React UI                   │  │  │
│  │  └────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────┘  │
│  (URL bar, tabs, bookmarks — full browser)      │
└────────────────────────────────────────────────┘
         ↕ HTTP
┌────────────────────────────────────────────────┐
│  Python process (FastAPI + uvicorn)              │
└────────────────────────────────────────────────┘
```

**How it works:**
Python starts FastAPI, then opens the user's default browser to `http://localhost:5000`. No native window — just a browser tab.

**Pros:**
- Simplest to implement (almost zero extra code)
- No extra dependencies
- Cross-platform by default

**Cons:**
- Doesn't look like a desktop app (has URL bar, tabs, bookmarks)
- User might accidentally close the tab
- Can't control window size or behavior
- Feels unprofessional

**When to use:** Quick prototypes, internal tools, developer-facing apps

---

### Option 2: PyWebView (our choice — Python native window)

```
┌────────────────────────────────────────────────┐
│  STREAM                              — □ ×     │ ← Native window
│  ┌──────────────────────────────────────────┐  │    (no URL bar,
│  │                                          │  │     no tabs,
│  │     STREAM React UI                      │  │     no bookmarks)
│  │     (renders inside OS WebView)          │  │
│  │                                          │  │    Uses OS's built-in
│  └──────────────────────────────────────────┘  │    WebView engine
└────────────────────────────────────────────────┘
         ↕ HTTP (localhost)
┌────────────────────────────────────────────────┐
│  Python process (FastAPI + uvicorn)              │
└────────────────────────────────────────────────┘
```

**How it works:**
Python starts FastAPI in a background thread, then opens a native OS window (using the OS's built-in WebView) pointed at `localhost:5000`. The window looks like a real desktop app.

**What WebView engine does each OS use:**
- macOS: WebKit (Safari's engine)
- Windows: Edge WebView2 (Chromium-based, pre-installed on Windows 10+)
- Linux: GTK WebKit2 (needs `libwebkit2gtk`)

**Pros:**
- Stays in Python ecosystem (no new language to learn)
- Tiny (~5 MB) — uses OS's built-in WebView, doesn't bundle a browser
- Looks like a real desktop app (native window frame)
- Simple API — about 10 lines of code to open a window
- BSD license — can distribute commercially, keep code proprietary

**Cons:**
- Fewer desktop-native features (no system tray, no custom menu bar)
- No built-in auto-updater
- WebView behavior varies slightly across OS (usually not an issue)
- Smaller community than Electron

**When to use:** Python-first projects that need a native look with minimal complexity. Perfect for STREAM.

---

### Option 3: Electron (heavyweight — bundles Chromium)

```
┌────────────────────────────────────────────────┐
│  STREAM                              — □ ×     │ ← Native window
│  ┌──────────────────────────────────────────┐  │
│  │                                          │  │    Bundles its OWN
│  │     STREAM React UI                      │  │    copy of Chromium
│  │     (renders in bundled Chromium)        │  │    (~150 MB!)
│  │                                          │  │
│  └──────────────────────────────────────────┘  │    Uses Node.js
└────────────────────────────────────────────────┘    for native APIs
         ↕ HTTP (localhost)
┌────────────────────────────────────────────────┐
│  Python process (FastAPI — spawned by Electron) │
└────────────────────────────────────────────────┘
```

**How it works:**
Electron is a framework by GitHub (used by VS Code, Slack, Discord). It bundles a full copy of Chromium (the Chrome browser engine) and Node.js into your app. Your React UI runs inside this bundled Chromium.

**Pros:**
- Most mature framework (used by VS Code, Slack, Discord, Figma, etc.)
- Massive ecosystem of plugins and tools
- Full desktop-native features (system tray, menus, notifications, auto-updater)
- Consistent behavior across all platforms (same Chromium everywhere)
- Excellent documentation and community

**Cons:**
- HEAVY: bundles Chromium (~150 MB just for the shell, on top of our app)
- HIGH MEMORY: each Electron app runs its own Chromium instance (~200+ MB RAM)
- Requires Node.js knowledge (not Python)
- Overkill for STREAM (we don't need system tray, custom menus, etc.)
- Two languages to manage: Python for backend + Node.js for desktop shell

**When to use:** Complex desktop apps that need full native features and have a Node.js team. Not ideal for Python-first projects.

---

### Option 4: Tauri (lightweight — Rust-based, uses OS WebView)

```
┌────────────────────────────────────────────────┐
│  STREAM                              — □ ×     │ ← Native window
│  ┌──────────────────────────────────────────┐  │
│  │                                          │  │    Uses OS's built-in
│  │     STREAM React UI                      │  │    WebView (like
│  │     (renders in OS WebView)              │  │    PyWebView, but
│  │                                          │  │    the shell is Rust)
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
         ↕ HTTP (localhost)
┌────────────────────────────────────────────────┐
│  Python process (FastAPI — spawned by Tauri)     │
└────────────────────────────────────────────────┘
```

**How it works:**
Tauri is a newer alternative to Electron, written in Rust. Like PyWebView, it uses the OS's built-in WebView (not a bundled Chromium). But its native shell is written in Rust, giving it more desktop-native features.

**Pros:**
- Lightweight (~10 MB shell vs Electron's ~150 MB)
- More desktop features than PyWebView (system tray, menus, auto-updater)
- Excellent security model (can restrict what the frontend can do)
- Growing rapidly, backed by the Tauri Foundation
- Uses OS WebView (same as PyWebView)

**Cons:**
- Requires Rust toolchain to build (Cargo, rustc, etc.)
- Adds a third language to the project (Python + TypeScript + Rust)
- The Rust shell must spawn Python as a subprocess (more complex lifecycle)
- Newer framework — less battle-tested than Electron
- Tauri v2 (latest) still has some rough edges

**When to use:** If you outgrow PyWebView and need more desktop-native features (auto-updater, system tray, etc.) but don't want Electron's weight.

---

### Comparison Summary

| Feature | Browser | PyWebView | Electron | Tauri |
|---------|---------|-----------|----------|-------|
| **Native window** | No (browser tab) | Yes | Yes | Yes |
| **Shell size** | 0 MB | ~5 MB | ~150 MB | ~10 MB |
| **RAM usage** | Low | Low | High (~200 MB) | Low |
| **Languages** | Python only | Python only | Python + Node.js | Python + Rust |
| **System tray** | No | No | Yes | Yes |
| **Auto-updater** | No | No | Yes (built-in) | Yes (built-in) |
| **Custom menus** | No | Limited | Full | Full |
| **Learning curve** | None | Low | Medium | High (Rust) |
| **Maturity** | N/A | Mature | Very mature | Maturing |
| **License** | N/A | BSD (permissive) | MIT (permissive) | MIT/Apache |

### Why PyWebView for STREAM (for now)

We chose PyWebView because:

1. **We're a Python project.** PyWebView stays in the Python ecosystem. No Rust toolchain, no Node.js, no new language to learn.

2. **We just need a window.** STREAM is a chat UI. We don't need system tray, custom native menus, or complex desktop integrations. PyWebView gives us exactly what we need: a native window showing our React app.

3. **It's tiny.** ~5 MB vs Electron's ~150 MB. Our users are already downloading ~2-4 GB of AI models — we don't need to add another 150 MB of browser engine on top.

4. **It's simple.** About 10 lines of Python to open a window. The complexity budget should go into the AI routing logic, not the desktop shell.

5. **BSD license.** We can distribute commercially and keep our code proprietary if needed.

### Future: When to Switch

PyWebView may not be enough forever. Here's when to consider switching:

- **Switch to Tauri if:** We need an auto-updater, system tray icon (e.g., "STREAM is running in background"), or native menu bar. Tauri gives these without Electron's weight.

- **Switch to Electron if:** We need the richest possible desktop integration, have a Node.js developer on the team, and don't mind the bundle size.

- **The good news:** The backend code (FastAPI, litellm, routing logic) stays exactly the same regardless of which shell we use. Only the thin launcher layer changes. So switching frameworks later is a contained, manageable change — not a rewrite.

---

## 4. Our Tech Stack

### 4.1 PyWebView — The Native Window

**What is it?**
PyWebView is a Python library that opens a native OS window containing a web page. It's NOT a browser — there's no URL bar, no tabs, no bookmarks. It looks like a real desktop app.

**How does it work?**

```
┌──────────────────────────────────────────────┐
│  STREAM                              — □ ×   │  ← Native window frame
│  ┌────────────────────────────────────────┐  │     (provided by OS)
│  │                                        │  │
│  │     Your React UI renders here         │  │  ← WebView component
│  │     (exactly like in a browser,        │  │     (WebKit on Mac,
│  │      but without the browser chrome)   │  │      Edge on Windows,
│  │                                        │  │      GTK WebKit on Linux)
│  │                                        │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

**Key PyWebView code pattern:**

```python
import webview

# This opens a native window pointing to our FastAPI server
window = webview.create_window(
    title='STREAM',                    # Window title
    url='http://127.0.0.1:5000',      # Our FastAPI server
    width=1200,                        # Window width in pixels
    height=800,                        # Window height in pixels
    min_size=(800, 600),              # Minimum resize dimensions
)

# This blocks until the window is closed by the user
webview.start()
```

---

### 4.2 PyInstaller — The Bundler

**What is it?**
PyInstaller takes your Python application and creates a standalone executable that includes:
- Python interpreter (so users don't need Python installed)
- All your pip packages
- Your application code
- Any data files you specify

**How does it work?**

```
INPUT:                              OUTPUT:

stream/middleware/app.py            ┌──────────────────────┐
stream/middleware/config.py         │                      │
stream/middleware/core/*.py    ───►  │   STREAM.app         │
stream/desktop/main.py              │   (directory with    │
+ Python 3.12 interpreter          │    executable +      │
+ fastapi, uvicorn, litellm...     │    support files)    │
+ React dist/ folder               │                      │
+ litellm_config.yaml              └──────────────────────┘
```

**Two output modes:**

1. **One-file mode** (`--onefile`):
   - Everything packed into one executable
   - Slower to start (must unpack to temp directory)
   - Cleaner for users

2. **One-directory mode** (`--onedir`): ← We use this
   - Creates a folder with the executable + support files
   - Faster to start (no unpacking needed)
   - Better for apps with many data files

**Key PyInstaller concepts:**

**Hidden imports:** Some Python packages import other packages dynamically (at runtime, not at the top of the file). PyInstaller can't detect these. We must tell it explicitly:

```python
# PyInstaller can see this:
import fastapi  # Static import, PyInstaller finds it

# PyInstaller CANNOT see this:
module = importlib.import_module("litellm.llms.anthropic")  # Dynamic import
```

**Data files:** Non-Python files (YAML configs, React HTML/JS, etc.) must be explicitly included:

```python
datas = [
    ('frontends/react/dist', 'frontend'),  # (source, destination in bundle)
]
```

---

## 5. Architecture: Docker vs Desktop

### 5.1 Current Architecture (Docker — 5 Separate Services)

```
┌──────────────────────────────── Docker Desktop ────────────────────────────────┐
│                                                                                │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ Postgres │   │   Ollama     │   │   LiteLLM    │   │ Lakeshore Proxy   │  │
│  │  :5432   │   │   :11434     │   │    :4000     │   │      :8001        │  │
│  │          │   │              │   │              │   │                   │  │
│  │  Logs &  │   │  Local AI    │   │  Cloud API   │   │  HPC via Globus   │  │
│  │  Costs   │   │  (CPU only   │   │  Gateway     │   │  Compute          │  │
│  │          │   │   on Mac!)   │   │              │   │                   │  │
│  └────┬─────┘   └──────┬──────┘   └──────┬───────┘   └────────┬──────────┘  │
│       │                │                  │                     │              │
│       │         ┌──────▼──────────────────▼─────────────────────▼──────┐      │
│       └────────►│              STREAM Middleware                       │      │
│                 │              FastAPI :5000                           │      │
│                 └─────────────────────┬───────────────────────────────┘      │
│                                       │                                      │
└───────────────────────────────────────┼──────────────────────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   Browser          │
                              │   React Frontend   │
                              │   localhost:3000    │
                              └────────────────────┘
```

**Problems with this architecture for desktop distribution:**
1. Requires Docker Desktop (large, complex for non-developers)
2. Ollama in Docker CAN'T use Mac GPU (Metal) — runs on CPU
3. 5 services to manage, configure, and keep running
4. PostgreSQL is overkill for a single-user desktop app
5. LiteLLM as a separate service adds complexity

---

### 5.2 Desktop Architecture (Single Process + Native Ollama)

```
┌──────────────────── STREAM Desktop App ─────────────────────┐
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PyWebView Window                         │   │
│  │              (native OS window)                       │   │
│  │                                                       │   │
│  │    Shows the React frontend — looks like a real       │   │
│  │    desktop app. No browser chrome, no URL bar.        │   │
│  │                                                       │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │           FastAPI Middleware (:5000)                   │   │
│  │           (same code as Docker version!)               │   │
│  │                                                       │   │
│  │  SERVES BOTH:                                         │   │
│  │  • API endpoints (/v1/chat, /health, etc.)            │   │
│  │  • React static files (index.html, JS, CSS)           │   │
│  │                                                       │   │
│  │  USES DIRECTLY (no separate services):                │   │
│  │  • litellm library (Python calls, not HTTP)           │   │
│  │  • SQLite (file-based, no server needed)              │   │
│  │  • Lakeshore proxy (mounted into same FastAPI app)    │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
              ┌───────────▼────────────┐
              │   Ollama (native)      │
              │   localhost:11434      │
              │                        │
              │   Metal on Mac         │
              │   CUDA on Windows      │
              │   CUDA/ROCm on Linux   │
              │                        │
              │   Direct GPU access!   │
              │   5-10x faster         │
              └────────────────────────┘
```

**What changed from Docker to Desktop:**

| Component | Docker Mode | Desktop Mode |
|-----------|------------|--------------|
| **Window** | Browser tab | PyWebView native window |
| **React** | Vite dev server (:3000) | Static files served by FastAPI |
| **FastAPI** | Docker container | Background thread in app |
| **Ollama** | Docker container (CPU on Mac) | Native binary (GPU!) |
| **LiteLLM** | Separate Docker service (:4000) | Python library calls (in-process) |
| **Database** | PostgreSQL container | SQLite file |
| **Lakeshore** | Separate Docker service (:8001) | Mounted into FastAPI (same process) |
| **Config** | `.env` file | `~/.stream/config.toml` |

**What stays THE SAME:**
- All FastAPI route handlers (chat.py, health.py, costs.py, etc.)
- All React frontend components
- The streaming protocol (SSE)
- The complexity judge logic
- The routing algorithm
- The fallback logic
- Globus Compute integration for Lakeshore

---

## 6. How GPU Access Works

This is the #1 reason we're building a desktop app.

### The Problem With Docker on Mac

Docker on Mac works like this:

```
Your Mac (Apple Silicon M1/M2/M3/M4)
└── Docker Desktop
    └── Linux Virtual Machine (HyperKit/QEMU)
        └── Docker containers run here
            └── Ollama runs on LINUX CPU (no GPU access!)
```

The Linux VM is a completely different operating system. Apple Silicon GPUs use "Metal" — Apple's proprietary GPU framework. Metal only works on macOS, not Linux. So the Linux VM inside Docker has NO way to access the GPU.

**Result:** Ollama in Docker on Mac = CPU only = SLOW

### The Solution With Native Ollama

When Ollama runs natively on macOS:

```
Your Mac (Apple Silicon M1/M2/M3/M4)
└── macOS (native)
    └── Ollama (native binary)
        └── Uses Metal framework directly
            └── Full GPU access! = FAST
```

The complexity judge goes from ~3-5 seconds to ~0.3-0.5 seconds. That's a 10x improvement in the most latency-sensitive part of STREAM.

### Other Platforms

- **Windows:** Ollama uses CUDA (NVIDIA GPUs). Docker CAN access NVIDIA GPUs on Windows via WSL2/NVIDIA Container Toolkit, but native is still simpler and avoids Docker dependency.
- **Linux:** Same as Windows. Docker CAN use NVIDIA GPUs, but native is simpler. Also supports AMD GPUs via ROCm (native only).

### Performance Comparison (approximate, on Apple M2)

| Operation | Docker (CPU) | Native (Metal GPU) |
|-----------|-------------|-------------------|
| Complexity judge | 3-5 seconds | 0.3-0.5 seconds |
| Local chat (3B) | 15-30 sec | 2-5 seconds |
| Local chat (1B) | 5-10 sec | 1-2 seconds |
| Model loading | 10-20 sec | 2-3 seconds |

---

## 7. Implementation Phases

We build this incrementally. Each phase produces a testable result. The Docker/server mode continues to work unchanged throughout.

### Phase 1: Safe Config Defaults

**What we're doing:**
The current `config.py` crashes if there's no `.env` file because it does things like `int(os.getenv("MIDDLEWARE_PORT"))` — if the env var doesn't exist, `os.getenv()` returns `None`, and `int(None)` crashes.

For a desktop app, we can't require users to create a `.env` file. We need sensible defaults.

**What we're changing:**

File: `stream/middleware/config.py`

```python
# BEFORE (crashes without .env):
MIDDLEWARE_HOST = os.getenv("MIDDLEWARE_HOST")                    # Returns None
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT"))               # int(None) crash!
CORS_ORIGINS = os.getenv("CORS_ORIGINS").split(",")              # None.split() crash!

# AFTER (safe defaults):
MIDDLEWARE_HOST = os.getenv("MIDDLEWARE_HOST", "127.0.0.1")      # Default: localhost
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT", "5000"))       # Default: port 5000
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5000").split(",")

# NEW: Mode detection
STREAM_MODE = os.getenv("STREAM_MODE", "server")  # "server" or "desktop"
```

**Why this is safe:**
In Docker/server mode, the `.env` file ALWAYS provides these values. The defaults only activate when `.env` is missing (desktop mode). So this change has zero impact on existing deployments.

**How to test:**

```bash
# Remove .env temporarily, then:
python -m stream.middleware.app
# Should start on localhost:5000 without crashing
```

---

### Phase 2: Fix Hardcoded Docker Hostnames

**What we're doing:**
Some files use Docker DNS names like `http://ollama:11434` — the word "ollama" is a Docker internal hostname that only works inside Docker's network. Outside Docker, Ollama runs on `localhost`.

**What Docker DNS names are:**
When you run `docker-compose up`, Docker creates a virtual network. Each service gets a DNS name matching its service name in docker-compose.yml:

```yaml
services:
  ollama:           # <- This becomes DNS name "ollama"
    image: ollama/ollama
  middleware:       # <- This becomes DNS name "middleware"
    ...
```

So inside Docker, the middleware can reach Ollama at `http://ollama:11434`. Outside Docker, you'd use `http://localhost:11434`.

**What we're changing:**

Files:
- `stream/middleware/core/tier_health.py`
- `stream/middleware/core/warm_ping.py`

Replace hardcoded `http://ollama:` with the `OLLAMA_HOST` env var (which `OllamaModelManager` already reads correctly).

**How to test:**

```bash
# Run middleware locally (no Docker):
python -m stream.middleware.app
# Health checks for Ollama should work via localhost:11434
```

---

### Phase 3: Direct LiteLLM Library Calls

**What we're doing:**
Currently, LiteLLM runs as a **separate HTTP server** in Docker:

```
Middleware -> HTTP request -> LiteLLM server (:4000) -> HTTP request -> Cloud API
```

In desktop mode, we skip the LiteLLM server and call the library directly:

```
Middleware -> litellm.acompletion() -> Cloud API
```

This is possible because `litellm` is already a Python dependency in our `pyproject.toml`. We've been using it as a server, but it also works as a regular Python library.

**What's the difference?**

As a server (current):

```python
# litellm_client.py - sends HTTP request to LiteLLM server
response = await httpx.post(
    "http://litellm:4000/v1/chat/completions",
    json={"model": "cloud-claude", "messages": messages}
)
```

As a library (desktop mode):

```python
# litellm_direct.py - calls litellm Python function directly
response = await litellm.acompletion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=messages,
    stream=True
)
```

**Model name mapping:**
The LiteLLM server uses `litellm_config.yaml` to map friendly names (like `cloud-claude`) to actual provider model names (like `anthropic/claude-sonnet-4-20250514`). In desktop mode, we load this mapping file in Python and do the translation ourselves.

**New file:** `stream/middleware/core/litellm_direct.py`

**Modified files:**
- `stream/middleware/core/litellm_client.py` — routes to direct mode
- `stream/middleware/core/complexity_judge.py` — same pattern

**How to test:**

```bash
STREAM_MODE=desktop python -m stream.middleware.app
# Send a chat message -> should work without LiteLLM Docker service
```

---

### Phase 4: SQLite Database

**What we're doing:**
PostgreSQL is a powerful database server, but it's overkill for a single-user desktop app. It requires:
- A separate running process
- Configuration (user, password, database name)
- ~100MB of disk space

SQLite is a file-based database built into Python. It requires:
- Nothing. It's already included in Python's standard library.
- Just a file path: `~/.stream/data/costs.db`

**What's the difference?**

```
PostgreSQL:
  Your App -> TCP connection -> PostgreSQL server process -> disk file

SQLite:
  Your App -> direct function call -> disk file
  (no server, no network, no configuration)
```

**New file:** `stream/middleware/core/database_sqlite.py` — same interface as `database.py` but using SQLite

**Modified file:** `stream/middleware/core/database.py` — adds a mode switch at initialization

**How to test:**

```bash
STREAM_MODE=desktop python -m stream.middleware.app
# Send messages -> cost data should appear in ~/.stream/data/costs.db
```

---

### Phase 5: React Static File Serving

**What we're doing:**
In development, the React frontend runs on its own server (Vite, port 3000). For desktop, we pre-build the React app into static HTML/JS/CSS files and serve them directly from FastAPI.

**How it works:**

```
Development (two servers):
  Browser -> Vite (:3000) -> serves React files
  Browser -> Vite proxy -> FastAPI (:5000) -> serves API

Desktop (one server):
  PyWebView -> FastAPI (:5000) -> serves BOTH React files AND API
```

**What "building" React means:**
When you run `npm run build` in the React project, Vite:
1. Compiles TypeScript to JavaScript
2. Bundles all JS files into a few optimized files
3. Processes Tailwind CSS into a single CSS file
4. Outputs everything to `frontends/react/dist/`:

```
dist/
├── index.html          # The single HTML page (SPA)
├── assets/
│   ├── index-abc123.js  # All React code, minified
│   ├── index-def456.css # All styles, minified
│   └── logo.svg         # Static assets
```

**SPA (Single Page Application) Routing:**
React uses client-side routing. When the user navigates to `/settings`, the browser doesn't request a new page from the server. Instead, React's JavaScript handles the URL change internally.

But if the user refreshes the page at `/settings`, the browser DOES request `/settings` from the server. FastAPI doesn't have a `/settings` route — that's a React route. So we need a "fallback": any URL that isn't an API route (`/v1/*`, `/health`) returns `index.html`, and React's JavaScript takes over from there.

**New file:** `stream/desktop/static_files.py`

**Modified file:** `stream/middleware/app.py`

**How to test:**

```bash
cd frontends/react && npm run build   # Build React
STREAM_MODE=desktop python -m stream.middleware.app
# Browse to http://localhost:5000 -> should see the React UI
```

---

### Phase 6: Desktop Launcher (PyWebView)

**What we're doing:**
This is the main entry point for the desktop app. It:
1. Sets up the environment
2. Starts Ollama (if not already running)
3. Starts FastAPI in a background thread
4. Opens a native window (PyWebView)

**The startup sequence:**

```
User double-clicks STREAM.app
         |
         v
+-----------------------------+
| 1. Set STREAM_MODE=desktop  | Apply desktop defaults
|    Apply config defaults    | (localhost URLs, SQLite, etc.)
+-------------+---------------+
              v
+-----------------------------+
| 2. Check Ollama             | Is it installed? Is it running?
|    Start if needed          | Run "ollama serve" as subprocess
+-------------+---------------+
              v
+-----------------------------+
| 3. Start FastAPI            | uvicorn.run() in a background
|    (background thread)      | thread (daemon=True)
+-------------+---------------+
              v
+-----------------------------+
| 4. Wait for server ready    | Poll http://127.0.0.1:5000/health
|    (poll health endpoint)   | until it returns 200 OK
+-------------+---------------+
              v
+-----------------------------+
| 5. Open PyWebView window    | Native OS window showing
|    -> http://127.0.0.1:5000 | the React frontend
|                             |
|    BLOCKS here until user   | webview.start() blocks the
|    closes the window        | main thread
+-------------+---------------+
              v
+-----------------------------+
| 6. Cleanup                  | Stop Ollama subprocess
|    (window was closed)      | (if we started it)
+-----------------------------+
```

**Why FastAPI runs in a background thread:**
PyWebView's `webview.start()` blocks the main thread (it runs an event loop for the native window). So we start FastAPI's uvicorn server in a separate thread. The `daemon=True` flag means the thread automatically dies when the main thread (PyWebView) exits.

```python
import threading
import uvicorn

# Start FastAPI in background
server_thread = threading.Thread(
    target=uvicorn.run,
    kwargs={
        'app': app,
        'host': '127.0.0.1',
        'port': 5000,
        'log_level': 'warning',  # Quiet in desktop mode
    },
    daemon=True,  # Dies when main thread exits
)
server_thread.start()
```

**New files:**
- `stream/desktop/__init__.py` — package marker
- `stream/desktop/config.py` — desktop environment defaults
- `stream/desktop/ollama_lifecycle.py` — manage Ollama process
- `stream/desktop/main.py` — entry point

**How to test:**

```bash
pip install pywebview
python -m stream.desktop.main
# Should: start Ollama, start FastAPI, open native window with STREAM UI
```

---

### Phase 7: Lakeshore in Desktop Mode

**What we're doing:**
Lakeshore (UIC's HPC cluster) works from ANYWHERE via Globus Compute — it's not campus-only. Desktop users can still authenticate with Globus and use Lakeshore for free GPU inference.

In Docker, the Lakeshore proxy runs as a separate container on port 8001. In desktop mode, we mount the proxy routes directly into the main FastAPI app (same process, no separate service).

**What "mounting" means:**
FastAPI lets you include one app's routes inside another:

```python
# Docker mode: Lakeshore proxy runs separately on port 8001
# Desktop mode: We import and mount its routes into our main app

from stream.proxy.app import router as lakeshore_router
app.include_router(lakeshore_router, prefix="/lakeshore")
```

**Modified files:**
- `stream/middleware/core/lifecycle.py` — mount proxy in desktop mode
- Config defaults: `LAKESHORE_PROXY_URL` points to self in desktop mode

**How to test:**

```bash
python -m stream.desktop.main
# In the UI, Lakeshore tier should appear and work (after Globus auth)
```

---

### Phase 8: PyInstaller Bundling

**What we're doing:**
Bundle everything into a distributable package for each platform.

**The build pipeline:**

```
Step 1: Build React
  cd frontends/react
  npm run build                    -> frontends/react/dist/

Step 2: Run PyInstaller
  pyinstaller stream.spec          -> dist/STREAM.app

Step 3: Install to Applications
  cp -r dist/STREAM.app /Applications/

Combined one-liner:
  pyinstaller stream.spec --noconfirm && cp -r dist/STREAM.app /Applications/
```

**The .spec file:**
PyInstaller uses a `.spec` file (Python script) to configure the build. Key settings:

```python
# stream.spec
a = Analysis(
    ['stream/desktop/main.py'],     # Entry point
    datas=[
        ('stream/desktop/_frontend', 'frontend'),     # React build
        ('stream/gateway/litellm_config.yaml', 'stream/gateway'),
    ],
    hiddenimports=[                 # Dynamic imports PyInstaller can't find
        'uvicorn.logging',
        'litellm.llms.anthropic',
        'litellm.llms.openai',
        'tiktoken_ext.openai_public',
        # ... many more
    ],
    excludes=[                      # Don't bundle these (save space)
        'streamlit', 'matplotlib', 'numpy', 'jupyter', 'pytest',
        'psycopg2',  # We use SQLite in desktop mode
    ],
)
```

**Size estimates:**

| Component | Approximate Size |
|-----------|-----------------|
| Python interpreter + stdlib | ~30 MB |
| pip packages (fastapi, litellm, etc.) | ~70 MB |
| React frontend | ~5 MB |
| PyWebView | ~5 MB |
| **Total app (no models)** | **~110 MB** |
| Ollama (not bundled, installed separately) | ~50 MB |
| Llama 3.2:1b model (downloaded on first run) | ~1.3 GB |
| Llama 3.2:3b model (downloaded on first run) | ~2.0 GB |

**Platform-specific packaging:**

**macOS (.dmg):**
- PyInstaller creates `STREAM.app` bundle
- `create-dmg` wraps it in a `.dmg` with drag-to-Applications

**Windows (.exe installer):**
- Use Inno Setup or NSIS to create an installer
- The installer copies files to Program Files and creates Start Menu shortcut

**Linux (.AppImage):**
- AppImage is a single file that runs on any Linux distribution
- No installation needed — user just makes it executable and runs it

**How to test:**

```bash
pip install pyinstaller
pyinstaller stream.spec --noconfirm && cp -r dist/STREAM.app /Applications/

# Run the bundled app:
open /Applications/STREAM.app                    # macOS (Launchpad)
dist/STREAM.app/Contents/MacOS/STREAM            # macOS (terminal, see logs)
```

---

### Phase 9: First-Run Experience

**What we're doing:**
When a user opens STREAM for the first time, we need to:
1. Check if Ollama is installed → offer to install it
2. Download AI models (~2-4 GB) → show progress
3. Optionally enter cloud API keys → save to config

We already have `OllamaModelManager` that handles model checking and downloading — we just need a desktop-friendly UI for it.

**New file:** `stream/desktop/first_run.py`

**How to test:**

```bash
# Delete ~/.stream/ to simulate a fresh install
rm -rf ~/.stream/
python -m stream.desktop.main
# Should show first-run setup wizard
```

---

## 8. New Files We'll Create

```
stream/desktop/                          # NEW PACKAGE
├── __init__.py                          # Package marker
├── main.py                              # Entry point (PyWebView + uvicorn)
├── config.py                            # Desktop default environment variables
├── ollama_lifecycle.py                  # Detect/install/start/stop Ollama
├── static_files.py                      # Mount React dist/ into FastAPI
├── first_run.py                         # First-run setup wizard
└── model_mapping.py                     # Load model names from litellm_config.yaml

stream/middleware/core/
├── litellm_direct.py                    # Direct litellm library calls (NEW)
└── database_sqlite.py                   # SQLite backend (NEW)

scripts/
└── build_desktop.py                     # Build orchestrator (NEW)

stream.spec                              # PyInstaller configuration (NEW)
```

---

## 9. Existing Files We'll Modify

```
stream/middleware/config.py              # Add safe defaults + STREAM_MODE
stream/middleware/app.py                 # Mount static files in desktop mode
stream/middleware/core/lifecycle.py      # Conditional startup steps
stream/middleware/core/litellm_client.py # Route to direct mode
stream/middleware/core/complexity_judge.py # Route to direct mode
stream/middleware/core/database.py       # SQLite/PostgreSQL switch
stream/middleware/core/tier_health.py    # Fix Docker hostnames
stream/middleware/core/warm_ping.py      # Fix Docker hostnames
pyproject.toml                           # Add desktop dependencies
```

---

## 10. How to Build and Test

### Development Testing (before PyInstaller)

```bash
# 1. Install desktop dependencies
pip install pywebview

# 2. Build React frontend
cd frontends/react && npm run build && cd ../..

# 3. Run desktop mode
python -m stream.desktop.main
```

### Building the Distributable

```bash
# 1. Install build tools
pip install pyinstaller

# 2. Build and install
pyinstaller stream.spec --noconfirm && cp -r dist/STREAM.app /Applications/

# 3. Test:
open /Applications/STREAM.app                    # macOS (Launchpad)
dist/STREAM.app/Contents/MacOS/STREAM            # macOS (terminal, see logs)
```

### Verification Checklist

- [ ] `python -m stream.middleware.app` works without `.env`
- [ ] Health checks work with native Ollama (not Docker)
- [ ] Chat works without LiteLLM Docker service
- [ ] Cost tracking works with SQLite
- [ ] React UI loads at localhost:5000
- [ ] `python -m stream.desktop.main` opens native window
- [ ] Chat works end-to-end in native window
- [ ] Lakeshore tier works (with Globus auth)
- [ ] PyInstaller bundled app launches successfully
- [ ] `docker-compose up` still works (no regression)

---

## 11. Distribution and Licensing

### PyWebView License: BSD 3-Clause

- Can distribute commercially
- Can charge licensing fees
- Can keep your code proprietary
- Can modify PyWebView source
- Must include BSD copyright notice (in About dialog or LICENSE file)
- Don't use PyWebView developers' names to endorse your product

### Distribution Channels

- **Direct download**: Host `.dmg`/`.exe`/`.AppImage` on your website or GitHub Releases
- **GitHub Releases**: Free hosting for open-source projects
- **Mac App Store**: Requires Apple Developer account ($99/year) and review
- **Windows Store**: Requires Microsoft Partner account and review
- **Linux**: Can publish to Flathub, Snap Store, or distribute `.deb`/`.rpm`

### Code Signing (recommended for production)

Without code signing:
- **macOS**: Users see "STREAM is from an unidentified developer" warning
- **Windows**: Users see SmartScreen "Windows protected your PC" warning
- **Linux**: No warnings (but also no verification)

With code signing:
- **macOS**: Apple Developer certificate ($99/year) + notarization
- **Windows**: Code signing certificate (~$100-400/year from DigiCert, Sectigo, etc.)

### Platform-Specific Notes

**macOS:**
- **GPU:** Metal (automatic via Ollama native binary)
- **Location:** `/Applications/STREAM.app`
- **Data:** `~/.stream/`

**Windows:**
- **GPU:** CUDA (if NVIDIA), CPU otherwise
- **Location:** `C:\Program Files\STREAM\`
- **Data:** `~/.stream/` (resolves to `C:\Users\<name>\.stream\`)

**Linux:**
- **GPU:** CUDA (if NVIDIA), ROCm (if AMD), CPU otherwise
- **Formats:** `.AppImage` (universal), `.deb` (Debian/Ubuntu), `.rpm` (Fedora)
- **Data:** `~/.stream/`
- **Note:** May need `libwebkit2gtk` for PyWebView:
  - Ubuntu/Debian: `sudo apt install libwebkit2gtk-4.0-dev`
  - Fedora: `sudo dnf install webkit2gtk4.0-devel`

---

## 12. Future Roadmap

### In-App API Key Entry

Let users enter cloud provider API keys directly in the Settings UI. No `.env` file or config file editing needed. Keys saved to `~/.stream/config.toml` securely.

### Mobile App

The React frontend can be wrapped in:
- **Capacitor** (by Ionic): wraps web apps in a native mobile shell
- **React Native WebView**: similar concept

Both would point to a hosted STREAM backend (cloud deployment).

### Cloud Deployment

Same FastAPI codebase with `STREAM_MODE=server`. Deploy via:
- Docker Compose (current setup, already works)
- Kubernetes (for scaling)
- Cloud VMs (AWS, GCP, Azure)

### Auto-Update

Add an auto-update mechanism so users always have the latest version. Options: Sparkle (Mac), Squirrel (Windows), or custom check-and-download. Note: This is one area where Tauri/Electron have built-in solutions; with PyWebView we'd build a custom updater.

### Sound Effects

Subtle completion chime when response finishes and tab is in background. Off by default, user-controllable toggle in Settings.

### Upgrade to Tauri (if needed)

If we need more desktop-native features (system tray, auto-updater, native menus), we can swap PyWebView for Tauri. The backend code (FastAPI + litellm + routing) stays identical — only the thin launcher layer (~100 lines) changes.

---

## 13. Troubleshooting

### "Ollama not found"

- Make sure Ollama is installed: https://ollama.com/download
- On Mac: check `/usr/local/bin/ollama` exists
- On Linux: `which ollama` should return a path

### "Module not found" in PyInstaller build

- Add the missing module to `hiddenimports` in `stream.spec`
- Common culprits: tiktoken, litellm provider modules, uvicorn internals

### "WebView not available" on Linux

- Install WebKit2GTK: `sudo apt install libwebkit2gtk-4.0-dev` (Ubuntu/Debian)
- Or: `sudo dnf install webkit2gtk4.0-devel` (Fedora)

### Large bundle size

- Check `excludes` in `stream.spec` — are we excluding test/dev dependencies?
- litellm pulls many optional provider SDKs — exclude unused ones

### "CORS error" in desktop mode

- This should not happen since FastAPI serves both API and static files (same origin = no CORS). If it does, check that `CORS_ORIGINS` includes `http://127.0.0.1:5000`

### Ollama using CPU instead of GPU

- Mac: Make sure you're running native Ollama, not Docker Ollama
- Windows: Make sure NVIDIA drivers are installed
- Linux: Make sure CUDA toolkit is installed: `nvidia-smi` should show GPU
- Check: `ollama run llama3.2:3b` — it should show "using Metal" or "using CUDA"

### Port 5000 already in use

A previous instance is still running. Kill it:

```bash
lsof -i :5000      # Find the PID
kill <PID>          # Stop it
```

Or on macOS: Cmd+Option+Esc → Force Quit STREAM.

---

## User Data Directory

```
~/.stream/
├── config.toml        # User config (API keys, preferences)
├── data/
│   └── costs.db       # SQLite cost tracking database
├── logs/
│   ├── stream.log     # Application logs
│   └── ollama.log     # Ollama subprocess logs
└── cache/
    └── tiktoken/      # Token counting cache files
```
