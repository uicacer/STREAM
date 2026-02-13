# STREAM Desktop App Packaging Plan

## Executive Summary

Package STREAM as a standalone desktop application that:
- Works on Mac (with Metal GPU), Windows (with CUDA), and Linux
- Requires no Docker installation
- Auto-detects and uses available GPU
- Single installer experience

---

## Architecture Comparison

### Current (Docker-based)
```
┌─────────────────────────────────────────────────────────┐
│  Docker Desktop (required)                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────────┐ │
│  │ Postgres │ │ Ollama  │ │ LiteLLM │ │  Middleware   │ │
│  │  :5432   │ │ :11434  │ │  :4000  │ │    :5000      │ │
│  └─────────┘ └─────────┘ └─────────┘ └───────────────┘ │
└─────────────────────────────────────────────────────────┘
         ❌ No Mac GPU    ✅ Windows/Linux GPU
```

### Proposed (Native Desktop)
```
┌─────────────────────────────────────────────────────────┐
│  STREAM Desktop App (Tauri)                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │  React Frontend (embedded WebView)              │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Python Backend (PyInstaller bundle)            │   │
│  │  - FastAPI middleware                           │   │
│  │  - SQLite (replaces Postgres)                   │   │
│  │  - LiteLLM (embedded)                           │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Ollama (native binary)                         │   │
│  │  - Auto-downloads models on first run           │   │
│  │  - Uses Metal (Mac) / CUDA (Win/Linux)          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         ✅ Mac GPU (Metal)    ✅ Windows/Linux GPU
```

---

## Component Packaging Strategy

### 1. Frontend (React → Tauri WebView)

**Current:** Vite dev server + React SPA
**Packaged:** Built React bundle served by Tauri WebView

```bash
# Build steps
cd frontends/react
npm run build          # Creates dist/ folder
# Tauri embeds dist/ into native app
```

**Changes needed:**
- Add `tauri.conf.json` configuration
- Update API URLs to use localhost (already works)
- No code changes to React components

**Size:** ~5MB (JS bundle + assets)

---

### 2. Backend (Python → PyInstaller)

**Current:** FastAPI running via uvicorn
**Packaged:** Single executable via PyInstaller

```bash
# Build steps
pyinstaller --onefile \
  --add-data "stream/gateway/litellm_config.yaml:stream/gateway" \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.protocols.http \
  stream/middleware/app.py
```

**Changes needed:**
- Replace PostgreSQL with SQLite (simpler, no separate service)
- Embed LiteLLM config
- Add startup checks for Ollama

**Size:** ~100MB (Python + dependencies)

---

### 3. Local LLM (Ollama → Native Binary)

**Current:** Docker container (ollama/ollama:0.14.1)
**Packaged:** Native Ollama binary bundled with app

Ollama releases pre-built binaries:
- macOS: `ollama-darwin-arm64` (M-series) or `ollama-darwin-amd64` (Intel)
- Windows: `ollama-windows-amd64.exe`
- Linux: `ollama-linux-amd64`

**Download URLs:** https://github.com/ollama/ollama/releases

**Changes needed:**
- Download correct binary for target platform during build
- Auto-start Ollama on app launch
- Auto-download models on first run (llama3.2:1b, llama3.2:3b)

**Size:** ~50MB (binary) + ~5GB (models, downloaded on first run)

---

### 4. Database (PostgreSQL → SQLite)

**Current:** PostgreSQL container for cost tracking
**Packaged:** SQLite file in app data directory

**Changes needed:**
```python
# Before (database.py)
DATABASE_URL = "postgresql://user:pass@host:5432/db"

# After
import sqlite3
DATABASE_PATH = os.path.join(APP_DATA_DIR, "stream.db")
```

**Migration:**
- Create SQLite schema matching current PostgreSQL tables
- Update database.py to use SQLite
- Remove psycopg2 dependency

**Size:** ~1MB (schema + data)

---

### 5. LiteLLM Gateway (Embedded)

**Current:** Separate Docker container
**Packaged:** Direct library calls (no separate server)

**Option A:** Keep as subprocess
```python
# Start LiteLLM in background
subprocess.Popen(["litellm", "--config", config_path, "--port", "4000"])
```

**Option B:** Direct library calls (recommended)
```python
# Skip the proxy, call litellm directly
from litellm import completion
response = completion(model="claude-3-5-sonnet", messages=[...])
```

**Changes needed:** Refactor `litellm_client.py` to use direct calls

---

## Build Pipeline

### Directory Structure
```
stream-desktop/
├── src-tauri/              # Tauri (Rust) native shell
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
│       └── main.rs         # Manages subprocess lifecycle
├── frontend/               # React (copied from frontends/react)
│   └── dist/               # Built React app
├── backend/                # Python backend bundle
│   └── stream-backend      # PyInstaller executable
├── ollama/                 # Ollama binary (per-platform)
│   └── ollama              # Native binary
└── models/                 # Downloaded on first run
    └── .ollama/            # Model cache
```

### Build Commands

```bash
# 1. Build React frontend
cd frontends/react && npm run build

# 2. Build Python backend
pyinstaller --onefile stream/middleware/app.py -n stream-backend

# 3. Download Ollama binary (per platform)
curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-darwin-arm64 -o ollama/ollama

# 4. Build Tauri app (bundles everything)
cd src-tauri && cargo tauri build
```

### Output
```
target/release/bundle/
├── macos/STREAM.app        # macOS app bundle (~200MB + models)
├── dmg/STREAM.dmg          # macOS installer
├── msi/STREAM.msi          # Windows installer
└── deb/stream.deb          # Linux package
```

---

## First-Run Experience

```
┌─────────────────────────────────────────────────────────┐
│                    Welcome to STREAM                     │
│                                                         │
│  Setting up your local AI assistant...                  │
│                                                         │
│  [============================] 45%                     │
│  Downloading Llama 3.2 1B model (1.3 GB)               │
│                                                         │
│  ℹ️ This only happens once. Models are cached locally.  │
│                                                         │
│  GPU Detected: Apple M2 Pro (Metal)                    │
│  Local inference will use GPU acceleration.             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Platform-Specific Considerations

### macOS
- **GPU:** Metal (automatic via Ollama native binary)
- **Signing:** Requires Apple Developer certificate for distribution
- **Notarization:** Required for Gatekeeper approval
- **Location:** `/Applications/STREAM.app`
- **Data:** `~/Library/Application Support/STREAM/`

### Windows
- **GPU:** CUDA (if NVIDIA), CPU otherwise
- **Signing:** Code signing certificate recommended
- **Location:** `C:\Program Files\STREAM\`
- **Data:** `%APPDATA%\STREAM\`

### Linux
- **GPU:** CUDA (if NVIDIA), ROCm (if AMD), CPU otherwise
- **Formats:** .deb, .rpm, .AppImage
- **Location:** `/opt/stream/` or `/usr/local/bin/`
- **Data:** `~/.local/share/stream/`

---

## Size Estimates

| Component | Size |
|-----------|------|
| Tauri shell | ~10 MB |
| React bundle | ~5 MB |
| Python backend | ~100 MB |
| Ollama binary | ~50 MB |
| **Total (no models)** | **~165 MB** |
| Llama 3.2 1B model | ~1.3 GB |
| Llama 3.2 3B model | ~2.0 GB |
| **Total (with models)** | **~3.5 GB** |

---

## Implementation Phases

### Phase 1: Simplify Backend (1-2 weeks)
- [ ] Replace PostgreSQL with SQLite
- [ ] Refactor LiteLLM to direct calls
- [ ] Remove Docker dependencies from middleware
- [ ] Test backend runs standalone

### Phase 2: Tauri Integration (1 week)
- [ ] Create Tauri project structure
- [ ] Configure React frontend embedding
- [ ] Add subprocess management for backend + Ollama
- [ ] Implement first-run model download UI

### Phase 3: Platform Builds (1 week)
- [ ] Set up CI/CD for multi-platform builds
- [ ] Test on macOS (Intel + M-series)
- [ ] Test on Windows (with/without NVIDIA)
- [ ] Test on Linux (Ubuntu, Fedora)

### Phase 4: Distribution (ongoing)
- [ ] Code signing (Mac + Windows)
- [ ] Auto-update mechanism
- [ ] Crash reporting
- [ ] Usage analytics (opt-in)

---

## Alternative: Simpler Electron Approach

If Tauri is too complex, Electron is simpler but heavier:

```
┌─────────────────────────────────────────────────────────┐
│  STREAM.app (Electron)                                  │
│  ├── Chromium (~120 MB)                                │
│  ├── Node.js runtime                                    │
│  ├── React frontend                                     │
│  └── Spawns:                                            │
│      ├── Python backend (PyInstaller)                  │
│      └── Ollama (native binary)                        │
└─────────────────────────────────────────────────────────┘
```

**Pros:** More familiar tooling, larger ecosystem
**Cons:** ~150MB larger than Tauri, more memory usage

---

## Recommendation

**Start with Phase 1** (simplify backend) regardless of frontend choice. This makes STREAM more portable and easier to deploy in any environment.

Then evaluate:
- **Tauri** if you want smallest size + best performance
- **Electron** if you want faster development + more libraries

Both will give you full GPU access on all platforms.
