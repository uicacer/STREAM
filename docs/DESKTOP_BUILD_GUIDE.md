# STREAM Desktop App — Build Guide

## What is PyInstaller?

PyInstaller is a tool that packages a Python application into a **standalone
executable** that users can run without installing Python or any dependencies.

Think of it like a shipping container for your code:

```
WITHOUT PyInstaller:
  User needs: Python 3.11 + pip install litellm fastapi uvicorn httpx ...
  Then runs:  python -m stream.desktop.main

WITH PyInstaller:
  User needs: nothing
  Just runs:  double-click STREAM.app
```

PyInstaller bundles everything into one package:
- Your Python code (all `.py` files)
- The Python interpreter itself
- All pip dependencies (litellm, fastapi, etc.)
- Data files you specify (React build, config files, icons)

The result is a native app:
- **macOS**: `STREAM.app` (a `.app` bundle, like any Mac app)
- **Windows**: `STREAM.exe` (a standard Windows executable)
- **Linux**: `STREAM` (an ELF binary)

## Key Concept: The `.spec` File

PyInstaller uses a **spec file** (`stream.spec`) as its recipe. This file tells
PyInstaller exactly what to include in the bundle:

```
stream.spec
├── Analysis       → Which Python files to include (entry point + auto-detected imports)
├── hiddenimports  → Modules PyInstaller can't auto-detect (e.g., litellm internals)
├── datas          → Non-Python files to bundle (React build, YAML configs, icons)
├── EXE            → How to build the executable
└── BUNDLE         → macOS-specific: wraps everything into a .app with icon + metadata
```

You only edit `stream.spec` when you need to add new data files or hidden imports.
For normal code changes, you don't touch it.

## How to Rebuild After Code Changes

### Quick Reference

```bash
# If you changed Python code only:
pyinstaller stream.spec --noconfirm && cp -r dist/STREAM.app /Applications/

# If you changed React frontend code:
cd frontends/react && npm run build && cd ../..
pyinstaller stream.spec --noconfirm && cp -r dist/STREAM.app /Applications/

# Test the result:
open /Applications/STREAM.app          # macOS (or find it in Launchpad)
# or
dist/STREAM.app/Contents/MacOS/STREAM  # see console output for debugging
```

### Step-by-Step

**Step 1: Build the React frontend (only if you changed frontend code)**

```bash
cd frontends/react
npm run build
cd ../..
```

This compiles TypeScript + React into optimized static files in `frontends/react/dist/`.
PyInstaller bundles this folder into the app so FastAPI can serve the UI.

If you only changed Python files, skip this step — the existing `dist/` folder
gets bundled as-is.

**Step 2: Run PyInstaller**

```bash
pyinstaller stream.spec --noconfirm
```

- `stream.spec` — The recipe file at the project root
- `--noconfirm` — Overwrite the previous build without asking

This takes about 60–90 seconds. The output goes to:
- `dist/STREAM.app` — The final macOS app bundle
- `build/stream/` — Intermediate cache (speeds up subsequent builds, safe to ignore)

**Step 3: Install to Applications (makes it appear in Launchpad)**

```bash
cp -r dist/STREAM.app /Applications/
```

This copies the built app into `/Applications`, which is where macOS looks for
apps to show in Launchpad (the grid of app icons). Without this step, the app
only exists in your project's `dist/` folder and won't appear in Launchpad or
Spotlight search.

You can combine the build and install into one command:

```bash
pyinstaller stream.spec --noconfirm && cp -r dist/STREAM.app /Applications/
```

**Note:** If an older version is already in `/Applications`, `cp -r` overwrites
it. This is what you want — it replaces the old build with the new one.

**Step 4: Test**

```bash
# Option A: Double-click in Finder or open from Launchpad
open /Applications/STREAM.app

# Option B: Run from terminal (see logs and errors in console)
dist/STREAM.app/Contents/MacOS/STREAM
```

Option B is better for debugging because you see all print statements and errors
directly in your terminal. Option A just shows the app window.

## Build Output Structure

```
dist/STREAM.app/
├── Contents/
│   ├── MacOS/
│   │   └── STREAM              ← The actual executable (runs when you open the app)
│   ├── Resources/
│   │   ├── icon.icns           ← App icon (Dock, Finder, Spotlight)
│   │   └── stream/             ← Your Python code + dependencies
│   │       ├── middleware/      ← FastAPI backend
│   │       ├── desktop/         ← Desktop-specific code (PyWebView, Ollama lifecycle)
│   │       ├── frontend/dist/   ← React build (HTML, JS, CSS)
│   │       └── gateway/         ← litellm_config.yaml
│   ├── Frameworks/              ← System libraries
│   └── Info.plist               ← macOS app metadata (name, version, icon)
```

## Updating the App Icon

The app icon is generated from `frontends/react/public/favicon.svg` and stored
as `assets/icon.icns` (macOS multi-resolution icon format).

To regenerate the icon (e.g., after changing the SVG):

```bash
# 1. Create a temporary iconset folder
mkdir -p /tmp/icon.iconset

# 2. Convert SVG to PNGs at all required sizes
#    (rsvg-convert comes from librsvg: brew install librsvg)
for size in 16 32 64 128 256 512 1024; do
  rsvg-convert -w $size -h $size frontends/react/public/favicon.svg \
    > /tmp/icon.iconset/icon_${size}x${size}.png
done

# 3. Create the @2x (Retina) variants by copying the next size up
cp /tmp/icon.iconset/icon_32x32.png   /tmp/icon.iconset/icon_16x16@2x.png
cp /tmp/icon.iconset/icon_64x64.png   /tmp/icon.iconset/icon_32x32@2x.png
cp /tmp/icon.iconset/icon_256x256.png /tmp/icon.iconset/icon_128x128@2x.png
cp /tmp/icon.iconset/icon_512x512.png /tmp/icon.iconset/icon_256x256@2x.png
cp /tmp/icon.iconset/icon_1024x1024.png /tmp/icon.iconset/icon_512x512@2x.png

# 4. Convert iconset → .icns (macOS built-in tool)
iconutil -c icns /tmp/icon.iconset -o assets/icon.icns

# 5. Rebuild the app
pyinstaller stream.spec --noconfirm
```

## What About Windows and Linux?

**PyInstaller builds are platform-specific.** You can only build for the OS
you're currently running on:

| Build on  | Produces          | Can't produce       |
|-----------|-------------------|---------------------|
| macOS     | `STREAM.app`      | .exe or Linux bin   |
| Windows   | `STREAM.exe`      | .app or Linux bin   |
| Linux     | `STREAM` (binary) | .app or .exe        |

This is because PyInstaller bundles the **native Python interpreter** and
**OS-specific libraries** for the current platform. A macOS Python can't
run on Windows, so it can't create a Windows bundle.

### To build on Windows:

1. Install Python 3.11+ and the project dependencies on a Windows machine
2. Copy the project files (or clone the git repo)
3. Run `pip install pyinstaller` (or `uv pip install pyinstaller`)
4. The `stream.spec` will need a few Windows-specific changes:
   - Remove the `BUNDLE(...)` section (that's macOS-only for `.app` bundles)
   - Change `icon.icns` to `icon.ico` (Windows uses `.ico` format, not `.icns`)
   - Adjust paths if needed (backslashes vs forward slashes)
5. Run `pyinstaller stream.spec --noconfirm`
6. Result: `dist/STREAM/STREAM.exe`

### To build on Linux:

Same process as Windows, but:
- No icon format change needed (Linux desktop entries use PNG)
- Remove the `BUNDLE(...)` section
- Result: `dist/STREAM/STREAM` (a Linux ELF binary)

### Cross-platform builds (future):

For automated multi-platform builds, you can use **GitHub Actions**:
- A macOS runner builds the `.app`
- A Windows runner builds the `.exe`
- A Linux runner builds the binary
- All three artifacts get uploaded to a GitHub Release

This is a common pattern for open-source desktop apps. We can set this up
when you're ready to distribute to users on other platforms.

## Troubleshooting

### "ModuleNotFoundError: No module named 'xyz'"

PyInstaller sometimes misses modules it can't auto-detect. Fix:
1. Open `stream.spec`
2. Add the missing module to `hiddenimports`:
   ```python
   hiddenimports=[
       'existing_module',
       'xyz',  # Add the missing one
   ],
   ```
3. Rebuild: `pyinstaller stream.spec --noconfirm`

### "FileNotFoundError: .../some_file.json"

A data file (not Python code) that your app needs at runtime isn't bundled. Fix:
1. Open `stream.spec`
2. Add the file to `datas`:
   ```python
   datas=[
       ('path/to/file.json', 'destination/folder'),
   ],
   ```
3. Rebuild

### App launches but shows blank/JSON instead of React UI

The React build (`frontends/react/dist/`) is missing or outdated:
```bash
cd frontends/react && npm run build && cd ../..
pyinstaller stream.spec --noconfirm
```

### Port 5000 already in use

A previous instance is still running. Kill it:
```bash
lsof -i :5000      # Find the PID
kill <PID>          # Stop it
```
Or on macOS: Cmd+Option+Esc → Force Quit STREAM.
