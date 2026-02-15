"""
Serve the React frontend from FastAPI (desktop mode only).

WHY THIS IS NEEDED:
-------------------
In Docker/server mode, the React frontend runs on its own dev server (Vite)
at port 3000. The user opens http://localhost:3000, and Vite serves the React
files. API calls get proxied to FastAPI at port 5000.

In desktop mode, there's no Vite dev server — just one FastAPI server.
So FastAPI needs to serve BOTH the API endpoints AND the React UI files.

HOW IT WORKS:
-------------
1. We point FastAPI at the pre-built React files in frontends/react/dist/.
   These files were created by running "npm run build" in the React project.

2. The dist/ folder contains:
     index.html           — The single HTML page that loads everything
     assets/index-XXX.js  — All React code bundled + minified into one file
     assets/index-XXX.css — All styles bundled into one file
     favicon.svg          — The browser tab icon

3. FastAPI mounts the assets/ folder so browsers can fetch JS, CSS, and fonts.

4. A "catch-all" route returns index.html for any URL that isn't an API route.
   This is called "SPA fallback" — because React is a Single Page Application
   that handles its own routing client-side (e.g., /settings, /about).
   Without this fallback, refreshing the page at /settings would give a 404
   because FastAPI has no /settings route — only React knows about it.

IMPORTANT — ROUTE ORDER:
-------------------------
FastAPI matches routes top-to-bottom. API routes (/v1/*, /health, /docs)
are registered BEFORE the catch-all, so they still work normally.
The catch-all only triggers for routes that don't match any API endpoint.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def find_react_dist() -> Path | None:
    """
    Locate the React build output (dist/ folder).

    We search several possible locations because the dist/ folder might be in
    different places depending on how the app was launched:
      - Development: relative to the project root (frontends/react/dist/)
      - PyInstaller bundle: next to the executable (_internal/frontend/dist/)

    Returns:
        Path to the dist/ folder, or None if not found
    """
    # List of places to look, in priority order
    candidates = [
        # 1. Development: running from project root with "python -m stream.middleware.app"
        #    The dist/ folder is at the top level of the project
        Path(__file__).resolve().parent.parent.parent / "frontends" / "react" / "dist",
        # 2. PyInstaller bundle (Phase 8, future): files are packaged inside the app
        #    __file__ will be inside _internal/, so we look for frontend/dist/ nearby
        Path(__file__).resolve().parent.parent / "frontend" / "dist",
    ]

    for path in candidates:
        if path.is_dir() and (path / "index.html").exists():
            logger.info(f"Found React build at: {path}")
            return path

    logger.warning("React dist/ folder not found — UI will not be available")
    logger.warning("To build: cd frontends/react && npm run build")
    return None


def mount_static_files(app: FastAPI) -> None:
    """
    Mount the React frontend onto the FastAPI app.

    This does two things:
      1. Mounts the assets/ folder at /assets so the browser can fetch
         JavaScript, CSS, fonts, and images.
      2. Adds a catch-all route that returns index.html for any URL
         that isn't already handled by an API route.

    Args:
        app: The FastAPI application instance
    """
    dist_path = find_react_dist()

    if dist_path is None:
        # No React build found — skip mounting.
        # The API still works fine; users just won't see the UI.
        return

    # -------------------------------------------------------------------------
    # Step 1: Mount the assets/ folder
    # -------------------------------------------------------------------------
    # StaticFiles is FastAPI's built-in way to serve files from a directory.
    # When the browser loads index.html, it finds <script src="/assets/index-XXX.js">
    # and <link href="/assets/index-XXX.css"> — those requests need to resolve
    # to the actual files in dist/assets/.
    #
    # "name" is just an internal label for FastAPI's URL generation — not visible to users.
    assets_path = dist_path / "assets"
    if assets_path.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="static-assets")
        logger.info(f"Mounted /assets -> {assets_path}")

    # -------------------------------------------------------------------------
    # Step 2: Serve favicon.svg at the root
    # -------------------------------------------------------------------------
    # Browsers automatically request /favicon.svg (or /favicon.ico) for the
    # tab icon. We serve it directly so it doesn't fall through to the SPA catch-all.
    favicon_path = dist_path / "favicon.svg"
    if favicon_path.exists():

        @app.get("/favicon.svg", include_in_schema=False)
        async def serve_favicon():
            """Serve the browser tab icon."""
            return FileResponse(str(favicon_path), media_type="image/svg+xml")

    # -------------------------------------------------------------------------
    # Step 3: SPA catch-all route
    # -------------------------------------------------------------------------
    # This is the key trick for Single Page Applications:
    # Any URL that doesn't match an API route gets index.html.
    #
    # Example:
    #   GET /settings        -> not an API route -> return index.html
    #   GET /v1/chat/...     -> matches API route -> handled by chat router (not this)
    #   GET /health          -> matches API route -> handled by health router (not this)
    #
    # React's JavaScript (loaded by index.html) then looks at the URL and
    # renders the correct page client-side. This is why it's called a
    # "Single Page Application" — there's literally one HTML page (index.html),
    # and JavaScript handles all the navigation.
    index_html = dist_path / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """
        SPA fallback: return index.html for any unmatched route.

        FastAPI tries all registered routes first. Only if nothing matches
        does this catch-all activate. So API routes (/v1/*, /health, etc.)
        are NOT affected — they still work normally.
        """
        # If the request is for a specific file in dist/ (e.g., a font or image
        # that isn't in assets/), try to serve it directly
        requested_file = dist_path / full_path
        if requested_file.is_file() and requested_file.resolve().is_relative_to(
            dist_path.resolve()
        ):
            return FileResponse(str(requested_file))

        # Otherwise, return index.html and let React handle the routing.
        # "no-store" tells the browser: "Never cache this page — always fetch
        # a fresh copy from the server." Without this, WebKit (macOS) may cache
        # an old response and keep showing it even after the server code changes.
        response = HTMLResponse(index_html.read_text())
        response.headers["Cache-Control"] = "no-store"
        return response

    logger.info("SPA catch-all route registered — React UI is active")
