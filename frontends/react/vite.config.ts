/**
 * Vite Configuration
 * ==================
 *
 * This file configures Vite, our frontend build tool and dev server.
 *
 * KEY CONCEPT: Development Proxy
 * ------------------------------
 * In development, we run TWO separate servers:
 *   1. Frontend (Vite): http://localhost:3000 - serves React app
 *   2. Backend (FastAPI): http://localhost:5000 - serves API
 *
 * Problem: Browser security (CORS) blocks frontend from calling a different port.
 * Solution: Vite's proxy intercepts API requests and forwards them to the backend.
 *
 * How it works:
 *   Browser → localhost:3000/v1/chat → Vite Proxy → localhost:5000/v1/chat → Backend
 *
 * The frontend code just calls "/v1/chat" (no port), and Vite handles the routing.
 *
 * PRODUCTION NOTE:
 * In production, both frontend and backend typically run behind a reverse proxy
 * (like Nginx) on the same domain, so this proxy config is only for development.
 *
 * BACKEND PORT:
 * The backend is expected to run on port 5000. If you change the backend port,
 * update the `target` URLs below to match.
 */

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      // Allows imports like: import { Button } from '@/components/Button'
      // Instead of: import { Button } from '../../../components/Button'
      '@': path.resolve(__dirname, './src'),
    },
  },

  server: {
    // Port for the Vite dev server (frontend)
    port: 3000,

    /**
     * Proxy Configuration
     * -------------------
     * Maps URL paths to the backend server.
     * Any request starting with these paths gets forwarded to the backend.
     *
     * Example: fetch('/v1/chat') → proxied to http://localhost:5000/v1/chat
     */
    proxy: {
      // Main API routes (chat, config, auth, etc.)
      '/v1': {
        target: 'http://localhost:5000',  // Backend server URL
        changeOrigin: true,                // Needed for virtual hosted sites
      },

      // Health check endpoints (tier availability status)
      '/health': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
