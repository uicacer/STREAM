# STREAM Production Deployment Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [Understanding the Current Architecture](#2-understanding-the-current-architecture)
3. [The Authentication Challenge](#3-the-authentication-challenge)
4. [Deployment Scenarios](#4-deployment-scenarios)
5. [Solution 1: Service Account Model](#5-solution-1-service-account-model)
6. [Solution 2: OAuth Redirect Flow](#6-solution-2-oauth-redirect-flow)
7. [Solution 3: Desktop Application](#7-solution-3-desktop-application)
8. [Choosing the Right Approach](#8-choosing-the-right-approach)
9. [Production Checklist](#9-production-checklist)
10. [Glossary](#10-glossary)

---

## 1. Introduction

This document explains how to deploy STREAM in different production environments. It's written for developers who may not have deployed containerized applications before, so we'll explain concepts as we go.

### What is "Production"?

**Development** = Running on your laptop for testing. It's okay if things break.

**Production** = Running for real users. Must be reliable, secure, and always available.

### Current Development Setup

```
Your Laptop (Development)
├── Docker Desktop
│   ├── postgres (database)
│   ├── ollama (local AI)
│   ├── lakeshore-proxy (HPC connection)
│   ├── litellm (AI gateway)
│   └── middleware (routing logic)
│
└── Host Machine (not in Docker)
    └── streamlit (web interface) ← Runs here for OAuth
```

**Why is Streamlit outside Docker?**

The Globus OAuth flow requires opening a browser and receiving a callback. This only works when the application runs on the same machine as the browser - your laptop. Docker containers are isolated and can't open your browser.

---

## 2. Understanding the Current Architecture

### How Requests Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT DEVELOPMENT FLOW                         │
└─────────────────────────────────────────────────────────────────────────┘

  YOU (Developer)
       │
       │ 1. Open browser to http://localhost:8501
       ▼
┌──────────────┐
│  Streamlit   │  ← Running on your laptop (NOT in Docker)
│  Frontend    │
└──────┬───────┘
       │ 2. Send chat message
       ▼
┌──────────────┐
│  Middleware  │  ← Running in Docker
│  (Routing)   │
└──────┬───────┘
       │ 3. Route based on complexity
       ▼
┌──────────────────────────────────────────────────────┐
│                    WHERE IT GOES                      │
├──────────────┬──────────────────┬────────────────────┤
│    LOCAL     │    LAKESHORE     │       CLOUD        │
│   (Ollama)   │  (HPC via Globus)│   (Claude/GPT)     │
│     FREE     │    LOW COST      │       PAID         │
└──────────────┴──────────────────┴────────────────────┘
```

### The Docker Network

Docker creates an isolated network where containers can talk to each other using service names:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DOCKER NETWORK (stream_network)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   middleware ──────────────► litellm ──────────────► ollama             │
│       │                         │                                        │
│       │                         │                                        │
│       └──────────────► lakeshore-proxy ──────────► Globus Compute       │
│                                 │                   (Internet)           │
│                                 │                                        │
│   postgres ◄────────────────────┘                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Container names ARE hostnames inside Docker:
- middleware can reach litellm at "http://litellm:4000"
- middleware can reach lakeshore-proxy at "http://lakeshore-proxy:8001"
```

---

## 3. The Authentication Challenge

### Why Authentication is Tricky

Globus Compute uses **OAuth 2.0** for authentication. Here's a simplified explanation:

```
OAUTH 2.0 FLOW (Simplified)
════════════════════════════

1. App says: "I need to access Globus on behalf of this user"
2. Browser opens: "Please log in to Globus"
3. User logs in and clicks "Allow"
4. Globus redirects browser back to app with a secret code
5. App exchanges code for access tokens
6. App can now use Globus API

The KEY REQUIREMENT: Steps 2-4 happen IN THE BROWSER
```

### The Problem with Docker

```
DEVELOPMENT (Works ✓)
═════════════════════

Your Laptop
    │
    ├── Browser ◄─────────────┐
    │                          │
    └── Streamlit ─────────────┘
        (can open browser,
         receive callback)


PRODUCTION SERVER (Broken ✗)
════════════════════════════

Cloud Server (headless - no monitor, no browser)
    │
    └── Docker
        └── Streamlit ──── ??? ──── Can't open browser!

User's Computer
    │
    └── Browser ◄─── Connected via internet, NOT localhost
```

**The core problem**: OAuth callbacks go to `localhost`, but on a cloud server, `localhost` is the server, not the user's computer.

---

## 4. Deployment Scenarios

### Scenario A: Single-User Development Machine

**Example**: Your laptop, a lab workstation

**Current setup works perfectly!** The user IS the machine operator.

```
┌─────────────────────────────────────────┐
│         SINGLE-USER WORKSTATION         │
├─────────────────────────────────────────┤
│                                         │
│  User sits at this computer             │
│       │                                 │
│       ▼                                 │
│  ┌─────────┐    ┌──────────────────┐   │
│  │ Browser │◄──►│ Streamlit (host) │   │
│  └─────────┘    └────────┬─────────┘   │
│                          │              │
│                 ┌────────▼────────┐    │
│                 │ Docker Services │    │
│                 └─────────────────┘    │
│                                         │
│  OAuth: Browser opens locally ✓         │
│                                         │
└─────────────────────────────────────────┘
```

### Scenario B: Cloud Server (Multi-User)

**Example**: AWS EC2, DigitalOcean Droplet, University VM

**Multiple users access via internet**. Server has no display.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLOUD SERVER DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   User A's Computer          User B's Computer                       │
│        │                          │                                  │
│        └──────────┬───────────────┘                                 │
│                   │ Internet                                         │
│                   ▼                                                  │
│   ┌───────────────────────────────────────────────┐                 │
│   │              CLOUD SERVER                      │                 │
│   │  ┌─────────────────────────────────────────┐  │                 │
│   │  │           Docker Services               │  │                 │
│   │  │  ┌──────────┐  ┌──────────┐            │  │                 │
│   │  │  │ Frontend │  │Middleware│  ...       │  │                 │
│   │  │  └──────────┘  └──────────┘            │  │                 │
│   │  └─────────────────────────────────────────┘  │                 │
│   │                                                │                 │
│   │  No browser here! OAuth redirect goes where?  │                 │
│   └───────────────────────────────────────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Scenario C: Desktop Application

**Example**: Electron app, Tauri app (like VS Code, Slack, Discord)

**User installs and runs locally**. App bundles everything.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DESKTOP APPLICATION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   User's Computer                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    STREAM Desktop App                        │   │
│   │  ┌─────────────────────────────────────────────────────┐    │   │
│   │  │              App Window (Chromium)                   │    │   │
│   │  │  ┌──────────────────────────────────────────────┐   │    │   │
│   │  │  │           Streamlit UI                        │   │    │   │
│   │  │  └──────────────────────────────────────────────┘   │    │   │
│   │  └─────────────────────────────────────────────────────┘    │   │
│   │                          │                                   │   │
│   │                          ▼                                   │   │
│   │  ┌─────────────────────────────────────────────────────┐    │   │
│   │  │           Docker Desktop (bundled or required)       │    │   │
│   │  │  ┌─────────┐ ┌──────────┐ ┌─────────┐              │    │   │
│   │  │  │ Ollama  │ │Middleware│ │ Proxy   │  ...         │    │   │
│   │  │  └─────────┘ └──────────┘ └─────────┘              │    │   │
│   │  └─────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   OAuth: Opens system browser, callback to localhost ✓               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Solution 1: Service Account Model

### Concept

Instead of each user authenticating with Globus, an **administrator authenticates once** during deployment. All users share this authentication.

```
SERVICE ACCOUNT MODEL
═════════════════════

DEPLOYMENT TIME (Once):
  Admin authenticates → Tokens saved to server

RUNTIME (Every request):
  Any user → Server uses admin's tokens → Globus Compute
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SERVICE ACCOUNT ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1: Admin Setup (One-time)                                     │
│  ════════════════════════════════                                   │
│                                                                      │
│  Admin's Computer                                                    │
│       │                                                              │
│       │ 1. Run: globus-compute-endpoint configure                   │
│       │ 2. Authenticate in browser                                   │
│       │ 3. Copy ~/.globus_compute/storage.db to server              │
│       ▼                                                              │
│  Cloud Server                                                        │
│       │                                                              │
│       └── /secure/globus_credentials/storage.db                     │
│                                                                      │
│                                                                      │
│  STEP 2: Runtime (Every request)                                    │
│  ═══════════════════════════════                                    │
│                                                                      │
│  Any User ──► Cloud Server ──► Uses admin's credentials ──► Globus  │
│                                                                      │
│  Users don't need to authenticate individually!                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Changes

**docker-compose.prod.yml**:
```yaml
services:
  lakeshore-proxy:
    # ... existing config ...
    volumes:
      # Mount pre-authenticated credentials (read-only for security)
      - /secure/globus_credentials:/root/.globus_compute:ro
    environment:
      # Disable interactive authentication
      - GLOBUS_COMPUTE_INTERACTIVE_AUTH=false
```

**Deployment script**:
```bash
#!/bin/bash
# deploy.sh - Run on admin's machine

# 1. Authenticate locally
echo "Authenticating with Globus..."
python -c "from globus_compute_sdk import Client; Client()"

# 2. Copy credentials to server
echo "Copying credentials to server..."
scp -r ~/.globus_compute user@server:/secure/globus_credentials

# 3. Start services
echo "Starting STREAM services..."
ssh user@server "cd /app && docker-compose -f docker-compose.prod.yml up -d"
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Simple setup | Single point of failure |
| Users don't need Globus accounts | Can't track per-user usage |
| Works on headless servers | Token refresh requires admin |
| No OAuth complexity | Security: shared credentials |

### When to Use

- Internal university deployment
- Research group with trusted users
- Proof-of-concept deployments
- When simplicity > fine-grained access control

---

## 6. Solution 2: OAuth Redirect Flow

### Concept

Each user authenticates individually. The OAuth callback is configured to return to the cloud server's public URL instead of localhost.

```
OAUTH REDIRECT FLOW
═══════════════════

1. User clicks "Login with Globus" in STREAM
2. Browser redirects to Globus login page
3. User authenticates
4. Globus redirects to https://stream.university.edu/auth/callback
5. Server receives auth code, exchanges for tokens
6. User's session is now authenticated
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OAUTH REDIRECT ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User's Browser                      Cloud Server                    │
│       │                                   │                          │
│       │ 1. Click "Login"                  │                          │
│       │─────────────────────────────────►│                          │
│       │                                   │                          │
│       │ 2. Redirect to Globus             │                          │
│       │◄─────────────────────────────────│                          │
│       │                                   │                          │
│       ▼                                   │                          │
│  ┌──────────┐                            │                          │
│  │  Globus  │ 3. User logs in            │                          │
│  │  Auth    │                            │                          │
│  └────┬─────┘                            │                          │
│       │                                   │                          │
│       │ 4. Redirect to callback URL       │                          │
│       │   https://stream.edu/callback     │                          │
│       │─────────────────────────────────►│                          │
│       │                                   │ 5. Exchange code         │
│       │                                   │    for tokens            │
│       │ 6. Session authenticated          │                          │
│       │◄─────────────────────────────────│                          │
│       │                                   │                          │
└───────┴───────────────────────────────────┴──────────────────────────┘
```

### Requirements

1. **Register a Globus Application**
   - Go to https://app.globus.org/settings/developers
   - Create new app
   - Set redirect URI: `https://your-domain.edu/auth/callback`

2. **SSL Certificate** (HTTPS required for OAuth)
   - Use Let's Encrypt (free)
   - Or university-provided certificate

3. **Public Domain Name**
   - `stream.cs.university.edu`
   - Or use a reverse proxy

### Implementation Outline

```python
# New file: stream/middleware/routes/auth.py

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
import globus_sdk

router = APIRouter()

# Globus app credentials (from environment)
CLIENT_ID = os.getenv("GLOBUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("GLOBUS_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GLOBUS_REDIRECT_URI")

@router.get("/auth/login")
async def login():
    """Redirect user to Globus login page."""
    client = globus_sdk.ConfidentialAppAuthClient(CLIENT_ID, CLIENT_SECRET)

    # Generate authorization URL
    auth_url = client.oauth2_get_authorize_url(
        redirect_uri=REDIRECT_URI,
        requested_scopes=["openid", "profile", "email",
                         "urn:globus:auth:scope:compute.api.globus.org:all"]
    )

    return RedirectResponse(auth_url)

@router.get("/auth/callback")
async def callback(request: Request, code: str):
    """Handle OAuth callback from Globus."""
    client = globus_sdk.ConfidentialAppAuthClient(CLIENT_ID, CLIENT_SECRET)

    # Exchange code for tokens
    token_response = client.oauth2_exchange_code_for_tokens(code)

    # Store tokens in user session
    # ... (session management code)

    return RedirectResponse("/")
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Per-user authentication | More complex setup |
| Proper access control | Requires Globus app registration |
| Industry standard | Requires SSL certificate |
| Audit trail per user | More code to maintain |

### When to Use

- Production deployment for many users
- When you need per-user tracking
- Enterprise/institutional deployments
- When security compliance is required

---

## 7. Solution 3: Desktop Application

### Concept

Package STREAM as a desktop application that users install and run locally. Docker services run in the background.

### Technology Options

| Framework | Language | Size | Notes |
|-----------|----------|------|-------|
| **Electron** | JavaScript | ~150MB | Most mature, used by VS Code, Slack |
| **Tauri** | Rust | ~10MB | Smaller, uses system webview |
| **PyInstaller + PyWebview** | Python | ~50MB | Stays in Python ecosystem |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DESKTOP APP ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STREAM.app / STREAM.exe                                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │              Native Window (Chromium/WebView)            │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │                                                    │  │  │  │
│  │  │  │              Streamlit Web Interface               │  │  │  │
│  │  │  │                                                    │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                            │                                   │  │
│  │                            │ HTTP (localhost)                  │  │
│  │                            ▼                                   │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │              Background Services                         │  │  │
│  │  │                                                          │  │  │
│  │  │  Option A: Docker Desktop (user must install)            │  │  │
│  │  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌───────────┐    │  │  │
│  │  │  │ Postgres│ │ Ollama   │ │ LiteLLM │ │ Middleware│    │  │  │
│  │  │  └─────────┘ └──────────┘ └─────────┘ └───────────┘    │  │  │
│  │  │                                                          │  │  │
│  │  │  Option B: Embedded services (larger app size)           │  │  │
│  │  │  - SQLite instead of Postgres                            │  │  │
│  │  │  - Ollama binary bundled                                 │  │  │
│  │  │  - Python services as executables                        │  │  │
│  │  │                                                          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  OAuth: Works! Browser opens on user's machine, callback to localhost│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Simplified Desktop Architecture (Recommended)

For a simpler desktop app, you can skip Docker entirely:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 SIMPLIFIED DESKTOP (No Docker)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STREAM.app                                                          │
│  ├── streamlit (bundled Python)                                     │
│  ├── middleware (bundled Python)                                    │
│  ├── sqlite (instead of Postgres)                                   │
│  └── ollama (native binary for Mac/Windows/Linux)                   │
│                                                                      │
│  Lakeshore & Cloud: Still accessed via internet                      │
│  Local AI: Ollama runs as subprocess                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Pros and Cons

| Pros | Cons |
|------|------|
| OAuth works perfectly | Complex packaging |
| No server costs | Must support Mac/Win/Linux |
| Data stays local | Updates are harder |
| Works offline (Local tier) | Large download size |

### When to Use

- Single-user research tool
- When data privacy is critical
- Offline-capable requirement
- Distributing to non-technical users

---

## 8. Choosing the Right Approach

### Decision Matrix

| Factor | Service Account | OAuth Redirect | Desktop App |
|--------|-----------------|----------------|-------------|
| **Setup Complexity** | Low | High | Medium |
| **Per-User Auth** | No | Yes | Yes |
| **Server Required** | Yes | Yes | No |
| **Offline Support** | No | No | Yes (Local tier) |
| **Multi-User** | Yes | Yes | No |
| **OAuth Works** | N/A (bypassed) | Yes | Yes |
| **Best For** | Internal/Research | Enterprise | Individual |

### Recommendation by Use Case

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DECISION FLOWCHART                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  START: Who will use STREAM?                                         │
│                │                                                     │
│                ▼                                                     │
│  ┌─────────────────────────┐                                        │
│  │ Single user (just me)?  │                                        │
│  └───────────┬─────────────┘                                        │
│              │                                                       │
│      Yes ◄───┴───► No                                               │
│       │              │                                               │
│       ▼              ▼                                               │
│  ┌─────────┐  ┌─────────────────────┐                               │
│  │ DESKTOP │  │ Is it a trusted     │                               │
│  │   APP   │  │ internal group?     │                               │
│  └─────────┘  └──────────┬──────────┘                               │
│                          │                                           │
│                  Yes ◄───┴───► No                                   │
│                   │              │                                   │
│                   ▼              ▼                                   │
│            ┌───────────┐  ┌─────────────┐                           │
│            │  SERVICE  │  │    OAUTH    │                           │
│            │  ACCOUNT  │  │   REDIRECT  │                           │
│            └───────────┘  └─────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Production Checklist

### General (All Deployments)

- [ ] **Environment Variables**: Never commit secrets to git
- [ ] **Logging**: Configure log aggregation (e.g., Loki, CloudWatch)
- [ ] **Monitoring**: Set up health check alerts
- [ ] **Backups**: Database backup strategy
- [ ] **Updates**: Plan for updating services

### Cloud Server Specific

- [ ] **SSL/TLS**: HTTPS everywhere (Let's Encrypt or institutional cert)
- [ ] **Firewall**: Only expose necessary ports (80, 443)
- [ ] **Reverse Proxy**: NGINX or Traefik in front
- [ ] **Rate Limiting**: Prevent abuse
- [ ] **Authentication**: User login (university SSO?)
- [ ] **Domain Name**: Configure DNS

### Security Hardening

```yaml
# docker-compose.prod.yml additions

services:
  middleware:
    # Run as non-root user
    user: "1000:1000"

    # Read-only filesystem where possible
    read_only: true
    tmpfs:
      - /tmp

    # Limit capabilities
    cap_drop:
      - ALL

    # Security options
    security_opt:
      - no-new-privileges:true
```

### Reverse Proxy Example (NGINX)

```nginx
# /etc/nginx/sites-available/stream

server {
    listen 80;
    server_name stream.university.edu;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name stream.university.edu;

    ssl_certificate /etc/letsencrypt/live/stream.university.edu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/stream.university.edu/privkey.pem;

    # Streamlit frontend
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Middleware API (if exposed directly)
    location /api/ {
        proxy_pass http://localhost:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **OAuth 2.0** | Industry standard protocol for authorization. Lets users grant apps access without sharing passwords. |
| **Headless Server** | A server without a monitor/display. Can't run graphical applications like browsers. |
| **Reverse Proxy** | Server that sits in front of your app, handling SSL, load balancing, etc. (NGINX, Traefik) |
| **SSL/TLS** | Encryption for HTTPS. Protects data in transit. Required for OAuth callbacks. |
| **Service Account** | A non-human account used by applications to authenticate. |
| **Docker Volume** | Persistent storage that survives container restarts. |
| **Container** | Isolated environment running a single service (like a lightweight VM). |
| **SSO** | Single Sign-On. Log in once, access multiple apps. Universities often use Shibboleth or CAS. |
| **Callback URL** | Where OAuth redirects after authentication. Must be pre-registered. |
| **Let's Encrypt** | Free SSL certificate authority. Certificates auto-renew. |

---

## Summary

| Deployment | OAuth Solution | Complexity | Best For |
|------------|----------------|------------|----------|
| Development (current) | Host frontend | Low | Your laptop |
| Cloud (simple) | Service account | Medium | Internal teams |
| Cloud (enterprise) | OAuth redirect | High | Many users |
| Desktop app | Native browser | Medium | Single users |

**Start simple** (service account), then add complexity (OAuth redirect) only when needed.

---

*This document is part of the STREAM project. Last updated: February 2025.*
