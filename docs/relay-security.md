# STREAM Relay — Security Guide

This document explains how the WebSocket relay is secured, what each protection
does, and what steps remain before opening STREAM to multiple users on campus.
It is written to be understandable without prior security expertise.

---

## Background: What is the relay and why does it need securing?

The relay (`stream/relay/server.py`) is a small server that sits between two
parties who cannot talk directly to each other:

- **Producer** — a Python function running on Lakeshore HPC, behind UIC's campus
  firewall. It cannot accept incoming connections.
- **Consumer** — your STREAM instance (laptop or server), possibly behind a home
  router NAT. It also cannot accept incoming connections.

Both sides connect *outward* to the relay, which forwards messages between them.
This is exactly how a phone call works through a switchboard: neither caller needs
to know the other's address; they both dial the operator.

Because the relay sits in the middle of every Lakeshore response, it is a
natural target for attack. The security measures below address this.

---

## Layer 1 — Channel isolation (UUID per request)

**What it does.**
Every time you send a message to Lakeshore, STREAM generates a fresh random
128-bit UUID (e.g., `a3f9c2d1-e8b7-4f6a-9c3d-...`). The producer and consumer
connect to the relay using *that specific UUID* as the channel name:

```
ws://relay.example.com/produce/a3f9c2d1-e8b7-4f6a-9c3d-...
ws://relay.example.com/consume/a3f9c2d1-e8b7-4f6a-9c3d-...
```

To read someone else's response, an attacker would need to guess *their exact*
UUID. There are $2^{122}$ possible values — more than the number of atoms in the
observable universe. Guessing one is computationally infeasible.

**Where in the code.**
UUID generation: `stream/middleware/core/litellm_direct.py` — the consumer generates
`channel_id = str(uuid.uuid4())` before submitting the Globus job.
Channel routing in the relay: `stream/relay/server.py` — `handle_connection()` parses
the path (`/produce/{channel_id}` or `/consume/{channel_id}`) and places each
connection in the right channel slot.

**What it protects against.**
Passive eavesdropping: an attacker who does not know the UUID cannot connect to
your channel.

**What it does NOT protect against.**
An attacker who has captured the UUID from the network (if traffic is unencrypted)
could race to connect. This is why TLS (Layer 5) is needed for multi-user deployments.

---

## Layer 2 — Data minimization (no persistence)

**What it does.**
The relay never writes anything to disk. Tokens exist in memory only while being
forwarded. As soon as a token is sent to the consumer, it is gone. When both sides
disconnect, the channel entry is deleted from the in-memory dictionary.

**Where in the code.**
`stream/relay/server.py` — `_handle_producer()`: each token is forwarded
immediately via `await consumer.send(message)`. If the consumer is not yet
connected, tokens are held in `channel["buffer"]` — a plain Python list in RAM,
never written anywhere. When the consumer connects, `_handle_consumer()` flushes
the buffer and clears it: `channel["buffer"].clear()`.
Cleanup: `_maybe_cleanup_channel()` deletes the channel dict entry as soon as
both sides have disconnected and the buffer is empty.

**What it protects against.**
A compromised relay server: an attacker who gains shell access to the relay VM
finds no conversation history, no user data, no logs of past tokens. They can
only observe tokens *currently in transit*, and only if they can also intercept
the WebSocket connection.

---

## Layer 3 — Credential separation

**What it does.**
Globus Compute requires OAuth2 authentication tokens to submit jobs. These tokens
grant access to the HPC cluster. They are used only on the *control plane*
(Globus's own AMQP network) and are never sent through the relay.

The relay only ever sees raw token text — the words the language model is generating.
It has no access to any credentials or user identity.

**Where in the code.**
`stream/middleware/core/litellm_direct.py` — Globus authentication happens in
`GlobusComputeClient.__init__()`, stored in the client object. The relay call
`submit_streaming_inference(relay_url=RELAY_URL, relay_secret=RELAY_SECRET)` passes
only the relay URL and secret — no auth tokens.
`stream/middleware/core/globus_compute_client.py` — `remote_vllm_streaming()` (the
function that runs on Lakeshore) receives `relay_url` and `relay_secret` as plain
string arguments. No Globus credentials are in scope inside that function.

**What it protects against.**
If the relay were compromised, the attacker cannot impersonate users or submit
new HPC jobs, because they never had the Globus credentials to begin with.

---

## Layer 4 — Shared-secret authentication ✅ IMPLEMENTED

**What it does.**
The relay server now requires a shared secret token. Only STREAM instances that
know the secret can connect as producers or consumers. Anyone else (random internet
scanners, unauthorized users) is rejected immediately.

Think of it like a password for the relay's door. Before your producer or consumer
is allowed inside, they whisper the password. Wrong password → door stays shut.

**How it works in practice.**
The secret is appended to the WebSocket URL as a query parameter:

```
ws://relay.example.com/produce/a3f9c2d1...?secret=a3f9c2d1e8b74f6a...
ws://relay.example.com/consume/a3f9c2d1...?secret=a3f9c2d1e8b74f6a...
```

The relay checks this before creating any channel state. Wrong or missing secret
→ connection closed with code 4003 (Forbidden).

The `/health` endpoint is exempt from auth — monitoring tools need to check if
the relay is running without knowing the secret.

**Where in the code.**

*Relay server* (`stream/relay/server.py`):
- Global `_RELAY_SECRET: str` holds the configured secret.
- `handle_connection()` — after parsing the path, if `_RELAY_SECRET` is set:
  ```python
  qs = parse_qs(parsed.query)
  provided = qs.get("secret", [None])[0]
  if provided != _RELAY_SECRET:
      await websocket.close(4003, "Forbidden: invalid or missing secret")
      return
  ```
- `start_relay(secret=...)` sets `_RELAY_SECRET` at startup.
- `main()` reads from `--secret` flag or `RELAY_SECRET` environment variable.

*Consumer — desktop mode* (`stream/middleware/core/litellm_direct.py`):
- Imports `RELAY_SECRET` from `stream/middleware/config.py`.
- Consumer URL: `f"{RELAY_URL}/consume/{channel_id}?secret={RELAY_SECRET}"` (when secret is set).
- Also passed to `submit_streaming_inference(relay_secret=RELAY_SECRET)`.

*Consumer — server mode* (`stream/proxy/app.py`):
- Same pattern: imports `RELAY_SECRET`, appends to consumer URL, passes to
  `submit_streaming_inference`.

*Producer — Lakeshore* (`stream/middleware/core/globus_compute_client.py`):
- `remote_vllm_streaming()` (the function serialized and sent to Lakeshore) now
  accepts `relay_secret=""`.
- Producer URL: `f"{relay_url}/produce/{channel_id}?secret={relay_secret}"` (when set).
- The secret travels to Lakeshore as a plain function argument inside Globus
  Compute's own encrypted serialization — it is not visible on the network.

*Configuration* (`stream/middleware/config.py`):
```python
RELAY_SECRET = os.getenv("RELAY_SECRET", "")
```

*Your `.env` file*:
```
RELAY_SECRET=         # set this to your generated secret
```

**How to generate and enable the secret.**

```bash
# 1. Generate a strong secret (32 random bytes = 64 hex characters)
python -c "import secrets; print(secrets.token_hex(32))"
# Example output: a3f9c2d1e8b74f6adc3e9f12b5...

# 2. Set it in .env on your machine:
RELAY_SECRET=a3f9c2d1e8b74f6adc3e9f12b5...

# 3. On the relay server VM, set the same value:
export RELAY_SECRET=a3f9c2d1e8b74f6adc3e9f12b5...
python -m stream.relay.server
# OR:
python -m stream.relay.server --secret a3f9c2d1e8b74f6adc3e9f12b5...
```

**What it protects against.**
Without this layer, anyone who discovers the relay's public IP address and port
can connect as a producer and inject arbitrary tokens into your session, or
connect as a consumer and read your response. With the secret, they are rejected
before touching any channel state.

---

## Layer 5 — Resource limits ✅ IMPLEMENTED

**What it does.**
Two limits prevent the relay from being abused to exhaust server memory:

**Buffer size cap** (default: 1,000 messages per channel).
If a producer sends tokens faster than the consumer connects — or a buggy producer
sends thousands of messages to a dead channel — the buffer is capped. When full,
the *oldest* message is dropped to make room for the newest. This prevents a
single misbehaving job from filling the relay's RAM.

```python
# stream/relay/server.py — _handle_producer()
if len(channel["buffer"]) >= _MAX_BUFFER_MESSAGES:
    channel["buffer"].pop(0)   # drop oldest
```

**Abandoned channel reaper** (default: 300-second timeout).
If only one side ever connects to a channel (e.g., the Globus job failed and
the producer never showed up), the channel would sit in memory forever. A
background task runs every 60 seconds and deletes channels where only one side
connected and the channel is older than the timeout.

```python
# stream/relay/server.py — _channel_reaper()
# Runs as asyncio.create_task() inside start_relay()
```

Configure both via CLI:
```bash
python -m stream.relay.server \
  --max-buffer 1000 \
  --channel-timeout 300
```

**What it protects against.**
Memory exhaustion attacks: a script that opens thousands of channels and sends
garbage data cannot crash the relay server.

---

## Layer 6 — TLS (`wss://`) ⬜ STILL NEEDED FOR MULTI-USER

**What it does.**
All five layers above protect *who can connect* and *what stays in memory*.
But the token text itself — the words of the AI's response — travels over the
network in plain text if the relay uses `ws://` (unencrypted WebSocket).

On a campus network or over the internet, a passive network observer (another
machine on the same network segment, an ISP, etc.) could read the tokens as they
flow.

TLS encrypts the entire connection, so tokens are unreadable in transit.

**How to add it (no code changes to STREAM required).**

Option A — Nginx reverse proxy:
```nginx
# /etc/nginx/sites-available/relay
server {
    listen 443 ssl;
    server_name relay.your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/relay.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/relay.your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600;
    }
}
```
Then in `.env`: `RELAY_URL=wss://relay.your-domain.com`

Option B — Cloudflare Tunnel (simplest, no domain needed):
```bash
# On the relay VM:
cloudflared tunnel --url http://localhost:8765
# Gives you a stable wss://your-name.trycloudflare.com URL automatically
```

Option C — Caddy (auto TLS via Let's Encrypt):
```
# /etc/caddy/Caddyfile
relay.your-domain.com {
    reverse_proxy localhost:8765
}
```

**The relay server does not change.** It keeps listening on `ws://localhost:8765`.
The reverse proxy handles TLS externally and forwards decrypted traffic to the relay.

---

## Summary: current security status

| Layer | Protection | Status |
|---|---|---|
| 1 — UUID channel isolation | Prevents guessing other sessions | ✅ Always present |
| 2 — Data minimization | No persistence; nothing to steal after the fact | ✅ Always present |
| 3 — Credential separation | Globus tokens never touch the relay | ✅ Always present |
| 4 — Shared-secret auth | Unauthorized connections rejected at handshake | ✅ **Implemented** |
| 5 — Resource limits | Buffer cap + channel reaper prevent memory abuse | ✅ **Implemented** |
| 6 — TLS (`wss://`) | Encrypts token text in transit | ⬜ Infrastructure step |

**The relay is safe for single-user and small-group research use today.**
Layers 1–5 are all in place. Add Layer 6 (TLS) before opening STREAM to
arbitrary campus users, since shared networks require encrypted transit.

---

## Files changed in this implementation

| File | What changed |
|---|---|
| `stream/relay/server.py` | `--secret` flag; `--max-buffer`; `--channel-timeout`; `_channel_reaper()` background task; auth check in `handle_connection()`; buffer cap in `_handle_producer()` |
| `stream/middleware/config.py` | Added `RELAY_SECRET = os.getenv("RELAY_SECRET", "")` |
| `stream/middleware/core/litellm_direct.py` | Imports `RELAY_SECRET`; appends to consumer URL; passes to `submit_streaming_inference()` |
| `stream/middleware/core/globus_compute_client.py` | `remote_vllm_streaming()` accepts `relay_secret`; appends to producer URL; `submit_streaming_inference()` accepts and passes through `relay_secret` |
| `stream/proxy/app.py` | Imports `RELAY_SECRET`; appends to consumer URL; passes to `submit_streaming_inference()` |
| `.env` | Added `RELAY_SECRET=` entry with generation instructions |
