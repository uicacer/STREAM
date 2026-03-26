"""
WebSocket Relay Server for STREAM — True Token Streaming from Lakeshore.

=============================================================================
WHAT THIS FILE DOES
=============================================================================
This is a lightweight WebSocket relay server that enables real-time token
streaming from Lakeshore HPC to the user's browser. It solves a fundamental
limitation of Globus Compute: Globus is batch-only (submit function → wait →
get complete result), so tokens can't flow progressively to the user.

=============================================================================
THE ARCHITECTURE (Control Plane vs Data Plane)
=============================================================================
We split the work into two channels:

  CONTROL PLANE (Globus Compute — existing):
    - Authentication (OAuth2 tokens)
    - Job submission (serialize function, send via AMQP)
    - Launching the remote function on the HPC compute node
    - Returning final status (success/error)

  DATA PLANE (this relay — new):
    - Real-time token delivery as the GPU generates them
    - Lightweight message forwarding, no computation
    - Both sides connect OUTBOUND to the relay (bypasses firewalls)

=============================================================================
HOW IT INTEGRATES WITH STREAM (Two Modes)
=============================================================================

  SERVER MODE (Docker — 5 containers):
  ─────────────────────────────────────
  The Lakeshore Proxy container is what submits jobs to Globus Compute.
  With the relay, the proxy also connects as a consumer to receive tokens:

    React UI → Middleware → LiteLLM → Lakeshore Proxy
                                           │
                                    (1) Generate channel_id
                                    (2) Submit to Globus Compute
                                        (passing relay_url + channel_id)
                                    (3) Connect to relay as CONSUMER
                                    (4) Stream tokens back as HTTP SSE
                                           │
    React UI ← Middleware ← LiteLLM ←─────┘ (streaming HTTP response)

    Meanwhile on Lakeshore:
      Remote function → vLLM stream=True → relay PRODUCER → relay → CONSUMER (proxy)


  DESKTOP MODE (single process):
  ──────────────────────────────
  litellm_direct.py calls the Globus Compute client directly. With the
  relay, it also connects as a consumer:

    React UI → Middleware → litellm_direct.py
                                 │
                          (1) Generate channel_id
                          (2) Call globus_client.submit_inference()
                              (passing relay_url + channel_id)
                          (3) Connect to relay as CONSUMER
                          (4) Yield tokens directly
                                 │
    React UI ← Middleware ←─────┘ (streaming response)

    Meanwhile on Lakeshore:
      Remote function → vLLM stream=True → relay PRODUCER → relay → CONSUMER

  In both modes, the PRODUCER (Lakeshore) and CONSUMER (proxy or litellm_direct)
  make OUTBOUND connections to the relay. The relay just forwards messages.

=============================================================================
THE DATA FLOW IN DETAIL
=============================================================================

    ┌─────────────────┐        ┌─────────────┐        ┌──────────────────┐
    │  Lakeshore HPC  │        │   Relay     │        │  Proxy or        │
    │  (a-001)        │        │   Server    │        │  litellm_direct  │
    │                 │        │             │        │                  │
    │  vLLM stream    │        │             │        │                  │
    │  ──→ token₁ ──────WS────→  forward  ──────WS────→  SSE/yield      │
    │  ──→ token₂ ──────WS────→  forward  ──────WS────→  SSE/yield      │
    │  ──→ token₃ ──────WS────→  forward  ──────WS────→  SSE/yield      │
    │  ──→ [done]  ─────WS────→  forward  ──────WS────→  [stream end]   │
    └─────────────────┘        └─────────────┘        └──────────────────┘
         PRODUCER               RELAY                    CONSUMER
      (outbound conn)                                 (outbound conn)

=============================================================================
WHY BOTH SIDES CONNECT OUTBOUND
=============================================================================
- Lakeshore compute nodes are behind UIC's campus firewall
  → No inbound connections allowed
  → But OUTBOUND connections work (that's how Globus AMQP works too)

- The user's laptop is behind a home NAT/router
  → Also can't accept inbound connections

- The relay runs on a public server (or localhost + tunnel for development)
  → Both sides connect OUT to the relay
  → The relay just forwards bytes between matched connections
  → Same principle as TURN servers for video calls

=============================================================================
TUNNELING (Development Only)
=============================================================================
A tunnel creates a temporary public URL for a server running on your laptop.
This is needed because the relay runs locally but Lakeshore needs to reach it.

Recommended: Cloudflare Tunnel (cloudflared)
  cloudflared tunnel --url http://localhost:8765
  → Gives you a URL like: https://random-words.trycloudflare.com
  → Stable connections, persists across multiple requests
  → Free, no account required

Alternatives:
  ngrok http 8765                                    → https://abc123.ngrok-free.app
  ssh -4 -R 80:localhost:8765 nokey@localhost.run     → https://xxxx.lhr.life
  (localhost.run drops connections after ~30s of inactivity — not recommended)

WHAT we need for production:
  - A real server with a public IP (options):
    1. UIC campus VM behind ACER's reverse proxy (ideal — data stays at UIC)
    2. A small cloud VM ($5/month on DigitalOcean/Hetzner)
    3. A dedicated machine in the ACER lab
  - The relay uses ~10 MB RAM and near-zero CPU — any server works
  - Replace the tunnel URL with the server's real URL in STREAM's config
  - Add TLS (wss://) via reverse proxy (nginx/caddy) for encrypted transit

=============================================================================
RELAY PROTOCOL (WebSocket Messages)
=============================================================================
All messages are JSON strings:

  Producer → Relay → Consumer:
    {"type": "token",  "content": "Hello"}     — a chunk of generated text
    {"type": "done"}                           — generation complete
    {"type": "error",  "message": "..."}       — something went wrong

  These match what vLLM returns in its SSE stream, just reformatted as JSON.

=============================================================================
SECURITY NOTES
=============================================================================
- The channel_id is a random UUID — unguessable (122 bits of entropy)
- Only someone who knows the exact channel_id can connect to a channel
- The relay doesn't execute code, access files, or store data
- Channels are cleaned up immediately when both sides disconnect
- For production: add TLS (wss://) and optionally a shared secret token

=============================================================================
USAGE
=============================================================================
  Development:
    python -m stream.relay.server                                # starts on ws://0.0.0.0:8765
    cloudflared tunnel --url http://localhost:8765                # creates public URL

  Production:
    python -m stream.relay.server --host 0.0.0.0 --port 8765
    (behind a reverse proxy with TLS termination)
"""

import argparse
import asyncio
import json
import logging
import time as _time
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.asyncio.server import serve

logger = logging.getLogger(__name__)

# Shared secret token for authenticating producers and consumers.
# Set via --secret CLI argument. Empty string = auth disabled (dev mode).
_RELAY_SECRET: str = ""

# Production limits — set via CLI flags.
# MAX_BUFFER_MESSAGES: how many messages to buffer before dropping.
#   Prevents a malicious or runaway producer from filling RAM.
#   Default 1000 is generous (~1MB at avg 1KB/message) but bounded.
# CHANNEL_TIMEOUT_SECONDS: how long to keep a channel alive when only
#   ONE side is connected. After this timeout the channel is abandoned
#   and cleaned up, preventing orphaned channels from accumulating.
#   Default 300s covers the worst-case Globus Compute cold-start delay.
_MAX_BUFFER_MESSAGES: int = 1000
_CHANNEL_TIMEOUT_SECONDS: int = 300


# =============================================================================
# CHANNEL REGISTRY
# =============================================================================
# A "channel" is a pair of WebSocket connections matched by a channel_id.
# When a producer connects to /produce/{channel_id} and a consumer connects
# to /consume/{channel_id}, the relay forwards messages from producer to
# consumer in real-time.
#
# The registry maps channel_id → {
#     "producer": ws | None,
#     "consumer": ws | None,
#     "buffer":   list[str],      # messages buffered before consumer connects
#     "created":  float,          # time.monotonic() timestamp for timeout tracking
# }.
#
# Typical timing:
#   1. Consumer connects first (STREAM proxy/litellm_direct — immediate)
#   2. Producer connects a few seconds later (Lakeshore — after Globus routing)
#   3. Producer sends tokens, relay forwards to consumer
#   4. Producer sends "done", both disconnect, channel cleaned up
#
# If the producer somehow connects first and sends tokens before the consumer
# is ready, messages are buffered and flushed when the consumer connects.

channels: dict[str, dict] = {}


# =============================================================================
# BACKGROUND CHANNEL REAPER
# =============================================================================


async def _channel_reaper():
    """
    Background task: periodically sweep for abandoned channels.

    An abandoned channel is one where only ONE side connected and the other
    side never showed up within CHANNEL_TIMEOUT_SECONDS. Without this reaper,
    a failed Globus job (or a consumer that crashed before the producer
    connected) would leave a channel entry in memory forever.

    Runs every 60 seconds. Overhead is negligible — just iterates over the
    channel dict and checks timestamps.
    """
    while True:
        await asyncio.sleep(60)
        now = _time.monotonic()
        stale = []
        for channel_id, ch in list(channels.items()):
            # A channel is stale if: only one side is connected (or neither)
            # AND it was created more than CHANNEL_TIMEOUT_SECONDS ago.
            one_sided = (ch["producer"] is None) != (ch["consumer"] is None)
            both_gone = ch["producer"] is None and ch["consumer"] is None
            age = now - ch["created"]
            if (one_sided or both_gone) and age > _CHANNEL_TIMEOUT_SECONDS:
                stale.append(channel_id)
        for channel_id in stale:
            channels.pop(channel_id, None)
            logger.warning(
                f"[{channel_id[:8]}] abandoned channel reaped after "
                f"{_CHANNEL_TIMEOUT_SECONDS}s (one side never connected)"
            )


# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================


async def handle_connection(websocket):
    """
    Handle an incoming WebSocket connection.

    The URL path determines the role:
      /produce/{channel_id}  — Lakeshore side, sending tokens
      /consume/{channel_id}  — STREAM proxy or litellm_direct, receiving tokens
      /health                — health check (returns JSON status, no auth required)

    Authentication: when --secret is set, the ?secret=<token> query parameter
    must be present on produce and consume connections. Health checks are exempt.

    The relay is completely stateless — it doesn't interpret the messages,
    just forwards them from producer to consumer.
    """
    # Extract the full request path (including query string) from the WebSocket request.
    # Examples: "/health", "/produce/abc123?secret=mytoken", "/consume/abc123?secret=mytoken"
    full_path = websocket.request.path
    parsed = urlparse(full_path)
    path = parsed.path  # just the path, without query string

    # ---- Health check endpoint ----
    # Monitoring tools (or STREAM's tier_health.py) hit /health to verify the
    # relay is running. We respond with a JSON status and immediately close.
    # Health checks do NOT require authentication — they carry no user data.
    if path == "/health":
        await websocket.send(
            json.dumps(
                {
                    "status": "healthy",
                    "active_channels": len(channels),  # how many streams are active right now
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        )
        await websocket.close()  # done — health check is a one-shot request
        return  # exit early: don't try to parse this as a produce/consume path

    # ---- Parse the path to determine role and channel ----
    # Valid paths: /produce/{channel_id} or /consume/{channel_id}
    # Split "/produce/abc123" → ["produce", "abc123"]
    parts = path.strip("/").split("/")
    if len(parts) != 2 or parts[0] not in ("produce", "consume"):
        # Invalid path — reject with WebSocket close code 4000 (custom error code)
        await websocket.close(4000, "Invalid path. Use /produce/{id} or /consume/{id}")
        return  # exit early: can't proceed without a valid role and channel

    role = parts[0]  # "produce" (Lakeshore sending tokens) or "consume" (proxy receiving)
    channel_id = parts[1]  # UUID that pairs this producer with its consumer

    # ---- Shared-secret authentication ----
    # When --secret is configured, every produce/consume connection must supply
    # the matching token as ?secret=<value> in the query string. Connections
    # without the correct secret are rejected before any channel state is created.
    if _RELAY_SECRET:
        qs = parse_qs(parsed.query)
        provided = qs.get("secret", [None])[0]
        if provided != _RELAY_SECRET:
            logger.warning(f"[{channel_id[:8]}] rejected {role}r: invalid or missing secret")
            await websocket.close(4003, "Forbidden: invalid or missing secret")
            return  # exit early: unauthenticated connection

    logger.info(f"[{channel_id[:8]}] {role}r connected")  # log first 8 chars for readability

    # ---- Initialize the channel if it doesn't exist ----
    # The first side to connect (usually consumer) creates the channel entry.
    # The second side (usually producer) finds it already here.
    if channel_id not in channels:
        channels[channel_id] = {
            "producer": None,  # will hold the producer's WebSocket connection
            "consumer": None,  # will hold the consumer's WebSocket connection
            # Messages queued before the consumer connects.
            # If the producer starts sending tokens before the proxy/litellm_direct
            # connects, we buffer them here so nothing is lost.
            "buffer": [],
            # Creation timestamp for the channel reaper (abandoned channel cleanup).
            "created": _time.monotonic(),
        }

    channel = channels[channel_id]

    # ---- Register this connection in the channel ----
    # Each channel allows exactly ONE producer and ONE consumer. If a second
    # producer (or consumer) tries to connect to the same channel_id, reject
    # it — something is wrong (duplicate request, stale connection, etc.)
    if role == "produce":
        if channel["producer"] is not None:
            await websocket.close(4001, "Producer already connected for this channel")
            return  # exit early: don't overwrite the existing producer
        channel["producer"] = websocket  # register this connection as the producer
        await _handle_producer(websocket, channel, channel_id)  # blocks until producer disconnects
    else:  # consume
        if channel["consumer"] is not None:
            await websocket.close(4001, "Consumer already connected for this channel")
            return  # exit early: don't overwrite the existing consumer
        channel["consumer"] = websocket  # register this connection as the consumer
        await _handle_consumer(websocket, channel, channel_id)  # blocks until consumer disconnects


async def _handle_producer(websocket, channel, channel_id):
    """
    Handle the producer side (Lakeshore compute node).

    The producer sends token messages. For each message:
    - If a consumer is connected → forward immediately
    - If no consumer yet → buffer the message (flushed when consumer connects)

    When the producer disconnects (or sends {"type": "done"}), we clean up.
    """
    try:
        # Loop over every message the producer sends (each message = one token
        # or a control signal like "done" or "error"). This loop runs until
        # the producer disconnects or we break out.
        async for message in websocket:
            consumer = channel.get("consumer")

            if consumer is not None:
                # Consumer is connected — forward the message immediately.
                # This is the fast path during normal operation: producer sends
                # a token → relay forwards it → consumer receives it instantly.
                try:
                    await consumer.send(message)
                except websockets.ConnectionClosed:
                    # Consumer disconnected mid-stream (e.g., user navigated away
                    # or closed the browser tab). The producer keeps running on
                    # Lakeshore (we can't stop GPU inference mid-generation),
                    # but there's no one to forward to, so we stop.
                    logger.warning(
                        f"[{channel_id[:8]}] consumer disconnected, " "dropping remaining tokens"
                    )
                    break  # stop the loop — no point reading more tokens
            else:
                # Consumer hasn't connected yet — buffer the message so it's
                # not lost. When the consumer connects, _handle_consumer()
                # will flush these buffered messages first.
                if len(channel["buffer"]) >= _MAX_BUFFER_MESSAGES:
                    # Buffer full — a runaway producer is sending faster than
                    # the consumer can connect. Drop oldest message (sliding
                    # window) to keep memory bounded.
                    channel["buffer"].pop(0)
                    logger.warning(
                        f"[{channel_id[:8]}] buffer full "
                        f"({_MAX_BUFFER_MESSAGES} messages) — dropping oldest"
                    )
                channel["buffer"].append(message)
                logger.debug(
                    f"[{channel_id[:8]}] buffered message "
                    f"(no consumer yet, buffer size: {len(channel['buffer'])})"
                )

    except websockets.ConnectionClosed:
        # Producer's WebSocket connection dropped (network issue, Lakeshore
        # job ended, etc.). This is normal — not an error.
        logger.info(f"[{channel_id[:8]}] producer disconnected")
    finally:
        # Always clean up: unregister the producer and try to remove the
        # channel if both sides are done.
        channel["producer"] = None
        _maybe_cleanup_channel(channel_id)


async def _handle_consumer(websocket, channel, channel_id):
    """
    Handle the consumer side (STREAM's Lakeshore Proxy or litellm_direct).

    When the consumer connects:
    1. Flush any buffered messages (tokens that arrived before we connected)
    2. Stay alive — new messages are pushed by _handle_producer() directly

    The consumer doesn't send messages to the producer (it's one-directional:
    Lakeshore → relay → proxy/litellm_direct). The consumer stays connected
    until the producer sends {"type": "done"} or disconnects.
    """
    try:
        # ---- Flush buffered messages ----
        # If the producer already sent some tokens before we connected
        # (race condition: producer was faster), deliver them now so the
        # consumer doesn't miss the beginning of the response.
        if channel["buffer"]:
            logger.debug(
                f"[{channel_id[:8]}] flushing {len(channel['buffer'])} "
                f"buffered messages to consumer"
            )
            for msg in channel["buffer"]:
                await websocket.send(msg)  # deliver each buffered token
            channel["buffer"].clear()  # buffer is now empty — future tokens go direct

        # ---- Keep the connection alive ----
        # The consumer doesn't actively receive tokens here — that's done by
        # _handle_producer() calling `consumer.send()` directly. This loop
        # just keeps the WebSocket alive and listens for any messages FROM
        # the consumer (e.g., a future "cancel" command). When the producer
        # closes the connection and sends the final "done" message, the
        # consumer's WebSocket will also close, ending this loop.
        async for message in websocket:
            # Future: handle cancel requests from the consumer
            # e.g., {"type": "cancel"} → tell producer to stop
            logger.debug(
                f"[{channel_id[:8]}] received message from consumer "
                f"(currently ignored): {message[:100]}"
            )

    except websockets.ConnectionClosed:
        # Consumer disconnected (user closed tab, network dropped, etc.)
        logger.info(f"[{channel_id[:8]}] consumer disconnected")
    finally:
        # Always clean up: unregister the consumer and try to remove the
        # channel if both sides are done.
        channel["consumer"] = None
        _maybe_cleanup_channel(channel_id)


def _maybe_cleanup_channel(channel_id):
    """
    Remove the channel from the registry if both sides have disconnected
    AND there are no buffered messages waiting for a consumer.

    This prevents memory leaks — without cleanup, every chat request would
    leave an empty channel entry in the dictionary forever.

    But we must NOT clean up if there are buffered messages: the producer
    might have sent tokens and disconnected before the consumer connected.
    Those buffered messages still need to be delivered when the consumer
    eventually connects.
    """
    channel = channels.get(channel_id)
    # Only clean up if BOTH producer and consumer have disconnected.
    # If one side is still connected, the channel is still in use.
    if channel and channel["producer"] is None and channel["consumer"] is None:
        if channel["buffer"]:
            # Edge case: producer sent tokens and disconnected, but consumer
            # hasn't connected yet. Those tokens are sitting in the buffer.
            # We MUST keep the channel alive so the consumer can still receive
            # them when it eventually connects. Deleting now = lost tokens.
            logger.info(
                f"[{channel_id[:8]}] both sides disconnected but "
                f"{len(channel['buffer'])} buffered messages remain — "
                f"keeping channel alive for consumer"
            )
            return  # don't delete — consumer still needs these messages
        # Both sides done, no buffered messages — safe to remove.
        # This prevents the `channels` dict from growing forever.
        del channels[channel_id]
        logger.info(f"[{channel_id[:8]}] channel cleaned up")


# =============================================================================
# SERVER STARTUP
# =============================================================================


async def start_relay(
    host: str = "0.0.0.0",
    port: int = 8765,
    secret: str = "",
    max_buffer: int = 1000,
    channel_timeout: int = 300,
):
    """
    Start the WebSocket relay server.

    Args:
        host:            Bind address. "0.0.0.0" = accept connections from anywhere.
        port:            Port to listen on.
        secret:          Shared secret token. When non-empty, all produce/consume
                         connections must supply ?secret=<value>. Disabled when empty.
        max_buffer:      Max messages buffered per channel before oldest is dropped.
        channel_timeout: Seconds before an abandoned one-sided channel is reaped.
    """
    global _RELAY_SECRET, _MAX_BUFFER_MESSAGES, _CHANNEL_TIMEOUT_SECONDS
    _RELAY_SECRET = secret
    _MAX_BUFFER_MESSAGES = max_buffer
    _CHANNEL_TIMEOUT_SECONDS = channel_timeout

    # Print connection info so the operator knows what URLs to use
    logger.info(f"Starting WebSocket relay on ws://{host}:{port}")
    logger.info(f"  Producer URL: ws://<host>:{port}/produce/{{channel_id}}")
    logger.info(f"  Consumer URL: ws://<host>:{port}/consume/{{channel_id}}")
    logger.info(f"  Health check: ws://<host>:{port}/health")
    if secret:
        logger.info("  Auth:         shared-secret enabled (set RELAY_SECRET in clients)")
    else:
        logger.warning("  Auth:         DISABLED — set --secret for production deployments")
    logger.info("")
    logger.info("For development, create a public tunnel:")
    logger.info(f"  cloudflared tunnel --url http://localhost:{port}   (recommended)")
    logger.info(f"  ngrok http {port}                                  (alternative)")
    logger.info("  Then use the tunnel URL as RELAY_URL in STREAM's .env")
    logger.info("")

    # Start the WebSocket server. `serve()` returns an async context manager
    # that listens on host:port and calls `handle_connection` for every new
    # WebSocket connection. `serve_forever()` blocks until the process is killed.
    async with serve(handle_connection, host, port) as server:
        # Start the background reaper task that cleans up abandoned channels.
        asyncio.create_task(_channel_reaper())
        await server.serve_forever()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
# Run the relay as:
#   python -m stream.relay.server
#   python -m stream.relay.server --port 9000
#   python -m stream.relay.server --host 127.0.0.1 --port 8765


def main():
    parser = argparse.ArgumentParser(
        description="STREAM WebSocket Relay — forwards tokens from "
        "Lakeshore HPC to STREAM in real-time."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0 = all interfaces)"
    )
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--secret",
        default="",
        help=(
            "Shared secret token for authentication. When set, all produce/consume "
            "connections must supply ?secret=<value>. Reads RELAY_SECRET env var "
            "if this flag is not provided."
        ),
    )
    parser.add_argument(
        "--max-buffer",
        type=int,
        default=1000,
        help=(
            "Max messages buffered per channel when consumer hasn't connected yet. "
            "Oldest messages are dropped when the limit is reached (default: 1000)."
        ),
    )
    parser.add_argument(
        "--channel-timeout",
        type=int,
        default=300,
        help=(
            "Seconds before an abandoned one-sided channel is reaped. "
            "Prevents memory leaks from failed Globus jobs (default: 300)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    args = parser.parse_args()

    # Allow RELAY_SECRET env var as an alternative to --secret flag.
    import os

    secret = args.secret or os.getenv("RELAY_SECRET", "")

    # Configure Python's logging system. DEBUG shows every buffered message,
    # INFO shows connections and disconnections, WARNING+ for errors only.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        # asyncio.run() starts the event loop and runs our async server.
        # This blocks until the server is stopped (Ctrl+C or kill signal).
        asyncio.run(
            start_relay(
                host=args.host,
                port=args.port,
                secret=secret,
                max_buffer=args.max_buffer,
                channel_timeout=args.channel_timeout,
            )
        )
    except KeyboardInterrupt:
        # Ctrl+C — graceful shutdown. Not an error.
        logger.info("Relay server stopped (Ctrl+C)")


if __name__ == "__main__":
    main()
