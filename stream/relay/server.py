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

- The relay runs on a public server (or localhost + ngrok for development)
  → Both sides connect OUT to the relay
  → The relay just forwards bytes between matched connections
  → Same principle as TURN servers for video calls

=============================================================================
WHAT IS NGROK? (Development Only)
=============================================================================
ngrok is a tool that creates a temporary public URL for a server running on
your laptop. When you run `ngrok http 8765`, it gives you a URL like:
  https://abc123.ngrok-free.app
that tunnels to localhost:8765 on your machine.

WHY we use it now:
  - During development, the relay runs on your laptop (localhost:8765)
  - But Lakeshore needs to reach it from outside your network
  - ngrok creates a public tunnel so Lakeshore can connect to your laptop
  - Free tier is sufficient (we only need one tunnel, low bandwidth)

WHAT we need for production:
  - A real server with a public IP (options):
    1. UIC campus VM behind ACER's reverse proxy (ideal — data stays at UIC)
    2. A small cloud VM ($5/month on DigitalOcean/Hetzner)
    3. A dedicated machine in the ACER lab
  - The relay uses ~10 MB RAM and near-zero CPU — any server works
  - Replace the ngrok URL with the server's real URL in STREAM's config
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
    python -m stream.relay.server                # starts on ws://0.0.0.0:8765
    ngrok http 8765                              # creates public URL

  Production:
    python -m stream.relay.server --host 0.0.0.0 --port 8765
    (behind a reverse proxy with TLS termination)
"""

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime

import websockets
from websockets.asyncio.server import serve

logger = logging.getLogger(__name__)


# =============================================================================
# CHANNEL REGISTRY
# =============================================================================
# A "channel" is a pair of WebSocket connections matched by a channel_id.
# When a producer connects to /produce/{channel_id} and a consumer connects
# to /consume/{channel_id}, the relay forwards messages from producer to
# consumer in real-time.
#
# The registry maps channel_id → {"producer": ws, "consumer": ws, "buffer": []}.
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
# WEBSOCKET HANDLER
# =============================================================================


async def handle_connection(websocket):
    """
    Handle an incoming WebSocket connection.

    The URL path determines the role:
      /produce/{channel_id}  — Lakeshore side, sending tokens
      /consume/{channel_id}  — STREAM proxy or litellm_direct, receiving tokens
      /health                — health check (returns JSON status)

    The relay is completely stateless — it doesn't interpret the messages,
    just forwards them from producer to consumer.
    """
    path = websocket.request.path

    # ---- Health check endpoint ----
    if path == "/health":
        await websocket.send(
            json.dumps(
                {
                    "status": "healthy",
                    "active_channels": len(channels),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        )
        await websocket.close()
        return

    # ---- Parse the path to determine role and channel ----
    parts = path.strip("/").split("/")
    if len(parts) != 2 or parts[0] not in ("produce", "consume"):
        await websocket.close(4000, "Invalid path. Use /produce/{id} or /consume/{id}")
        return

    role = parts[0]  # "produce" or "consume"
    channel_id = parts[1]  # UUID string

    logger.info(f"[{channel_id[:8]}] {role}r connected")

    # ---- Initialize the channel if it doesn't exist ----
    if channel_id not in channels:
        channels[channel_id] = {
            "producer": None,
            "consumer": None,
            # Messages queued before the consumer connects.
            # If the producer starts sending tokens before the proxy/litellm_direct
            # connects, we buffer them here so nothing is lost.
            "buffer": [],
        }

    channel = channels[channel_id]

    # ---- Register this connection in the channel ----
    if role == "produce":
        if channel["producer"] is not None:
            await websocket.close(4001, "Producer already connected for this channel")
            return
        channel["producer"] = websocket
        await _handle_producer(websocket, channel, channel_id)
    else:  # consume
        if channel["consumer"] is not None:
            await websocket.close(4001, "Consumer already connected for this channel")
            return
        channel["consumer"] = websocket
        await _handle_consumer(websocket, channel, channel_id)


async def _handle_producer(websocket, channel, channel_id):
    """
    Handle the producer side (Lakeshore compute node).

    The producer sends token messages. For each message:
    - If a consumer is connected → forward immediately
    - If no consumer yet → buffer the message (flushed when consumer connects)

    When the producer disconnects (or sends {"type": "done"}), we clean up.
    """
    try:
        async for message in websocket:
            consumer = channel.get("consumer")

            if consumer is not None:
                # Consumer is connected — forward the message immediately.
                # This is the fast path during normal operation.
                try:
                    await consumer.send(message)
                except websockets.ConnectionClosed:
                    # Consumer disconnected mid-stream (e.g., user navigated away).
                    # The producer keeps running on Lakeshore (we can't stop GPU
                    # inference mid-generation), but we stop forwarding.
                    logger.warning(
                        f"[{channel_id[:8]}] consumer disconnected, " "dropping remaining tokens"
                    )
                    break
            else:
                # Consumer hasn't connected yet — buffer the message.
                channel["buffer"].append(message)
                logger.debug(
                    f"[{channel_id[:8]}] buffered message "
                    f"(no consumer yet, buffer size: {len(channel['buffer'])})"
                )

    except websockets.ConnectionClosed:
        logger.info(f"[{channel_id[:8]}] producer disconnected")
    finally:
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
        # If the producer already sent some tokens before we connected,
        # deliver them now so no tokens are lost.
        if channel["buffer"]:
            logger.debug(
                f"[{channel_id[:8]}] flushing {len(channel['buffer'])} "
                f"buffered messages to consumer"
            )
            for msg in channel["buffer"]:
                await websocket.send(msg)
            channel["buffer"].clear()

        # ---- Keep the connection alive ----
        # The consumer stays connected, receiving messages forwarded by
        # _handle_producer(). We listen for any messages FROM the consumer
        # (like a cancel request), though currently we don't act on them.
        async for message in websocket:
            # Future: handle cancel requests from the consumer
            # e.g., {"type": "cancel"} → tell producer to stop
            logger.debug(
                f"[{channel_id[:8]}] received message from consumer "
                f"(currently ignored): {message[:100]}"
            )

    except websockets.ConnectionClosed:
        logger.info(f"[{channel_id[:8]}] consumer disconnected")
    finally:
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
    if channel and channel["producer"] is None and channel["consumer"] is None:
        if channel["buffer"]:
            # There are still undelivered messages — keep the channel alive
            # so the consumer can receive them when it connects.
            logger.info(
                f"[{channel_id[:8]}] both sides disconnected but "
                f"{len(channel['buffer'])} buffered messages remain — "
                f"keeping channel alive for consumer"
            )
            return
        del channels[channel_id]
        logger.info(f"[{channel_id[:8]}] channel cleaned up")


# =============================================================================
# SERVER STARTUP
# =============================================================================


async def start_relay(host: str = "0.0.0.0", port: int = 8765):
    """
    Start the WebSocket relay server.

    Args:
        host: Bind address. "0.0.0.0" = accept connections from anywhere
              (needed for ngrok and production). "127.0.0.1" = local only.
        port: Port to listen on. 8765 is the conventional WebSocket dev port.
    """
    logger.info(f"Starting WebSocket relay on ws://{host}:{port}")
    logger.info(f"  Producer URL: ws://<host>:{port}/produce/{{channel_id}}")
    logger.info(f"  Consumer URL: ws://<host>:{port}/consume/{{channel_id}}")
    logger.info(f"  Health check: ws://<host>:{port}/health")
    logger.info("")
    logger.info("For development with ngrok:")
    logger.info(f"  ngrok http {port}")
    logger.info("  Then use the ngrok URL as RELAY_URL in STREAM's .env")
    logger.info("")

    async with serve(handle_connection, host, port) as server:
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(start_relay(host=args.host, port=args.port))
    except KeyboardInterrupt:
        logger.info("Relay server stopped (Ctrl+C)")


if __name__ == "__main__":
    main()
