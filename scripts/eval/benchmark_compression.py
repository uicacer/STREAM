#!/usr/bin/env python3
"""
STREAM Compression Impact Benchmark

Tests whether tier-aware rolling summarization keeps simple queries on the
local (free) tier even after a long conversation fills the context window.

Strategy: inject synthetic long user/assistant pairs as conversation history
(no API calls) to fill the 32K context window quickly. Only probe turns make
real API calls to measure which tier handles them.

Each synthetic filler pair is ~1,050 actual tokens (padded to 2340 chars/side).
Key context sizes:
  Turn 10: ~10,500 tokens  (baseline, well below threshold)
  Turn 20: ~21,000 tokens  (approaching 80% threshold at 26,214)
  Turn 30: ~31,500 tokens  (past 30,720 max input — FAILS without compression)
  Turn 35: ~36,750 tokens  (well past limit)
  Turn 40: ~42,000 tokens  (well past limit)

With compression:    context is summarized to ~4,300 tokens → stays LOCAL at all turns
Without compression: turn 30+ exceeds local tier's max input (32K - 2K output reserve = 30,720) → forced upgrade

Usage:
    # Run the full comparison (40 turns):
    # Step 1 — with compression (normal STREAM):
    python scripts/eval/benchmark_compression.py --label with
    # Step 2 — stop STREAM, set ROLLING_SUMMARIZATION_ENABLED=false, restart, then:
    python scripts/eval/benchmark_compression.py --label without
    # Step 3 — re-enable compression and restart STREAM

    # Quick single-conversation test:
    python scripts/eval/benchmark_compression.py --conversations 1 --verbose

Results are saved to scripts/eval/results/compression_<label>_<timestamp>.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STREAM_URL = os.getenv("STREAM_URL", "http://localhost:5000")
CHAT_ENDPOINT = f"{STREAM_URL}/v1/chat/completions"

# Compression triggers at 80% of local context limit (32,768 tokens).
COMPRESSION_THRESHOLD_TOKENS = int(32768 * 0.80)  # 26,214
LOCAL_CONTEXT_HARD_LIMIT = 32768

# Default conversation length for the with/without comparison experiment.
# 40 turns gives ~42,000 injected tokens at the final probe — well past the
# 30,720 max input (32K - 2K output reserve), so the without-compression run
# will fail at turn 30+.
DEFAULT_NUM_TURNS = 40

# Probe turns for DEFAULT_NUM_TURNS=40.
# Turn 10: baseline (pre-threshold)
# Turn 20: approaching threshold
# Turn 30: just past threshold (compression fires here)
# Turn 35: past 32K hard limit — the critical test
# Turn 40: final state
DEFAULT_PROBE_TURNS = [10, 20, 30, 35, 40]

PROBE_QUERY = "What is 2 + 2?"

# ---------------------------------------------------------------------------
# Synthetic long filler content
#
# Each filler pair must contribute ~1,050 real tokens so that ~25 filler turns
# reach the 26,214-token compression threshold (25 × 1,050 ≈ 26,250).
#
# Llama tokenizer averages ~4 chars/token for English prose, so each side
# needs ~2,100 chars (≈ 525 tokens × 2 sides = ~1,050 tokens/pair).
#
# We build pairs from a core text padded with a technical sentence to reach
# the target character count.
# ---------------------------------------------------------------------------

# Padding sentence appended repeatedly to reach target length (~50 chars each)
_PAD = (
    " Each component communicates through well-defined interfaces that enforce"
    " separation of concerns and enable independent scaling of subsystems."
)

_TARGET_CHARS = 2340  # chars per side ≈ 585 tokens per side ≈ 1,050 tokens/pair (actual tokenizer)


def _pad(text: str) -> str:
    """Pad text to _TARGET_CHARS by appending _PAD repeatedly."""
    while len(text) < _TARGET_CHARS:
        text += _PAD
    return text[:_TARGET_CHARS]


# Five topic pairs — rotated across filler turns for variety
_PAIR_CORES = [
    (
        "Can you explain how the TCP/IP protocol stack works? I'd like to understand "
        "each layer and how data flows from an application down through the network and "
        "back up on the receiving side. Please cover the application layer, transport layer, "
        "internet layer, and link layer, including what headers are added at each step and "
        "why. I'm particularly interested in how the transport layer handles reliable "
        "delivery, flow control, and congestion avoidance. Also explain how IP routing "
        "decisions are made, how ARP resolves IP addresses to MAC addresses, and what "
        "happens when a packet needs to cross multiple router hops to reach its destination. "
        "Include details about MTU, fragmentation, reassembly, DNS resolution, and the full "
        "TCP handshake and TLS negotiation sequence a browser performs before its first byte.",
        "The TCP/IP stack has four layers, each adding a header as data flows downward. "
        "The Application layer generates the HTTP request. TCP (Transport) segments data, "
        "numbers each byte, and ensures reliable delivery via three-way handshake and "
        "sliding-window flow control; CUBIC backs off on loss. IP (Internet) wraps segments "
        "in datagrams; routers forward hop by hop using longest-prefix matching. Datagrams "
        "exceeding path MTU are fragmented and reassembled at the destination. ARP resolves "
        "next-hop IP to MAC by broadcasting on the LAN. The Link layer adds an Ethernet "
        "frame with FCS checksum. TLS adds one RTT for cipher negotiation. Total before "
        "first content byte: DNS (1 RTT) + TCP handshake (1 RTT) + TLS 1.3 (1 RTT) + "
        "HTTP (1 RTT) = 4 RTTs minimum on a cold connection with no cached DNS entry.",
    ),
    (
        "I want a thorough explanation of how relational database query optimization works. "
        "Explain what a query execution plan is, how the planner chooses between nested loop "
        "join, hash join, and merge join. Cover cardinality estimation from table statistics, "
        "B-tree versus hash versus GiST indexes, when a sequential scan beats an index scan, "
        "and multi-column index column ordering. Explain EXPLAIN ANALYZE output in PostgreSQL "
        "including what the cost units represent and how to interpret actual versus estimated "
        "row counts. Cover anti-patterns: implicit type casts that prevent index usage, N+1 "
        "queries from ORMs, unbounded scans on large tables, and missing partial indexes on "
        "low-cardinality columns. Also explain what VACUUM and ANALYZE do, why dead-tuple "
        "bloat accumulates under high UPDATE/DELETE load, and how autovacuum should be tuned.",
        "Query optimization in PostgreSQL: the parser builds a query tree; the planner "
        "enumerates join orders and access paths, assigns cost (I/O + CPU units), picks the "
        "cheapest. Nested loop: O(N×M) but great when inner is small or indexed. Hash join: "
        "builds a hash table on the smaller relation then probes — O(N+M), good for large "
        "equijoins. Merge join: requires sorted inputs, efficient for ordered equijoins. "
        "Cardinality estimates use per-column histograms and most-common-value lists updated "
        "by ANALYZE. Estimates go wrong with correlated columns or stale stats. B-tree covers "
        "range queries; hash is O(1) equality-only; GiST handles geometric and text-search "
        "operators. Index scan wins when selectivity is above ~5-10% of table rows. VACUUM "
        "reclaims dead tuples left by MVCC without full table lock. Autovacuum fires when "
        "dead tuples exceed scale_factor × reltuples + autovacuum_vacuum_threshold.",
    ),
    (
        "Please give me an in-depth explanation of how modern operating systems manage "
        "virtual memory. Cover virtual address spaces, multi-level page tables, and MMU "
        "address translation. Describe page fault handling: loading from disk, swap eviction "
        "via LRU or clock algorithms, and restarting the faulting instruction. Cover "
        "copy-on-write in fork(), demand paging, memory-mapped files, TLB shootdowns in "
        "multi-core systems, huge pages and their trade-offs, NUMA memory locality, the "
        "buddy allocator for physical frame management, the slab allocator for kernel "
        "objects, Linux memory overcommit policy, and OOM killer scoring heuristics. I need "
        "enough depth to understand why NUMA-unaware allocation causes latency spikes and "
        "how transparent huge pages interact with the khugepaged compaction daemon.",
        "Virtual memory gives each process a private 64-bit address space. The MMU walks "
        "a four-level page table (PGD→PUD→PMD→PTE on x86-64) translating virtual to "
        "physical; the TLB caches recent entries. Page fault: kernel checks validity, loads "
        "from disk (demand paging) or swap, restarts the faulting instruction. CoW: fork() "
        "marks all writable pages shared and read-only; first write copies the frame. mmap "
        "maps file pages directly into virtual space without explicit read calls. TLB "
        "shootdown: unmapping a page on one CPU requires IPIs to all cores to invalidate "
        "stale TLB entries — expensive at high core counts; huge 2MiB pages reduce pressure. "
        "NUMA: allocate on the node local to the accessing CPU; remote memory adds 40-100ns. "
        "Buddy allocator manages physical frames in power-of-2 blocks and coalesces buddies. "
        "Slab allocator carves pages into fixed-size caches for kernel structs. Overcommit "
        "allows more virtual than physical RAM; OOM killer SIGKILLs the highest-scoring "
        "process when physical memory is exhausted, scoring by RSS and oom_score_adj.",
    ),
    (
        "Explain in detail how distributed consensus algorithms work, specifically Raft. "
        "I want to understand: the FLP impossibility result, the roles of leader, follower, "
        "and candidate, leader election with randomized timeouts, log replication and "
        "commitment rules, how Raft handles split votes, network partitions, and leader "
        "crashes. Explain the joint consensus membership change protocol, how it prevents "
        "two independent majorities from forming. Contrast Raft with Paxos on "
        "understandability and practical implementation complexity. Explain linearizability "
        "in a Raft key-value store and how read-only queries can be served via leader leases "
        "without going through the log. Cover throughput and latency: batching, pipelining "
        "AppendEntries RPCs, and how performance differs on a LAN versus a WAN cluster.",
        "Raft decomposes consensus into leader election, log replication, and safety. "
        "Servers start as followers; no heartbeat within 150-300ms random timeout → "
        "increment term, become candidate, request votes. A server votes for the first "
        "candidate whose log is at least as up-to-date; majority wins. Leader appends "
        "entries and sends AppendEntries in parallel; entry is committed on majority ACK. "
        "Committed entries are never overwritten — the key safety invariant. Isolated leader "
        "eventually loses election; uncommitted entries are overwritten on rejoin. Joint "
        "consensus for membership changes: transition phase requires both old and new "
        "quorum agreement, preventing split-brain. Raft vs Paxos: Raft chose a strong "
        "leader for understandability; Paxos multi-decree form has no canonical "
        "implementation. Linearizability: route all writes through leader; reads use "
        "read-index commit or leader lease to avoid going through the log. Throughput: "
        "pipeline RPCs without waiting for prior ACKs, batch entries per heartbeat interval "
        "— thousands of ops/second on LAN, hundreds on cross-datacenter WAN.",
    ),
    (
        "Give me a thorough explanation of modern transformer neural networks. Start with "
        "scaled dot-product attention: what problem does self-attention solve that RNNs "
        "struggled with, and how is Q, K, V computed? Explain multi-head attention and why "
        "multiple heads are useful. Walk through the encoder block: LayerNorm, FFN with "
        "GELU, residual connections, positional encodings. Explain the decoder's causal mask "
        "and cross-attention. Cover BERT masked LM, GPT causal LM, and T5 span denoising "
        "pre-training objectives. Explain RLHF with PPO for alignment. Cover Chinchilla "
        "scaling laws, KV cache for O(n) autoregressive inference, FlashAttention's SRAM "
        "tiling, speculative decoding with a draft model, and 3D model parallelism: tensor, "
        "pipeline, and data parallelism as used in frontier model training and inference.",
        "Self-attention lets every token attend to all others in O(1) path length, solving "
        "RNN vanishing gradients over long sequences. Scaled dot-product: Q=XW_Q, K=XW_K, "
        "V=XW_V; output = softmax(QKT/sqrt(d_k))V; scaling prevents softmax saturation. "
        "Multi-head runs h independent heads with smaller d_k, concatenates and projects — "
        "different heads specialize on syntactic, semantic, and positional relations. "
        "Encoder block: LayerNorm → MultiHeadAttn → residual → LayerNorm → FFN(GELU) → "
        "residual. Decoder adds a causal mask and cross-attention over encoder output. "
        "BERT: mask 15%, predict bidirectionally. GPT: predict next token causally. T5: "
        "corrupt spans, reconstruct. RLHF: train reward model from human preferences, "
        "maximize reward with PPO. Chinchilla: optimal training scales model and tokens "
        "proportionally; loss follows power law in compute. KV cache avoids recomputing "
        "past K/V — O(n) per decoding step. FlashAttention tiles QKV in SRAM, cutting HBM "
        "bandwidth 5-10x. Speculative decoding: draft model proposes k tokens, large model "
        "verifies in one parallel pass, accepting all agreeing tokens at once.",
    ),
]

# Build padded pairs — each side padded to _TARGET_CHARS (~525 tokens/side)
FILLER_PAIRS = [(_pad(u), _pad(a)) for u, a in _PAIR_CORES]


# ---------------------------------------------------------------------------
# Probe API call
# ---------------------------------------------------------------------------


def send_probe(messages: list[dict], timeout: float = 60.0) -> dict:
    """
    Send the probe query with model='local' (bypasses complexity judge) and
    return tier, model, token count, and error.
    """
    payload = {
        "model": "local",  # Bypass judge — test local tier directly
        "messages": messages + [{"role": "user", "content": PROBE_QUERY}],
        "stream": True,
    }

    content_parts = []
    tier = None
    actual_model = None
    total_tokens = 0
    error = None

    try:
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream("POST", CHAT_ENDPOINT, json=payload) as response,
        ):
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "stream_metadata" in data:
                    meta = data["stream_metadata"]
                    tier = meta.get("tier", tier)
                    actual_model = meta.get("model", actual_model)
                    if "cost" in meta:
                        c = meta["cost"]
                        total_tokens = c.get("input_tokens", 0) + c.get("output_tokens", 0)
                    continue

                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    c = delta.get("content")
                    if c:
                        content_parts.append(c)

    except httpx.TimeoutException:
        error = "Request timed out"
    except httpx.ConnectError:
        error = f"Cannot connect to STREAM at {STREAM_URL}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return {
        "content": "".join(content_parts),
        "tier": tier,
        "model": actual_model,
        "total_tokens": total_tokens,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Conversation Simulation
# ---------------------------------------------------------------------------


def run_conversation(
    conversation_id: int,
    num_turns: int,
    probe_turns: list[int],
    verbose: bool = False,
) -> dict:
    """
    Simulate a num_turns conversation using synthetic filler history.

    Filler turns: inject pre-built ~1,050-token user+assistant pairs directly
    into messages — no API call. Context grows by ~1,050 tokens per filler turn.

    Probe turns: real API call with model='local' to measure whether compression
    kept context under the 32K hard limit. 'actual_tokens' is the real
    input+output count Ollama processed — the proof that compression is working.

    Without compression:
      turn 30 → 26,250 tokens (past 80% threshold, but still under 32K)
      turn 35 → 32,550 tokens → EXCEEDS Ollama 32K limit → error / upgrade
    With compression:
      turn 35 → actual ~5K tokens (summarized) → stays LOCAL
    """
    print(f"\n  --- Conversation {conversation_id} ---")

    messages: list[dict] = []
    probe_results = []
    filler_index = 0
    injected_tokens = 0  # cumulative injected chars / 4 ≈ tokens

    for turn in range(1, num_turns + 1):
        is_probe = turn in probe_turns

        if is_probe:
            # Real API call — measure tier and actual token count
            if verbose:
                print(
                    f"    [turn {turn:2d} PROBE]  injected≈{injected_tokens:,} tokens  ",
                    end="",
                    flush=True,
                )

            result = send_probe(messages)

            if result.get("error"):
                if verbose:
                    print(f"ERROR: {result['error']}")
                probe_results.append(
                    {
                        "turn": turn,
                        "tier": None,
                        "model": None,
                        "actual_tokens": 0,
                        "injected_tokens": injected_tokens,
                        "stayed_local": False,
                        "error": result["error"],
                    }
                )
                # Still extend history so later turns are realistic
                messages.append({"role": "user", "content": PROBE_QUERY})
                messages.append({"role": "assistant", "content": "4"})
            else:
                stayed_local = result["tier"] == "local"
                probe_results.append(
                    {
                        "turn": turn,
                        "tier": result["tier"],
                        "model": result["model"],
                        "actual_tokens": result["total_tokens"],
                        "injected_tokens": injected_tokens,
                        "stayed_local": stayed_local,
                        "error": None,
                    }
                )
                messages.append({"role": "user", "content": PROBE_QUERY})
                messages.append({"role": "assistant", "content": result["content"] or "4"})

                tier_str = result["tier"] or "unknown"
                status = "LOCAL" if stayed_local else f"UPGRADED → {tier_str.upper()}"
                actual = result["total_tokens"]
                if verbose:
                    print(f"{status}  actual_tokens={actual}")
                else:
                    print(
                        f"    [turn {turn:2d} PROBE]  "
                        f"injected≈{injected_tokens:,}  actual={actual}  {status}"
                    )
        else:
            # Synthetic filler — inject directly, no API call
            pair = FILLER_PAIRS[filler_index % len(FILLER_PAIRS)]
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})
            filler_index += 1
            pair_tokens = (len(pair[0]) + len(pair[1])) // 4
            injected_tokens += pair_tokens
            if verbose:
                print(
                    f"    [turn {turn:2d} filler] injected≈{injected_tokens:,} tokens "
                    f"(+{pair_tokens} this turn, no API call)"
                )

    return {
        "conversation_id": conversation_id,
        "probes": probe_results,
    }


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_compression_benchmark(
    num_conversations: int = 5,
    num_turns: int = DEFAULT_NUM_TURNS,
    probe_turns: list[int] = None,
    label: str = "",
    verbose: bool = False,
) -> dict:
    """Run compression impact benchmark across multiple conversations."""

    if probe_turns is None:
        probe_turns = DEFAULT_PROBE_TURNS

    # Measure actual pair sizes so we can report accurately
    pair_tokens = sum((len(p[0]) + len(p[1])) // 4 for p in FILLER_PAIRS) // len(FILLER_PAIRS)
    filler_count = num_turns - len(probe_turns)
    max_injected = filler_count * pair_tokens

    print(f"\n{'='*60}")
    print(f"  Compression Impact Benchmark{' [' + label + ']' if label else ''}")
    print(f"  Conversations: {num_conversations}")
    print(f"  Turns: {num_turns} ({filler_count} synthetic + {len(probe_turns)} probes)")
    print(f"  Filler pair size: ~{pair_tokens} tokens each (padded to {_TARGET_CHARS} chars/side)")
    print(f"  Max injected context: ~{max_injected:,} tokens (turn {num_turns})")
    print(f"  Compression threshold: {COMPRESSION_THRESHOLD_TOKENS:,} tokens (80% of 32K)")
    print(f"  Ollama hard limit:     {LOCAL_CONTEXT_HARD_LIMIT:,} tokens (32K)")
    print(f"  Threshold crossed at:  turn ~{COMPRESSION_THRESHOLD_TOKENS // pair_tokens + 1}")
    print(f"  Hard limit crossed at: turn ~{LOCAL_CONTEXT_HARD_LIMIT // pair_tokens + 1}")
    print(f"  Probe turns: {probe_turns}")
    print(f'  Probe query: "{PROBE_QUERY}"')
    print(f"{'='*60}")

    all_conversations = []
    for conv_id in range(1, num_conversations + 1):
        conv_result = run_conversation(conv_id, num_turns, probe_turns, verbose)
        all_conversations.append(conv_result)

    # Aggregate probe results by turn
    turn_stats = {}
    for turn in probe_turns:
        probes_at_turn = [p for c in all_conversations for p in c["probes"] if p["turn"] == turn]
        stayed_local = sum(1 for p in probes_at_turn if p["stayed_local"])
        total = len(probes_at_turn)
        avg_injected = sum(p["injected_tokens"] for p in probes_at_turn) / total if total > 0 else 0
        avg_actual = sum(p["actual_tokens"] for p in probes_at_turn) / total if total > 0 else 0
        turn_stats[turn] = {
            "stayed_local": stayed_local,
            "total": total,
            "local_pct": stayed_local / total * 100 if total > 0 else 0,
            "avg_injected_tokens": int(avg_injected),
            "avg_actual_tokens": int(avg_actual),
            "tiers_used": [p["tier"] for p in probes_at_turn],
        }

    total_probes = sum(len(c["probes"]) for c in all_conversations)
    total_stayed_local = sum(1 for c in all_conversations for p in c["probes"] if p["stayed_local"])
    forced_upgrades = total_probes - total_stayed_local

    summary = {
        "label": label,
        "num_conversations": num_conversations,
        "num_turns": num_turns,
        "compression_threshold_tokens": COMPRESSION_THRESHOLD_TOKENS,
        "local_context_hard_limit": LOCAL_CONTEXT_HARD_LIMIT,
        "filler_pair_tokens_avg": pair_tokens,
        "probe_turns": probe_turns,
        "probe_query": PROBE_QUERY,
        "per_turn": turn_stats,
        "total_probes": total_probes,
        "total_stayed_local": total_stayed_local,
        "forced_upgrades": forced_upgrades,
        "local_retention_pct": total_stayed_local / total_probes * 100 if total_probes > 0 else 0,
        "conversations": all_conversations,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  COMPRESSION IMPACT RESULTS{' [' + label + ']' if label else ''}")
    print(f"{'='*60}")
    print("\n  injected = cumulative synthetic tokens in history before probe")
    print("  actual   = real input+output tokens Ollama processed")
    print("  Without compression: actual ≈ injected (grows unbounded → fails at 32K)")
    print("  With compression:    actual << injected after threshold (stays local)\n")
    print(
        f"  {'Turn':>4}  {'Injected':>10}  {'Actual':>8}  {'Reduction':>10}  {'Local':>7}  {'%':>5}"
    )
    print(f"  {'-'*57}")
    for turn in probe_turns:
        ts = turn_stats[turn]
        inj = ts["avg_injected_tokens"]
        act = ts["avg_actual_tokens"]
        if inj > 0 and act < inj:
            reduction = f"{(1 - act/inj)*100:.0f}%"
        elif ts["total"] > 0 and ts["stayed_local"] == 0:
            reduction = "FAILED"
        else:
            reduction = "none"
        markers = []
        if inj > COMPRESSION_THRESHOLD_TOKENS:
            markers.append("≥threshold")
        if inj > LOCAL_CONTEXT_HARD_LIMIT:
            markers.append("≥32K LIMIT")
        marker_str = f"  ← {', '.join(markers)}" if markers else ""
        print(
            f"  {turn:>4}  {inj:>10,}  {act:>8,}  {reduction:>10}  "
            f"  {ts['stayed_local']}/{ts['total']}  {ts['local_pct']:>5.0f}%"
            f"{marker_str}"
        )

    print(f"\n  Forced tier upgrades: {forced_upgrades}/{total_probes}")
    print(f"  Local retention:     {summary['local_retention_pct']:.0f}%")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="STREAM Compression Impact Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
With/without comparison (Option B):
  Step 1: python benchmark_compression.py --label with
  Step 2: Stop STREAM, restart with ROLLING_SUMMARIZATION_ENABLED=false
  Step 3: python benchmark_compression.py --label without
  Step 4: Restart STREAM normally to re-enable compression
        """,
    )
    parser.add_argument(
        "--conversations", type=int, default=5, help="Number of conversations (default: 5)"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_NUM_TURNS,
        help=f"Turns per conversation (default: {DEFAULT_NUM_TURNS}). "
        f"Use 40 for the with/without comparison.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Label for output file: 'with' or 'without' (default: '')",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-turn details including filler injections"
    )
    parser.add_argument("--url", type=str, help="STREAM URL (default: http://localhost:5000)")
    args = parser.parse_args()

    if args.url:
        global STREAM_URL, CHAT_ENDPOINT
        STREAM_URL = args.url
        CHAT_ENDPOINT = f"{STREAM_URL}/v1/chat/completions"

    # Build probe turns dynamically from --turns value.
    # Always probe at turn 10 (baseline), 20 (approaching threshold),
    # 30 (just past compression threshold), and the last two turns.
    num_turns = args.turns
    if num_turns >= 35:
        candidates = [10, 20, 30, 35, num_turns]
        # Deduplicate while preserving order, drop any turn > num_turns
        seen = set()
        probe_turns = []
        for t in candidates:
            if t <= num_turns and t not in seen:
                probe_turns.append(t)
                seen.add(t)
    else:
        # Short run: spread probes evenly
        step = max(1, num_turns // 5)
        probe_turns = list(range(step, num_turns + 1, step))[:5]
        if num_turns not in probe_turns:
            probe_turns.append(num_turns)

    # Check connectivity
    print(f"Connecting to STREAM at {STREAM_URL}...")
    try:
        r = httpx.get(f"{STREAM_URL}/health", timeout=5)
        print(f"  Connected! Status: {r.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to STREAM: {e}")
        print(f"  Make sure STREAM is running at {STREAM_URL}")
        sys.exit(1)

    summary = run_compression_benchmark(
        num_conversations=args.conversations,
        num_turns=num_turns,
        probe_turns=probe_turns,
        label=args.label,
        verbose=args.verbose,
    )

    # Save results — label goes in filename for easy side-by-side comparison
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    label_part = f"_{args.label}" if args.label else ""
    output_file = results_dir / f"compression{label_part}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "stream_url": STREAM_URL,
                "summary": summary,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
