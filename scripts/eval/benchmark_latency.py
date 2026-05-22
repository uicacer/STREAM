#!/usr/bin/env python3
"""
STREAM Latency & Throughput Benchmark (Day 1)

Measures TTFT, total response time, and throughput across all tiers.
Also compares relay streaming vs batch mode for the Lakeshore tier.

Usage:
    python scripts/eval/benchmark_latency.py                  # Full benchmark (20 runs)
    python scripts/eval/benchmark_latency.py --runs 3          # Quick test (3 runs)
    python scripts/eval/benchmark_latency.py --tiers local     # Specific tier only
    python scripts/eval/benchmark_latency.py --relay-comparison-only  # Lakeshore relay vs batch

Results are saved to scripts/eval/results/latency_<timestamp>.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STREAM_URL = os.getenv("STREAM_URL", "http://localhost:5000")
CHAT_ENDPOINT = f"{STREAM_URL}/v1/chat/completions"

# Pool of distinct benchmark queries. Each run draws a different query so that
# prefix caching (vLLM, Ollama, cloud) does not artificially lower TTFT by
# reusing cached KV computations from a previous identical prompt.
# All queries are comparable in length (~10-15 tokens) so prompt-length effects
# do not confound latency comparisons across runs.
BENCHMARK_QUERIES = [
    # --- Block A: History / Social Science / Economics (indices 0-49) ---
    "What were the main causes of the First World War and how did the alliance system contribute?",
    "How did the printing press change the spread of knowledge in Europe?",
    "What is the tragedy of the commons and give a real-world example?",
    "How did the Cold War shape the foreign policy of non-aligned nations?",
    "Explain what supply and demand curves represent in economics.",
    "What were the key ideas of the Enlightenment and how did they influence revolutions?",
    "How does inflation erode purchasing power over time?",
    "What is the difference between a parliamentary and a presidential system of government?",
    "How did the Silk Road facilitate cultural exchange between civilizations?",
    "Explain what game theory is and describe the prisoner's dilemma.",
    "What caused the Great Depression and what policies helped end it?",
    "How does comparative advantage explain why countries trade with each other?",
    "What is the role of central banks in managing monetary policy?",
    "How did colonialism shape the economies of African nations?",
    "Explain what GDP measures and what it fails to capture about well-being.",
    "What was the significance of the Magna Carta in the development of constitutional law?",
    "How does behavioral economics differ from classical economics?",
    "What is the difference between a market economy and a command economy?",
    "How did the Industrial Revolution change social structures in Britain?",
    "Explain what the social contract theory argues about the origin of government.",
    "What is a carbon tax and how does it aim to reduce emissions?",
    "How did the Green Revolution transform food production in developing countries?",
    "What is the difference between a recession and a depression?",
    "How does gerrymandering affect political representation?",
    "Explain what the Marshall Plan was and why it was implemented.",
    "What is the difference between civil law and common law traditions?",
    "How did the civil rights movement in the US use nonviolent resistance?",
    "What is universal basic income and what are the main arguments for and against it?",
    "How did the Ottoman Empire maintain control over such a diverse territory?",
    "Explain what Keynesian economics argues about government spending in recessions.",
    "What is the difference between a tariff and a quota in trade policy?",
    "How did the invention of writing change human civilization?",
    "What is the role of the IMF in the global financial system?",
    "How does the electoral college system work in US presidential elections?",
    "Explain what the Arab Spring was and why its outcomes varied so much by country.",
    "What is the difference between a federal and a unitary state?",
    "How did decolonization after World War II reshape the map of Africa?",
    "What is moral hazard in economics and give a financial example?",
    "How does the World Trade Organization resolve trade disputes?",
    "Explain what the Gini coefficient measures about income inequality.",
    "What caused the 2008 financial crisis and what were its main effects?",
    "How did the Renaissance change European art and intellectual life?",
    "What is the difference between socialism and communism?",
    "How does the United Nations Security Council make binding decisions?",
    "Explain what stagflation is and why it is difficult to fix with standard policy tools.",
    "What was the significance of the Treaty of Westphalia for modern state sovereignty?",
    "How does propaganda work to shape public opinion?",
    "What is the difference between a bond and a stock?",
    "How did the Black Death change European society and the economy?",
    "Explain what the Bretton Woods system established and why it collapsed.",
    # --- Block B: Philosophy / Ethics / Psychology (indices 50-99) ---
    "What is the trolley problem and what does it reveal about moral intuitions?",
    "How does Kant's categorical imperative differ from utilitarian ethics?",
    "What is the difference between deductive and inductive reasoning?",
    "Explain Plato's allegory of the cave and what it says about knowledge.",
    "What is cognitive dissonance and how do people typically resolve it?",
    "How does confirmation bias affect how we evaluate new information?",
    "Explain what free will means and describe one argument against it.",
    "What is the difference between epistemology and metaphysics?",
    "How does Maslow's hierarchy of needs describe human motivation?",
    "Explain what the Milgram obedience experiment revealed about authority.",
    "What is the hard problem of consciousness and why is it considered hard?",
    "How does classical conditioning differ from operant conditioning?",
    "Explain what the Socratic method is and how it works in practice.",
    "What is the difference between a belief, knowledge, and justified true belief?",
    "How does the bystander effect explain why people sometimes fail to help?",
    "Explain what existentialism claims about meaning and human freedom.",
    "What is the difference between intrinsic and extrinsic motivation?",
    "How does the availability heuristic lead to systematic errors in judgment?",
    "Explain what the philosophy of mind problem of other minds consists of.",
    "What is the difference between empiricism and rationalism?",
    "How does attachment theory explain adult relationship patterns?",
    "Explain what the utilitarian calculus tries to maximize and its main criticism.",
    "What is the Dunning-Kruger effect and what does research actually show?",
    "How does the social learning theory of Bandura explain behavior acquisition?",
    "Explain what epistemic humility means and why philosophers value it.",
    "What is the difference between moral realism and moral relativism?",
    "How does working memory differ from long-term memory in cognitive psychology?",
    "Explain what Rawls's veil of ignorance thought experiment is designed to do.",
    "What is the placebo effect and what does it reveal about mind-body interaction?",
    "How does the philosophy of language address how words acquire meaning?",
    "Explain what virtue ethics focuses on that consequentialism and deontology neglect.",
    "What is the difference between a psychological disorder and normal variation?",
    "How does priming influence subsequent perception and behavior?",
    "Explain what the problem of induction is and why Hume raised it.",
    "What is the difference between explicit and implicit memory?",
    "How does the concept of cognitive load apply to learning and instruction design?",
    "Explain what nihilism claims and how it differs from pessimism.",
    "What is the difference between correlation and causation in psychology research?",
    "How does the theory of mind develop in children and what happens when it doesn't?",
    "Explain what the social contract theory of Rousseau emphasizes.",
    "What is the difference between consciousness and self-awareness?",
    "How does schema theory explain how humans organize and recall knowledge?",
    "Explain what the is-ought problem in ethics states.",
    "What is the difference between a longitudinal and a cross-sectional study?",
    "How does the concept of flow in psychology describe optimal experience?",
    "Explain what Wittgenstein meant by language games.",
    "What is the difference between phenomenology and behaviorism as approaches to psychology?",
    "How does the halo effect bias our judgments of people?",
    "Explain what the paradox of tolerance argues about free societies.",
    "What is the difference between a delusion and a strongly held false belief?",
    # --- Block C: Engineering / Technology / Architecture (indices 100-149) ---
    "How does a transformer architecture process sequential data differently from an RNN?",
    "What is the difference between a CPU cache miss and a cache hit?",
    "How does RAID storage provide redundancy and what are its levels?",
    "Explain what a finite state machine is and give a practical example.",
    "What is the difference between lossy and lossless compression?",
    "How does a compiler perform register allocation during code generation?",
    "Explain what a digital twin is and how it is used in engineering.",
    "What is the difference between a 3NF and BCNF database normalization form?",
    "How does a PID controller work to maintain a target value?",
    "Explain what branch prediction does in a CPU pipeline.",
    "What is the difference between a von Neumann and a Harvard architecture?",
    "How does error correction code memory detect and fix bit flips?",
    "Explain what the actor model of concurrency is and how it avoids shared state.",
    "What is the difference between polling and interrupts for I/O handling?",
    "How does a write-ahead log ensure database durability after a crash?",
    "Explain what a software design pattern is and describe the observer pattern.",
    "What is the difference between symmetric and asymmetric multiprocessing?",
    "How does a compiler implement function call conventions using a stack frame?",
    "Explain what a hypervisor does and the difference between Type 1 and Type 2.",
    "What is the difference between a JIT compiler and an AOT compiler?",
    "How does the copy-on-write mechanism work in operating systems?",
    "Explain what a memory barrier instruction does in concurrent programming.",
    "What is the difference between a sparse and a dense matrix representation?",
    "How does a lock-free data structure avoid using mutexes?",
    "Explain what a type system provides beyond catching simple errors.",
    "What is the difference between synchronous and asynchronous exceptions?",
    "How does a linker resolve symbol references between object files?",
    "Explain what software entropy is and how refactoring addresses it.",
    "What is the difference between mutable and immutable data structures?",
    "How does a relational database execute a join operation efficiently?",
    "Explain what SIMD instructions do and why they speed up certain computations.",
    "What is the difference between a stack overflow and a heap overflow?",
    "How does a garbage collector handle cyclic references?",
    "Explain what the SOLID principles of object-oriented design describe.",
    "What is the difference between a monad and a functor in functional programming?",
    "How does a neural network overfit and what techniques prevent it?",
    "Explain what a protocol buffer is and how it differs from JSON serialization.",
    "What is the difference between a primary key and a unique key in SQL?",
    "How does the OAuth2 authorization flow work?",
    "Explain what a circuit board PCB trace resistance affects at high frequencies.",
    "What is the difference between a semaphore and a monitor in concurrent programming?",
    "How does a distributed tracing system track requests across services?",
    "Explain what chaos engineering is and why Netflix pioneered it.",
    "What is the difference between blue-green and canary deployments?",
    "How does a content-addressable storage system like Git store objects?",
    "Explain what a software bill of materials is and why it matters for security.",
    "What is the difference between horizontal pod autoscaling and vertical pod autoscaling?",
    "How does a probabilistic data structure trade accuracy for memory efficiency?",
    "Explain what a long-tail latency problem is in distributed systems.",
    "What is the difference between a compiler warning and a compiler error?",
    # --- Block D: Art / Literature / Culture / Language (indices 150-199) ---
    "What is magical realism in literature and which authors are most associated with it?",
    "How does meter work in poetry and what is the difference between iambic and trochaic?",
    "Explain what the unreliable narrator technique does in fiction.",
    "What is the difference between modernism and postmodernism in literature?",
    "How did jazz develop from blues and ragtime in the early twentieth century?",
    "Explain what stream of consciousness writing is and give a famous example.",
    "What is the difference between a metaphor and a simile?",
    "How does color theory in visual art describe complementary colors?",
    "Explain what the Socratic dialogue format achieves that a lecture cannot.",
    "What is the difference between tragedy and comedy in classical drama?",
    "How does the Sapir-Whorf hypothesis argue language shapes thought?",
    "Explain what minimalism in art reacts against.",
    "What is the difference between connotation and denotation in language?",
    "How does the hero's journey structure appear across different mythologies?",
    "Explain what impressionism tried to capture that academic painting did not.",
    "What is the difference between a dialect and a language?",
    "How does foreshadowing build tension in narrative fiction?",
    "Explain what the uncanny valley effect is in robotics and animation.",
    "What is the difference between a sonnet and a haiku?",
    "How does typography affect how readers perceive and process text?",
    "Explain what surrealism was reacting to in early twentieth century art.",
    "What is the difference between a theme and a motif in literature?",
    "How does the concept of negative space function in visual design?",
    "Explain what oral tradition is and how stories change as they are retold.",
    "What is the difference between diegetic and non-diegetic sound in film?",
    "How does point of view affect what a reader knows and feels in a novel?",
    "Explain what the Bauhaus school tried to achieve by uniting art and craft.",
    "What is the difference between a simile and an analogy?",
    "How does repetition function differently in music versus in poetry?",
    "Explain what the canon in literature means and why it is contested.",
    "What is the difference between satire and parody?",
    "How did Renaissance perspective change the way painters depicted space?",
    "Explain what code-switching is in linguistics.",
    "What is the difference between a protagonist and an antihero?",
    "How does the Gestalt principle of grouping work in visual perception?",
    "Explain what dialect continuum means in linguistics.",
    "What is the difference between an allegory and a symbol?",
    "How does genre fiction differ from literary fiction in its goals?",
    "Explain what the pathetic fallacy is in literature.",
    "What is the difference between rhythm and tempo in music?",
    "How did cubism challenge traditional representation in painting?",
    "Explain what intertextuality means and give an example.",
    "What is the difference between a reliable and an unreliable narrator?",
    "How does the concept of cultural capital explain social inequality?",
    "Explain what the Turing test was designed to measure.",
    "What is the difference between a pidgin and a creole language?",
    "How does defamiliarization work as a literary technique?",
    "Explain what the term avant-garde means in the context of art movements.",
    "What is the difference between a myth and a legend?",
    "How does dramatic irony differ from situational irony?",
]

CACHED_QUERY = BENCHMARK_QUERIES[0]  # Fixed query used for KV cache warm runs


def get_benchmark_query(run_index: int, cached: bool = False) -> str:
    """Return a query for the given run. Same query every run when cached=True."""
    if cached:
        return CACHED_QUERY
    return BENCHMARK_QUERIES[run_index % len(BENCHMARK_QUERIES)]


# Use exact tier names ("local", "lakeshore", "cloud") to bypass the complexity
# judge and measure pure tier inference latency. Routing overhead (judge) is
# reported separately from the routing benchmark (avg ~0.68s).
TIER_MODELS = {
    "local": "local",
    "lakeshore": "lakeshore",  # tier keyword bypasses complexity judge; middleware uses default lakeshore model
    "cloud": "cloud",  # tier keyword bypasses complexity judge; middleware uses default cloud model (cloud-or-claude)
}

# ---------------------------------------------------------------------------
# Hardware Detection
# ---------------------------------------------------------------------------


def detect_hardware() -> dict:
    """Auto-detect local hardware for reproducibility reporting."""
    hw = {
        "os": "",
        "cpu": "",
        "ram": "",
        "gpu": None,
    }

    # OS
    system = platform.system()
    if system == "Darwin":
        ver = platform.mac_ver()[0]
        hw["os"] = f"macOS {ver} ({platform.machine()})"
    elif system == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME"):
                        hw["os"] = line.split("=")[1].strip().strip('"')
                        break
        except Exception:
            hw["os"] = f"Linux {platform.release()}"
    else:
        hw["os"] = f"{system} {platform.release()}"

    # CPU
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                hw["cpu"] = result.stdout.strip()
            else:
                # Apple Silicon
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in result.stdout.split("\n"):
                    if "Chip" in line and ":" in line:
                        hw["cpu"] = f"Apple {line.split(':')[1].strip()}"
                        break
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line.lower():
                        hw["cpu"] = line.split(":")[1].strip()
                        break
    except Exception:
        hw["cpu"] = platform.processor() or "Unknown"

    # RAM
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                ram_gb = int(result.stdout.strip()) / (1024**3)
                hw["ram"] = f"{ram_gb:.0f} GB"
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        hw["ram"] = f"{kb / (1024**2):.0f} GB"
                        break
    except Exception:
        hw["ram"] = "Unknown"

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                hw["gpu"] = f"{parts[0].strip()} {int(parts[1].strip()) // 1024}GB"
    except FileNotFoundError:
        pass

    if not hw["gpu"] and system == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split("\n"):
                if "Chipset Model" in line:
                    hw["gpu"] = f"{line.split(':')[1].strip()} (Metal)"
                    break
        except Exception:
            pass

    return hw


# ---------------------------------------------------------------------------
# SSE Parsing Helpers
# ---------------------------------------------------------------------------


def send_streaming_request(model: str, query: str, timeout: float = 180.0) -> dict:
    """
    Send a streaming chat request to STREAM and measure timing.

    Returns:
        {
            "ttft": float,          # Time to first content token (seconds)
            "total_time": float,    # Total response time (seconds)
            "output_tokens": int,   # Number of output tokens (from cost summary)
            "throughput": float,    # Tokens per second
            "tier": str,            # Which tier actually handled it
            "model": str,           # Which model was used
            "content_length": int,  # Response length in characters
            "error": str | None,    # Error message if failed
        }
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    content_parts = []
    output_tokens = 0
    tier = None
    actual_model = None
    error = None

    try:
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream("POST", CHAT_ENDPOINT, json=payload) as response,
        ):
            if response.status_code != 200:
                response.read()  # must read before accessing .text on a streaming response
                return {"error": f"HTTP {response.status_code}: {response.text}"}

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Strip "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Stream metadata (tier, model info)
                if "stream_metadata" in data:
                    meta = data["stream_metadata"]
                    if "tier" in meta:
                        tier = meta["tier"]
                    if "model" in meta:
                        actual_model = meta["model"]
                    # Cost summary at the end
                    if "cost" in meta:
                        cost_info = meta["cost"]
                        output_tokens = cost_info.get("output_tokens", 0)
                    continue

                # Content tokens
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        content_parts.append(content)

    except httpx.TimeoutException:
        error = "Request timed out"
    except httpx.ConnectError:
        error = f"Cannot connect to STREAM at {STREAM_URL}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_total = time.perf_counter() - t_start
    content_text = "".join(content_parts)

    # Calculate throughput
    throughput = output_tokens / t_total if output_tokens > 0 and t_total > 0 else 0

    return {
        "ttft": ttft,
        "total_time": t_total,
        "output_tokens": output_tokens,
        "throughput": throughput,
        "tier": tier,
        "model": actual_model,
        "content_length": len(content_text),
        "error": error,
    }


def send_batch_request(model: str, query: str, timeout: float = 180.0) -> dict:
    """
    Simulate batch mode by consuming the full SSE stream before recording TTFT.

    The STREAM server always returns SSE (even with stream=False). In real batch
    mode (no relay), Globus Compute returns the complete response at once, so the
    user sees nothing until all tokens arrive. We simulate this by reading the
    entire stream and setting TTFT = total time.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "stream": True,
    }

    t_start = time.perf_counter()
    output_tokens = 0
    content_parts = []
    error = None
    actual_model = model

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
                    if "model" in meta:
                        actual_model = meta["model"]
                    if "cost" in meta:
                        output_tokens = meta["cost"].get("output_tokens", 0)
                    continue

                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        content_parts.append(content)

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_total = time.perf_counter() - t_start

    return {
        "ttft": t_total,  # In batch mode, TTFT = total time (user waits for everything)
        "total_time": t_total,
        "output_tokens": output_tokens,
        "throughput": output_tokens / t_total if output_tokens > 0 else 0,
        "tier": model.split("-")[0] if "-" in model else "unknown",
        "model": actual_model,
        "content_length": len("".join(content_parts)),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_tier_benchmark(
    tier: str,
    model: str,
    runs: int,
    warmup: bool = True,
    cached: bool = False,
    query_offset: int = 0,
) -> dict:
    """Run latency benchmark for a single tier."""
    cache_note = " [CACHED — same query, KV cache warm]" if cached else ""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {tier} ({model}){cache_note}")
    print(f"  Runs: {runs} (+1 warmup)" if warmup else f"  Runs: {runs}")
    print(f"{'='*60}")

    results = []

    total_runs = runs + (1 if warmup else 0)
    for i in range(total_runs):
        is_warmup = warmup and i == 0
        label = "warmup" if is_warmup else f"run {i if not warmup else i}"
        query = get_benchmark_query(query_offset + i, cached=cached)

        print(f"  [{label}] ", end="", flush=True)
        result = send_streaming_request(model, query)

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            if not is_warmup:
                results.append(result)
            continue

        ttft_val = result.get("ttft")
        print(
            f"TTFT={ttft_val:.2f}s  " if ttft_val is not None else "TTFT=N/A  ",
            f"Total={result['total_time']:.2f}s  "
            f"Tokens={result['output_tokens']}  "
            f"Throughput={result['throughput']:.1f} tok/s",
            sep="",
        )

        if not is_warmup:
            results.append(result)

    # Calculate statistics
    successful = [r for r in results if not r.get("error") and r.get("ttft") is not None]
    failed = len(results) - len(successful)

    if not successful:
        return {
            "tier": tier,
            "model": model,
            "runs": runs,
            "successful": 0,
            "failed": failed,
            "error": "All runs failed",
        }

    ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
    totals = [r["total_time"] for r in successful]
    throughputs = [r["throughput"] for r in successful if r["throughput"] > 0]

    def _p95(data: list) -> float | None:
        if len(data) < 2:
            return None
        return statistics.quantiles(data, n=100)[94]  # index 94 = p95

    stats = {
        "tier": tier,
        "model": model,
        "runs": runs,
        "cached_query": cached,
        "successful": len(successful),
        "failed": failed,
        "ttft": {
            "median": statistics.median(ttfts) if ttfts else None,
            "mean": statistics.mean(ttfts) if ttfts else None,
            "stdev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            "p95": _p95(ttfts),
            "min": min(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
            "values": ttfts,
        },
        "total_time": {
            "median": statistics.median(totals),
            "mean": statistics.mean(totals),
            "stdev": statistics.stdev(totals) if len(totals) > 1 else 0,
            "p95": _p95(totals),
            "min": min(totals),
            "max": max(totals),
            "values": totals,
        },
        "throughput": {
            "median": statistics.median(throughputs) if throughputs else None,
            "mean": statistics.mean(throughputs) if throughputs else None,
            "stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            "values": throughputs,
        },
    }

    print("\n  Summary:")
    ttft_p95 = stats["ttft"]["p95"]
    print(
        f"    TTFT:       {stats['ttft']['median']:.2f}s (median) +/- {stats['ttft']['stdev']:.2f}s"
        + (f"  p95={ttft_p95:.2f}s" if ttft_p95 else "")
    )
    total_p95 = stats["total_time"]["p95"]
    print(
        f"    Total:      {stats['total_time']['median']:.2f}s (median) +/- {stats['total_time']['stdev']:.2f}s"
        + (f"  p95={total_p95:.2f}s" if total_p95 else "")
    )
    if stats["throughput"]["median"]:
        print(f"    Throughput: {stats['throughput']['median']:.1f} tok/s (median)")

    return stats


def run_cloud_thinking_benchmark(
    runs: int, query_offset: int = 0, thinking_only: bool = False
) -> dict:
    """
    Benchmark cloud tier with and without extended thinking at all effort levels.

    Tests two configurations via OpenRouter → Claude Sonnet 4:
    - no thinking (baseline)
    - thinking at medium effort (~5K thinking tokens, representative of typical use)

    Returns a dict keyed by configuration name with median TTFT, total time, and
    throughput. Useful for quantifying the latency cost of extended thinking.
    """
    # All configs use cloud-or-claude (OpenRouter → Anthropic) for apples-to-apples
    # comparison. Thinking is controlled via OpenRouter's native `reasoning` param
    # (passed as extra_body). effort=None means no reasoning param sent at all.
    thinking_configs = [
        {
            "label": "no_thinking",
            "model": "cloud-claude",
            "reasoning_effort": None,
            "description": "No thinking (direct Anthropic)",
        },
        # medium: ~500-1000 thinking tokens, minimal TTFT impact
        # {"label": "effort_medium", "model": "cloud-claude", "reasoning_effort": "medium",
        #  "description": "Thinking: medium (direct Anthropic)"},
        {
            "label": "effort_high",
            "model": "cloud-claude",
            "reasoning_effort": "high",
            "description": "Thinking: high (direct Anthropic, ~16K token budget)",
        },
    ]

    # Filter configs based on --thinking-only flag
    if thinking_only:
        thinking_configs = [c for c in thinking_configs if c["reasoning_effort"] is not None]

    results = {}

    for cfg_idx, cfg in enumerate(thinking_configs):
        label = cfg["label"]
        model = cfg["model"]
        effort = cfg["reasoning_effort"]
        desc = cfg["description"]
        # Each config gets its own non-overlapping query block
        config_offset = query_offset + cfg_idx * (runs + 1)

        print(f"\n{'='*60}")
        print(f"  Cloud Thinking: {desc}")
        print(f"  Model: {model}  effort={effort}")
        print(f"  Runs: {runs} (+1 warmup, discarded)")
        print(f"{'='*60}")

        run_results = []
        for i in range(runs + 1):
            is_warmup = i == 0
            label_str = "warmup" if is_warmup else f"run {i}/{runs}"
            query = get_benchmark_query(config_offset + i)
            print(f"  [{label_str}] ", end="", flush=True)

            # Always use "cloud" as tier bypass keyword; cloud_provider selects
            # the specific model within the cloud tier when needed.
            payload = {
                "model": "cloud",
                "messages": [{"role": "user", "content": query}],
                "stream": True,
            }
            if model != "cloud":
                payload["cloud_provider"] = model
            if effort is not None:
                payload["reasoning_effort"] = effort

            t_start = time.perf_counter()
            ttft = None
            content_parts = []
            output_tokens = 0
            error = None

            try:
                with (
                    httpx.Client(timeout=300.0) as client,
                    client.stream("POST", CHAT_ENDPOINT, json=payload) as response,
                ):
                    if response.status_code != 200:
                        body = response.read().decode("utf-8", errors="replace")[:200]
                        error = f"HTTP {response.status_code}: {body}"
                    else:
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
                                if "cost" in meta:
                                    output_tokens = meta["cost"].get("output_tokens", 0)
                                continue
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    if ttft is None:
                                        ttft = time.perf_counter() - t_start
                                    content_parts.append(content)
            except httpx.TimeoutException:
                error = "Request timed out"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

            t_total = time.perf_counter() - t_start
            result = {
                "ttft": ttft,
                "total_time": t_total,
                "output_tokens": output_tokens,
                "throughput": output_tokens / t_total if output_tokens > 0 else 0,
                "error": error,
            }

            ttft_val = result.get("ttft")
            if error:
                print(f"ERROR: {error}")
            elif ttft_val is None:
                print(
                    f"TTFT=N/A  Total={t_total:.2f}s  Tokens={output_tokens}  (no content received)"
                )
            else:
                print(
                    f"TTFT={ttft_val:.2f}s  " f"Total={t_total:.2f}s  Tokens={output_tokens}",
                )

            if not is_warmup:
                run_results.append(result)

            # Pause between requests to avoid provider rate limiting.
            # Thinking requests consume far more tokens, so need longer gap.
            sleep_time = 8 if effort is not None else 2
            time.sleep(sleep_time)

        successful = [r for r in run_results if not r.get("error") and r.get("ttft") is not None]

        def _p95(data: list) -> float | None:
            return statistics.quantiles(data, n=100)[94] if len(data) >= 2 else None

        if successful:
            ttfts = [r["ttft"] for r in successful]
            totals = [r["total_time"] for r in successful]
            throughputs = [r["throughput"] for r in successful if r["throughput"] > 0]
            results[label] = {
                "description": desc,
                "model": model,
                "reasoning_effort": effort,
                "successful": len(successful),
                "failed": runs - len(successful),
                "ttft_median": statistics.median(ttfts),
                "ttft_stdev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
                "ttft_p95": _p95(ttfts),
                "total_median": statistics.median(totals),
                "total_stdev": statistics.stdev(totals) if len(totals) > 1 else 0,
                "total_p95": _p95(totals),
                "throughput_median": statistics.median(throughputs) if throughputs else None,
            }
            p95_str = (
                f"  p95={results[label]['ttft_p95']:.2f}s" if results[label]["ttft_p95"] else ""
            )
            print(
                f"\n  TTFT: {results[label]['ttft_median']:.2f}s (median) "
                f"+/- {results[label]['ttft_stdev']:.2f}s{p95_str}"
            )
            print(f"  Total: {results[label]['total_median']:.2f}s (median)")
        else:
            results[label] = {"description": desc, "model": model, "error": "All runs failed"}

    # Print summary comparison table
    print(f"\n{'='*60}")
    print("  Cloud Thinking Effort Comparison")
    print(f"{'='*60}")
    print(f"  {'Configuration':<35} {'TTFT (s)':<20} {'Total (s)':<15}")
    print(f"  {'-'*70}")
    for _lbl, r in results.items():
        if r.get("error"):
            print(f"  {r['description']:<35} ERROR")
            continue
        p95_str = f" p95={r['ttft_p95']:.2f}" if r.get("ttft_p95") else ""
        print(
            f"  {r['description']:<35} "
            f"{r['ttft_median']:.2f}+/-{r['ttft_stdev']:.2f}{p95_str:<12}  "
            f"{r['total_median']:.2f}+/-{r['total_stdev']:.2f}"
        )

    return results


def run_relay_comparison(model: str, runs: int, skip_batch: bool = False) -> dict:
    """Compare relay streaming vs batch mode for Lakeshore."""
    print(f"\n{'='*60}")
    print("  Relay vs Batch Comparison (lakeshore-qwen-vl-72b)")
    print(f"  Runs: {runs} per mode (+1 warmup each, discarded)")
    print(f"{'='*60}")

    batch_results = []
    if not skip_batch:
        # Batch first (cold cache) — must run before relay to get reproducible cold-start numbers.
        # Running relay first warms vLLM's prefix cache and artificially lowers batch TTFT (~11s vs ~15s).
        print("\n  --- Batch Mode (cold cache) ---")
        for i in range(runs + 1):
            is_warmup = i == 0
            label = "warmup" if is_warmup else f"batch {i}/{runs}"
            print(f"  [{label}] ", end="", flush=True)
            result = send_batch_request(model, get_benchmark_query(i))
            if result.get("error") or result.get("ttft") is None:
                err = result.get("error") or "no tokens received"
                print(f"ERROR: {err}")
            else:
                print(f"TTFT={result['ttft']:.2f}s (= total, all at once)")
            if not is_warmup:
                batch_results.append(result)

    # Relay streaming (after batch — cache state doesn't affect relay TTFT)
    print("\n  --- Relay Streaming ---")
    streaming_results = []
    for i in range(runs + 1):
        is_warmup = i == 0
        label = "warmup" if is_warmup else f"relay {i}/{runs}"
        print(f"  [{label}] ", end="", flush=True)
        result = send_streaming_request(model, get_benchmark_query(runs + 1 + i))
        if result.get("error") or result.get("ttft") is None:
            err = result.get("error") or "no tokens received (vLLM not running?)"
            print(f"ERROR: {err}")
        else:
            print(f"TTFT={result['ttft']:.2f}s  Total={result['total_time']:.2f}s")
        if not is_warmup:
            streaming_results.append(result)

    # Statistics
    stream_ok = [r for r in streaming_results if not r.get("error")]
    batch_ok = [r for r in batch_results if not r.get("error")]

    stream_ttfts = [r["ttft"] for r in stream_ok if r["ttft"] is not None]
    batch_ttfts = [r["ttft"] for r in batch_ok if r["ttft"] is not None]

    def _p95(data: list) -> float | None:
        return statistics.quantiles(data, n=100)[94] if len(data) >= 2 else None

    comparison = {
        "streaming": {
            "ttft_median": statistics.median(stream_ttfts) if stream_ttfts else None,
            "ttft_stdev": statistics.stdev(stream_ttfts) if len(stream_ttfts) > 1 else 0,
            "ttft_p95": _p95(stream_ttfts),
            "total_median": statistics.median([r["total_time"] for r in stream_ok])
            if stream_ok
            else None,
            "successful": len(stream_ok),
        },
        "batch": {
            "ttft_median": statistics.median(batch_ttfts) if batch_ttfts else None,
            "ttft_stdev": statistics.stdev(batch_ttfts) if len(batch_ttfts) > 1 else 0,
            "ttft_p95": _p95(batch_ttfts),
            "total_median": statistics.median([r["total_time"] for r in batch_ok])
            if batch_ok
            else None,
            "successful": len(batch_ok),
        },
    }

    if stream_ttfts and batch_ttfts:
        speedup = statistics.median(batch_ttfts) / statistics.median(stream_ttfts)
        comparison["ttft_speedup"] = speedup
        s_p95 = comparison["streaming"]["ttft_p95"]
        b_p95 = comparison["batch"]["ttft_p95"]
        print(f"\n  TTFT Speedup: {speedup:.1f}x faster with relay streaming")
        print(
            f"    Relay:  {comparison['streaming']['ttft_median']:.2f}s (median) +/- {comparison['streaming']['ttft_stdev']:.2f}s"
            + (f"  p95={s_p95:.2f}s" if s_p95 else "")
        )
        print(
            f"    Batch:  {comparison['batch']['ttft_median']:.2f}s (median) +/- {comparison['batch']['ttft_stdev']:.2f}s"
            + (f"  p95={b_p95:.2f}s" if b_p95 else "")
        )

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="STREAM Latency & Throughput Benchmark")
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of runs per tier (default: 50)"
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=["local", "lakeshore", "cloud"],
        help="Tiers to benchmark (default: all)",
    )
    parser.add_argument(
        "--relay-comparison-only", action="store_true", help="Only run relay vs batch comparison"
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip batch mode in relay comparison (relay streaming only)",
    )
    parser.add_argument(
        "--skip-relay", action="store_true", help="Skip relay vs batch comparison entirely"
    )
    parser.add_argument(
        "--cloud-thinking",
        action="store_true",
        help="Run cloud thinking benchmark (no thinking + high effort thinking)",
    )
    parser.add_argument(
        "--thinking-only",
        action="store_true",
        help="With --cloud-thinking: skip no-thinking baseline, run only thinking configs",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Send the same query every run to warm the KV cache (measures best-case TTFT)",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")
    parser.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="Start query pool at this index (avoids provider semantic cache across sessions)",
    )
    parser.add_argument("--url", type=str, help="STREAM URL (default: http://localhost:5000)")
    args = parser.parse_args()

    if args.url:
        global STREAM_URL, CHAT_ENDPOINT
        STREAM_URL = args.url
        CHAT_ENDPOINT = f"{STREAM_URL}/v1/chat/completions"

    # Detect hardware
    print("Detecting hardware...")
    hardware = detect_hardware()
    print(f"  OS:  {hardware['os']}")
    print(f"  CPU: {hardware['cpu']}")
    print(f"  RAM: {hardware['ram']}")
    print(f"  GPU: {hardware['gpu'] or 'None detected'}")

    # Check connectivity
    print(f"\nConnecting to STREAM at {STREAM_URL}...")
    try:
        r = httpx.get(f"{STREAM_URL}/health", timeout=5)
        print(f"  Connected! Status: {r.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to STREAM: {e}")
        print(f"  Make sure STREAM is running at {STREAM_URL}")
        sys.exit(1)

    tiers = args.tiers or list(TIER_MODELS.keys())
    results = {
        "timestamp": datetime.now().isoformat(),
        "stream_url": STREAM_URL,
        "hardware": {
            "local": hardware,
            "lakeshore": {
                "gpu": "NVIDIA H100 NVL 94 GB",
                "node": "ghi2-002",
                "partition": "batch_gpu2",
                "model": "Qwen2.5-VL-72B-Instruct-AWQ",
                "framework": "vLLM with Marlin kernels",
            },
            "cloud": {
                "note": "Infrastructure not disclosed by providers",
                "model": "Claude Sonnet 4.6 (claude-sonnet-4-6-20250514) via OpenRouter",
                "path": "STREAM cloud tier → LiteLLM gateway → OpenRouter → Anthropic",
                "thinking": "disabled for standard benchmark; use --cloud-thinking for no-thinking vs medium effort comparison",
            },
        },
        "benchmark_queries": f"{len(BENCHMARK_QUERIES)} distinct queries (rotating pool)",
        "runs_per_tier": args.runs,
        "tiers": {},
        "relay_comparison": None,
        "cloud_thinking": None,
    }

    if not args.relay_comparison_only and not (args.cloud_thinking and not args.tiers):
        for tier in tiers:
            model = TIER_MODELS.get(tier)
            if not model:
                print(f"\nSkipping unknown tier: {tier}")
                continue
            if tier == "local":
                print("\n  *** WARNING: Local tier benchmark must be run in DESKTOP mode.")
                print("  ***   The Docker Ollama container uses CPU only (no Apple GPU/Metal).")
                print(
                    "  ***   Run: STREAM_MODE=desktop python scripts/eval/benchmark_latency.py --tiers local"
                )
                print("  ***   Continuing anyway — results will reflect CPU inference, not GPU.\n")
            tier_offset = args.query_offset + list(TIER_MODELS.keys()).index(tier) * (args.runs + 1)
            stats = run_tier_benchmark(
                tier,
                model,
                args.runs,
                warmup=not args.no_warmup,
                cached=args.cached,
                query_offset=tier_offset,
            )
            results["tiers"][tier] = stats

    # Cloud thinking effort benchmark
    # Offset queries past those already used by the tier benchmarks to avoid
    # hitting OpenRouter's semantic cache with repeated identical queries.
    if args.cloud_thinking:
        n_tiers_run = len([t for t in tiers if t in results["tiers"]])
        thinking_offset = args.query_offset + n_tiers_run * (args.runs + 1)
        cloud_thinking = run_cloud_thinking_benchmark(
            args.runs, query_offset=thinking_offset, thinking_only=args.thinking_only
        )
        results["cloud_thinking"] = cloud_thinking

    # Relay vs batch comparison for Lakeshore
    if ("lakeshore" in tiers or args.relay_comparison_only) and not args.skip_relay:
        model = TIER_MODELS["lakeshore"]
        comparison = run_relay_comparison(model, args.runs, skip_batch=args.skip_batch)
        results["relay_comparison"] = comparison

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_file = results_dir / f"latency_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY TABLE (for paper)")
    print(f"{'='*60}")
    print(f"  {'Tier':<30} {'TTFT (s)':<15} {'Total (s)':<15} {'tok/s':<10}")
    print(f"  {'-'*70}")
    for tier, stats in results["tiers"].items():
        if stats.get("error"):
            print(f"  {tier:<30} ERROR: {stats['error']}")
            continue
        ttft = stats["ttft"]["median"]
        ttft_std = stats["ttft"]["stdev"]
        total = stats["total_time"]["median"]
        total_std = stats["total_time"]["stdev"]
        tp = stats["throughput"]["median"]
        print(
            f"  {tier:<30} "
            f"{ttft:.2f} +/- {ttft_std:.2f}  "
            f"{total:.2f} +/- {total_std:.2f}  "
            f"{tp:.1f}"
            if tp
            else "---"
        )

    if results["relay_comparison"]:
        rc = results["relay_comparison"]
        print("\n  Relay vs Batch (Lakeshore):")
        if rc["streaming"]["ttft_median"] and rc["batch"]["ttft_median"]:
            print(f"    Streaming TTFT: {rc['streaming']['ttft_median']:.2f}s")
            print(f"    Batch TTFT:     {rc['batch']['ttft_median']:.2f}s")
            print(f"    Speedup:        {rc.get('ttft_speedup', 0):.1f}x")


if __name__ == "__main__":
    main()
