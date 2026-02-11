#!/usr/bin/env python3
"""
================================================================================
STREAM Latency Benchmark Suite - Publication Quality
================================================================================

A comprehensive latency benchmarking tool designed for academic publication.
Supports all models, auto-detects hardware, and generates publication figures.

================================================================================
HOW TO MANUALLY DETERMINE HARDWARE SPECIFICATIONS
================================================================================

If auto-detection fails or you need to verify, here's how to manually check
hardware for each tier:

LOCAL TIER (Your Machine Running Ollama)
----------------------------------------
macOS:
    # CPU info
    sysctl -n machdep.cpu.brand_string
    # For Apple Silicon:
    system_profiler SPHardwareDataType | grep "Chip"

    # RAM
    sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'

    # GPU (Apple Silicon uses Metal)
    system_profiler SPDisplaysDataType | grep "Chipset Model"

    # For NVIDIA eGPU (if connected)
    /usr/local/cuda/bin/nvidia-smi

Linux:
    # CPU
    cat /proc/cpuinfo | grep "model name" | head -1

    # RAM
    free -h | grep Mem

    # GPU (NVIDIA)
    nvidia-smi --query-gpu=name,memory.total --format=csv

    # GPU (AMD)
    rocm-smi

Windows:
    # All info via PowerShell
    Get-ComputerInfo | Select CsProcessors, OsTotalVisibleMemorySize
    nvidia-smi  # if NVIDIA GPU present

LAKESHORE TIER (UIC HPC Cluster)
--------------------------------
The Lakeshore proxy reports GPU info via its health endpoint:
    curl http://localhost:8001/health

To manually check on Lakeshore (when logged into the cluster):
    # SSH to Lakeshore
    ssh <netid>@lakeshore.cs.uic.edu

    # Check GPU allocation
    squeue -u $USER  # See your job allocation

    # On compute node with GPU
    nvidia-smi  # Shows GPU model and memory

Known Lakeshore GPU configurations:
    - NVIDIA A100 80GB (ga-001 through ga-008)
    - NVIDIA V100 32GB (older nodes)

To verify your allocation:
    scontrol show job <job_id>

CLOUD TIER (Anthropic/OpenAI APIs)
----------------------------------
Cloud providers do NOT expose infrastructure details for security reasons.
We know from public information:

Anthropic (Claude):
    - Uses custom TPU and GPU clusters
    - Inference runs on Google Cloud infrastructure
    - No specific hardware details disclosed

OpenAI (GPT):
    - Runs on Microsoft Azure
    - Uses NVIDIA A100 and H100 GPUs (per public announcements)
    - Specific configuration undisclosed

For benchmarking, report as:
    - "Anthropic Cloud Infrastructure" for Claude models
    - "OpenAI/Azure Infrastructure" for GPT models

================================================================================
METHODOLOGY
================================================================================

This benchmark follows MLPerf inference benchmark guidelines:

1. WARMUP PHASE:
   - Run N warmup requests (default: 2)
   - Excluded from measurements
   - Ensures models are loaded, connections established, caches warm

2. MEASUREMENT PHASE:
   - Run M iterations (default: 10)
   - Each request measures:
     * TTFB (Time To First Byte): When first data arrives from server
     * TTFT (Time To First Token): When first AI content appears ← KEY METRIC
     * Total Time: Complete response time

3. STATISTICAL ANALYSIS:
   - Calculate P50 (median), P95, P99 percentiles
   - P95 is the standard metric for SLAs
   - Raw data preserved for custom analysis

================================================================================
USAGE EXAMPLES
================================================================================

# Basic benchmark with all defaults
python tests/latency_benchmark.py

# Full benchmark with figures
python tests/latency_benchmark.py -n 10 --figures

# Test all available models
python tests/latency_benchmark.py --all-models --figures

# Test specific models
python tests/latency_benchmark.py --models local-llama cloud-claude cloud-gpt

# Compare Lakeshore Globus vs SSH
python tests/latency_benchmark.py --tier lakeshore --lakeshore-label globus -o globus
python tests/latency_benchmark.py --tier lakeshore --lakeshore-label ssh -o ssh
python tests/latency_benchmark.py --compare globus.json ssh.json --figures

# Export CSV for R/Python statistical analysis
python tests/latency_benchmark.py -o results --csv

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports for core functionality
import argparse  # Command-line argument parsing
import csv  # CSV export for statistical analysis
import json  # JSON serialization for results
import platform  # Platform/OS detection
import statistics  # Statistical calculations (mean, stdev)
import subprocess  # Running system commands for hardware detection
import time  # Timing measurements
from dataclasses import dataclass, field  # Structured data classes
from datetime import datetime  # Timestamps
from pathlib import Path  # Path manipulation

# Third-party imports
import httpx  # HTTP client for API requests
import yaml  # YAML parsing for litellm config

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================
# These provide additional functionality but aren't strictly required

try:
    # matplotlib: Required for generating publication figures
    # Install: pip install matplotlib numpy
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not installed. Figures will not be generated.")
    print("   Install with: pip install matplotlib numpy")

try:
    # psutil: Provides cross-platform hardware info
    # Install: pip install psutil
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not installed. Using fallback hardware detection.")
    print("   Install with: pip install psutil")


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# API endpoints
MIDDLEWARE_URL = "http://localhost:5000/v1/chat/completions"  # STREAM middleware
LAKESHORE_PROXY_URL = "http://localhost:8001"  # Lakeshore proxy (for health check)

# File paths
LITELLM_CONFIG_PATH = Path("stream/gateway/litellm_config.yaml")  # Model definitions
FIGURES_DIR = Path("tests/figures")  # Output directory for figures

# Benchmark configuration
TEST_QUERY = "What is Python?"  # Default query (can be overridden)

# IMPORTANT: Using the same query repeatedly causes CACHE HITS in the judge!
# For realistic latency measurement, use unique queries.
# The --unique-queries flag generates a new query for each iteration.

# List of unique queries for realistic testing (avoids cache hits)
UNIQUE_QUERIES = [
    "What is Python?",
    "Explain machine learning",
    "How does a hash table work?",
    "What is recursion in programming?",
    "Describe the OSI model",
    "What is a binary search tree?",
    "Explain RESTful APIs",
    "How does HTTPS encryption work?",
    "What is containerization?",
    "Explain microservices architecture",
    "What is functional programming?",
    "How does garbage collection work?",
    "What is a neural network?",
    "Explain SQL joins",
    "What is version control?",
    "How does DNS work?",
    "What is Big O notation?",
    "Explain the CAP theorem",
    "What is OAuth?",
    "How does caching work?",
]


# =============================================================================
# DATA CLASSES
# =============================================================================
# These define the structure of our configuration and results


@dataclass
class HardwareSpec:
    """
    Hardware specification for a tier.

    Stores hardware details for documentation and figure annotations.

    Attributes:
        description: Human-readable summary (e.g., "Apple M2 Pro, 16GB RAM")
        cpu: CPU model if known
        ram: RAM amount if known
        gpu: GPU model if known
        os: Operating system if known
        detection_method: How this info was obtained ("auto", "queried", "manual")
    """

    description: str
    cpu: str | None = None
    ram: str | None = None
    gpu: str | None = None
    os: str | None = None
    detection_method: str = "manual"


@dataclass
class ModelSpec:
    """
    Model specification from litellm_config.yaml.

    Stores model metadata for documentation.

    Attributes:
        name: Model identifier in STREAM (e.g., "local-llama")
        tier: Which tier this model belongs to ("local", "lakeshore", "cloud")
        underlying_model: Actual model name (e.g., "llama3.2:3b")
        params: Parameter count if known (e.g., "3B")
        input_cost: Cost per input token (for cost estimation)
        output_cost: Cost per output token (for cost estimation)
    """

    name: str
    tier: str
    underlying_model: str
    params: str | None = None
    input_cost: float = 0.0
    output_cost: float = 0.0


@dataclass
class BenchmarkConfig:
    """
    Complete benchmark configuration.

    This holds all settings for a benchmark run, making results reproducible.
    """

    iterations: int = 10  # Number of measurement iterations
    warmup: int = 2  # Warmup iterations (excluded from stats)
    test_query: str = TEST_QUERY  # Query used for all tests

    # Which models/judges to test
    models: list = field(default_factory=list)
    judges: list = field(default_factory=lambda: ["ollama-1b", "ollama-3b", "haiku"])

    # Hardware specs (populated during init)
    hardware: dict = field(default_factory=dict)

    # Model specs (loaded from litellm config)
    model_specs: dict = field(default_factory=dict)


# =============================================================================
# HARDWARE AUTO-DETECTION
# =============================================================================
# These functions automatically detect hardware specifications.
# Each function includes comments on the detection method and fallbacks.


def detect_local_cpu() -> str:
    """
    Auto-detect the local CPU model.

    DETECTION METHODS BY PLATFORM:

    macOS:
        1. Try: sysctl -n machdep.cpu.brand_string (Intel Macs)
        2. Try: system_profiler SPHardwareDataType (Apple Silicon)
        3. Fallback: platform.processor()

    Linux:
        1. Read /proc/cpuinfo, find "model name" line
        2. Fallback: platform.processor()

    Windows:
        1. Use platform.processor()
        2. Alternative: wmic cpu get name

    MANUAL CHECK:
        macOS:  sysctl -n machdep.cpu.brand_string
        Linux:  cat /proc/cpuinfo | grep "model name" | head -1
        Windows: wmic cpu get name

    Returns:
        CPU model string, e.g., "Apple M2 Pro" or "Intel Core i7-12700K"
    """
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            # ----------------------------------------------------------------
            # Method 1: Intel-style brand string (works on Intel Macs)
            # ----------------------------------------------------------------
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            # ----------------------------------------------------------------
            # Method 2: Apple Silicon - use system_profiler
            # ----------------------------------------------------------------
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Look for "Chip:" line in output
                for line in result.stdout.split("\n"):
                    if "Chip" in line and ":" in line:
                        chip_name = line.split(":")[1].strip()
                        return f"Apple {chip_name}"

        elif system == "Linux":
            # ----------------------------------------------------------------
            # Linux: Read /proc/cpuinfo
            # ----------------------------------------------------------------
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line.lower():
                        # Line format: "model name    : Intel(R) Core(TM) i7-..."
                        return line.split(":")[1].strip()

        elif system == "Windows":
            # ----------------------------------------------------------------
            # Windows: Use platform module
            # For more detail: wmic cpu get name
            # ----------------------------------------------------------------
            return platform.processor() or "Unknown Windows CPU"

    except Exception as e:
        # Log the error but don't fail - return fallback
        print(f"   ⚠️  CPU detection failed: {e}")

    # Fallback for all platforms
    return platform.processor() or "Unknown CPU"


def detect_local_ram() -> str:
    """
    Auto-detect total system RAM.

    DETECTION METHODS:

    All platforms (preferred):
        Use psutil.virtual_memory().total if psutil is installed

    macOS fallback:
        sysctl -n hw.memsize

    Linux fallback:
        Read /proc/meminfo, find MemTotal line

    MANUAL CHECK:
        macOS:  sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'
        Linux:  free -h | grep Mem
        Windows: systeminfo | findstr "Total Physical Memory"

    Returns:
        RAM string, e.g., "16GB"
    """
    # ----------------------------------------------------------------
    # Preferred method: psutil (cross-platform, accurate)
    # ----------------------------------------------------------------
    if PSUTIL_AVAILABLE:
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024**3)  # Convert bytes to GB
        return f"{ram_gb:.0f}GB"

    # ----------------------------------------------------------------
    # macOS fallback: sysctl
    # ----------------------------------------------------------------
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                ram_bytes = int(result.stdout.strip())
                ram_gb = ram_bytes / (1024**3)
                return f"{ram_gb:.0f}GB"
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Linux fallback: /proc/meminfo
    # ----------------------------------------------------------------
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        # Format: "MemTotal:       16384000 kB"
                        kb = int(line.split()[1])
                        gb = kb / (1024**2)
                        return f"{gb:.0f}GB"
        except Exception:
            pass

    return "Unknown RAM"


def detect_local_gpu() -> str | None:
    """
    Auto-detect local GPU if present.

    DETECTION METHODS:

    NVIDIA GPUs (all platforms):
        nvidia-smi --query-gpu=name,memory.total --format=csv

    AMD GPUs (Linux):
        rocm-smi

    Apple Silicon (macOS):
        system_profiler SPDisplaysDataType (reports Metal GPU)

    Intel iGPU (Linux):
        lspci | grep VGA

    MANUAL CHECK:
        NVIDIA: nvidia-smi
        AMD:    rocm-smi
        macOS:  system_profiler SPDisplaysDataType | grep "Chipset"
        Linux:  lspci | grep -i vga

    Returns:
        GPU string or None, e.g., "NVIDIA RTX 3080 10GB" or "Apple M2 Pro GPU (Metal)"
    """
    # ----------------------------------------------------------------
    # Try NVIDIA first (most common for ML workloads)
    # ----------------------------------------------------------------
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    mem_mb = int(parts[1].strip())
                    mem_gb = mem_mb / 1024
                    gpus.append(f"{name} {mem_gb:.0f}GB")
            if gpus:
                return ", ".join(gpus)
    except FileNotFoundError:
        pass  # nvidia-smi not found - no NVIDIA GPU
    except Exception:
        pass

    # ----------------------------------------------------------------
    # macOS: Check for Apple Silicon GPU (Metal)
    # ----------------------------------------------------------------
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Chipset Model" in line:
                        gpu_name = line.split(":")[1].strip()
                        return f"{gpu_name} (Metal)"
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Linux: Try AMD ROCm
    # ----------------------------------------------------------------
    if platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Parse ROCm output
                for line in result.stdout.split("\n"):
                    if "Card" in line:
                        return line.split(":")[1].strip() if ":" in line else line
        except FileNotFoundError:
            pass

    return None  # No GPU detected


def detect_local_os() -> str:
    """
    Detect operating system and version.

    MANUAL CHECK:
        macOS:   sw_vers
        Linux:   cat /etc/os-release
        Windows: ver

    Returns:
        OS string, e.g., "macOS 14.2" or "Ubuntu 22.04"
    """
    system = platform.system()

    if system == "Darwin":
        # macOS - get version number
        ver = platform.mac_ver()[0]  # Returns ('14.2', ('', '', ''), 'arm64')
        return f"macOS {ver}"

    elif system == "Linux":
        # Linux - try to get distro name from /etc/os-release
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME"):
                        # Format: PRETTY_NAME="Ubuntu 22.04.3 LTS"
                        return line.split("=")[1].strip().strip('"')
        except Exception:
            pass
        return f"Linux {platform.release()}"

    elif system == "Windows":
        return f"Windows {platform.release()}"

    return f"{system} {platform.release()}"


def detect_local_hardware() -> HardwareSpec:
    """
    Auto-detect all local hardware and combine into HardwareSpec.

    This function combines CPU, RAM, GPU, and OS detection.
    Results are used for:
    1. Figure annotations
    2. Reproducibility documentation
    3. JSON output metadata
    """
    print("   Detecting local hardware...")

    cpu = detect_local_cpu()
    print(f"      CPU: {cpu}")

    ram = detect_local_ram()
    print(f"      RAM: {ram}")

    gpu = detect_local_gpu()
    if gpu:
        print(f"      GPU: {gpu}")
    else:
        print("      GPU: None detected")

    os_ver = detect_local_os()
    print(f"      OS:  {os_ver}")

    # Build description string
    parts = [cpu, ram]
    if gpu:
        parts.append(gpu)
    parts.append(os_ver)

    return HardwareSpec(
        description=", ".join(parts), cpu=cpu, ram=ram, gpu=gpu, os=os_ver, detection_method="auto"
    )


def query_lakeshore_hardware() -> HardwareSpec:
    """
    Query Lakeshore hardware from the proxy health endpoint.

    The Lakeshore proxy exposes hardware info at GET /health:
        Response: {"status": "healthy", "gpu_info": "NVIDIA A100 80GB", ...}

    MANUAL CHECK (if proxy is unreachable):
        1. SSH to Lakeshore:
           ssh <netid>@lakeshore.cs.uic.edu

        2. Check GPU on compute node:
           srun --gres=gpu:1 nvidia-smi

        3. Known configurations:
           - ga-001 to ga-008: NVIDIA A100 80GB
           - Older nodes: NVIDIA V100 32GB

    Falls back to known defaults if proxy is unreachable.
    """
    print("   Querying Lakeshore hardware...")

    try:
        # ----------------------------------------------------------------
        # Query the Lakeshore proxy health endpoint
        # ----------------------------------------------------------------
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{LAKESHORE_PROXY_URL}/health")

            if response.status_code == 200:
                data = response.json()
                gpu_info = data.get("gpu_info", "NVIDIA A100 80GB")
                vllm_model = data.get("vllm_model", "Unknown")

                print(f"      GPU:   {gpu_info}")
                print(f"      Model: {vllm_model}")

                return HardwareSpec(
                    description=f"UIC Lakeshore HPC - {gpu_info}",
                    gpu=gpu_info,
                    detection_method="queried from proxy",
                )

    except httpx.ConnectError:
        print("      ⚠️  Proxy unreachable (is it running?)")
    except Exception as e:
        print(f"      ⚠️  Query failed: {e}")

    # ----------------------------------------------------------------
    # Fallback to known defaults
    # ----------------------------------------------------------------
    print("      Using default: NVIDIA A100 80GB")
    return HardwareSpec(
        description="UIC Lakeshore HPC - NVIDIA A100 80GB (default)",
        gpu="NVIDIA A100 80GB",
        detection_method="manual (proxy unreachable)",
    )


def get_cloud_hardware(provider: str) -> HardwareSpec:
    """
    Return documented cloud provider hardware specs.

    Cloud providers don't expose infrastructure details publicly.
    We document what's known from official announcements.

    MANUAL VERIFICATION:
        - Anthropic: No public infrastructure details
        - OpenAI: Runs on Azure, uses A100/H100 (per announcements)

    For publication, cite provider documentation where available.
    """
    specs = {
        "anthropic": HardwareSpec(
            description="Anthropic Cloud Infrastructure (undisclosed)",
            detection_method="manual (proprietary)",
        ),
        "openai": HardwareSpec(
            description="Microsoft Azure Infrastructure (NVIDIA A100/H100)",
            detection_method="manual (public announcements)",
        ),
    }
    return specs.get(
        provider, HardwareSpec(description="Unknown cloud provider", detection_method="unknown")
    )


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


def load_models_from_litellm_config(config_path: Path = LITELLM_CONFIG_PATH) -> dict:
    """
    Load all model definitions from litellm_config.yaml.

    This parses the LiteLLM configuration to discover all available models.
    Each model is categorized by tier (local, lakeshore, cloud) based on
    its naming convention (local-*, lakeshore-*, cloud-*).

    Returns:
        Dict mapping model_name to ModelSpec
    """
    models = {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # ----------------------------------------------------------------
        # Parse each model definition
        # ----------------------------------------------------------------
        for model_def in config.get("model_list", []):
            name = model_def.get("model_name")
            if not name:
                continue

            litellm_params = model_def.get("litellm_params", {})
            model_info = model_def.get("model_info", {})

            # Determine tier from model name prefix
            if name.startswith("local-"):
                tier = "local"
            elif name.startswith("lakeshore-"):
                tier = "lakeshore"
            elif name.startswith("cloud-"):
                tier = "cloud"
            else:
                tier = "unknown"

            # Get underlying model name (remove provider prefix)
            underlying = litellm_params.get("model", "")
            if "/" in underlying:
                underlying = underlying.split("/")[-1]

            models[name] = ModelSpec(
                name=name,
                tier=tier,
                underlying_model=underlying,
                input_cost=model_info.get("input_cost_per_token", 0),
                output_cost=model_info.get("output_cost_per_token", 0),
            )

        print(f"   Loaded {len(models)} models from {config_path}")

    except FileNotFoundError:
        print(f"   ⚠️  Config not found: {config_path}")
    except Exception as e:
        print(f"   ⚠️  Error loading config: {e}")

    return models


def get_default_models() -> list:
    """Return default models for benchmarking (one per tier)."""
    return ["local-llama", "lakeshore-qwen", "cloud-claude"]


# =============================================================================
# STATISTICAL CALCULATIONS
# =============================================================================


def calculate_percentiles(data: list) -> dict:
    """
    Calculate statistical metrics for latency measurements.

    Following MLPerf and industry standards, we report:
    - P50 (median): Typical user experience
    - P95: Standard SLA metric - "95% of requests are faster than this"
    - P99: Tail latency - important for reliability guarantees

    INTERPRETATION:
    - P50 = median (half of requests are faster/slower)
    - P95 = 95th percentile (only 5% of requests are slower)
    - P99 = 99th percentile (only 1% of requests are slower)

    For publication, P95 is typically the headline metric.

    Args:
        data: List of latency measurements in seconds

    Returns:
        Dict with p50, p95, p99, mean, std, min, max, count, raw
    """
    if not data:
        return {
            "p50": 0,
            "p95": 0,
            "p99": 0,
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "count": 0,
            "raw": [],
        }

    # Sort data for percentile calculation
    sorted_data = sorted(data)
    n = len(sorted_data)

    def percentile(p):
        """
        Calculate p-th percentile using linear interpolation.

        This matches numpy.percentile(data, p, interpolation='linear')
        """
        if n == 1:
            return sorted_data[0]
        # Calculate position in sorted array
        k = (n - 1) * p / 100
        f = int(k)  # Floor index
        c = min(f + 1, n - 1)  # Ceiling index
        # Linear interpolation between surrounding values
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    return {
        "p50": round(percentile(50), 4),
        "p95": round(percentile(95), 4),
        "p99": round(percentile(99), 4),
        "mean": round(statistics.mean(data), 4),
        "std": round(statistics.stdev(data), 4) if n > 1 else 0,
        "min": round(min(data), 4),
        "max": round(max(data), 4),
        "count": n,
        "raw": [round(x, 4) for x in data],  # Keep raw for box plots
    }


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================


def run_single_request(
    model: str, judge_strategy: str | None = None, timeout: float = 120.0, query: str = None
) -> dict:
    """
    Execute a single chat completion request and measure latency.

    MEASUREMENT POINTS:
    1. Start timer
    2. Send HTTP request
    3. TTFB: When first byte arrives (any data from server)
    4. TTFT: When first AI-generated TOKEN arrives ← KEY METRIC
    5. Total: When response completes

    Args:
        model: Model to test (e.g., "local-llama") or "auto" for routing
        judge_strategy: If using "auto", which judge to use
        timeout: Request timeout in seconds

    Returns:
        Dict with ttfb, ttft, total_time, success, routing info, error
    """
    # ----------------------------------------------------------------
    # Build request payload (OpenAI-compatible format)
    # ----------------------------------------------------------------
    # Use provided query or fall back to default
    test_query = query or TEST_QUERY

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": test_query}],
        "temperature": 0.7,
        "stream": True,  # Must be True to measure TTFT
    }

    # Add judge strategy for auto mode
    if judge_strategy:
        payload["judge_strategy"] = judge_strategy

    # ----------------------------------------------------------------
    # Initialize timing variables
    # ----------------------------------------------------------------
    start_time = time.time()
    ttfb = None  # Time To First Byte
    ttft = None  # Time To First Token ← KEY METRIC
    actual_tier = None
    actual_model = None
    response_text = ""
    error_msg = None

    try:
        # ----------------------------------------------------------------
        # Send streaming request
        # ----------------------------------------------------------------
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream("POST", MIDDLEWARE_URL, json=payload) as response,
        ):
            # Process Server-Sent Events (SSE)
            for line in response.iter_lines():
                # Record TTFB on first data received
                if ttfb is None:
                    ttfb = time.time() - start_time

                # SSE format: "data: {json_payload}"
                if not line.startswith("data: "):
                    continue

                data_str = line[6:].strip()  # Remove "data: " prefix

                # Handle stream end marker
                if data_str == "[DONE]" or not data_str:
                    continue

                try:
                    data = json.loads(data_str)

                    # Check for error
                    if "error" in data:
                        error_msg = str(data["error"])
                        continue

                    # Extract routing metadata
                    if "stream_metadata" in data:
                        meta = data["stream_metadata"]
                        actual_tier = meta.get("tier")
                        actual_model = meta.get("model")

                    # Extract AI-generated content
                    # THIS IS WHERE WE MEASURE TTFT
                    if "choices" in data:
                        for choice in data["choices"]:
                            content = choice.get("delta", {}).get("content", "")
                            if content:
                                # Record TTFT on FIRST token
                                if ttft is None:
                                    ttft = time.time() - start_time
                                response_text += content

                except json.JSONDecodeError:
                    pass  # Skip malformed JSON

        # Calculate total time
        total_time = time.time() - start_time

        return {
            "ttfb": ttfb or total_time,
            "ttft": ttft or total_time,
            "total_time": total_time,
            "success": len(response_text) > 0 and error_msg is None,
            "actual_tier": actual_tier,
            "actual_model": actual_model,
            "error": error_msg,
        }

    except httpx.TimeoutException:
        return {
            "ttfb": ttfb or 0,
            "ttft": ttft or 0,
            "total_time": time.time() - start_time,
            "success": False,
            "error": f"Timeout after {timeout}s",
        }
    except httpx.ConnectError:
        return {
            "ttfb": 0,
            "ttft": 0,
            "total_time": time.time() - start_time,
            "success": False,
            "error": "Connection refused - is middleware running?",
        }
    except Exception as e:
        return {
            "ttfb": ttfb or 0,
            "ttft": ttft or 0,
            "total_time": time.time() - start_time,
            "success": False,
            "error": str(e),
        }


def run_benchmark(
    model: str, judge: str | None, iterations: int, warmup: int, label: str | None = None
) -> dict:
    """
    Run a complete benchmark for one configuration.

    METHODOLOGY:
    1. Run 'warmup' requests (excluded from measurements)
       - Ensures model is loaded
       - Establishes connections
       - Warms caches

    2. Run 'iterations' measured requests
       - Record TTFB, TTFT, total time for each
       - Calculate percentile statistics

    Args:
        model: Model to test or "auto"
        judge: Judge strategy for auto mode
        iterations: Number of measured iterations
        warmup: Number of warmup iterations
        label: Custom label for this benchmark

    Returns:
        Dict with label, config, raw results, and statistics
    """
    # ----------------------------------------------------------------
    # Determine label for this benchmark
    # ----------------------------------------------------------------
    if label:
        config_label = label
    elif judge:
        config_label = f"auto + {judge}"
    else:
        config_label = model

    print(f"\n{'─' * 60}")
    print(f"📊 {config_label.upper()}")
    print(f"{'─' * 60}")

    # ----------------------------------------------------------------
    # Warmup phase
    # IMPORTANT: Each warmup uses a DIFFERENT query to avoid cache hits
    # ----------------------------------------------------------------
    if warmup > 0:
        print(f"   Warmup: {warmup} requests (unique queries)...", end=" ", flush=True)
        for i in range(warmup):
            # Use unique query to avoid caching the same result
            query = UNIQUE_QUERIES[i % len(UNIQUE_QUERIES)]
            run_single_request(model if not judge else "auto", judge, query=query)
        print("done")

    # ----------------------------------------------------------------
    # Measurement phase
    # CRITICAL: Each iteration uses a UNIQUE query to prevent judge cache hits
    # This measures REAL latency, not cached latency
    # ----------------------------------------------------------------
    results = []
    ttft_times = []  # Collect TTFT values for statistics

    print(f"   Measuring: {iterations} iterations (unique queries - no cache)")

    for i in range(iterations):
        # Use unique query - offset by warmup count to avoid repeating warmup queries
        query_idx = (warmup + i) % len(UNIQUE_QUERIES)
        query = UNIQUE_QUERIES[query_idx]

        # Run single request with unique query
        result = run_single_request(model if not judge else "auto", judge, query=query)
        results.append(result)

        if result["success"]:
            ttft_times.append(result["ttft"])
            print(f"      [{i+1:2d}/{iterations}] ✅ TTFT: {result['ttft']:.3f}s")
        else:
            print(f"      [{i+1:2d}/{iterations}] ❌ {result.get('error', 'Error')[:40]}")

    # ----------------------------------------------------------------
    # Calculate statistics
    # ----------------------------------------------------------------
    stats = calculate_percentiles(ttft_times)
    success_rate = (len(ttft_times) / iterations * 100) if iterations > 0 else 0

    if stats["count"] > 0:
        print(
            f"\n   → Results: P50={stats['p50']:.3f}s | P95={stats['p95']:.3f}s | P99={stats['p99']:.3f}s"
        )

    return {
        "label": config_label,
        "model": model,
        "judge": judge,
        "iterations": iterations,
        "warmup": warmup,
        "results": results,
        "ttft_stats": stats,
        "success_rate": success_rate,
    }


# =============================================================================
# OUTPUT: CSV EXPORT
# =============================================================================


def export_csv(all_results: list, output_path: Path):
    """
    Export raw latency data to CSV for statistical analysis.

    The CSV format is designed for easy import into R, Python (pandas),
    or spreadsheet software.

    Columns:
    - configuration: Test configuration label
    - iteration: 1-indexed iteration number
    - ttfb: Time To First Byte (seconds)
    - ttft: Time To First Token (seconds) ← KEY METRIC
    - total_time: Total response time (seconds)
    - success: Whether request succeeded (True/False)
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(["configuration", "iteration", "ttfb", "ttft", "total_time", "success"])

        # Data rows
        for result in all_results:
            for i, r in enumerate(result["results"]):
                writer.writerow(
                    [
                        result["label"],
                        i + 1,
                        f"{r['ttfb']:.4f}",
                        f"{r['ttft']:.4f}",
                        f"{r['total_time']:.4f}",
                        r["success"],
                    ]
                )

    print(f"📄 CSV exported: {output_path}")


# =============================================================================
# OUTPUT: FIGURE GENERATION
# =============================================================================


def generate_figures(results: list, config: BenchmarkConfig, output_dir: Path):
    """
    Generate publication-quality figures.

    FIGURES GENERATED:
    1. ttft_comparison.png - Bar chart comparing TTFT P50/P95 across configs
    2. latency_boxplot.png - Box plot showing full distribution
    3. judge_overhead.png - Judge strategy overhead analysis

    All figures include:
    - Hardware configuration annotations
    - Clear axis labels and titles
    - Publication-ready resolution (150 DPI)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Skipping figures (matplotlib not installed)")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to successful results only
    successful = [r for r in results if r["success_rate"] > 0]
    if not successful:
        print("⚠️  No successful results to plot")
        return

    # ----------------------------------------------------------------
    # Configure matplotlib for publication
    # ----------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )

    # ================================================================
    # FIGURE 1: TTFT Comparison Bar Chart
    # ================================================================
    print("   Generating ttft_comparison.png...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Extract data
    labels = [r["label"] for r in successful]
    p50_values = [r["ttft_stats"]["p50"] for r in successful]
    p95_values = [r["ttft_stats"]["p95"] for r in successful]

    # Create bar positions
    x = np.arange(len(labels))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(
        x - width / 2, p50_values, width, label="P50 (Median)", color="#27ae60", edgecolor="black"
    )
    bars2 = ax.bar(
        x + width / 2, p95_values, width, label="P95", color="#c0392b", edgecolor="black"
    )

    # Labels and title
    ax.set_ylabel("Time To First Token (seconds)")
    ax.set_xlabel("Configuration")
    ax.set_title("STREAM Latency Benchmark: TTFT Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper left")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add hardware info annotation box
    hw_lines = ["Hardware Configuration:"]
    for tier, spec in config.hardware.items():
        if hasattr(spec, "description"):
            desc = spec.description[:45] + "..." if len(spec.description) > 45 else spec.description
        else:
            desc = str(spec)[:45]
        hw_lines.append(f"• {tier}: {desc}")

    hw_text = "\n".join(hw_lines)
    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
    ax.text(
        0.98,
        0.98,
        hw_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "ttft_comparison.png")
    plt.close()

    # ================================================================
    # FIGURE 2: Box Plot (Distribution)
    # ================================================================
    print("   Generating latency_boxplot.png...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Collect raw data for box plots
    box_data = []
    box_labels = []
    for r in successful:
        raw = r["ttft_stats"].get("raw", [])
        if raw:
            box_data.append(raw)
            box_labels.append(r["label"])

    if box_data:
        # Create box plot
        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels)

        # Color each box differently
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)

        ax.set_ylabel("Time To First Token (seconds)")
        ax.set_xlabel("Configuration")
        ax.set_title("STREAM Latency Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "latency_boxplot.png")

    plt.close()

    # ================================================================
    # FIGURE 3: Judge Overhead Analysis
    # ================================================================
    print("   Generating judge_overhead.png...")

    # Separate direct tests from judge tests
    direct_tests = {r["model"]: r for r in results if not r["judge"] and r["success_rate"] > 0}
    judge_tests = [r for r in results if r["judge"] and r["success_rate"] > 0]

    if judge_tests and direct_tests:
        fig, ax = plt.subplots(figsize=(10, 6))

        judge_names = []
        baseline_values = []
        overhead_values = []

        for jt in judge_tests:
            # Find which tier the auto mode routed to
            if jt["results"] and jt["results"][0].get("actual_tier"):
                routed_tier = jt["results"][0]["actual_tier"]

                # Find matching direct test
                for model, dt in direct_tests.items():
                    if routed_tier in model:
                        baseline = dt["ttft_stats"]["p95"]
                        total = jt["ttft_stats"]["p95"]
                        overhead = max(0, total - baseline)

                        judge_names.append(jt["judge"])
                        baseline_values.append(baseline)
                        overhead_values.append(overhead)
                        break

        if judge_names:
            x = np.arange(len(judge_names))
            width = 0.6

            # Stacked bar chart
            ax.bar(
                x,
                baseline_values,
                width,
                label="Tier Baseline (P95)",
                color="#3498db",
                edgecolor="black",
            )
            ax.bar(
                x,
                overhead_values,
                width,
                bottom=baseline_values,
                label="Judge Overhead",
                color="#f39c12",
                edgecolor="black",
            )

            ax.set_ylabel("Time To First Token (seconds)")
            ax.set_xlabel("Judge Strategy")
            ax.set_title("LLM Judge Strategy Overhead Analysis")
            ax.set_xticks(x)
            ax.set_xticklabels(judge_names)
            ax.legend()

            # Add overhead labels
            for i, (base, over) in enumerate(zip(baseline_values, overhead_values, strict=False)):
                total = base + over
                ax.annotate(
                    f"{total:.2f}s\n(+{over:.2f}s)",
                    xy=(i, total),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

            plt.tight_layout()
            plt.savefig(output_dir / "judge_overhead.png")

    plt.close()

    print(f"   ✅ Figures saved to: {output_dir}/")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Main entry point for the benchmark suite.

    Parses command-line arguments and orchestrates the benchmark.
    """
    # ----------------------------------------------------------------
    # Parse command-line arguments
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="STREAM Latency Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/latency_benchmark.py -n 10 --figures
  python tests/latency_benchmark.py --all-models --csv -o results
  python tests/latency_benchmark.py --tier lakeshore --lakeshore-label globus
        """,
    )

    # Basic options
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=10,
        help="Number of measured iterations (default: 10)",
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=2, help="Warmup iterations (default: 2)"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file base name (creates .json and optionally .csv)"
    )
    parser.add_argument("-f", "--figures", action="store_true", help="Generate publication figures")
    parser.add_argument("--csv", action="store_true", help="Export raw data as CSV")

    # Model selection
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument(
        "--all-models", action="store_true", help="Test ALL models defined in litellm config"
    )
    parser.add_argument(
        "--tier", choices=["local", "lakeshore", "cloud"], help="Test only models from this tier"
    )
    parser.add_argument(
        "--judges-only",
        action="store_true",
        help="Only test judge strategies (skip direct tier tests)",
    )
    parser.add_argument("--no-judges", action="store_true", help="Skip judge strategy tests")

    # Labels
    parser.add_argument(
        "--lakeshore-label", type=str, help="Custom label for Lakeshore (e.g., 'globus' or 'ssh')"
    )

    # Comparison mode
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE1", "FILE2"), help="Compare two result files"
    )

    # Hardware overrides
    parser.add_argument("--local-hardware", type=str, help="Override local hardware description")
    parser.add_argument(
        "--lakeshore-hardware", type=str, help="Override Lakeshore hardware description"
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Handle comparison mode
    # ----------------------------------------------------------------
    if args.compare:
        print("Comparison mode not fully implemented yet")
        # TODO: Implement comparison
        return

    # ----------------------------------------------------------------
    # Initialize configuration
    # ----------------------------------------------------------------
    config = BenchmarkConfig(
        iterations=args.iterations,
        warmup=args.warmup,
    )

    # ----------------------------------------------------------------
    # Detect hardware for all tiers
    # ----------------------------------------------------------------
    print("\n🔍 HARDWARE DETECTION")
    print("=" * 60)

    # Local hardware (auto-detected)
    local_hw = detect_local_hardware()
    if args.local_hardware:
        local_hw.description = args.local_hardware
        local_hw.detection_method = "manual override"

    # Lakeshore hardware (queried from proxy)
    lakeshore_hw = query_lakeshore_hardware()
    if args.lakeshore_hardware:
        lakeshore_hw.description = args.lakeshore_hardware
        lakeshore_hw.detection_method = "manual override"

    # Cloud hardware (documented)
    cloud_hw = get_cloud_hardware("anthropic")

    config.hardware = {
        "local": local_hw,
        "lakeshore": lakeshore_hw,
        "cloud": cloud_hw,
    }

    # ----------------------------------------------------------------
    # Load model definitions
    # ----------------------------------------------------------------
    print("\n📦 LOADING MODELS")
    print("=" * 60)
    config.model_specs = load_models_from_litellm_config()

    # Determine which models to test
    if args.all_models:
        config.models = list(config.model_specs.keys())
    elif args.models:
        config.models = args.models
    elif args.tier:
        config.models = [m for m, s in config.model_specs.items() if s.tier == args.tier]
    else:
        config.models = get_default_models()

    # ----------------------------------------------------------------
    # Print benchmark configuration
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🌊 STREAM LATENCY BENCHMARK")
    print("=" * 80)
    print(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations:  {config.iterations}")
    print(f"Warmup:      {config.warmup}")
    print(f'Test Query:  "{config.test_query}"')
    print("\n📋 Hardware Configuration:")
    print(f"   Local:     {config.hardware['local'].description}")
    print(f"   Lakeshore: {config.hardware['lakeshore'].description}")
    print(f"   Cloud:     {config.hardware['cloud'].description}")
    print(f"\n📦 Models to test: {', '.join(config.models)}")

    all_results = []

    # ----------------------------------------------------------------
    # Phase 1: Direct model tests (no judge)
    # ----------------------------------------------------------------
    if not args.judges_only:
        print("\n" + "=" * 80)
        print("📍 PHASE 1: DIRECT MODEL TESTS (No LLM Judge)")
        print("=" * 80)
        print("Testing each model directly to establish baseline latency.")

        for model in config.models:
            # Use custom label for Lakeshore if provided
            if "lakeshore" in model and args.lakeshore_label:
                label = f"lakeshore-{args.lakeshore_label}"
            else:
                label = model

            result = run_benchmark(
                model=model,
                judge=None,
                iterations=config.iterations,
                warmup=config.warmup,
                label=label,
            )
            all_results.append(result)

    # ----------------------------------------------------------------
    # Phase 2: Auto mode with different judge strategies
    # ----------------------------------------------------------------
    if not args.no_judges:
        print("\n" + "=" * 80)
        print("🤖 PHASE 2: AUTO MODE WITH LLM JUDGE STRATEGIES")
        print("=" * 80)
        print("Testing automatic routing with different complexity judges.")
        print("The overhead vs direct tests shows the judge latency cost.")

        for judge in ["ollama-1b", "ollama-3b", "haiku"]:
            result = run_benchmark(
                model="auto",
                judge=judge,
                iterations=config.iterations,
                warmup=config.warmup,
            )
            all_results.append(result)

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\n{'Configuration':<30} {'P50':>10} {'P95':>10} {'P99':>10} {'Success':>10}")
    print("─" * 75)

    for r in all_results:
        s = r["ttft_stats"]
        p50 = f"{s['p50']:.3f}s" if s["count"] > 0 else "N/A"
        p95 = f"{s['p95']:.3f}s" if s["count"] > 0 else "N/A"
        p99 = f"{s['p99']:.3f}s" if s["count"] > 0 else "N/A"
        success = f"{r['success_rate']:.0f}%"
        print(f"{r['label']:<30} {p50:>10} {p95:>10} {p99:>10} {success:>10}")

    # ----------------------------------------------------------------
    # Export results
    # ----------------------------------------------------------------
    if args.output:
        base = Path(args.output)

        # JSON output (complete results with metadata)
        json_path = Path(f"{base}.json")
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "iterations": config.iterations,
                "warmup": config.warmup,
                "test_query": config.test_query,
                "hardware": {
                    tier: {
                        "description": spec.description,
                        "cpu": spec.cpu,
                        "ram": spec.ram,
                        "gpu": spec.gpu,
                        "os": spec.os,
                        "detection_method": spec.detection_method,
                    }
                    for tier, spec in config.hardware.items()
                },
                "models_tested": config.models,
            },
            "results": all_results,
        }

        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 JSON saved: {json_path}")

        # CSV output (for statistical analysis)
        if args.csv:
            csv_path = Path(f"{base}.csv")
            export_csv(all_results, csv_path)

    # ----------------------------------------------------------------
    # Generate figures
    # ----------------------------------------------------------------
    if args.figures:
        print("\n📊 GENERATING PUBLICATION FIGURES")
        print("=" * 60)
        generate_figures(all_results, config, FIGURES_DIR)

    # ----------------------------------------------------------------
    # Done!
    # ----------------------------------------------------------------
    print(f"\n✅ Benchmark completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
