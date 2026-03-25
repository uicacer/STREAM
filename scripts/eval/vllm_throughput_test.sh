#!/bin/bash
# =============================================================================
# vLLM Throughput Benchmark — Measures tok/s for Baseline vs Optimized
# =============================================================================
#
# Sends identical requests to a running vLLM server and measures throughput.
# Run this AFTER the vLLM server is up (either baseline or optimized).
#
# Usage:
#   bash scripts/eval/vllm_throughput_test.sh <host> <port> <label>
#
# Examples:
#   bash scripts/eval/vllm_throughput_test.sh ghi2-002 8000 baseline
#   bash scripts/eval/vllm_throughput_test.sh ghi2-002 8000 optimized
#
# Full verification workflow:
#   1. sbatch scripts/eval/vllm_baseline.sh         # Start baseline server
#   2. Wait ~60-90s for "Application startup complete"
#   3. bash scripts/eval/vllm_throughput_test.sh ghi2-002 8000 baseline
#   4. scancel <job_id>                               # Stop baseline
#   5. sbatch scripts/eval/vllm_optimized.sh          # Start optimized server
#   6. Wait ~60-90s for "Application startup complete"
#   7. bash scripts/eval/vllm_throughput_test.sh ghi2-002 8000 optimized
#   8. Compare results in scripts/eval/results/
#
# Output: JSON results saved to scripts/eval/results/throughput_<label>_<timestamp>.json
# =============================================================================

set -euo pipefail

HOST="${1:?Usage: $0 <host> <port> <label>}"
PORT="${2:?Usage: $0 <host> <port> <label>}"
LABEL="${3:?Usage: $0 <host> <port> <label>}"

BASE_URL="http://${HOST}:${PORT}"
MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
RUNS=5
WARMUP=1
MAX_TOKENS=200

# Diverse prompts — one per run to prevent prefix cache hits across runs.
# All are similar-length technical questions expected to produce ~200 token responses.
# Using diverse queries simulates realistic workloads and prevents artificial
# inflation of results from repeated identical queries.
WARMUP_PROMPT="Explain the concept of recursion in programming. Give a clear example."
PROMPTS=(
    "What is the difference between machine learning and deep learning?"
    "Describe how a relational database works and when to use it."
    "Explain how TCP/IP works in computer networking."
    "What is the difference between a process and a thread in an operating system?"
    "Explain how gradient descent works in neural network training."
)

RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
OUTPUT_FILE="${RESULTS_DIR}/throughput_${LABEL}_${TIMESTAMP}.json"

echo "=========================================="
echo "  vLLM Throughput Benchmark"
echo "  Label:     ${LABEL}"
echo "  Server:    ${BASE_URL}"
echo "  Model:     ${MODEL}"
echo "  Runs:      ${RUNS} (+${WARMUP} warmup)"
echo "  Max tokens: ${MAX_TOKENS}"
echo "=========================================="

# Check if server is ready
echo ""
echo "Checking server health..."
if ! curl -s --max-time 5 "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding at ${BASE_URL}"
    echo "Make sure the SLURM job is running and vLLM has finished loading."
    echo ""
    echo "Check with: squeue -u \$USER"
    echo "Check logs: tail -f logs/bench-${LABEL}-*.log"
    exit 1
fi
echo "Server is healthy."

# Get GPU info from server (vLLM exposes this)
echo ""
echo "Server info:"
curl -s "${BASE_URL}/v1/models" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('data', []):
    print(f\"  Model: {m['id']}\")
" 2>/dev/null || echo "  (could not fetch model info)"

echo ""
echo "Running benchmark..."
echo ""

# Arrays to store results
declare -a THROUGHPUTS
declare -a TOTAL_TIMES
declare -a OUTPUT_TOKENS_ARR

TOTAL_RUNS=$((WARMUP + RUNS))

for i in $(seq 1 $TOTAL_RUNS); do
    IS_WARMUP=$( [ "$i" -le "$WARMUP" ] && echo "true" || echo "false" )
    RUN_LABEL=$( [ "$IS_WARMUP" = "true" ] && echo "warmup" || echo "run $((i - WARMUP))" )

    # Use warmup prompt for warmup run, diverse prompts for actual runs
    if [ "$IS_WARMUP" = "true" ]; then
        CURRENT_PROMPT="$WARMUP_PROMPT"
    else
        PROMPT_IDX=$(( (i - WARMUP - 1) % ${#PROMPTS[@]} ))
        CURRENT_PROMPT="${PROMPTS[$PROMPT_IDX]}"
    fi

    printf "  [%-8s] " "$RUN_LABEL"

    # Send request and capture timing + response
    START_TIME=$(python3 -c "import time; print(time.time())")

    RESPONSE=$(curl -s --max-time 300 \
        -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${CURRENT_PROMPT}\"}],
            \"max_tokens\": ${MAX_TOKENS},
            \"temperature\": 0.7,
            \"stream\": false
        }")

    END_TIME=$(python3 -c "import time; print(time.time())")

    # Parse response
    PARSE_RESULT=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'error' in data:
        print(f'ERROR|0|0|{data[\"error\"].get(\"message\", \"unknown error\")}')
    else:
        usage = data.get('usage', {})
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        content = data['choices'][0]['message']['content']
        print(f'OK|{output_tokens}|{total_tokens}|{len(content)}')
except Exception as e:
    print(f'ERROR|0|0|{e}')
" 2>/dev/null)

    STATUS=$(echo "$PARSE_RESULT" | cut -d'|' -f1)
    OUTPUT_TOKENS=$(echo "$PARSE_RESULT" | cut -d'|' -f2)
    TOTAL_TOKENS=$(echo "$PARSE_RESULT" | cut -d'|' -f3)
    EXTRA=$(echo "$PARSE_RESULT" | cut -d'|' -f4)

    ELAPSED=$(python3 -c "print(round($END_TIME - $START_TIME, 2))")

    if [ "$STATUS" = "ERROR" ]; then
        echo "ERROR: $EXTRA"
        continue
    fi

    # Calculate throughput
    THROUGHPUT=$(python3 -c "
tokens = $OUTPUT_TOKENS
elapsed = $ELAPSED
tp = tokens / elapsed if elapsed > 0 and tokens > 0 else 0
print(f'{tp:.1f}')
")

    echo "tokens=${OUTPUT_TOKENS}  time=${ELAPSED}s  throughput=${THROUGHPUT} tok/s"

    # Store results (skip warmup)
    if [ "$IS_WARMUP" = "false" ]; then
        THROUGHPUTS+=("$THROUGHPUT")
        TOTAL_TIMES+=("$ELAPSED")
        OUTPUT_TOKENS_ARR+=("$OUTPUT_TOKENS")
    fi
done

echo ""

# Calculate statistics
if [ ${#THROUGHPUTS[@]} -eq 0 ]; then
    echo "ERROR: No successful runs. Cannot compute statistics."
    exit 1
fi

python3 << STATS_EOF
import json
import statistics
from datetime import datetime

throughputs = [$(IFS=,; echo "${THROUGHPUTS[*]}")]
total_times = [$(IFS=,; echo "${TOTAL_TIMES[*]}")]
output_tokens = [$(IFS=,; echo "${OUTPUT_TOKENS_ARR[*]}")]

results = {
    "timestamp": datetime.now().isoformat(),
    "label": "${LABEL}",
    "server": "${BASE_URL}",
    "model": "${MODEL}",
    "prompt": "diverse (5 unique queries per run to prevent prefix cache hits)",
    "max_tokens": ${MAX_TOKENS},
    "num_runs": ${RUNS},
    "num_warmup": ${WARMUP},
    "hardware": {
        "gpu": "NVIDIA H100 NVL 94 GB",
        "node": "ghi2-002",
        "driver": "550.163.01 (CUDA 12.4)",
    },
    "config": {
        "baseline": {
            "container": "vllm-0.15.1 (CUDA 12.9)",
            "quantization": "awq (plain)",
            "context_length": 32768,
            "gpu_memory_utilization": 0.85,
            "prefix_caching": False,
            "chunked_prefill": False,
            "torch_compile": False,
        },
        "optimized": {
            "container": "vllm-cu124 (CUDA 12.4, custom build)",
            "quantization": "awq_marlin (Marlin kernels)",
            "context_length": 65536,
            "gpu_memory_utilization": 0.90,
            "prefix_caching": True,
            "chunked_prefill": True,
            "torch_compile": False,
            "note": "torch.compile disabled due to Triton autotuning crash on driver 550",
        },
    },
    "results": {
        "throughput": {
            "median": round(statistics.median(throughputs), 1),
            "mean": round(statistics.mean(throughputs), 1),
            "stdev": round(statistics.stdev(throughputs), 1) if len(throughputs) > 1 else 0,
            "min": round(min(throughputs), 1),
            "max": round(max(throughputs), 1),
            "values": throughputs,
        },
        "total_time": {
            "median": round(statistics.median(total_times), 2),
            "mean": round(statistics.mean(total_times), 2),
            "values": total_times,
        },
        "output_tokens": {
            "median": round(statistics.median(output_tokens)),
            "mean": round(statistics.mean(output_tokens)),
            "values": output_tokens,
        },
    },
}

# Save JSON
with open("${OUTPUT_FILE}", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
tp = results["results"]["throughput"]
print("=" * 60)
print(f"  RESULTS: {results['label'].upper()}")
print("=" * 60)
print(f"  Throughput: {tp['median']:.1f} tok/s (median)")
print(f"             {tp['mean']:.1f} tok/s (mean)")
print(f"             +/- {tp['stdev']:.1f} tok/s (stdev)")
print(f"             range: {tp['min']:.1f} - {tp['max']:.1f} tok/s")
print(f"  Output tokens: {results['results']['output_tokens']['median']} (median)")
print(f"  Total time:    {results['results']['total_time']['median']:.2f}s (median)")
print()
print(f"  Results saved to: ${OUTPUT_FILE}")
print("=" * 60)
STATS_EOF

echo ""
echo "Next steps:"
if [ "$LABEL" = "baseline" ]; then
    echo "  1. Stop this server: scancel \$(squeue -u \$USER -h -o %i | head -1)"
    echo "  2. Start optimized: sbatch scripts/eval/vllm_optimized.sh"
    echo "  3. Wait ~60s, then: bash scripts/eval/vllm_throughput_test.sh ${HOST} ${PORT} optimized"
    echo "  4. Compare: python3 scripts/eval/compare_throughput.py"
elif [ "$LABEL" = "optimized" ]; then
    echo "  Compare results: python3 scripts/eval/compare_throughput.py"
fi
