# STREAM: PEARC 2026 Short Paper Outline

## Conference Details

- **Conference**: PEARC 2026 (Practice and Experience in Advanced Research Computing)
- **Track**: Track 2: Systems — Infrastructure and Middleware
- **Format**: Short paper (6 pages, ACM format)
- **Deadline**: March 30, 2026
- **Paper type**: Practice and experience report (deployed system)

---

## Paper Title (Options)

1. "STREAM: A Three-Tier Middleware for Cost-Aware LLM Inference Across Local, Campus HPC, and Cloud Resources"
2. "STREAM: Bridging Local, Campus, and Cloud LLM Inference with Tier-Aware Routing and Context Management"
3. "Democratizing LLM Access: A Multi-Tier Architecture with Modality-Aware Routing and Real-Time HPC Streaming"

---

## Contributions (4 Claims)

### C1: Three-Tier Routing Architecture (PRIMARY — systems contribution)
- **Claim**: STREAM is the first system that combines Local (user's laptop via Ollama), Campus HPC (Globus Compute + vLLM), and Cloud (multi-provider APIs) into a unified inference platform with automatic complexity-based routing.
- **Why novel**: No existing system combines all three tiers. FIRST (Argonne, SC'25) does two tiers (local server + HPC) but their "local" is still a server, not a user's laptop. Dartmouth Chat and Purdue GenAI Studio (PEARC'25) are campus-only. Hybrid LLM and RouteLLM route between models, not infrastructure tiers.
- **Evidence**: Deployed at UIC, serving students with $0 cost for Local and Lakeshore tiers.

### C2: Dual-Channel HPC Streaming (WebSocket Relay)
- **Claim**: STREAM solves the unsolved problem of real-time token streaming through Globus Compute by separating control plane (Globus Compute for auth + job submission) from data plane (WebSocket relay for token delivery).
- **Why novel**: Globus Compute is fundamentally batch-oriented (submit → wait → get result). No existing system streams tokens through it. FIRST (Argonne) uses Globus Compute but does NOT stream tokens — they initially polled every 2 seconds. Our relay enables ChatGPT-like real-time UX from HPC resources.
- **Evidence**: Implemented and working. See `stream/relay/server.py` and `docs/WEBSOCKET_RELAY_TECHNICAL_REPORT.md`.

### C3: Tier-Aware Context Management (TO BE IMPLEMENTED)
- **Claim**: STREAM introduces tier-aware context compression — compressing conversation history differently based on the target tier's context window, enabling small models to handle queries even in long conversations.
- **Why novel**: The routing literature (RouteLLM, Hybrid LLM) treats routing as a per-query decision and ignores context windows. The compression literature (LLMLingua, rolling summarization) compresses to a single target. Nobody has studied differential compression for multi-tier routing.
- **Formulation**: `route(query, history, tier_health, context_size) → (tier, compression_level)` — a joint optimization of routing and context management.
- **Evidence**: Implement rolling summarization with per-tier compression levels, measure tier distribution with/without compression.

### C4: Modality-Aware Routing (IMPLEMENTED)
- **Claim**: STREAM's router considers input modality (text vs. image+text) alongside query complexity, automatically directing multimodal queries to vision-capable tiers/models while respecting explicit user choices.
- **Why novel**: No campus LLM system handles modality-aware routing. The routing decision becomes richer: `route(query, complexity, modality, tier_health, model_capabilities) → (tier, model)`.
- **Evidence**: Implemented across all three tiers:
  - **Local**: Gemma 3 4B (vision) alongside Llama 3.2 3B (text)
  - **Lakeshore**: Qwen2.5-VL-72B (vision, 96 GiB H100)
  - **Cloud**: Claude Sonnet 4 and GPT-4 Turbo (both natively multimodal)
- **Routing rules**: AUTO mode auto-selects vision models; explicit model selection returns error if text-only model receives images (respects user intent, never silently overrides).
- **Vision judge**: Optional Gemma Vision 4B judge strategy can analyze images for complexity classification.
- **Globus constraint**: 8 MB payload validation with frontend compression (max 1024px, JPEG 85%).
- **Implementation**: See `docs/MULTIMODAL_SUPPORT.md` for detailed technical report.

### Model Table (for paper Section 3c)

| Tier | Model | Type | Parameters | Context | Use Case |
|------|-------|------|-----------|---------|----------|
| Local | Llama 3.2 3B | Text | 3B | 32K | General text queries |
| Local | Gemma 3 4B | Vision+Text | 4B | 32K | Image analysis, local multimodal |
| Lakeshore | Qwen2.5-VL-72B-AWQ | Vision+Text | 72B (4-bit) | 32K | Complex image+text queries |
| Cloud | Claude Sonnet 4 | Vision+Text | — | 200K | Highest quality multimodal |
| Cloud | GPT-4 Turbo | Vision+Text | — | 128K | General multimodal |

---

## Related Work

### Comparable Systems

| System | Institution | Tiers | Routing | Streaming | Multimodal | Context Mgmt |
|--------|-------------|-------|---------|-----------|------------|--------------|
| **STREAM** | UIC | Local + HPC + Cloud | LLM judge + modality + context | WebSocket relay | Yes (VL 72B) | Tier-aware compression |
| **FIRST** | Argonne (SC'25) | Local server + HPC | Static | No (batch return) | No | No |
| **Dartmouth Chat** | Dartmouth (PEARC'25) | Campus only | None (direct) | Standard SSE | No | No |
| **Purdue GenAI Studio** | Purdue (PEARC'25) | Campus only | None (Open WebUI) | Standard SSE | No | No |
| **Tufts LLM-Hub** | Tufts (PEARC'24) | Campus only | None | Unknown | No | No |
| **Hybrid LLM** | (ICLR'24) | 2 models | BERT classifier | N/A (API) | No | No |
| **RouteLLM** | UC Berkeley (ICLR'25) | 2 models | 4 classifiers | N/A (API) | No | No |

### Key Differentiators
- FIRST is closest but: (1) doesn't stream tokens, (2) routes to models not infrastructure tiers, (3) no local-device tier
- Dartmouth/Purdue are campus-only — no local tier, no cloud fallback, no routing
- Hybrid LLM/RouteLLM route between models (quality dimension only) not infrastructure tiers (cost + privacy + availability + context window)

---

## Routing Taxonomy (Related Work Section)

Seven methods exist in the literature for LLM query routing. STREAM uses Method 2 (LLM-as-judge) because it requires no training data, adapts via prompt editing, and adds acceptable latency (~1-2s with a local 3B model). The routing classifier is not STREAM's contribution — the multi-signal decision combining complexity + modality + context window + tier health is the novelty.

| # | Method | Paper | Latency | Training Data | STREAM? |
|---|--------|-------|---------|---------------|---------|
| 1 | Rule-based / keyword heuristics | Baseline in all papers | ~0ms | None | Fallback |
| 2 | **LLM-as-a-judge** | Router-R1 (2025) | ~1-2s | None | **Primary** |
| 3 | BERT encoder classifier | Hybrid LLM (ICLR'24) | ~5ms | Hundreds of labeled queries | No |
| 4 | Matrix factorization | RouteLLM (ICLR'25) | ~10ms | Preference pairs | No |
| 5 | kNN router | (May 2025 paper) | ~10ms | Reference dataset | No |
| 6 | Causal LLM classifier | RouteLLM (ICLR'25) | ~1-3s | Preference pairs | No |
| 7 | Cascading / self-verification | AutoMix (2024), FrugalGPT (2023) | Variable | None | No |

**Paper framing**: "Unlike model-level routing systems that optimize a single binary decision (strong vs. weak model), STREAM's routing engine must jointly consider query complexity, accumulated conversation context, input modality, and real-time tier availability to select both the target infrastructure tier and the appropriate context compression level — a fundamentally richer optimization problem."

---

## Context Compression Taxonomy (Related Work Section)

Five strategies exist for managing conversation context. STREAM uses Strategy 3 (rolling summarization) with tier-aware differential compression — producing different compression levels from the same history depending on the target tier's context window.

| # | Strategy | Key Paper | Compression | Extra LLM Calls | STREAM? |
|---|----------|-----------|-------------|-----------------|---------|
| 1 | Sliding window (truncation) | Baseline | Drop old turns | None | Building block |
| 2 | Observation masking | NeurIPS'25 workshop | Hide verbose outputs | None | No |
| 3 | **Rolling incremental summarization** | Acon (2025), ReSum (2025) | Summarize older turns | 1 per compression event | **Primary** |
| 4 | Token-level compression (LLMLingua) | EMNLP'23, ACL'24 | Drop low-info tokens | 1 (7B scorer) | Future work |
| 5 | Semantic memory / RAG | Mem0 | Extract facts as embeddings | Embedding per turn | Future work |

**Novel aspect**: All existing compression methods produce a single compressed output for a single target model. STREAM's tier-aware compression is the first to produce differential compression levels (aggressive for LOCAL 4K, moderate for Lakeshore 32K, raw for Cloud 200K) from the same conversation history, enabling the router to keep simple queries on cheap tiers even in long conversations.

### The Context-Routing Mismatch Problem

At turn 1, a simple question ("what is a for loop?") routes to LOCAL (free, fast). By turn 15, the raw history might be 5,000 tokens — now that same simple question can't run on LOCAL (4K context) even though LOCAL could answer it perfectly. Current behavior: force-upgrade to a higher tier, defeating the purpose of tiered routing.

**STREAM's solution**: Instead of upgrading the tier, compress the history to fit:
- Routing to LOCAL (4K context) → aggressive summary + last 2 turns verbatim
- Routing to Lakeshore (32K context) → lighter summary + last 5 turns verbatim
- Routing to Cloud (200K context) → pass everything raw

---

## WebSocket Relay Architecture (Section in Paper)

### The Problem
Globus Compute is batch-only: `submit(function, args) → wait → get_result()`. No mechanism for partial results, no streaming API, no yield-based delivery. Verified by searching SDK source, API docs, and GitHub issues — no streaming support exists or is planned.

### STREAM's Solution: Control Plane / Data Plane Separation

```
CONTROL PLANE (Globus Compute — authentication + job management):
  User → Middleware → Globus Compute → HPC Worker
  - OAuth2 authentication via Globus Auth
  - Job submission via AMQP queues
  - Status tracking (success/failure)

DATA PLANE (WebSocket Relay — real-time token delivery):
  HPC Worker → vLLM stream=True → WebSocket Relay → Middleware → Browser SSE
  - Lightweight forwarding, no computation
  - Both sides connect OUTBOUND (bypasses HPC firewalls)
  - Channel-based isolation (unique ID per request)
```

### Why This Works
1. The Globus Compute function launches vLLM inference with `stream=True`
2. As tokens arrive from vLLM, the function pushes them to the relay via WebSocket
3. The middleware consumes tokens from the relay and forwards them to the browser as Server-Sent Events (SSE)
4. Globus Compute still handles auth and error reporting — the relay only carries token data
5. Fallback: if the relay is unavailable, STREAM reverts to batch mode (fake streaming)

### Implementation
- Relay server: `stream/relay/server.py` (~200 lines, FastAPI + WebSockets)
- Producer: Inside the Globus Compute remote function on Lakeshore
- Consumer: `stream/proxy/app.py` (connects to relay, yields tokens as SSE)
- Full technical report: `docs/WEBSOCKET_RELAY_TECHNICAL_REPORT.md`

---

## Paper Structure (6 Pages)

### Page 1: Introduction (~0.75 pages)
- The problem: students need LLM access but cloud APIs cost money, campus GPUs are underutilized, and local laptops have limited models
- The gap: no system combines all three tiers with intelligent routing
- Contributions: list all 4 contributions (C1-C4)
- One paragraph on deployment at UIC

### Page 2: Architecture (~1.25 pages)
- Three-tier overview diagram (LOCAL → Lakeshore → Cloud)
- Data flow: Browser → React → Middleware → Router → [Tier] → Response
- LLM-as-judge complexity classification with keyword fallback
- Multi-signal routing: complexity + modality + context + tier health
- Figure: Architecture diagram showing all three tiers with data flow

### Pages 3-4: Key Innovations (~2 pages)

**3a. WebSocket Relay (~0.75 pages)**
- The Globus Compute streaming limitation
- Control plane / data plane separation
- Figure: Dual-channel architecture diagram
- Fallback to batch mode

**3b. Tier-Aware Context Management (~0.75 pages)**
- The context-routing mismatch problem
- Rolling summarization with differential compression per tier
- Example: same conversation compressed to 3 levels
- Integration with the routing decision

**3c. Modality-Aware Routing (~0.5 pages)**
- Image queries auto-routed to vision-capable tiers
- Qwen2.5-VL-72B on H100 handles both text and images
- Router checks model capabilities before routing

### Page 5: Evaluation (~1 page)
- **Latency**: Time-to-first-token and total response time per tier (with/without relay)
- **Cost**: Per-query cost comparison (Local=$0, Lakeshore=$0, Cloud=$X)
- **Tier distribution**: How queries distribute across tiers in normal usage vs. with compression
- **Context compression impact**: "Without tier-aware compression, X% of queries force-upgrade to Cloud after turn 10. With it, LOCAL handles Y% of simple queries even at turn 30."
- Table: Metrics across 30 multi-turn synthetic conversations

### Page 6: Related Work + Conclusion (~1 page)
- Compare to FIRST, Dartmouth Chat, Purdue GenAI Studio, Hybrid LLM, RouteLLM
- Key differentiation table (condensed from above)
- Future work: full user study, BERT classifier training, LLMLingua integration, cross-session memory
- Conclusion: 3-4 sentences summarizing contributions and impact

---

## Evaluation Plan

### Metrics to Collect

| Metric | How | Expected Result |
|--------|-----|-----------------|
| Time-to-first-token per tier | Timestamp logging in middleware | LOCAL: <1s, Lakeshore: ~5s (Globus overhead), Cloud: ~1-2s |
| Total response time per tier | End-to-end timing | LOCAL: 2-5s, Lakeshore: 10-30s, Cloud: 3-10s |
| Streaming vs batch latency (Lakeshore) | Compare relay mode vs fake streaming | Relay: TTFT ~5s, Batch: TTFT ~15-30s |
| Cost per conversation | SQLite cost tracking (already implemented) | 10-turn convo: $0 (local/lakeshore) vs $0.03-0.10 (cloud) |
| Tier distribution | Log which tier handles each query | auto mode: ~60% local, ~25% lakeshore, ~15% cloud |
| Compression impact on tier distribution | Run same conversations with/without compression | Without: 80% cloud after turn 10. With: 45% local at turn 30 |
| Routing accuracy | Label 50 queries manually, compare to judge | Expected ~85% accuracy with 3B judge |

### Synthetic Conversation Dataset
- 10 short conversations (5 turns) — simple Q&A
- 10 medium conversations (15 turns) — mixed complexity
- 10 long conversations (30 turns) — sustained interaction with topic evolution
- Each conversation includes a mix of LOW/MEDIUM/HIGH complexity queries
- Some conversations include image queries (for modality-aware routing evaluation)

---

## Implementation Timeline

| Period | Task | Paper Impact |
|--------|------|--------------|
| Feb 20-25 | Implement rolling summarization (tier-aware compression) | C3 |
| Feb 25-28 | Add multimodal support (image upload + routing) | C4 |
| Mar 1-5 | Run evaluation — collect latency/cost/tier metrics | Evaluation section |
| Mar 5-10 | Run compression experiment (with/without, tier distribution) | Key result |
| Mar 10-20 | Write paper (architecture + innovations + evaluation) | Full paper |
| Mar 20-25 | Figures, tables, formatting | Polish |
| Mar 25-30 | Review, feedback, revise, submit | Final |

---

## Key References

1. Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing," ICLR 2024
2. Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," ICLR 2025
3. Chard et al., "Globus Compute: A Federated Function-as-a-Service Platform," Future Generation Computer Systems, 2024
4. FIRST (Argonne), "Federated Inference Resource Scheduling Toolkit for Scientific AI Model Access," SC'25 Workshops
5. Dartmouth Chat, "Providing On-Prem GenAI Inference Services to a Campus Community," PEARC 2025
6. Purdue GenAI Studio, PEARC 2025
7. Tufts LLM-Hub, PEARC 2024
8. Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of LLMs," EMNLP 2023
9. "The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management," NeurIPS 2025 Workshop
10. "Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers," May 2025
11. Chen et al., "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," 2023

---

## STREAM System Details (for Methods section)

### Hardware
- **LOCAL**: User's laptop (Apple Silicon Mac, consumer GPU, or CPU-only)
- **Lakeshore HPC**: H100 NVL 96GB GPU (full), A100 40GB MIG slices
- **Cloud**: Anthropic Claude Sonnet 4, OpenAI GPT-4 Turbo, GPT-3.5 Turbo

### Models per Tier
| Tier | Model | Parameters | Context | Multimodal |
|------|-------|------------|---------|------------|
| LOCAL (tiny) | Llama 3.2 1B (Ollama) | 1B | 4K | No |
| LOCAL (default) | Llama 3.2 3B (Ollama) | 3B | 32K | No |
| LOCAL (quality) | Llama 3.1 8B (Ollama) | 8B | 4K | No |
| Lakeshore | Qwen 2.5 VL 72B AWQ (vLLM) | 72B | 32K | **Yes** |
| Cloud | Claude Sonnet 4 | Unknown | 200K | Yes |
| Cloud | GPT-4 Turbo | Unknown | 128K | Yes |

### Software Stack
- **Frontend**: React 19 + TypeScript + Tailwind CSS
- **Middleware**: FastAPI (Python) — routing, context management, health checks
- **Gateway**: LiteLLM (normalizes APIs across Ollama, vLLM, Claude, OpenAI)
- **HPC Integration**: Globus Compute SDK (authentication + remote execution)
- **Streaming Relay**: FastAPI + WebSockets (lightweight token forwarder)
- **Local Inference**: Ollama (packaged with desktop app)
- **HPC Inference**: vLLM 0.15.1 in Apptainer containers on SLURM-managed GPUs
- **Desktop Packaging**: PyWebView + embedded FastAPI server

### Deployment Modes
- **Desktop mode**: Single-user app (PyWebView), Ollama local, Lakeshore via Globus, Cloud via API keys
- **Server mode**: Multi-user Docker Compose (5 containers: React, Middleware, LiteLLM Gateway, Ollama, Lakeshore Proxy)

---

## Novelty Positioning (How to Frame for Reviewers)

### What NOT to claim
- Don't claim the LLM-as-judge routing classifier is novel (it's not — RouteLLM and others have better classifiers)
- Don't claim cost savings alone is novel (every campus system claims this)
- Don't claim "we built a chatbot" (reviewers will reject this immediately)

### What TO claim
1. **Infrastructure-tier routing** (not just model routing) — the routing decision accounts for infrastructure properties (latency, availability, context window, privacy, cost) not just model quality
2. **The joint optimization formulation**: `route(query, history, modality, tier_health) → (tier, model, compression_level)` — nobody has formalized this
3. **WebSocket relay solving Globus Compute's streaming limitation** — verified novel, FIRST doesn't do this
4. **Tier-aware differential context compression** — at the intersection of two well-studied areas (routing + compression) in a way neither community has explored
5. **Modality-aware routing** to infrastructure tiers — no campus system handles this

### The one-sentence pitch
"STREAM is a middleware that jointly optimizes LLM routing, context compression, and modality awareness across heterogeneous infrastructure tiers — local, campus HPC, and cloud — providing students with free, private, high-quality AI access while keeping simple queries on cheap resources even as conversations grow."
