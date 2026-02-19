# STREAM — Presentation Slides
### Copy each slide into Google Slides. Titles go in the title box, bullets go in the body.

---

## SLIDE 1 — Title Slide

**Title:**
STREAM: Smart Tiered Routing Engine for AI Models

**Subtitle:**
Connecting a Chat Interface to HPC GPUs Through Globus Compute

**Bottom:**
Nassar — UIC ACER Lab


---

## SLIDE 2 — The Problem We're Solving

**Title:**
The Problem

**Body:**
- AI inference is expensive — cloud APIs charge per token ($3–$15 per million tokens)
- UIC has free GPU resources on Lakeshore (NVIDIA A100s) — but they're behind a firewall
- Students and researchers shouldn't need to know how to SSH into an HPC cluster just to ask an AI a question
- What if we could route AI queries to the cheapest appropriate backend automatically?

**Speaker notes:**
The core insight is that not every question needs a $0.01 API call. "What is photosynthesis?" can be answered by a small local model. Only complex design or research questions need expensive cloud models. And medium questions can go to our university's free GPUs — if we can figure out how to reach them.


---

## SLIDE 3 — What is STREAM?

**Title:**
STREAM: Three Tiers, One Interface

**Body:**

| Tier | Where It Runs | Model | Cost |
|------|--------------|-------|------|
| Local | Your laptop (Ollama) | Llama 3.2 3B | Free |
| Lakeshore | UIC HPC GPUs (vLLM) | 5 models (1.5B–32B) | Free |
| Cloud | Anthropic / OpenAI APIs | Claude, GPT-4 | Pay-per-token |

- A complexity judge (small LLM) analyzes each question
- Simple → Local, Medium → Lakeshore, Complex → Cloud
- Automatic fallback: if a tier is down, the next one takes over
- User sees one chat interface — routing is invisible

**Speaker notes:**
STREAM stands for Smart Tiered Routing Engine for AI Models. The user types a question and STREAM decides where to send it. The hard engineering is in making this feel seamless — especially the Lakeshore connection, which is the focus of this talk.


---

## SLIDE 4 — Why Lakeshore Is the Hard Part

**Title:**
Why Lakeshore Is the Hard Part

**Body:**

```
Local tier:    Your laptop → Ollama on localhost        ✅ Easy
Cloud tier:    Your laptop → HTTPS to Anthropic/OpenAI  ✅ Easy
Lakeshore:     Your laptop → ??? → HPC behind firewall  🔒 Hard
```

- Lakeshore's GPU nodes are behind UIC's campus firewall
- No public IP address — you can't just send an HTTP request to them
- vLLM (the model server) listens on `http://ga-002:8000` — only reachable from inside the cluster
- Traditional solution: SSH tunnel — but that requires SSH keys, VPN, and technical knowledge

**Speaker notes:**
Local and Cloud tiers are straightforward networking — localhost or public HTTPS. Lakeshore is fundamentally different. The GPU node is invisible to the internet. You cannot reach it from your laptop, your Docker container, or anywhere outside UIC's network. This is standard HPC security — minimize attack surface. But it means we need a creative solution to bridge the gap.


---

## SLIDE 5 — Enter Globus Compute

**Title:**
The Solution: Globus Compute

**Body:**
- Globus Compute (formerly funcX) — Function-as-a-Service for research computing
- Developed by University of Chicago + Argonne National Laboratory
- Instead of connecting directly to the GPU, we send a *function* through Globus's cloud infrastructure
- Globus handles authentication, routing through firewalls, and delivering results

```
Traditional HPC:  SSH → login node → sbatch job.sh → wait → scp results
Globus Compute:   Python SDK → Globus cloud → function runs on HPC → result returned
```

- No SSH keys needed
- No VPN needed
- No firewall modifications needed
- Works from anywhere with internet

**Speaker notes:**
Globus Compute is like a postal service for code. You write a Python function, hand it to Globus, and Globus delivers it to your HPC cluster, runs it, and brings the result back. The user never touches SSH, never configures a VPN, never even knows Lakeshore exists behind a firewall. This is what makes STREAM accessible to students who have never used an HPC cluster.


---

## SLIDE 6 — How Globus Compute Works (Conceptual)

**Title:**
How Globus Compute Works

**Body:**
Think of it as a postal service for code:

1. **You** (STREAM) write a function: "call vLLM with this prompt and return the answer"
2. **You** hand it to the post office (Globus Cloud on AWS)
3. The **post office** routes it to the destination (Lakeshore endpoint daemon)
4. The **recipient** (endpoint on Lakeshore) opens the letter, runs the function, writes the result
5. The **result** comes back through the post office to you

Key components:
- **Globus Compute SDK** — Python library on your machine (the sender)
- **Globus Cloud** — AWS-hosted coordination service (the post office)
- **Globus Endpoint** — Daemon on Lakeshore that runs functions (the recipient)
- **AMQP** — The messaging protocol connecting them all (the mail trucks)

**Speaker notes:**
AMQP stands for Advanced Message Queuing Protocol — it's the same protocol that RabbitMQ uses. Globus runs an AMQP broker in the cloud. Our SDK publishes tasks to it, and the Lakeshore endpoint picks them up. The key insight is that both sides make OUTBOUND connections to the cloud broker — neither side needs to accept incoming connections. This is how it bypasses the firewall.


---

## SLIDE 7 — The Complete Journey of a Query

**Title:**
The Complete Journey of a Lakeshore Query

**Body:**

```
Your Laptop                      AWS (Globus Cloud)              Lakeshore HPC
============                     ==================              ==============

1. User asks a question
   in the chat UI

2. STREAM judges complexity
   → "medium" → route to
   Lakeshore

3. Serialize function + args
   with dill (Python)
   Send via AMQP ──────────────→ 4. Broker receives task
                                    Routes to endpoint
                                    queue ──────────────────→ 5. Endpoint picks up task
                                                                 Deserializes function

                                                              6. Function calls vLLM
                                                                 on ga-002:<port>
                                                                 GPU generates answer

                                  8. Broker receives          7. Serialize result
   9. SDK picks up result ←──────    result ←───────────────     Send via AMQP
      Deserialize

10. Convert to streaming
    text (word by word)
    Display in chat UI
```

**Speaker notes:**
Let me walk through each step. The user types "Explain photosynthesis." STREAM's judge model says this is medium complexity — route to Lakeshore. We serialize a Python function and its arguments using dill (a serialization library), publish it as an AMQP message to the Globus broker on AWS. The broker routes it to our Lakeshore endpoint. The endpoint deserializes the function, runs it — and inside that function, it makes an HTTP POST to vLLM on the GPU node. vLLM generates the response. The result travels back through the same AMQP channel to our SDK. We then convert the complete response into a streaming display for the user.


---

## SLIDE 8 — What Runs on Lakeshore (The Remote Function)

**Title:**
The Remote Function: What Actually Runs on the GPU Node

**Body:**

```python
def remote_vllm_inference(vllm_url, model, messages,
                          temperature, max_tokens, stream=False):
    import requests  # Must import inside — remote env is different

    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120
    )
    return response.json()
```

- This function is serialized on your laptop and deserialized on Lakeshore
- It runs INSIDE the Lakeshore network — so it CAN reach the vLLM server
- The function is self-contained: imports must be inside (Lakeshore doesn't have STREAM installed)

**Speaker notes:**
This is the actual function that Globus delivers to Lakeshore. Notice it imports `requests` inside the function body — because the remote environment is a bare Python installation, it doesn't have STREAM's modules. The function is completely self-contained. When it runs on Lakeshore, it's inside the firewall, so `ga-002:8000` is reachable. It makes a simple HTTP POST to vLLM — the same API format that OpenAI uses — and returns the JSON response.


---

## SLIDE 9 — The Proxy: Why LiteLLM Can't Talk to Globus

**Title:**
The Protocol Problem: Why We Need a Proxy

**Body:**

STREAM uses LiteLLM as its AI gateway — it speaks **HTTP** (OpenAI API format).
Globus Compute speaks **AMQP** (message queue protocol).
These are fundamentally incompatible. LiteLLM cannot submit a Globus function.

```
LiteLLM Gateway                              Globus Compute
─────────────────                             ─────────────────
Speaks: HTTP                                  Speaks: AMQP
Format: POST /v1/chat/completions             Format: serialize(function) → AMQP publish
Expects: streaming SSE response               Returns: Future → result (one shot)

           ❌ Cannot talk to each other directly
```

**The Solution: A Proxy that translates between the two worlds**

```
LiteLLM ──HTTP──→ Lakeshore Proxy ──AMQP──→ Globus Compute ──→ Lakeshore GPU
                  (the translator)
         ←─HTTP──                  ←─AMQP──                 ←── vLLM response
```

The proxy receives an OpenAI-format HTTP request, submits it as a Globus Compute function, waits for the result, and returns it as an HTTP response.

**Speaker notes:**
This is a critical piece of the architecture. LiteLLM is a powerful gateway that normalizes all AI providers into one API format — OpenAI's HTTP format. It knows how to talk to Claude, GPT, Ollama, and hundreds of providers. But it has no idea what Globus Compute is. Globus Compute uses AMQP — a completely different protocol where you serialize Python functions and submit them through a message queue. There's no way to make LiteLLM speak AMQP. So we built a proxy — a translation layer that looks like an OpenAI-compatible HTTP server to LiteLLM, but internally submits everything through Globus Compute. LiteLLM thinks it's talking to a normal AI server. It has no idea the request is being serialized, routed through AWS, and executed on an HPC cluster behind a firewall.


---

## SLIDE 10 — Server Mode: The 5-Container Path

**Title:**
Server Mode: How the Request Reaches Lakeshore

**Body:**

In Docker, the proxy runs as its own container — a separate process:

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                            │
│                                                             │
│  React UI ──→ Middleware ──→ LiteLLM GW ──→ Lakeshore Proxy │
│   (host)      (container)   (container)    (container)      │
│                                                             │
│  "Is this        "Medium      "lakeshore-    Receives HTTP  │
│   complex?"      → Lakeshore"  qwen maps     Translates     │
│                               to proxy"      to Globus      │
│                                              Submits fn     │
│                                              via AMQP       │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                              ┌────────────────┴────────────────┐
                              │         Globus Cloud (AWS)       │
                              │         AMQP Broker              │
                              └────────────────┬────────────────┘
                                               │
                              ┌────────────────┴────────────────┐
                              │         Lakeshore HPC            │
                              │     vLLM on ga-002:8000-8004     │
                              └─────────────────────────────────┘
```

Each arrow is an HTTP call between separate processes — no deadlocks possible.
The proxy container mounts `~/.globus_compute` from the host for authentication.

**Speaker notes:**
In server mode, everything is clean. Five Docker containers on a virtual network. The React UI sends a chat request to the middleware. The middleware judges complexity and decides on Lakeshore. It calls LiteLLM, which reads its config and sees that Lakeshore models point to the proxy container on port 8001. LiteLLM sends a standard HTTP request to the proxy. The proxy receives it, extracts the model and messages, and submits them as a Globus Compute function through AMQP. When the result comes back, the proxy converts it to an HTTP response and sends it back through the chain. Each arrow is between separate containers with separate processes and separate event loops. No deadlocks, no conflicts. This is the straightforward architecture.


---

## SLIDE 11 — Desktop Mode: The Single-Process Challenge

**Title:**
Desktop Mode: Why the Same Approach Breaks

**Body:**

In desktop mode, everything is ONE process. The proxy routes are embedded in the same FastAPI server:

```
┌──────────────────────────────────────────────────┐
│          Single Process (STREAM.app)              │
│                                                   │
│  React UI ──→ Middleware ──→ LiteLLM lib          │
│  (static         │                │               │
│   files)         │          tries HTTP POST       │
│                  │          to localhost:5000      │
│                  │                │               │
│                  │                ▼               │
│                  │     ┌──────────────────┐       │
│                  │     │  SAME FastAPI    │       │
│                  │     │  SAME event loop │       │
│                  │     │  DEADLOCK! 💀    │       │
│                  │     └──────────────────┘       │
│                  │                                │
│  Our solution:   │                                │
│                  ▼                                │
│  litellm_direct.py ──→ globus_client.submit()    │
│  (direct Python call, no HTTP, no deadlock)       │
└──────────────────────────────────────┼───────────┘
                                       │
                          Globus Cloud → Lakeshore
```

**The fix:** Detect desktop mode → skip HTTP entirely → call Globus client directly as a Python function

**Speaker notes:**
Here's where it gets tricky. In the desktop app, there's no Docker. Everything runs in one process — one FastAPI server, one event loop. The proxy routes are mounted into the same app at /lakeshore. Now, if LiteLLM tries to make an HTTP POST to localhost:5000/lakeshore — it's posting to ITSELF. The event loop needs to simultaneously send the request AND handle the incoming request. With a single-worker server, this creates a deadlock — the outgoing request blocks waiting for a response, but the server can't process the incoming request because it's blocked on the outgoing one. Our solution: in desktop mode, we completely bypass HTTP. Instead of LiteLLM making an HTTP call to the proxy, we call the Globus Compute client directly as a Python function call within the same process. No HTTP, no network, no deadlock. The function litellm_direct.py detects that the model starts with "lakeshore" and calls globus_client.submit_inference() directly. This is why the desktop mode required significant additional engineering beyond the server mode.


---

## SLIDE 12 — Technical Challenge: Serialization

**Title:**
Technical Challenge: The Serialization Problem

**Body:**
- Globus Compute serializes your function into bytes using `dill` (like `pickle` but for functions)
- These bytes are sent to Lakeshore and deserialized there

**The Problem (Desktop Mode):**
- STREAM's desktop app is built with PyInstaller (bundles Python into a .app/.exe)
- PyInstaller uses custom bytecode with references to `pyimod02_importers`
- Lakeshore doesn't have PyInstaller → deserialization fails:
  `ModuleNotFoundError: No module named 'pyimod02_importers'`

**The Solution:**
- Define the function from a source STRING using `exec()` at runtime
- Python's compiler generates clean bytecode — no PyInstaller references
- Works in both development (normal Python) and production (PyInstaller bundle)

**4 attempts before finding the solution:**
1. `CombinedCode` strategy → `inspect.getsource()` fails (no .py files in bundle)
2. `AllCodeStrategies` with normal `def` → dill by-reference → `"No module named 'stream'"`
3. `__module__ = '__main__'` → dill by-value → `"No module named 'pyimod02_importers'"`
4. `exec()` from source string → clean bytecode ✅

**Speaker notes:**
This was one of the hardest bugs to solve. When you build a desktop app with PyInstaller, it bundles your Python code as compiled bytecode. But that bytecode has references to PyInstaller's internal import system. When Globus tries to deserialize this on Lakeshore — which is just regular Python — it fails because PyInstaller doesn't exist there. We tried four different approaches before landing on the solution: define the function from a raw source string using exec(). This produces clean bytecode at runtime, regardless of whether Python is running inside PyInstaller or not.


---

## SLIDE 13 — What exec() Actually Does (The Fix in Detail)

**Title:**
What Does `exec()` From a Source String Mean?

**Body:**

**Normal way** — define a function directly in your code:
```python
# In globus_compute_client.py

def remote_vllm_inference(url, model, messages):
    import requests
    return requests.post(url, json={...}).json()
```
Python compiles this into bytecode when the file loads.
If PyInstaller loaded the file → the bytecode contains PyInstaller's import system.
If Lakeshore deserializes this bytecode → it crashes (no PyInstaller there).

---

**Our way** — store the function as a text string, compile it fresh at runtime:
```python
# In globus_compute_client.py

_REMOTE_FN_SOURCE = """
def remote_vllm_inference(url, model, messages):
    import requests
    return requests.post(url, json={...}).json()
"""

_ns = {}
exec(compile(_REMOTE_FN_SOURCE, "<remote_vllm_inference>", "exec"), _ns)
remote_vllm_inference = _ns["remote_vllm_inference"]
```
`exec()` compiles the string at runtime using Python's standard compiler.
The result: clean bytecode with zero PyInstaller references.
Lakeshore deserializes it successfully.

**Speaker notes:**
Let me show you exactly what the fix looks like. Normally you just write `def my_function` in your Python file. Python compiles that into bytecode when the file loads — and if PyInstaller loaded the file, the bytecode is contaminated with PyInstaller's custom import system. Our fix: we don't write the function directly. Instead, we store the entire function definition as a plain text string — just characters, not compiled code. Then at runtime, we call `exec()` on that string. `exec()` is a built-in Python function that takes a string, compiles it, and executes it. When exec() compiles our string, it uses Python's standard compiler — not PyInstaller's. So the resulting bytecode is clean. When Globus serializes this function and sends it to Lakeshore, Lakeshore sees standard Python bytecode and deserializes it without any issues. The function works identically in both cases — but the bytecode is compiled differently. This is the actual code in our repository (globus_compute_client.py, lines 73-126).


---

## SLIDE 14 — Why Not Just Use Docker for Desktop?

**Title:**
"Why Not Use Docker for Desktop Too?"

**Body:**
Docker would solve both technical challenges (serialization + deadlock) — but it defeats the purpose:

| | Desktop (PyInstaller) | Desktop (via Docker) |
|---|---|---|
| User installs | Nothing — drag .app to Applications | Docker Desktop (~1 GB), then STREAM |
| User starts app | Double-click | Start Docker daemon, then run command |
| Runs as | Native process | Linux VM + containers |
| Startup time | ~2 seconds | ~10-15 seconds |
| RAM overhead | ~200 MB | ~1-2 GB (Docker VM + containers) |
| Requires admin/root | No | Yes (Docker installation) |
| Distributable via | .app / .exe installer | docker-compose file + README |

The target user is a student who has never opened a terminal.
"Install Docker Desktop, then run `docker compose up`" is exactly the barrier STREAM eliminates.

**Speaker notes:**
A natural question is: if Docker mode already works, why not just use Docker for the desktop app too? Because the whole point of desktop mode is zero-dependency simplicity. Docker Desktop is a 1 GB install that requires admin privileges and runs a hidden Linux virtual machine on macOS. Our PyInstaller app is a native process — drag it to Applications and double-click. The technical challenges we solved (exec-based serialization, direct Globus calls) exist specifically to avoid pushing that Docker complexity onto end users. We could also have used multiple processes within the app — spawn the proxy as a subprocess — but a single-process direct function call is the simplest solution with the fewest moving parts.


---

## SLIDE 15 — Authentication: How Users Log In

**Title:**
Authentication: Globus OAuth2 Flow

**Body:**

**First Time Setup:**
1. User runs `globus-compute-endpoint configure` on Lakeshore
2. Opens a browser link → logs in with UIC credentials (CILogon/SSO)
3. Globus stores OAuth2 tokens in `~/.globus_compute/storage.db`
4. These tokens are what STREAM uses to submit functions

**On Each Request:**
- SDK reads tokens from `storage.db`
- Includes them in AMQP authentication with the broker
- If tokens expire → SDK refreshes automatically
- If refresh fails → STREAM shows "Re-authenticate with Globus" in the UI

**Two Independent Auth Systems:**
| System | How | Who Has It |
|--------|-----|-----------|
| SSH keys | `~/.ssh/id_ed25519` | Researchers with HPC accounts |
| Globus OAuth2 | CILogon + browser SSO | Anyone with a UIC account |

Globus is what makes STREAM accessible — no SSH keys, no terminal skills needed.

**Speaker notes:**
There are two completely separate authentication systems for reaching Lakeshore. SSH key authentication is what traditional HPC users know — you have a private key on your machine, and the public key is on Lakeshore. But that requires terminal skills and key management. Globus uses OAuth2 through CILogon — you log in with your university credentials in a browser, just like logging into Blackboard. This is what makes STREAM accessible to students who have never used an HPC cluster. The tokens get stored in a SQLite database on the user's machine, and STREAM reads them automatically.


---

## SLIDE 16 — Authentication in Desktop vs Docker

**Title:**
Authentication Across Deployment Modes

**Body:**

**Desktop Mode:**
- STREAM reads tokens directly from `~/.globus_compute/storage.db` on your machine
- If tokens expire, the UI shows a re-authentication prompt
- User clicks a link, logs in via browser, tokens refresh

**Docker Mode:**
- The proxy container mounts credentials from the host:
  ```yaml
  volumes:
    - ${HOME}/.globus_compute:/root/.globus_compute:rw
  ```
- Authenticate once on the host machine → container uses those tokens
- `/reload-auth` endpoint lets the app re-read credentials after re-authentication

**Key Design Decision:**
- Authentication errors are NEVER silently swallowed
- If Globus auth fails → the error propagates to the user with clear instructions
- No silent fallback to a different tier when it's an auth problem

**Speaker notes:**
In Docker, we mount the Globus credentials directory from the host machine into the container as a volume. This means you authenticate once on your machine, and the containerized proxy can use those same tokens. The reload-auth endpoint is important — if tokens expire and you re-authenticate on the host, the container needs to know to re-read the new tokens from disk. We also made a deliberate design choice: authentication failures are never silently swallowed. If your Globus tokens are expired, STREAM tells you explicitly and shows you how to fix it, rather than quietly falling back to the Cloud tier and charging you money.


---

## SLIDE 17 — Security: What About the Stored Tokens?

**Title:**
Security: What If Tokens Are Stolen?

**Body:**
Globus stores OAuth2 tokens in `~/.globus_compute/storage.db` on the user's machine.

**If an attacker gets these tokens, they could:**
- Submit arbitrary functions to the Globus Compute endpoint on Lakeshore
- Those functions execute inside the cluster network (behind the firewall)
- Roughly equivalent to stealing someone's SSH private key

**But this is the standard model for ALL credential storage:**

| Credential | Stored as | Same risk? |
|---|---|---|
| Globus tokens | `~/.globus_compute/storage.db` | Yes |
| SSH private keys | `~/.ssh/id_ed25519` | Yes |
| AWS credentials | `~/.aws/credentials` (plain text) | Yes |
| Browser cookies | SQLite in browser profile | Yes |

**Built-in mitigations:**
- Access tokens expire in 30-60 minutes (short-lived)
- Refresh tokens can be **revoked remotely** via `app.globus.org`
- File permissions: only the owner can read `storage.db`
- No password stored — only tokens (revoking doesn't require a password change)

**Speaker notes:**
This is a fair concern. If someone gains read access to your home directory, they can steal the Globus tokens and submit functions to Lakeshore from anywhere. But this is the same security model used by SSH keys, AWS credentials, Kubernetes configs, and browser cookies — all stored as files protected by OS permissions. If an attacker can read your home directory, you have a bigger problem than Globus tokens. The key mitigation is that access tokens are short-lived (30-60 minutes) and refresh tokens can be revoked instantly through the Globus web interface, which kills all access without needing to change your password.


---

## SLIDE 18 — Multi-Model Architecture on Lakeshore

**Title:**
5 Models on Lakeshore

**Body:**

| Model | Size | Port | Use Case |
|-------|------|------|----------|
| Qwen 2.5 1.5B | 1.5B params | 8000 | General purpose (fast) |
| Qwen 2.5 Coder 1.5B | 1.5B params | 8001 | Code generation |
| DeepSeek R1 1.5B | 1.5B params | 8002 | Deep reasoning |
| QwQ 1.5B | 1.5B params | 8003 | Reasoning |
| Qwen 2.5 32B AWQ | 32B params | 8004 | High quality |

- Each model runs as a separate vLLM instance on its own MIG slice (39.5 GB VRAM)
- GPU node: ga-002 with NVIDIA A100 (Multi-Instance GPU partitions)
- STREAM dynamically routes to the correct port based on the selected model
- Context windows: 32K tokens (1.5B models), 8K tokens (32B — KV cache limits)

**Speaker notes:**
We run five different models simultaneously on Lakeshore, each on its own MIG partition of an A100 GPU. MIG stands for Multi-Instance GPU — NVIDIA's technology for splitting one physical GPU into isolated virtual GPUs. Each partition gets 39.5 GB of VRAM. The 32B model is quantized with AWQ (Activation-aware Weight Quantization) to fit in a single partition, which limits its context window to 8K tokens instead of 32K. When you select a model in STREAM's settings, it constructs the right URL — same host but different port.


---

## SLIDE 19 — Per-Model Health Checks

**Title:**
Per-Model Health Checks: Real Inference Tests

**Body:**

**The Problem:**
- Old health check only verified "are your Globus tokens valid?"
- Tokens could be valid but the vLLM instance crashed → green indicator, failed request

**The Solution:**
- STREAM sends a real 1-token inference test through the FULL Globus Compute path
- Test: `messages=[{"role": "user", "content": "hi"}], max_tokens=1`
- Traverses: SDK → AMQP → Globus Cloud → Lakeshore → vLLM on correct port → back
- If ANY part of the chain is broken for that model, the health check catches it

**Per-model cache keys:**
- `"lakeshore"` → just checks auth (fast but incomplete)
- `"lakeshore:lakeshore-qwen-32b"` → real inference test for THAT model

20-second timeout (generous: full round-trip takes ~5s, 32B cold start can add more)

**Speaker notes:**
This was an important improvement. Previously, if the 32B model's vLLM instance crashed but the 1.5B models were fine, STREAM would show Lakeshore as healthy — and then fail when you tried to use the 32B model. Now, when you select a model, the health check sends an actual inference request through the entire pipeline. It's like pinging the model with "hi" and asking for one token back. If it works, we know the whole chain is operational for that specific model.


---

## SLIDE 20 — Performance: Where Does Time Go?

**Title:**
Performance: Anatomy of a 5-Second Request

**Body:**

```
Total: ~5 seconds
│
├── Executor creation (AMQP connect)     ~0.3s  (first request only)
├── Task submission (serialize + send)   ~0.6s  (first request only)
│
├── AMQP routing (SDK → Broker → HPC)   ~0.3s
├── Endpoint scheduling                  ~0.2s
├── Function deserialization             ~0.1s
├── vLLM inference (GPU generation)      ~2-3s  ← The actual work
├── Result return path                   ~0.4s
└── SDK overhead                         ~0.2s
```

- ~2-3s is actual GPU work (irreducible)
- ~2-3s is Globus overhead (the price of firewall transparency)
- Subsequent requests: ~5.2s (reuse AMQP connection — save ~0.9s)

**Speaker notes:**
Let's break down where time goes. About half is actual GPU inference — that's irreducible, the model needs time to generate tokens. The other half is Globus overhead — serialization, AMQP routing through AWS, deserialization, and the return trip. This is the price we pay for not needing SSH keys or VPN. On subsequent requests, we save about a second by reusing the persistent AMQP connection instead of opening a new one each time.


---

## SLIDE 21 — The Streaming Challenge

**Title:**
The Streaming Challenge

**Body:**

**Local & Cloud tiers:** True token-by-token streaming
```
token₁ → token₂ → token₃ → ... → token_N     (text appears progressively)
```

**Lakeshore tier:** Globus Compute is batch-only (submit → wait → complete result)
```
[======= 5 second wait =======] [all tokens arrive at once]
```

**Our Solution: Simulated Streaming**
- When the complete result arrives, split it into 2-word chunks
- Yield each chunk with a 50ms delay
- Creates a "typing" effect at ~40 words/second
- User experience feels consistent across all tiers

**The Limitation:**
- User still waits ~5 seconds before seeing ANY text
- The "streaming" is fake — GPU already finished, we're replaying
- Can't cancel mid-generation (GPU already done)

**Speaker notes:**
This is an inherent limitation of the Function-as-a-Service model. Globus Compute submits a function, it runs to completion, and you get the result. There's no way to get partial results while the function is running. So we simulate streaming — when the complete response arrives, we split it into word groups and deliver them progressively with small delays. It looks like streaming to the user, but it's actually a replay. The real downside is the 5-second blank screen before anything appears.


---

## SLIDE 22 — Future: WebSocket Relay (True Streaming)

**Title:**
Next Step: WebSocket Relay — True Token Streaming

**Body:**

**The Insight:** Separate the control plane from the data plane

- **Control plane (Globus Compute):** Authentication, job submission, launching the function
- **Data plane (WebSocket relay):** Real-time token delivery

```
                     ┌─────────────────┐
                     │ WebSocket Relay  │
                     │ (public server)  │
                     └───┬─────────┬───┘
                  outbound│       │outbound
                         │       │
   Lakeshore GPU ────────┘       └──────── User's laptop
   tokens flow ─────→→→→→→→→→→→→→→→→→──→  displays tokens
   as generated                            progressively
```

- Globus still handles auth + job launch (the hard part)
- But tokens flow through a lightweight WebSocket relay (the fast part)
- Both sides connect OUTBOUND to the relay → works through firewalls
- Relay is ~50 lines of Python, stateless, negligible resources

**Speaker notes:**
The key insight is that Globus Compute is excellent at solving the authentication and firewall problem — but it doesn't have to carry every single token. We use Globus for what it's good at: getting through the firewall and launching the function on the GPU. Then the function opens a WebSocket to a relay server and streams tokens through that side channel. The user's STREAM app connects to the same relay. Both connections are outbound, so both work through firewalls. The relay itself is tiny — it just forwards bytes between connections.


---

## SLIDE 23 — WebSocket Relay: The Improvement

**Title:**
Before and After: WebSocket Relay

**Body:**

**Current (Globus batch):**
```
Time:  0s        1s        2s        3s        4s        5s
       ├─ Globus routing ─┤                              │
                           ├── GPU generates all tokens ──┤
                                                          ├─ All tokens
User:  [waiting...         waiting...          waiting... │ arrive at once]
                                                    First token: ~5s
```

**With WebSocket relay:**
```
Time:  0s        1s        2s        3s     3.1s   3.2s  ...  5s
       ├─ Globus routing ─┤                 │      │          │
                           ├── GPU streams ─┼──────┼──────────┤
                                  token₁ ───→      │          │
                                  token₂ ──────────→          │
                                  token_N ────────────────────→
User:  [waiting...         waiting  │ text appears progressively...]
                                First token: ~3s
```

| Metric | Current | With Relay |
|--------|---------|------------|
| Time to first token | ~5s | ~3s |
| True streaming | No (simulated) | Yes (real) |
| Cancel mid-generation | No | Yes |

**Speaker notes:**
The total generation time is similar — the GPU still takes 2-3 seconds to produce all tokens. But the user experience is dramatically different. Currently, you see nothing for 5 seconds, then everything appears. With the relay, text starts appearing at 3 seconds and keeps flowing progressively — just like ChatGPT. Research on perceived latency shows that users tolerate longer total waits when they see progressive feedback. A 5-second blank screen feels much slower than 3 seconds of waiting followed by 2 seconds of text appearing.


---

## SLIDE 24 — The Relay Firewall Solution

**Title:**
Why the Relay Works Through Firewalls

**Body:**

**The Problem:**
- Lakeshore: firewall blocks all inbound connections (except SSH)
- User's laptop: behind home NAT/router, also blocks inbound
- Neither side can accept connections from the other

**The Solution:**
- Both sides connect OUTBOUND to the relay (a public server)
- Outbound connections work through any firewall (that's how web browsing works)

```
Lakeshore  ──outbound──→  Relay server  ←──outbound──  User's laptop
(firewall allows         (public IP,    (NAT allows
 outbound)                ~$5/month)     outbound)
```

**Where to host the relay:**
1. UIC service VM on Lakeshore infrastructure (ideal — all data stays at UIC)
2. Behind ACER's existing reverse proxy (no new firewall rules needed)
3. External cloud VM ($5/month fallback)

**Security:** The relay is a pure message forwarder — it doesn't execute code, access files, or connect to any other service. It exposes far less than SSH (port 22), which is already open.

**Speaker notes:**
This is the same principle that makes video calls work. When you and a friend are both behind home routers, neither of you can accept incoming connections. But you can both connect outbound to a TURN server, and the server relays your video. Our relay does the same thing for AI tokens. The relay itself is incredibly lightweight — under 10MB of memory, near-zero CPU, no disk usage. It just forwards bytes. We need to work with ACER to host it, either on Lakeshore's infrastructure or as a small cloud VM.


---

## SLIDE 25 — The Big Picture: Two Deployment Modes

**Title:**
Two Deployment Modes, Same Core

**Body:**

**Server Mode (Docker):**
```
┌─────────────────────────────────────────────┐
│              Docker Network                  │
│  React UI → Middleware → LiteLLM → Proxy    │
│               ↕             ↕        ↕      │
│           PostgreSQL    Ollama   Globus →    │
└────────────────────────────────────┼─────────┘
                                     ↓
                              Lakeshore HPC
```
- 5 containers, production deployment
- PostgreSQL for chat history

**Desktop Mode (PyInstaller):**
```
┌──────────────────────────────┐
│    Single Process (.app)     │
│  FastAPI + React + Ollama    │
│  + Globus client (direct) →  │───→ Lakeshore HPC
│  SQLite for chat history     │
└──────────────────────────────┘
```
- One native app, double-click to run
- No Docker, no terminal, no setup

**Speaker notes:**
STREAM works in two modes. Server mode uses Docker with five microservices — good for deploying on a server that multiple users share. Desktop mode bundles everything into a single native application — double-click STREAM.app and you're running. Both modes use the same Globus Compute client code to reach Lakeshore. The main engineering challenge was making the desktop mode work — single process, no self-connection deadlocks, PyInstaller serialization compatibility.


---

## SLIDE 26 — Other STREAM Features (Brief)

**Title:**
Other STREAM Features

**Body:**

- **Complexity Judge:** Small LLM analyzes each query → routes to the right tier automatically
- **Automatic Tier Fallback:** If Lakeshore is down → Cloud takes over (transparent to user)
- **Cost Tracking:** Real-time cost display per message (exact for completed, estimated for stopped)
- **Context Window Management:** Automatic conversation trimming per model's token limits
- **Thinking Blocks:** Display reasoning chains from models like DeepSeek R1
- **Conversation Summarization:** Compress old messages to fit more context
- **Dark Mode:** Optimized for long reading sessions (reduced halation, Geist fonts)
- **~21,900 lines of code** (Python: ~13,900 | TypeScript: ~7,200)

**Speaker notes:**
I want to briefly mention some other features beyond the Lakeshore connection. STREAM has a complexity judge that uses a small LLM to classify each question before routing it. If a tier goes down, fallback is automatic and transparent. We track costs in real-time for cloud requests. Context windows are managed per model — the 32B model only has 8K tokens, so we trim conversation history automatically. The total codebase is about 22,000 lines across Python and TypeScript.


---

## SLIDE 27 — Summary: Why This Matters

**Title:**
Why This Matters

**Body:**

**What we built:**
- A system that routes AI queries to free university GPUs — transparently
- Works through firewalls without SSH, VPN, or any user setup
- Runs as a native desktop app (double-click) or Docker deployment
- 5 models on Lakeshore, with per-model health monitoring

**Technical challenges solved:**
- Globus Compute integration (AMQP-based FaaS through firewall)
- PyInstaller serialization incompatibility (exec()-based remote functions)
- Desktop self-connection deadlock (direct Python calls bypassing HTTP)
- Persistent AMQP connections (0.6s saved per request)
- Per-model health verification (real inference tests, not just auth checks)

**What's next:**
- WebSocket relay for true token streaming (~3s first token instead of ~5s)
- More Lakeshore models as demand grows
- PEARC paper submission documenting this architecture

**Speaker notes:**
To summarize — STREAM makes university GPU resources accessible to anyone with a UIC account, without requiring any HPC knowledge. The Lakeshore connection through Globus Compute is the hardest part of the architecture, and we solved several non-trivial technical challenges to make it work reliably across both Docker and desktop deployments. The next milestone is the WebSocket relay, which will bring the user experience to parity with commercial chat interfaces like ChatGPT.


---

## SLIDE 28 — Demo Time

**Title:**
Demo

**Body:**
(Keep this slide minimal — transition to the live demo)

- Live demo of STREAM desktop app
- Show: tier routing, Lakeshore inference, model selection, cost tracking

**Speaker notes:**
Now let's see it in action. I'll open the STREAM desktop app and show you how it routes queries to different tiers. Pay attention to the metadata below each response — it shows which tier handled the query, which model was used, how long it took, and what it cost.
