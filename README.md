# STREAM

STREAM = Smart Tiered Routing Engine for AI Models (Middleware that smartly routes AI requests across local, cloud, and campus LLMs)

Smart - Intelligent routing decisions
Tiered - 3 tiers (Local, Cloud, Campus)
Routing - Routes requests to best backend
Engine - Automated system
AI - Artificial Intelligence
Models - LLMs (GPT, Claude, Llama, etc.)


## Quick Start

### 1. Fill in API Keys in `.env` file in the root directory. Then run:

```bash
chmod +x scripts/sync-env.sh (if not already executable)
bash scripts/sync-env.sh
```

### 2. If you want to connect to Lakeshore as well, do port forwarding first:
Connect to Lakeshore for Campus Tier access. Then submit the job script called vllm-ga001.sh available in the scripts directory.

### 3. Then, do port forwarding first:

Open a terminal and run:
```bash
ssh -L 8000:ga-001:8000 <YourNetID>@lakeshore.acer.uic.edu -N
```
The terminal will just hang. This is expected. Leave it open while using STREAM.

### 4. Start Gateway Services
```bash
cd stream/gateway
docker-compose up -d
```

### 5. Start Middleware
```bash
stream-middleware
```

### 6. Start Frontend
```bash
cd frontend
streamlit run streamlit_app.py
```
