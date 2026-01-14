# STREAM Production Architecture
## SLURM Triage Engine for AI-Assisted Management

**Version:** 1.0.0
**Author:** Anas Nassar (nassar@uic.edu)
**Organization:** ACER Technology Solutions, University of Illinois Chicago
**Last Updated:** December 31, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Principles](#core-principles)
4. [Component Breakdown](#component-breakdown)
   - [Layer 1: User/Application Interface](#layer-1-userapplication-interface)
   - [Layer 2: Authentication](#layer-2-authentication)
   - [Layer 3: AI Middleware Hub](#layer-3-ai-middleware-hub)
   - [Layer 4a: Model Gateway](#layer-4a-model-gateway)
   - [Layer 4b: Model Backends](#layer-4b-model-backends)
   - [Layer 5: MCP Tool Gateway](#layer-5-mcp-tool-gateway)
   - [Layer 6: Observability](#layer-6-observability)
5. [Data Flow Examples](#data-flow-examples)
6. [Security Model](#security-model)
7. [Cost Optimization Strategy](#cost-optimization-strategy)
8. [Technology Stack](#technology-stack)
9. [Deployment Architecture](#deployment-architecture)
10. [Future Roadmap](#future-roadmap)

---

## Executive Summary

### What is STREAM?

STREAM is a **production-grade AI middleware platform** that provides UIC researchers, faculty, and students with intelligent access to multiple AI model backends through a unified interface. It implements intelligent routing, policy enforcement, cost optimization, and comprehensive observability.

### The Problem STREAM Solves

**Before STREAM:**
- Researchers pay $0.015/1k tokens for cloud AI (Claude/GPT)
- No cost control or budgeting
- No integration with campus resources
- Each application manages its own AI connections
- No audit trail for compliance (HIPAA/FERPA)
- No tooling integration (SLURM, LDAP, etc.)

**With STREAM:**
- Intelligent routing: Local (free) → Campus GPU ($0.0005/1k) → Cloud ($0.015/1k)
- **30x cost savings** by using campus resources
- Centralized policy enforcement and quotas
- Unified interface for all AI models
- Complete audit trail for compliance
- AI can use campus tools (SLURM, LDAP, library search)

### Key Metrics

- **Cost Reduction:** Up to 97% (free local) or 67% (campus vs cloud)
- **Supported Users:** Faculty, staff, graduate students, undergraduates (tiered access)
- **Model Backends:** 4 tiers (Local Ollama, Lakeshore vLLM, Azure OpenAI, AWS Bedrock)
- **Tool Integrations:** LDAP, Library/KB, SLURM, Research Data Catalog
- **Compliance:** HIPAA, FERPA, SOC 2 ready

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  USERS & APPLICATIONS                                           │
│  - LibreChat / OpenWebUI (web interfaces)                      │
│  - Custom apps (department tools)                               │
│  - IDE agents (VS Code, GitHub Copilot alternative)            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: AUTHENTICATION (Shibboleth OIDC)                      │
│  Input:  User credentials                                       │
│  Output: JWT with attributes (NetID, affiliation, department)   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: AI MIDDLEWARE HUB (Policy + Telemetry)                │
│  - Request validation & correlation ID assignment               │
│  - Policy enforcement (quotas, tier restrictions)               │
│  - Request routing decisions                                    │
│  - Telemetry collection                                         │
└───────────┬────────────────────────┬────────────────────────────┘
            │                        │
            ↓ (2a)                   ↓ (2b)
┌───────────────────────┐  ┌─────────────────────────────────────┐
│  MODEL GATEWAY        │  │  MCP TOOL GATEWAY                   │
│  (LiteLLM)            │  │  - Tool registry                    │
│  - OAuth2/OIDC        │  │  - ABAC enforcement                 │
│  - Routing policy     │  │  - Rate limiting                    │
│  - Cost tracking      │  │  - Secret management                │
│  - Quotas/budgets     │  └────────┬────────────────────────────┘
└─────────┬─────────────┘           │
          │                         │
          ↓                         ↓
┌─────────────────────────┐  ┌─────────────────────────────────────┐
│  MODEL BACKENDS         │  │  TOOL IMPLEMENTATIONS               │
│  1. Local (Ollama)      │  │  - Directory/LDAP APIs              │
│     FREE                │  │  - Library/KB search                │
│  2. Lakeshore (vLLM)    │  │  - Student/course metadata          │
│     $0.0005/1k tokens   │  │  - Research data catalog            │
│  3. Azure OpenAI        │  │  - HPC/SLURM actions                │
│     HIPAA/FERPA         │  └─────────────────────────────────────┘
│  4. AWS Bedrock         │
│     Model variety       │
└─────────────────────────┘
            │                         │
            └─────────┬───────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: OBSERVABILITY (Splunk HEC)                            │
│  All components emit structured events                          │
│  - Authentication events                                        │
│  - LLM request/response events                                  │
│  - Tool call events                                             │
│  - Policy enforcement events                                    │
│  - Error events                                                 │
│  - Performance metrics                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow with Correlation ID

Every request gets a unique correlation ID that flows through all layers:

```
User Request
    ↓
correlation-id: abc123-def456 (assigned)
    ↓
[14:30:00.000] Authentication     | correlation-id=abc123 | user=nassar | status=success
[14:30:00.100] Middleware        | correlation-id=abc123 | tier=auto | routing_decision=lakeshore
[14:30:00.150] Model Gateway     | correlation-id=abc123 | backend=lakeshore-vllm
[14:30:01.200] AI Decision       | correlation-id=abc123 | action=use_tool | tool=ldap_search
[14:30:01.250] Tool Gateway      | correlation-id=abc123 | tool=ldap_search | abac=allowed
[14:30:01.380] Tool Result       | correlation-id=abc123 | status=success | records=1
[14:30:02.500] LLM Response      | correlation-id=abc123 | tokens=45 | cost=$0.0000225
[14:30:02.520] Request Complete  | correlation-id=abc123 | total_duration=2520ms | status=success
    ↓
Response to User
```

This enables **complete traceability** - search Splunk for one ID and see the entire journey.

---

## Core Principles

### 1. **Intelligent Cost Optimization**

STREAM automatically routes queries to the most cost-effective backend that can handle them:

```python
def route_query(query, user):
    if simple_query(query):
        return "local-ollama"        # FREE
    elif medium_complexity(query):
        return "lakeshore-vllm"      # 30x cheaper than cloud
    elif requires_compliance(query):
        return "azure-openai"        # HIPAA/FERPA
    else:
        return "aws-bedrock"         # Best quality/variety
```

**Cost Comparison:**
```
Query: "Explain quantum computing in detail" (500 tokens)

Local (Ollama):       $0.00000    (FREE)
Lakeshore (vLLM):     $0.00025    (campus GPU)
Cloud (Claude):       $0.00750    (30x more expensive)
Azure (GPT-4):        $0.01500    (60x more expensive)
```

### 2. **Policy-Driven Access Control**

Every request is evaluated against multiple policies:

```python
policies = {
    "tier_restrictions": {
        "undergraduate": ["local", "lakeshore"],
        "graduate": ["local", "lakeshore", "cloud"],
        "faculty": ["local", "lakeshore", "cloud", "azure"]
    },
    "quotas": {
        "undergraduate": {"daily_requests": 100, "monthly_cost": 10.00},
        "graduate": {"daily_requests": 500, "monthly_cost": 50.00},
        "faculty": {"daily_requests": 1000, "monthly_cost": 200.00}
    },
    "compliance": {
        "PHI_data": "azure-openai",  # HIPAA-compliant only
        "student_records": "azure-openai"  # FERPA-compliant only
    }
}
```

### 3. **Comprehensive Observability**

Every action generates structured events sent to Splunk:

```json
{
  "timestamp": "2025-12-31T14:30:01Z",
  "correlation_id": "abc123-def456",
  "event_type": "llm_request",
  "user": {"netid": "nassar", "affiliation": "staff"},
  "request": {"tier": "lakeshore", "model": "llama-2-7b"},
  "cost": {"total": 0.00025, "tokens_in": 150, "tokens_out": 300},
  "performance": {"duration_ms": 2340},
  "status": "success"
}
```

This enables:
- Real-time dashboards
- Cost tracking and chargeback
- Security monitoring
- Compliance auditing
- Performance optimization

### 4. **Multi-Backend Resilience**

Automatic failover ensures high availability:

```
Request → Lakeshore vLLM
          ↓ (timeout)
          Fallback → AWS Bedrock
                     ↓ (success)
                     Response
```

Users never see failures - the system automatically tries alternative backends.

### 5. **Tool Integration via MCP**

AI models can use external tools through a secure gateway:

```
AI: "I need to find Professor Smith's email"
    ↓
Tool Gateway:
  - Checks user has permission to query LDAP
  - Checks rate limit not exceeded
  - Routes to LDAP server
  - Returns result safely
    ↓
AI: "Professor Smith's email is smith@uic.edu"
```

---

## Component Breakdown

### Layer 1: User/Application Interface

**Purpose:** Multiple entry points for different user types

**Supported Interfaces:**

1. **LibreChat** (Open-source ChatGPT clone)
   - Web-based chat interface
   - Supports markdown, code highlighting
   - Multi-model selection
   - Conversation history

2. **OpenWebUI** (Alternative web interface)
   - Different UX, same backend
   - Plugin ecosystem
   - Admin dashboard

3. **Custom Applications**
   - Department-specific tools
   - Research workflows
   - Automated pipelines
   - Example: Medical imaging analysis tool

4. **IDE Agents**
   - VS Code extension
   - JupyterLab integration
   - GitHub Copilot alternative using campus resources
   - Example: Code completion powered by Lakeshore

**Technical Implementation:**
All interfaces communicate via HTTP REST API to middleware:

```http
POST https://stream.uic.edu/v1/chat/completions
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Explain quantum computing"}
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

### Layer 2: Authentication

**Technology:** Shibboleth OIDC (OpenID Connect)

**Authentication Flow:**

```
1. User accesses STREAM interface
2. App redirects to UIC Shibboleth:
   https://shibboleth.uic.edu/idp/profile/oidc/authorize
3. User logs in with NetID credentials
4. Shibboleth validates credentials
5. Shibboleth generates JWT with user attributes
6. User redirected back with JWT
7. App includes JWT in all API requests
```

**JWT Structure:**

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "nassar",
    "iss": "https://shibboleth.uic.edu",
    "aud": "stream-production",
    "exp": 1735689600,
    "iat": 1735686000,
    "netid": "nassar",
    "email": "nassar@uic.edu",
    "eduPersonAffiliation": ["staff"],
    "department": "ACER",
    "eduPersonEntitlement": [
      "urn:mace:uic.edu:entitlement:gpu:access",
      "urn:mace:uic.edu:entitlement:slurm:submit"
    ]
  },
  "signature": "<cryptographic_signature>"
}
```

**Why OIDC vs Basic Auth:**

| Feature | Basic Auth | OIDC |
|---------|-----------|------|
| Passwords | App sees passwords | App never sees passwords |
| Single Sign-On | No | Yes |
| Token expiry | Manual | Automatic |
| Attributes | None | Rich (affiliation, department) |
| Federation | No | Yes (can work with other universities) |
| Compliance | Poor | Strong (NIST 800-63B) |

**Security Properties:**
- JWT signed with RS256 (asymmetric cryptography)
- Public key published at JWKS endpoint
- Tokens expire after 1 hour
- Refresh tokens for long-lived sessions
- Revocation possible via token blacklist

### Layer 3: AI Middleware Hub

**Purpose:** Central orchestration and policy enforcement

**Technology:** FastAPI (Python async framework)

**Core Responsibilities:**

1. **Request Validation**
   ```python
   async def validate_request(request):
       # Check JWT is valid
       user = await validate_jwt(request.headers["Authorization"])

       # Check quota not exceeded
       if user.cost_this_month >= user.monthly_quota:
           raise HTTPException(429, "Monthly quota exceeded")

       # Check rate limit
       if user.requests_this_hour >= 100:
           raise HTTPException(429, "Rate limit exceeded")

       return user
   ```

2. **Correlation ID Assignment**
   ```python
   import uuid

   correlation_id = str(uuid.uuid4())
   # Attach to all downstream requests and logs
   ```

3. **Tier Routing Decision**
   ```python
   def decide_tier(query, user_preference, user):
       # User explicitly chose tier
       if user_preference in ["local", "lakeshore", "cloud", "azure"]:
           return user_preference

       # Query complexity analysis
       word_count = len(query.split())

       if word_count < 50:
           return "local"  # Simple queries
       elif word_count < 200:
           return "lakeshore"  # Medium complexity
       else:
           return "cloud"  # Complex queries
   ```

4. **Policy Enforcement**
   ```python
   policies = {
       # Tier access by affiliation
       "tier_access": {
           "undergraduate": ["local", "lakeshore"],
           "graduate": ["local", "lakeshore", "cloud"],
           "faculty": ["all"]
       },

       # Special handling for sensitive data
       "data_sensitivity": {
           "PHI": "azure-openai",  # HIPAA
           "FERPA": "azure-openai",  # Student records
           "public": "any"
       },

       # Time-based policies
       "time_policies": {
           "off_peak": {  # 10 PM - 6 AM
               "hours": range(22, 24) + range(0, 6),
               "bonus_tier_access": ["cloud"]  # Allow students to use cloud off-peak
           }
       }
   }
   ```

5. **Telemetry Collection**
   ```python
   async def emit_event(event):
       await splunk.log({
           "timestamp": datetime.utcnow().isoformat(),
           "correlation_id": event["correlation_id"],
           "event_type": event["type"],
           "user": event["user"],
           "details": event["data"]
       })
   ```

**Middleware API Endpoints:**

```
POST   /v1/chat/completions          # Main LLM endpoint
POST   /v1/tools/{tool_name}          # Tool execution
GET    /v1/models                     # List available models
GET    /v1/user/quota                 # Check quota
GET    /v1/user/usage                 # Usage statistics
GET    /health                        # Health check
GET    /metrics                       # Prometheus metrics
```

### Layer 4a: Model Gateway

**Technology:** LiteLLM (with enterprise features built in middleware)

**Purpose:** Unified interface to multiple AI providers

**Why LiteLLM?**

Different providers have incompatible APIs:

```python
# OpenAI format
openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Anthropic format (completely different!)
anthropic.messages.create(
    model="claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello"}]
)

# Azure format (different again!)
azure_openai.Completion.create(
    deployment_id="gpt-4-deployment",
    prompt="Hello"
)
```

LiteLLM provides one unified format:

```python
litellm.completion(
    model="gpt-4",  # or "claude-3-sonnet" or "azure/gpt-4"
    messages=[{"role": "user", "content": "Hello"}]
)
# LiteLLM handles translation to provider-specific format
```

**Configuration Example:**

```yaml
# litellm_config.yaml
model_list:
  - model_name: local-llama
    litellm_params:
      model: ollama/llama3.2:3b
      api_base: http://localhost:11434

  - model_name: lakeshore-llama
    litellm_params:
      model: openai/meta-llama/Llama-2-7b-hf
      api_base: http://lakeshore.acer.uic.edu:8000/v1

  - model_name: azure-gpt4
    litellm_params:
      model: azure/gpt-4
      api_base: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      api_version: "2024-02-15-preview"

  - model_name: bedrock-claude
    litellm_params:
      model: bedrock/anthropic.claude-v2
      aws_region_name: us-east-1

router_settings:
  fallbacks:
    - lakeshore-llama: [azure-gpt4, bedrock-claude]
    - azure-gpt4: [bedrock-claude]
```

**Key Features:**

1. **Automatic Retries**
   ```
   Request → lakeshore-llama
             ↓ (timeout after 30s)
             Retry → azure-gpt4
                     ↓ (success)
                     Response
   ```

2. **Load Balancing**
   ```yaml
   - model_name: cloud-gpt4
     litellm_params:
       model: azure/gpt-4
       api_base:
         - https://uic-openai-1.azure.com
         - https://uic-openai-2.azure.com
         - https://uic-openai-3.azure.com
   ```

3. **Cost Tracking**
   ```python
   response = litellm.completion(...)

   # LiteLLM automatically calculates:
   print(response.usage.prompt_tokens)      # 150
   print(response.usage.completion_tokens)  # 300
   print(response.usage.total_cost)         # $0.00495
   ```

4. **Streaming Support**
   ```python
   response = litellm.completion(
       model="lakeshore-llama",
       messages=[...],
       stream=True  # Enable streaming
   )

   for chunk in response:
       print(chunk.choices[0].delta.content, end="")
   ```

### Layer 4b: Model Backends

**1. Local Ollama (Tier 1 - FREE)**

**What:** LLM inference server running on user's laptop

**When to use:**
- Simple queries (definitions, basic Q&A)
- Offline work
- Privacy-sensitive data (never leaves device)
- No budget available

**Models:**
```yaml
- llama3.2:1b   # 1.3 GB, very fast, basic quality
- llama3.2:3b   # 2 GB, fast, good quality (default)
- llama3.1:8b   # 4.7 GB, slower, excellent quality
```

**Performance:**
```
Hardware: MacBook Pro M2
Model: llama3.2:3b
Speed: ~30 tokens/second
Latency: 100-200ms first token
Quality: Good for simple tasks
```

**Cost:** $0.00 (uses laptop electricity ~$0.0001/hour)

**Limitations:**
- Can't handle complex reasoning
- Limited context window (8k tokens)
- Single user only
- Requires local resources

---

**2. Lakeshore vLLM (Tier 2 - CHEAP)**

**What:** High-performance LLM server on UIC's GPU cluster

**Hardware:**
```
Cluster: Lakeshore HPC
Nodes: ga-001, ga-002
GPUs: NVIDIA A100 80GB (MIG partitions)
Allocation: 40GB MIG slice (gpu:3g.40gb:1)
```

**When to use:**
- Medium complexity queries
- Research workloads
- Batch processing
- When local is too slow but cloud is too expensive

**Models:**
```yaml
- Llama-2-7b-hf        # Default, general purpose
- CodeLlama-7b-hf      # Code generation
- Mistral-7B-v0.1      # Alternative, good quality
```

**Performance:**
```
Hardware: A100 GPU (40GB slice)
Model: Llama-2-7b
Speed: ~100-150 tokens/second
Latency: 50-100ms first token
Quality: Excellent for 7B model
```

**Cost:** $0.0005 per 1000 tokens
```
Example:
  Query: 150 tokens
  Response: 300 tokens
  Total: 450 tokens
  Cost: 450 * $0.0005 / 1000 = $0.000225 (~$0.0002)
```

**Deployment:**
```bash
# vLLM running in Apptainer container on Lakeshore
apptainer exec --nv vllm-openai_v0.13.0.sif \
  vllm serve meta-llama/Llama-2-7b-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

**Access:**
- Internal: `http://ga-001:8000/v1`
- Via nginx: `http://lakeshore.acer.uic.edu:8000/v1`

---

**3. Azure OpenAI (Tier 3 - REGULATED)**

**What:** OpenAI models hosted on Microsoft Azure cloud

**When to use:**
- Sensitive data (PHI, PII, student records)
- Compliance requirements (HIPAA, FERPA)
- Enterprise SLA needed
- Data residency requirements (must stay in US)

**Models:**
```
- GPT-4 Turbo
- GPT-3.5 Turbo
- (Future: GPT-4o, o1)
```

**Compliance:**
```
✅ HIPAA (Health Insurance Portability and Accountability Act)
✅ FERPA (Family Educational Rights and Privacy Act)
✅ SOC 2 Type II
✅ ISO 27001
✅ FedRAMP (Federal Risk and Authorization Management Program)
```

**Cost:** Same as OpenAI
```
GPT-4 Turbo:
  Input:  $10 / 1M tokens
  Output: $30 / 1M tokens

GPT-3.5 Turbo:
  Input:  $0.50 / 1M tokens
  Output: $1.50 / 1M tokens
```

**Configuration:**
```python
azure_config = {
    "endpoint": "https://uic-openai.openai.azure.com",
    "deployment": "gpt-4-turbo",
    "api_version": "2024-02-15-preview",
    "api_key": os.getenv("AZURE_OPENAI_KEY")
}
```

**Data Handling:**
- Data encrypted in transit (TLS 1.3)
- Data encrypted at rest (AES-256)
- No data used for training
- Data retention: 30 days (configurable)
- Data residency: US East region

---

**4. AWS Bedrock (Tier 3 - VARIETY)**

**What:** Amazon's AI model marketplace

**When to use:**
- Need multiple models for comparison
- Already using AWS infrastructure
- Want to try latest open models
- Research requiring model variety

**Available Models:**
```
Anthropic:
  - Claude 3 Opus
  - Claude 3 Sonnet
  - Claude 3 Haiku

Meta:
  - Llama 2 70B
  - Llama 3 70B

Mistral:
  - Mistral 7B
  - Mixtral 8x7B

Cohere:
  - Command
  - Command Light

Amazon:
  - Titan Text
  - Titan Embeddings
```

**Cost:** Varies by model
```
Claude 3 Sonnet:
  Input:  $3 / 1M tokens
  Output: $15 / 1M tokens

Llama 2 70B:
  Input:  $1 / 1M tokens
  Output: $1 / 1M tokens
```

**Use Case Example:**
```python
# Compare multiple models for same query
models = ["claude-3-sonnet", "llama-2-70b", "mistral-8x7b"]
results = []

for model in models:
    response = bedrock.invoke(model, query)
    results.append({
        "model": model,
        "response": response,
        "quality_score": evaluate(response)
    })

# Return best response
return max(results, key=lambda x: x["quality_score"])
```

---

**Backend Comparison Table:**

| Backend | Cost/1k tokens | Speed | Quality | Use Case |
|---------|---------------|-------|---------|----------|
| Local Ollama | $0.00 | Medium | Good | Simple, offline, private |
| Lakeshore vLLM | $0.0005 | Fast | Excellent | Research, medium tasks |
| Azure OpenAI | $10-30 | Fast | Best | Sensitive data, compliance |
| AWS Bedrock | $1-15 | Fast | Varies | Model variety, research |

**Cost Optimization Example:**

```
Scenario: 1000 queries/day, 500 tokens each

All on Cloud (GPT-4):
  1000 * 500 * $0.03 / 1000 = $15/day = $450/month

Smart Routing (STREAM):
  - 600 queries → Local (simple)     = $0
  - 300 queries → Lakeshore (medium) = $0.075/day
  - 100 queries → Cloud (complex)    = $1.50/day

  Total: $1.575/day = $47.25/month

Savings: $402.75/month (89% reduction!)
```

### Layer 5: MCP Tool Gateway

**Purpose:** Secure interface for AI to use external tools

**Technology:** FastAPI + MCP (Model Context Protocol)

**Architecture:**

```
┌─────────────────────────────────────────┐
│  AI Model                               │
│  "I need Professor Smith's email"       │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  MCP Tool Gateway                       │
│                                         │
│  1. Parse request                       │
│     tool: ldap_search                   │
│     params: {query: "cn=*Smith*"}       │
│                                         │
│  2. Check user permissions (ABAC)       │
│     ✓ User is staff                     │
│     ✓ Staff can search LDAP             │
│                                         │
│  3. Check rate limit                    │
│     45/100 requests this hour ✓         │
│                                         │
│  4. Validate parameters                 │
│     ✓ Query is safe (no injection)      │
│     ✓ Attributes requested are allowed  │
│                                         │
│  5. Call tool server                    │
│     → LDAP Server                       │
│                                         │
│  6. Return result                       │
│     {mail: "smith@uic.edu"}            │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  LDAP Server                            │
│  cn=John Smith,ou=faculty,dc=uic,dc=edu│
│  mail: smith@uic.edu                    │
└─────────────────────────────────────────┘
```

**Tool Registry:**

```json
{
  "tools": [
    {
      "name": "ldap_search",
      "description": "Search UIC directory for people",
      "server": "http://ldap-tool:9001",
      "parameters": {
        "query": {
          "type": "string",
          "description": "LDAP filter (e.g., cn=*Smith*)",
          "required": true
        },
        "attributes": {
          "type": "array",
          "description": "Fields to return",
          "default": ["cn", "mail", "office"]
        }
      },
      "security": {
        "requires_auth": true,
        "allowed_affiliations": ["staff", "faculty"],
        "rate_limit": "100/hour",
        "sensitive_data": true
      }
    },

    {
      "name": "library_search",
      "description": "Search UIC library catalog",
      "server": "http://library-tool:9002",
      "parameters": {
        "query": "string",
        "type": "books|articles|all"
      },
      "security": {
        "requires_auth": false,
        "allowed_affiliations": ["all"],
        "rate_limit": "1000/hour",
        "sensitive_data": false
      }
    },

    {
      "name": "slurm_status",
      "description": "Check SLURM job status",
      "server": "http://slurm-tool:9003",
      "parameters": {
        "job_id": "integer (optional)",
        "user": "string (optional)"
      },
      "security": {
        "requires_auth": true,
        "allowed_affiliations": ["faculty", "staff", "graduate"],
        "rate_limit": "50/hour",
        "sensitive_data": false
      }
    }
  ]
}
```

**ABAC (Attribute-Based Access Control):**

```python
def check_tool_access(user, tool, params):
    """
    Evaluate multiple attributes to determine access
    """
    # Attribute 1: Affiliation
    if user.affiliation not in tool.allowed_affiliations:
        return AccessDenied("Your affiliation cannot use this tool")

    # Attribute 2: Time of day
    if tool.name == "slurm_submit" and (hour < 8 or hour > 18):
        return AccessDenied("Job submission only allowed 8 AM - 6 PM")

    # Attribute 3: Data sensitivity
    if tool.sensitive_data and not user.has_training("data_privacy"):
        return AccessDenied("Privacy training required")

    # Attribute 4: Department quota
    if user.department_tool_quota < 1:
        return AccessDenied("Department quota exceeded")

    # Attribute 5: Parameter validation
    if params.get("scope") == "all_users" and user.role != "admin":
        return AccessDenied("Only admins can query all users")

    return AccessAllowed()
```

**Rate Limiting:**

```python
# Redis-based rate limiter
import redis

class RateLimiter:
    def __init__(self):
        self.redis = redis.Redis()

    def check_limit(self, user_id, tool_name, limit="100/hour"):
        key = f"ratelimit:{user_id}:{tool_name}"

        # Get current count
        count = self.redis.get(key) or 0

        # Parse limit (e.g., "100/hour")
        max_count, period = parse_limit(limit)  # (100, 3600)

        if int(count) >= max_count:
            ttl = self.redis.ttl(key)
            raise RateLimitExceeded(f"Try again in {ttl} seconds")

        # Increment and set expiry
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, period)
        pipe.execute()

        return True
```

**Secret Management:**

```python
# Tools need credentials, but apps never see them

class ToolGateway:
    def __init__(self):
        # Load secrets from HashiCorp Vault or AWS Secrets Manager
        self.secrets = {
            "ldap": {
                "host": "ldap.uic.edu",
                "bind_dn": "cn=stream-app,ou=applications,dc=uic,dc=edu",
                "password": vault.get("ldap_password")
            },
            "slurm": {
                "api_key": vault.get("slurm_api_key")
            }
        }

    def call_ldap_search(self, params):
        # Gateway uses secret, caller never sees it
        conn = ldap.connect(
            self.secrets["ldap"]["host"],
            self.secrets["ldap"]["bind_dn"],
            self.secrets["ldap"]["password"]
        )

        result = conn.search(params["query"])
        return result
```

**Tool Implementation Example:**

```python
# tools/ldap_server.py
from fastapi import FastAPI, HTTPException
import ldap

app = FastAPI()

@app.post("/execute")
async def execute_ldap_search(request: dict):
    tool = request["tool"]
    params = request["parameters"]
    user = request["user"]
    correlation_id = request["correlation_id"]

    # Validate
    if tool != "ldap_search":
        raise HTTPException(400, "Unknown tool")

    # Connect to LDAP
    conn = ldap.initialize("ldap://ldap.uic.edu")
    conn.simple_bind_s(BIND_DN, BIND_PASSWORD)

    # Search
    results = conn.search_s(
        "ou=people,dc=uic,dc=edu",
        ldap.SCOPE_SUBTREE,
        params["query"],
        params.get("attributes", ["cn", "mail"])
    )

    # Log to Splunk
    await log_tool_call({
        "correlation_id": correlation_id,
        "tool": "ldap_search",
        "user": user,
        "query": params["query"],
        "results_count": len(results)
    })

    return {
        "status": "success",
        "results": format_results(results)
    }
```

### Layer 6: Observability

**Technology:** Splunk (with HEC - HTTP Event Collector)

**Why Splunk?**

Without centralized logging:
```
Error occurs → Check 10 different log files on 5 servers
Debug time: 2-4 hours
```

With Splunk:
```
Error occurs → Search correlation_id in Splunk
Debug time: 30 seconds
```

**Event Types:**

1. **Authentication Events**
```json
{
  "event_type": "authentication",
  "timestamp": "2025-12-31T14:30:00Z",
  "user": "nassar",
  "method": "shibboleth_oidc",
  "ip_address": "128.248.2.59",
  "status": "success"
}
```

2. **LLM Request Events**
```json
{
  "event_type": "llm_request",
  "correlation_id": "abc123",
  "user": "nassar",
  "tier": "lakeshore",
  "model": "llama-2-7b",
  "tokens": {"input": 150, "output": 300},
  "cost": 0.000225,
  "duration_ms": 2340,
  "status": "success"
}
```

3. **Tool Call Events**
```json
{
  "event_type": "tool_call",
  "correlation_id": "abc123",
  "tool": "ldap_search",
  "user": "nassar",
  "abac_result": "allowed",
  "rate_limit_remaining": 85,
  "duration_ms": 120,
  "status": "success"
}
```

4. **Policy Events**
```json
{
  "event_type": "policy_enforcement",
  "correlation_id": "abc123",
  "policy": "tier_restriction",
  "user_affiliation": "undergraduate",
  "requested_tier": "azure",
  "decision": "denied",
  "reason": "Azure requires faculty status"
}
```

5. **Error Events**
```json
{
  "event_type": "error",
  "correlation_id": "abc123",
  "component": "vllm-backend",
  "error_type": "ConnectionError",
  "error_message": "ga-001:8000 timeout",
  "recovery_action": "fallback_to_cloud",
  "severity": "warning"
}
```

**Splunk Dashboards:**

```
Dashboard 1: Real-Time Monitoring
- Requests per minute (timechart)
- Active users (gauge)
- Error rate (percentage)
- Tier distribution (pie chart)
- Cost today vs budget (progress bar)

Dashboard 2: Cost Analysis
- Cost by user (table)
- Cost by tier (bar chart)
- Cost trend (line chart)
- Potential savings (calculation)
- Top expensive queries (table)

Dashboard 3: Performance
- Average latency by tier (timechart)
- P95 latency (timechart)
- Throughput (requests/second)
- Backend health (status indicators)

Dashboard 4: Security & Compliance
- Authentication failures (table)
- Unusual access patterns (table)
- ABAC denials (table)
- Sensitive data access (audit log)
```

**Splunk Queries:**

```spl
# Find all events for one request
index="stream_production" correlation_id="abc123"
| table timestamp, component, event_type, message

# Cost by user this month
index="stream_production" event_type="llm_request"
| stats sum(cost) as total_cost by user
| sort - total_cost

# Identify queries that should use Lakeshore instead of cloud
index="stream_production" event_type="llm_request" tier="cloud"
| where tokens.output < 500
| stats count, sum(cost) as wasted_cost
| eval potential_savings = wasted_cost * 0.97

# Detect unusual behavior
index="stream_production" event_type="tool_call"
| stats count by user
| where count > 100
| eval severity = if(count > 200, "critical", "warning")
```

**Alerting:**

```yaml
alerts:
  - name: "High Error Rate"
    query: index="stream_production" status="error"
    condition: count > 100 in 5 minutes
    action:
      - slack: "#stream-alerts"
      - pagerduty: oncall-engineer

  - name: "Budget Warning"
    query: index="stream_production" event_type="quota_check"
    condition: percentage_used > 75
    action:
      - email: user
      - message: "You've used 75% of your monthly quota"

  - name: "Suspicious Activity"
    query: index="stream_production" event_type="authentication" status="failed"
    condition: count > 10 from same IP in 1 hour
    action:
      - security_team: notify
      - account: temporary_lock
```

---

## Data Flow Examples

### Example 1: Simple Query (Local Tier)

```
User: "What is Python?"

┌─────────────────────────────────────────────────────────────────┐
│  1. User Interface (Streamlit)                                  │
│     User types query                                            │
└───────────┬─────────────────────────────────────────────────────┘
            │ HTTPS POST /v1/chat/completions
            │ Authorization: Bearer <JWT>
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. Authentication                                              │
│     ✓ JWT valid                                                 │
│     ✓ User: nassar, Affiliation: staff                          │
│     correlation-id: abc123 assigned                             │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. Middleware Hub                                              │
│     Query: "What is Python?" (3 words)                          │
│     Routing: simple → local tier                                │
│     ✓ User quota: 47/200 remaining                              │
│     ✓ Rate limit: 12/100 this hour                              │
└───────────┬─────────────────────────────────────────────────────┘
            │ Route to local-ollama
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. Model Gateway (LiteLLM)                                     │
│     Backend: local-ollama                                       │
│     Model: llama3.2:3b                                          │
└───────────┬─────────────────────────────────────────────────────┘
            │ HTTP to localhost:11434
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. Ollama (Local)                                              │
│     Processing on laptop CPU/GPU                                │
│     Model: llama3.2:3b                                          │
│     Response: "Python is a high-level programming language..."  │
│     Tokens: 45                                                  │
│     Duration: 1.2 seconds                                       │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. Response Path (reverse)                                     │
│     Gateway → Middleware → User                                 │
│     Cost: $0.00 (FREE)                                          │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. Observability (Splunk)                                      │
│     Event logged:                                               │
│     - correlation_id: abc123                                    │
│     - tier: local                                               │
│     - cost: 0.00                                                │
│     - duration: 1200ms                                          │
│     - status: success                                           │
└─────────────────────────────────────────────────────────────────┘

Total time: 1.2 seconds
Total cost: $0.00
```

### Example 2: Tool-Using Query (Lakeshore + LDAP)

```
User: "What's Professor Smith's email?"

┌─────────────────────────────────────────────────────────────────┐
│  1-3. Auth + Middleware (same as Example 1)                     │
│     correlation-id: def456                                      │
│     Routing: medium complexity → lakeshore                      │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. Model Gateway → Lakeshore vLLM                              │
│     Backend: lakeshore-llama                                    │
│     Model: Llama-2-7b-hf                                        │
└───────────┬─────────────────────────────────────────────────────┘
            │ HTTP to lakeshore.uic.edu:8000
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. vLLM on Lakeshore                                           │
│     Model analyzes query                                        │
│     Decision: "I need to search LDAP for this person"           │
│     Tool request: ldap_search(query="cn=*Smith*")               │
└───────────┬─────────────────────────────────────────────────────┘
            │ Tool call to middleware
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. MCP Tool Gateway                                            │
│     Tool: ldap_search                                           │
│     ABAC Check:                                                 │
│       ✓ User is staff (allowed)                                 │
│       ✓ Rate limit: 12/100 (ok)                                 │
│       ✓ Parameters valid                                        │
└───────────┬─────────────────────────────────────────────────────┘
            │ Call LDAP server
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. LDAP Server                                                 │
│     Search: cn=*Smith*                                          │
│     Found: John Smith                                           │
│     Result: {cn: "John Smith", mail: "smith@uic.edu"}          │
└───────────┬─────────────────────────────────────────────────────┘
            │ Return to AI
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  8. vLLM Final Response                                         │
│     Tool result received                                        │
│     Generate: "Professor Smith's email is smith@uic.edu"        │
│     Tokens: 20                                                  │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  9. Splunk Logging                                              │
│     Two events:                                                 │
│     - llm_request (correlation_id: def456)                      │
│     - tool_call (correlation_id: def456, tool: ldap_search)     │
└─────────────────────────────────────────────────────────────────┘

Total time: 2.5 seconds
Total cost: $0.00001 (Lakeshore)
Tools used: 1 (ldap_search)
```

### Example 3: Failover Scenario

```
User: "Analyze this medical imaging data..."

┌─────────────────────────────────────────────────────────────────┐
│  1-3. Auth + Middleware                                         │
│     correlation-id: ghi789                                      │
│     Detection: Medical data (PHI) → Requires Azure              │
│     Routing: azure-gpt4                                         │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. Model Gateway → Azure OpenAI                                │
│     Attempt 1: azure-gpt4                                       │
│     Error: 429 Rate Limit Exceeded                              │
│     [Splunk: Error event logged]                                │
└───────────┬─────────────────────────────────────────────────────┘
            │ Automatic fallback
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. Fallback: AWS Bedrock Claude                                │
│     Model: bedrock/claude-3-sonnet                              │
│     Status: Success ✓                                           │
│     [Splunk: Fallback event logged]                             │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. Response Returned                                           │
│     User sees: Analysis of imaging data                         │
│     User does NOT see: Fallback occurred (transparent)          │
│     Cost: Bedrock pricing instead of Azure                      │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. Splunk Events                                               │
│     Search correlation_id="ghi789" shows:                       │
│     - 14:30:01 | azure-gpt4 | attempt | 429 error               │
│     - 14:30:02 | fallback   | bedrock-claude | initiated        │
│     - 14:30:05 | bedrock-claude | success | 200                 │
└─────────────────────────────────────────────────────────────────┘

Total time: 5 seconds (includes retry)
Total cost: $0.00495 (Bedrock)
Fallbacks: 1
User experience: Seamless ✓
```

---

## Security Model

### Defense in Depth

STREAM implements multiple security layers:

```
Layer 1: Network
  ├─ HTTPS/TLS 1.3 for all connections
  ├─ UIC firewall (campus network only)
  └─ DDoS protection

Layer 2: Authentication
  ├─ Shibboleth OIDC (SSO)
  ├─ JWT with RS256 signatures
  ├─ Token expiration (1 hour)
  └─ Refresh token rotation

Layer 3: Authorization
  ├─ ABAC (Attribute-Based Access Control)
  ├─ Policy enforcement engine
  ├─ Quota management
  └─ Rate limiting

Layer 4: Application
  ├─ Input validation
  ├─ Output sanitization
  ├─ SQL injection prevention
  ├─ LDAP injection prevention
  └─ Secret management (Vault)

Layer 5: Data
  ├─ Encryption at rest (AES-256)
  ├─ Encryption in transit (TLS 1.3)
  ├─ Data retention policies
  └─ Right to deletion (GDPR)

Layer 6: Monitoring
  ├─ Real-time threat detection
  ├─ Anomaly detection
  ├─ Audit logging (Splunk)
  └─ Incident response
```

### Threat Model

**Threats & Mitigations:**

1. **Unauthorized Access**
   - Threat: Non-UIC users accessing system
   - Mitigation: Shibboleth SSO, JWT validation, firewall

2. **Privilege Escalation**
   - Threat: Undergraduate accessing faculty-only tiers
   - Mitigation: ABAC policy enforcement, JWT claims verification

3. **Data Exfiltration**
   - Threat: Bulk downloading LDAP directory
   - Mitigation: Rate limiting, ABAC, audit logging

4. **Cost Abuse**
   - Threat: User running expensive queries infinitely
   - Mitigation: Per-user quotas, cost limits, alerts

5. **Prompt Injection**
   - Threat: Malicious prompts trying to bypass restrictions
   - Mitigation: Input validation, output sanitization, tool ABAC

6. **Side-Channel Attacks**
   - Threat: Timing attacks to infer sensitive data
   - Mitigation: Constant-time operations, response randomization

### Compliance

**HIPAA Compliance (Health Data):**
```
✓ Data encryption (TLS 1.3, AES-256)
✓ Access controls (ABAC)
✓ Audit trails (Splunk)
✓ Data retention policies
✓ Business Associate Agreement with Azure
✓ Regular security assessments
```

**FERPA Compliance (Student Records):**
```
✓ Authorization before access
✓ Minimum necessary disclosure
✓ Audit logging of all access
✓ Secure storage and transmission
✓ Annual security training required
```

**SOC 2 Type II:**
```
✓ Security controls documented
✓ Availability monitoring
✓ Confidentiality measures
✓ Processing integrity
✓ Privacy safeguards
```

---

## Cost Optimization Strategy

### Intelligent Routing Algorithm

```python
def route_request(query, user, preferences):
    """
    Multi-factor routing decision
    """
    # Factor 1: Explicit user preference
    if preferences.tier != "auto":
        return preferences.tier

    # Factor 2: Data sensitivity
    if contains_phi(query) or contains_ferpa(query):
        return "azure"  # Compliance required

    # Factor 3: Query complexity
    complexity = analyze_complexity(query)

    if complexity == "simple":
        return "local"  # Free tier

    # Factor 4: User budget remaining
    if user.budget_remaining < 5.00:
        return "lakeshore"  # Conserve budget

    # Factor 5: Query length
    if len(query.split()) > 200:
        return "cloud"  # Long queries need quality

    # Factor 6: Time of day
    if is_off_peak_hours():
        return "lakeshore"  # Campus GPU available

    # Factor 7: Historical quality
    if user.satisfaction_score[query_type] < 0.7:
        return "cloud"  # User prefers quality for this type

    # Default: Lakeshore (best balance)
    return "lakeshore"
```

### Cost Analysis

**Monthly Cost Projection:**

```python
# Scenario: 1000 users, average 10 queries/day

# All cloud (baseline):
users = 1000
queries_per_user_per_day = 10
days = 30
avg_tokens = 500
cloud_cost_per_token = 0.000015

total_cloud = users * queries_per_user_per_day * days * avg_tokens * cloud_cost_per_token
# = 1000 * 10 * 30 * 500 * 0.000015
# = $2,250/month

# With STREAM intelligent routing:
# 60% local (free)
# 30% lakeshore ($0.0005/1k)
# 10% cloud ($0.015/1k)

local_cost = 0
lakeshore_cost = 0.3 * total_cloud * (0.0005 / 0.015)
cloud_cost = 0.1 * total_cloud

total_stream = local_cost + lakeshore_cost + cloud_cost
# = 0 + (0.3 * 2250 * 0.033) + (0.1 * 2250)
# = 0 + $22.28 + $225
# = $247.28/month

# Savings
savings = total_cloud - total_stream
# = $2,250 - $247.28
# = $2,002.72/month (89% reduction!)
```

### Cost Monitoring

**Real-time cost tracking:**

```python
class CostTracker:
    def track_request(self, user, tier, tokens_in, tokens_out):
        cost = calculate_cost(tier, tokens_in, tokens_out)

        # Update user stats
        user.cost_today += cost
        user.cost_this_month += cost
        user.requests_today += 1

        # Check thresholds
        if user.cost_this_month > user.monthly_quota * 0.75:
            send_warning_email(user, "75% quota used")

        if user.cost_this_month > user.monthly_quota * 0.90:
            send_alert_email(user, "90% quota used")

        if user.cost_this_month >= user.monthly_quota:
            suspend_access(user, "Monthly quota exceeded")

        # Department-level tracking
        dept = get_department(user)
        dept.cost_this_month += cost

        # Log to Splunk
        log_cost_event({
            "user": user.netid,
            "department": dept.name,
            "tier": tier,
            "cost": cost,
            "quota_remaining": user.monthly_quota - user.cost_this_month
        })
```

---

## Technology Stack

### Programming Languages
- **Python 3.11+** (Middleware, tools, backends)
- **TypeScript/React** (Frontend - future)
- **Bash** (Deployment scripts)

### Frameworks
- **FastAPI** (Middleware Hub, Tool Gateway)
- **LiteLLM** (Model Gateway)
- **Streamlit** (Initial UI)

### AI/ML
- **Ollama** (Local inference)
- **vLLM** (Lakeshore inference)
- **OpenAI SDK** (Cloud APIs)
- **Anthropic SDK** (Claude)
- **AWS Boto3** (Bedrock)

### Authentication
- **python-jose** (JWT handling)
- **httpx** (HTTP client for OIDC)

### Databases
- **PostgreSQL** (User data, quotas, history)
- **Redis** (Rate limiting, caching)

### Observability
- **Splunk** (Logging, monitoring, alerting)
- **Prometheus** (Metrics - future)
- **Grafana** (Dashboards - future)

### Infrastructure
- **Docker** (Containerization)
- **Nginx** (Reverse proxy)
- **SLURM** (HPC job scheduling)
- **Apptainer/Singularity** (HPC containers)

### Development Tools
- **uv** (Python package manager)
- **Git** (Version control)
- **VS Code** (IDE)
- **pytest** (Testing)

---

## Deployment Architecture

### Development Environment
```
Developer Laptop
├─ Ollama (local models)
├─ Docker Compose
│  ├─ LiteLLM Gateway
│  ├─ PostgreSQL
│  └─ Redis
├─ Streamlit UI (dev server)
└─ Middleware (uvicorn --reload)
```

### Staging Environment
```
UIC Staging Server
├─ Nginx (reverse proxy)
├─ Docker Swarm
│  ├─ Middleware (3 replicas)
│  ├─ LiteLLM (2 replicas)
│  ├─ PostgreSQL (primary + replica)
│  └─ Redis (cluster)
├─ Mock Shibboleth (test auth)
└─ Splunk (test index)
```

### Production Environment
```
UIC Production Infrastructure

┌─────────────────────────────────────────┐
│  Load Balancer (nginx)                  │
│  stream.uic.edu                         │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    ↓                     ↓
┌─────────────┐    ┌─────────────┐
│ Middleware  │    │ Middleware  │
│ Replica 1   │    │ Replica 2   │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                ↓
┌─────────────────────────────────────────┐
│  LiteLLM Gateway Cluster                │
│  (3 replicas, auto-scaling)             │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼───────────────┐
    ↓          ↓               ↓
┌────────┐ ┌───────────┐ ┌──────────┐
│ Local  │ │ Lakeshore │ │  Cloud   │
│ Ollama │ │ vLLM      │ │ Backends │
└────────┘ └───────────┘ └──────────┘

Data Layer:
├─ PostgreSQL (primary + 2 replicas)
├─ Redis (3-node cluster)
└─ Splunk (enterprise deployment)

Security:
├─ UIC Shibboleth (SSO)
├─ HashiCorp Vault (secrets)
└─ Firewall (campus network only)
```

### Lakeshore Integration
```
Lakeshore HPC Cluster

Login Node (lakeshore.acer.uic.edu)
├─ Nginx Reverse Proxy :8000
│  └─ Routes to active GPU node
│
GPU Nodes (ga-001, ga-002)
├─ SLURM managed
├─ vLLM container (Apptainer)
│  ├─ Model: Llama-2-7b-hf
│  ├─ Port: 8000
│  └─ API: OpenAI-compatible
└─ Health monitoring
```

---

## Future Roadmap

### Phase 1: MVP (Completed)
- ✅ Basic Streamlit UI
- ✅ 3-tier routing (Local, Lakeshore, Cloud)
- ✅ LiteLLM integration
- ✅ Ollama local deployment
- ✅ vLLM on Lakeshore

### Phase 2: Authentication & Policy (Week 1-2)
- [ ] Shibboleth OIDC integration
- [ ] JWT validation
- [ ] Per-user quotas
- [ ] Policy enforcement engine
- [ ] Basic cost tracking

### Phase 3: Observability (Week 2-3)
- [ ] Splunk HEC integration
- [ ] Structured event logging
- [ ] Real-time dashboards
- [ ] Alerting rules
- [ ] Cost analytics

### Phase 4: MCP Tools (Week 3-4)
- [ ] Tool registry
- [ ] ABAC enforcement
- [ ] Rate limiting (Redis)
- [ ] LDAP tool
- [ ] Library search tool
- [ ] Secret management (Vault)

### Phase 5: Multi-Backend (Week 4-5)
- [ ] Azure OpenAI integration
- [ ] AWS Bedrock integration
- [ ] Compliance tagging
- [ ] Automatic failover

### Phase 6: Production Hardening (Week 5-6)
- [ ] Load testing (1000+ concurrent users)
- [ ] Security audit
- [ ] Disaster recovery plan
- [ ] Backup strategy
- [ ] Documentation

### Phase 7: Advanced Features (Month 2-3)
- [ ] SLURM tools (job submission, monitoring)
- [ ] Research data catalog integration
- [ ] Multi-model comparison
- [ ] Fine-tuning support
- [ ] RAG (Retrieval-Augmented Generation)

### Phase 8: Scale & Optimization (Month 3-6)
- [ ] Kubernetes deployment
- [ ] Auto-scaling
- [ ] Multi-region support
- [ ] Advanced caching
- [ ] Model serving optimization

### Phase 9: Desktop Application (Month 4-6)
- [ ] Tauri/Electron app
- [ ] Embedded SSH tunneling
- [ ] System tray integration
- [ ] Auto-updates
- [ ] Offline mode

### Phase 10: Enterprise Features (Month 6-12)
- [ ] Department-level analytics
- [ ] Chargeback automation
- [ ] Custom model deployment
- [ ] API rate limiting tiers
- [ ] White-label support

---

## Development Guidelines

### Code Organization
```
STREAM/
├── backend/
│   ├── middleware/          # Middleware Hub
│   │   ├── app.py
│   │   ├── auth.py
│   │   ├── policy.py
│   │   └── telemetry.py
│   ├── tools/               # MCP Tool implementations
│   │   ├── ldap_server.py
│   │   ├── library_server.py
│   │   └── slurm_server.py
│   └── config.py            # Configuration
├── gateway/
│   ├── litellm_config.yaml
│   └── docker-compose.yml
├── frontend/
│   └── streamlit_app.py
├── docs/
│   ├── ARCHITECTURE.md      # This file
│   ├── API.md
│   └── DEPLOYMENT.md
└── tests/
    ├── test_auth.py
    ├── test_routing.py
    └── test_tools.py
```

### Coding Standards
- **Type hints** for all functions
- **Docstrings** (Google style)
- **Error handling** with proper exceptions
- **Logging** at appropriate levels
- **Testing** with >80% coverage

### Git Workflow
```bash
main           # Production-ready code
├── develop    # Integration branch
    ├── feature/auth-oidc
    ├── feature/mcp-tools
    └── feature/splunk-logging
```

---

## References

### External Documentation
- [LiteLLM Docs](https://docs.litellm.ai/)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [vLLM Docs](https://docs.vllm.ai/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Splunk HEC](https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector)

### UIC Resources
- UIC Shibboleth: `https://shibboleth.uic.edu`
- Lakeshore HPC: `lakeshore.acer.uic.edu`
- ACER Documentation: `https://acer.uic.edu/docs`

### Contact
- **Developer:** Anas Nassar (nassar@uic.edu)
- **Organization:** ACER Technology Solutions
- **Department:** Advanced Cyberinfrastructure for Education and Research

---

**Document Version:** 1.0.0
**Last Updated:** December 31, 2025
**Next Review:** February 1, 2026
