# STREAM

**S**mart **T**iered **R**outing **E**ngine for **A**I **M**odels

Intelligent middleware that routes AI requests across local, cloud, and campus LLMs based on complexity, cost, and availability.

## ✨ Features

- 🎯 **Smart Routing**: Automatically routes requests to the best AI tier (LOCAL → LAKESHORE → CLOUD)
- 💰 **Cost Optimization**: Minimizes API costs by preferring local models when possible
- 🔄 **Automatic Fallback**: Seamlessly falls back to cloud providers when local/campus services fail
- 📊 **Usage Tracking**: Tracks costs, tokens, and performance across all tiers
- 🚀 **Easy Setup**: One command to install and start everything

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- At least 5GB free disk space (for AI models)
- Internet connection (for downloading models and cloud API access)

### 1. Clone and Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd STREAM

# Create .env file with your API keys
cp .env.example .env
nano .env  # Add your ANTHROPIC_API_KEY and other keys
```

**Required environment variables:**
- `ANTHROPIC_API_KEY` - For Claude (cloud tier)
- `OPENAI_API_KEY` - Optional, for GPT models
- Other keys as needed (see `.env.example`)

### 2. Run Setup Script

**That's it! One command does everything:**

```bash
bash scripts/setup-stream.sh
```

This script will:
1. ✅ Check prerequisites (Docker, docker-compose)
2. 🧹 Clean up any existing services
3. 📥 Download required AI models (~3GB, takes 5-10 minutes first time)
4. 🚀 Start all services (Ollama, PostgreSQL, LiteLLM, Middleware, Frontend)
5. 🏥 Wait for services to be healthy

### 3. Access STREAM

Once setup completes, open your browser:

**🌐 Main Interface:** http://localhost:8501

---

## 🎓 Optional: Lakeshore Campus Access

If you have access to UIC's Lakeshore cluster, you can enable the campus tier:

### 1. Connect to Lakeshore

```bash
ssh <YourNetID>@lakeshore.acer.uic.edu
```

### 2. Submit vLLM Job

```bash
# On Lakeshore
cd /path/to/your/workspace
sbatch scripts/vllm-ga001.sh
```

### 3. Set Up Port Forwarding

Open a **new terminal** on your local machine:

```bash
ssh -L 8000:ga-001:8000 <YourNetID>@lakeshore.acer.uic.edu -N
```

**Keep this terminal open** while using STREAM. The Lakeshore tier will now be available!

---

## 📊 Usage

### Chat Interface

1. Open http://localhost:8501
2. Type your question in the chat input
3. STREAM automatically:
   - Analyzes question complexity
   - Routes to best available tier
   - Shows which model was used
   - Tracks cost and tokens

### Tier Selection

STREAM intelligently routes based on:

| Tier | Models | When Used | Cost |
|------|--------|-----------|------|
| 🏠 **LOCAL** | llama3.2:1b/3b | Simple questions | FREE |
| 🏫 **LAKESHORE** | llama-3.1-70b | Medium questions | FREE |
| ☁️ **CLOUD** | Claude Sonnet 4 | Complex questions | $3/$15 per 1M tokens |

### Manual Tier Override

In the sidebar, you can:
- Force a specific tier
- View cost estimates
- See tier availability
- Check model status

---

## 🛠️ Management Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f middleware
docker-compose logs -f frontend
```

### Check Service Status

```bash
docker-compose ps
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart middleware
```

### Stop STREAM

```bash
docker-compose down
```

### Restart STREAM

```bash
docker-compose up -d
```

### Update Models

```bash
# Pull latest version of a model
docker exec -it stream-ollama ollama pull llama3.2:3b

# List installed models
docker exec -it stream-ollama ollama list
```

---

## 🐛 Troubleshooting

### "Model not found" errors

```bash
# Re-run model download
docker exec -it stream-ollama ollama pull llama3.2:1b
docker exec -it stream-ollama ollama pull llama3.2:3b
```

### Services won't start

```bash
# Check what's wrong
docker-compose logs

# Clean restart
docker-compose down
docker-compose up -d --build
```

### Port already in use

```bash
# Find what's using the port
lsof -i :8501  # or :5000, :4000

# Kill the process
kill -9 <PID>

# Or change ports in .env:
FRONTEND_PORT=8502
MIDDLEWARE_PORT=5001
```

### Out of disk space

```bash
# Clean up Docker
docker system prune -a --volumes

# Then re-run setup
bash scripts/setup-stream.sh
```

### Reset everything

```bash
# Nuclear option: delete all data and start fresh
docker-compose down -v  # -v removes volumes
bash scripts/setup-stream.sh
```

---

## 📈 Monitoring

### Health Checks

```bash
# Middleware health
curl http://localhost:5000/health

# Detailed health (shows all tiers)
curl http://localhost:5000/health/detailed
```

### Cost Tracking

View real-time costs in:
- Frontend sidebar (per-session)
- PostgreSQL database (historical)
- Middleware logs

---

## 🔧 Development

### Run Services Individually

```bash
# Start only infrastructure
docker-compose up -d ollama postgres litellm

# Run middleware locally (for development)
cd stream
python -m stream.middleware.app

# Run frontend locally
cd frontend
streamlit run streamlit_app.py
```

### Hot Reload

- **Middleware**: Use `RELOAD=true` in `.env`
- **Frontend**: Streamlit auto-reloads on file changes

---

## 📝 Configuration

Key settings in `.env`:

```bash
# Model Selection
LOCAL_MODEL=llama3.2:3b
LAKESHORE_MODEL=llama-3.1-70b
CLOUD_MODEL=claude-sonnet-4

# Routing Thresholds
SIMPLE_THRESHOLD=5      # LOCAL tier
MEDIUM_THRESHOLD=15     # LAKESHORE tier
# Above 15 → CLOUD tier

# Ports
FRONTEND_PORT=8501
MIDDLEWARE_PORT=5000
LITELLM_PORT=4000

# API Keys
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

[Your License Here]

---

## 🆘 Support

- **Issues**: Open a GitHub issue
- **Email**: [your-email]
- **Documentation**: [link-to-docs]

---

## 🎉 Acknowledgments

- Built for researchers at University of Illinois Chicago
- Uses Ollama for local inference
- Powered by LiteLLM for unified API access
