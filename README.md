# STREAM - Complete Beginner's Guide

**S**mart **T**iered **R**outing **E**ngine for **A**I **M**odels

A tool that helps UIC researchers use AI for free (or cheap) by automatically choosing between free local AI, free campus AI, and paid cloud AI based on your question's complexity or user manual preference.

---

## 📖 Table of Contents

1. [What is STREAM?](#what-is-stream)
2. [What You'll Need](#what-youll-need)
3. [Part 1: Local Setup (Your Computer)](#part-1-local-setup-your-computer)
4. [Part 2: Cloud Setup (Optional - Best Quality AI)](#part-2-cloud-setup-optional---best-quality-ai)
5. [Part 3: Lakeshore Setup (Optional - Free Campus AI)](#part-3-lakeshore-setup-optional---free-campus-ai)
6. [Using STREAM](#using-stream)
7. [Troubleshooting](#troubleshooting)
8. [Getting Help](#getting-help)

---

## What is STREAM?

STREAM is like having three AI assistants:

1. **🏠 Local AI** (Free, Fast) - Runs on your computer, answers simple questions
2. **🏫 Lakeshore AI** (Free, Powerful) - UIC's campus supercomputer, answers medium questions
3. **☁️ Cloud AI** (Paid, Best) - Professional AI like Claude, answers complex questions

**STREAM automatically picks the best one for each question** when in Auto mode, saving you money while getting great answers!

### Why Use STREAM?

- **Save Money**: Uses free AI when possible, only uses paid AI when necessary
- **Always Available**: Falls back to cloud AI if campus is down
- **No Thinking Required**: Automatically routes based on complexity (when in Auto mode)
- **Track Costs**: See exactly how much you're spending

---

## What You'll Need

### Required (Everyone)
- [ ] A computer (Mac, Linux, or Windows)
- [ ] 10 GB of free disk space
- [ ] Internet connection
- [ ] About 60 minutes for first-time setup (sometimes, setup in lakeshore alone can take ~60 mins)

### Optional (For Free Campus AI)
- [ ] UIC NetID and access to Lakeshore cluster
- [ ] VPN if working from home

### Optional (For Best Quality AI)
- [ ] Claude API key (for complex questions)
- [ ] OpenAI API key (if you want GPT models)

**Don't worry if you don't know what "API keys" are - we'll explain everything!**

---

## Part 1: Local Setup (Your Computer)

This part sets up STREAM on your computer with free local AI. Takes about 20-30 minutes.

### Step 1: Install Docker Desktop

**What is Docker?** Docker is like a "virtual computer" that runs STREAM and all its components in an isolated environment. This means STREAM won't mess with your computer's settings. Think of it as a self-contained app that brings everything it needs with it.

**Why we need it:** STREAM has many components (database, AI models, web server) that need to work together. Docker makes sure they all work perfectly without conflicting with your computer's software.

**Installation:**

#### For Mac:
1. Go to https://www.docker.com/products/docker-desktop
2. Click "Download for Mac"
3. Choose your Mac type:
   - **Apple Silicon (M1/M2/M3)**: Download "Mac with Apple chip"
   - **Intel Mac**: Download "Mac with Intel chip"

   *Don't know which you have?* Click Apple menu () → "About This Mac" → Look at "Chip" or "Processor"

4. Open the downloaded file and drag Docker to Applications
5. Open Docker from Applications
6. Click "Accept" on the agreement
7. **Wait for Docker to start** - you'll see a Docker icon in your menu bar (top right)
8. Docker might ask for your password - this is normal

**Test it worked:**
Open Terminal (press Cmd+Space, type "terminal", press Enter) and run:
```bash
docker --version
```

You should see something like: `Docker version 24.0.6`

#### For Windows:
1. Go to https://www.docker.com/products/docker-desktop
2. Click "Download for Windows"
3. Run the installer
4. Follow the installation wizard
5. **Restart your computer** when prompted
6. Open Docker Desktop from Start menu
7. Click "Accept" on the agreement

**Test it worked:**
Open PowerShell (press Windows key, type "powershell", press Enter) and run:
```bash
docker --version
```

#### For Linux (Ubuntu):
```bash
# Update system
sudo apt update

# Install Docker
sudo apt install docker.io docker-compose -y

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add yourself to docker group (so you don't need sudo)
sudo usermod -aG docker $USER
```

**Important:** You need to log out and log back in for the docker group changes to take effect.

**How to log out and back in:**
- **Desktop Ubuntu**: Click your name in top-right corner → "Log Out" → Log back in with your password
- **Ubuntu Server (SSH)**: Type `exit` to close your session, then reconnect with `ssh your-username@your-server`
- **Alternative (if you don't want to log out)**: Run `newgrp docker` to activate the group in your current session

**After logging back in, test it worked:**
```bash
docker --version
```

---

### Step 2: Get STREAM

**What is Git?** A tool for downloading code from the internet. Most Macs/Linux have it already.

**Download STREAM:**

```bash
# Open Terminal (Mac/Linux) or PowerShell (Windows)

# Navigate to where you want STREAM (e.g., Desktop)
cd ~/Desktop

# Download STREAM
git clone https://github.com/uicacer/STREAM.git

# Go into the STREAM folder
cd STREAM
```

**Don't have Git?**
- **Mac**: Open Terminal and type `git`. If not installed, Mac will ask to install it - click "Install"
- **Windows**: Download from https://git-scm.com/download/win
- **Alternative**: Download ZIP from GitHub and unzip it

---

### Step 3: Configure STREAM

**What are we doing?** Telling STREAM your settings and API keys (if you have them).

```bash
# Make sure you're in the STREAM folder
cd ~/Desktop/STREAM  # (adjust path if you put it elsewhere)

# Create configuration file
cp .env.example .env

# Edit the file
nano .env
```

**You'll see a file that looks like this:**

```bash
# =============================================================================
# API Keys (Optional - Leave empty if you don't have them)
# =============================================================================
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# =============================================================================
# Lakeshore vLLM (Campus HPC) - Leave as-is for now
# =============================================================================
LAKESHORE_VLLM_HOST=host.docker.internal
LAKESHORE_VLLM_PORT=8000
LAKESHORE_VLLM_ENDPOINT=http://${LAKESHORE_VLLM_HOST}:${LAKESHORE_VLLM_PORT}
```

**What to do:**

1. **If you have cloud API keys** (see Part 2 for how to get them):
   - Move cursor to the `ANTHROPIC_API_KEY=` line
   - Add your key after the `=`: `ANTHROPIC_API_KEY=sk-ant-api03-your-key-here`
   - Do the same for `OPENAI_API_KEY=` if you have one

2. **If you DON'T have API keys yet:**
   - Just leave them blank! STREAM will work with local AI only
   - You can add them later

3. **Lakeshore settings:**
   - Leave these as-is for now
   - We'll configure Lakeshore in Part 3 if you want campus AI

4. **Save and exit:**
   - Press `Ctrl+X`
   - Press `Y` to confirm
   - Press `Enter` to save

---

### Step 4: Run STREAM Setup

**This is the easy part!** One command does everything.

```bash
# Make sure you're in STREAM folder
cd ~/Desktop/STREAM

# Run the setup script
bash scripts/setup-stream.sh
```

**What will happen:**

```
   _____ _______ _____  ______          __  __
  / ____|__   __|  __ \|  ____|   /\   |  \/  |
 | (___    | |  | |__) | |__     /  \  | \  / |
  \___ \   | |  |  _  /|  __|   / /\ \ | |\/| |
  ____) |  | |  | | \ \| |____ / ____ \| |  | |
 |_____/   |_|  |_|  \_\______/_/    \_\_|  |_|

  Smart Tiered Routing Engine for AI Models

Version 1.0.0 | Interactive Setup

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1/6: Checking Prerequisites
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Checking Docker installation...
✅ Docker is running
   Checking docker-compose...
✅ docker-compose is available (v2.40.3)
...
```

The script will:

1. ✅ **Check prerequisites** - Make sure Docker is working
2. 🧹 **Clean up** - Remove any old versions
3. 📥 **Download AI models** - This takes 10-15 minutes (downloading ~3 GB)

   You'll see:
   ```
   ❓ Download models now? [Y/n]:
   ```

   **Press Enter** (or type `y` then Enter)

   Then wait while it downloads:
   ```
   ⠋ Building ollama...
   ⠙ Building middleware...
   ...
   ```

4. 🚀 **Start services** - Launch all STREAM components
5. 🏥 **Health check** - Make sure everything is working

**When it's done, you'll see:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ STREAM is Ready!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Installation Complete!

🌐 Access Points:
   Frontend (UI):   http://localhost:8501
   Middleware API:  http://localhost:5000
   LiteLLM Gateway: http://localhost:4000

🎉 Happy researching with STREAM!
```

---

### Step 5: Test STREAM

**Open your web browser** and go to:

```
http://localhost:8501
```

You should see the STREAM interface! Try asking:

```
What is Python?
```

STREAM will answer using your local AI (free!).

**🎉 Congratulations! You have STREAM running locally!**

At this point:
- ✅ Local AI (free) is working
- ❌ Cloud AI - not configured yet (optional - see Part 2)
- ❌ Lakeshore AI (free campus) - not set up yet (optional - see Part 3)

---

## Part 2: Cloud Setup (Optional - Best Quality AI)

This part shows you how to get API keys for the highest quality AI (Claude and GPT). This is **optional** - STREAM works fine with just local AI.

### What are API Keys?

**API Keys** are like passwords that let STREAM use professional cloud AI services. Without them, STREAM uses only local and campus AI (which is free but less powerful for complex questions).

### Why Use Cloud AI?

- **Highest Quality**: Best answers for complex research questions
- **Latest Models**: Access to cutting-edge AI like Claude Sonnet 4
- **Reliability**: Always available, never down
- **Speed**: Faster responses than local models

### Cost Breakdown

**Claude API (Anthropic):**
- Model used: **Claude Sonnet 4** (latest and most powerful)
- Cost: **$3 per million input tokens** / **$15 per million output tokens**
- What this means:
  - This entire README is ~7,000  tokens
  - You get ~140 conversations like this for $3
  - STREAM only uses Claude for complex questions
- Realistic monthly cost: $5-15 for moderate use

**OpenAI API:**
- Model used: **GPT-4** or **GPT-4 Turbo** (latest available)
- Cost: Similar to Claude, ~$2-$10 per million tokens depending on model
- Optional - only if you specifically want GPT models

---

### Get Claude API Key (Recommended)

**Step-by-step:**

1. **Create Anthropic Account:**
   - Go to https://console.anthropic.com
   - Click "Sign Up" (or "Sign In" if you have an account)
   - Verify your email

2. **Add Payment Method:**
   - Go to "Billing" in the left menu
   - Click "Add payment method"
   - Enter credit card details
   - **Note:** Anthropic requires a payment method but only charges for what you use

3. **Create API Key:**
   - Go to "API Keys" in the left menu
   - Click "Create Key"
   - Give it a name (e.g., "STREAM-Key")
   - **Copy the key** - it looks like: `sk-ant-api03-...`
   - **Save it somewhere safe** - you can't see it again!

4. **Add to STREAM:**
   ```bash
   # On your computer
   cd ~/Desktop/STREAM
   nano .env

   # Find this line:
   ANTHROPIC_API_KEY=

   # Add your key:
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

   # Save: Ctrl+X, Y, Enter
   ```

5. **Restart STREAM:**
   ```bash
   docker-compose restart middleware
   ```

**Done!** STREAM can now use Claude for complex questions.

---

### Get OpenAI API Key (Optional)

Only do this if you specifically want GPT models in addition to Claude. For now, STREAM uses Claude by default for cloud AI and using OpenAI is not possible without code changes. This will be added in future versions.

**Step-by-step:**

1. Go to https://platform.openai.com/signup
2. Sign up or sign in
3. Go to "API Keys" in left menu
4. Click "Create new secret key"
5. **Copy the key** - looks like: `sk-proj-...`
6. Save it securely

7. **Add to STREAM:**
   ```bash
   nano ~/Desktop/STREAM/.env

   # Add your key:
   OPENAI_API_KEY=sk-proj-your-key-here

   # Save and restart
   docker-compose restart middleware
   ```

---

### Cost Management Tips

**Set Spending Limits:**
- In Anthropic Console → Billing → Set monthly budget
- Get email alerts at 50%, 80%, 100%
- Auto-disable key if limit reached

**Keep Costs Low:**
- Use Model Tier: "Local" or "Lakeshore" for most questions
- Only switch to "Cloud" for truly complex queries
- Or don't add API keys at all - free tiers work great!

---

## Part 3: Lakeshore Setup (Optional - Free Campus AI)

This part sets up access to UIC's powerful GPU cluster. This is **optional** but recommended for researchers who want more powerful AI without paying.

### What is Lakeshore?

**Lakeshore** is UIC's high-performance computing (HPC) cluster - think of it as UIC's supercomputer. It has powerful NVIDIA GPUs that can run large AI models for **free**.

**Why use it:** Get high-quality answers without paying, using UIC's research computing resources.

### Prerequisites for Lakeshore

- [ ] UIC NetID and password
- [ ] Access to Lakeshore cluster (request from ACER if you don't have it)
- [ ] VPN connection (if working from home)

**Need Lakeshore access?** Email acer@uic.edu and request access to the Lakeshore cluster.

---

### Step 1: Connect to UIC VPN (If Working from Home)

**What is VPN?** A secure connection to UIC's network that makes your computer act like it's on campus.

1. Go to https://vpn.uic.edu
2. Download and install Cisco AnyConnect
3. Open AnyConnect
4. Connect to: `vpn.uic.edu`
5. Enter your NetID and password
6. Approve the Duo push notification

**On campus?** Skip this step - you're already on UIC's network!

---

### Step 2: Test SSH Connection to Lakeshore

**What is SSH?** A way to control a remote computer from your terminal. Think of it like remote desktop, but text-only. This is how you'll interact with Lakeshore.

**Why we need it:** To access Lakeshore's computing resources and run AI models on their GPUs.

**Test your connection:**

```bash
# Open Terminal (Mac/Linux) or PowerShell (Windows)

# Connect to Lakeshore (replace YOUR_NETID with your actual NetID)
ssh YOUR_NETID@lakeshore.acer.uic.edu
```

**First time connecting?**
```
The authenticity of host 'lakeshore.acer.uic.edu' can't be established.
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` and press Enter.

**Enter your UIC password** when prompted.

**Success looks like:**
```
[YOUR_NETID@lakeshore-login-01 ~]$
```

You're now on Lakeshore! Type `exit` to return to your computer.

**Troubleshooting:**
- **Connection refused**: Make sure you're on VPN (if off-campus)
- **Permission denied**: Check your NetID and password
- **No Lakeshore access**: Email acer@uic.edu

---

### Step 3: Set Up STREAM on Lakeshore

**Upload STREAM files to Lakeshore:**

#### Option A: Upload from Your Computer

```bash
# On your computer (NOT on Lakeshore)
cd ~/Desktop/STREAM

# Upload the vllm script to Lakeshore
scp scripts/vllm-ga001.sh YOUR_NETID@lakeshore.acer.uic.edu:~/
```

#### Option B: Clone STREAM on Lakeshore

```bash
# Connect to Lakeshore
ssh YOUR_NETID@lakeshore.acer.uic.edu

# Download STREAM
cd ~
git clone https://github.com/uicacer/STREAM.git
cd STREAM
```

**Create necessary directories:**

```bash
# On Lakeshore
cd ~/STREAM

# Create containers directory
mkdir -p containers

# Create deploy directory and move the vLLM script there
mkdir -p deploy
mv ~/vllm-ga001.sh deploy/  # If you uploaded it to home directory
# OR if you cloned STREAM:
cp scripts/vllm-ga001.sh deploy/
```

---

### Step 4: Build vLLM Container on Lakeshore

**What is vLLM?** vLLM (Virtual Large Language Model) is specialized software that runs AI models efficiently on GPUs. It's like a turbo-charged engine for running AI models, making them much faster than normal.

**Why we need it:** To serve AI models on Lakeshore's GPUs efficiently. Without vLLM, the models would run much slower.

**Important:** This step could take about **60 minutes** because the container is ~8 GB and needs to download many dependencies. It might also take much lesser depending on network speed and cluster load.

#### Request GPU Node

```bash
# On Lakeshore - request interactive GPU session
# Request 2 hours to be safe (build takes ~60 min)
salloc --job-name="vllm-build" --nodes=1 --time=2:00:00 \
  --partition=batch_gpu --account=<YOUR_ACCOUNT> --gres=gpu:1g.10gb:1
```

Wait for allocation:
```
salloc: Pending job allocation 198482
salloc: job 198482 queued and waiting for resources
salloc: job 198482 has been allocated resources
salloc: Granted job allocation 198482
salloc: Nodes ga-001 are ready for job
```

#### Build the Container

```bash
# Load Apptainer (container software on HPC)
module load apptainer

# Go to containers directory
cd ~/STREAM/containers

# Build vLLM container (could take ~60 minutes, ~8 GB download)
apptainer build vllm-openai_v0.13.0.sif docker://vllm/vllm-openai:v0.13.0
```

**What you'll see:**
```
INFO:    Starting build...
Getting image source signatures
Copying blob 9f2b12f755d0 done
Copying blob 6e8af4fd0a07 done
...
INFO:    Creating SIF file...
```

**This could take 45-75 minutes.** You can:
- Leave it running
- The process downloads ~8 GB of data
- Unpacks and converts it to Singularity format
- Creates final `.sif` file

**When complete:**
```
INFO:    Build complete: vllm-openai_v0.13.0.sif
```

**Exit the GPU node:**
```bash
exit  # Return to login node
```

**Verify the container:**
```bash
ls -lh ~/STREAM/containers/
# Should show: vllm-openai_v0.13.0.sif (~8 GB)
```

---

### Step 5: Start vLLM on Lakeshore

**What we're doing:** Submitting a job to Lakeshore that runs the vLLM server on a GPU node.

```bash
# On Lakeshore
cd ~/STREAM

# Submit the vLLM job
sbatch deploy/vllm-ga001.sh
```

**You'll see:**
```
Submitted batch job 123456
```

**Check if it's running:**
```bash
squeue -u $USER
```

**Success looks like:**
```
JOBID   PARTITION     NAME           USER      ST  TIME  NODES NODELIST
123456  batch_gpu     stream-vllm    yournetid  R  0:30      1 ga-001
```

- `R` = Running ✅
- `PD` = Pending (waiting for GPU) ⏳

**Wait about 5 minutes for the model to load.** Check the logs:

```bash
# View the log file (replace 123456 with your actual job ID)
cat logs/stream-vllm-ga-001-123456.log

# Or continuously monitor:
tail -f logs/stream-vllm-ga-001-123456.log
```

**When ready, you'll see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Press Ctrl+C** to stop viewing the log (the job keeps running).

**Important Notes:**
- The job runs for 30 minutes by default (see `--time=00:30:00` in the script)
- For longer research sessions, edit `deploy/vllm-ga001.sh` and change the time to more hours: `--time=04:00:00`
- The model currently used is **Qwen2.5-1.5B-Instruct** (fast, lightweight, good quality)

**If your job fails:**
Check the logs for errors:
```bash
cat logs/stream-vllm-ga-001-123456.log
```

Common issues:
- Container not found: Make sure you built it in Step 4
- Out of memory: The 1.5B model should work fine on 1g.10gb GPU
- Node not available: Try `sbatch deploy/vllm-ga001.sh` again

---

### Step 6: Set Up Port Forwarding

**What is port forwarding?** It creates a secure tunnel from your computer to Lakeshore, so STREAM on your computer can talk to the AI running on Lakeshore's GPU.

**Why we need it:** STREAM runs on your local computer, but the AI model runs on Lakeshore. Port forwarding connects them securely.

**On your computer (NOT on Lakeshore), open a NEW terminal:**

```bash
# Create SSH tunnel to ga-001 node
# Keep this terminal open while using STREAM
ssh -L 8000:ga-001:8000 YOUR_NETID@lakeshore.acer.uic.edu -N
```

**What this does:**
- `-L 8000:ga-001:8000` = Forward local port 8000 to ga-001's port 8000
- `-N` = Don't open a shell, just create the tunnel
- **This terminal will appear "frozen" - that's normal!** The tunnel is working.
- **Keep this terminal open while using STREAM**

**Testing the connection:**

**Open another terminal on your computer:**

```bash
# Test if vLLM is accessible
curl http://localhost:8000/health

# You should see:
{"status":"ok"}
```

**Success!** Your computer can now talk to Lakeshore's AI.

**Troubleshooting:**
- **Connection refused**: Make sure your vLLM job is running on Lakeshore (check `squeue -u $USER`)
- **Port already in use**: Something else is using port 8000 on your computer. Kill it or use a different port
- **Tunnel keeps disconnecting**: Add keep-alive:
  ```bash
  ssh -L 8000:ga-001:8000 YOUR_NETID@lakeshore.acer.uic.edu -N \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3
  ```

---

### Step 7: Test Lakeshore Integration

**Open STREAM in your browser:**

```
http://localhost:8501
```

**In the STREAM interface:**
1. Look at the sidebar on the left
2. Under "Model Tier", select **"🏫 Lakeshore (Campus GPU)"**
3. Ask a question:
   ```
   Explain the difference between supervised and unsupervised learning.
   ```

**You should see:**
- The answer appears
- At the bottom: "🏫 Answered by: Lakeshore"
- Response time shown (e.g., "⏱️ 2.5s")
- Cost: **FREE**

**🎉 Success! You now have:**
- ✅ Local AI (free, fast)
- ✅ Lakeshore AI (free, powerful)
- ✅ Cloud AI (paid, best quality - if you added API keys in Part 2)

---

## Using STREAM

### Basic Usage

1. **Open STREAM:** http://localhost:8501
2. **Select Model Tier** in the sidebar:
   - **🤖 Auto (Smart Routing)**: STREAM picks best tier based on complexity
   - **🏠 Local (Ollama - Free)**: Always use your computer (fast, simple questions)
   - **🏫 Lakeshore (Campus GPU)**: Always use UIC's supercomputer (free, better quality)
   - **☁️ Cloud (Claude/GPT - Paid)**: Always use professional AI (best quality)

3. **Type your question** in the chat box
4. **Press Enter** or click Send
5. **See the response** with:
   - The answer
   - Which tier was used (shown at bottom: "🏫 Answered by: Lakeshore")
   - Response time (e.g., "⏱️ 2.5s")
   - Cost (if cloud tier: shows "$0.XX")

### Understanding Model Tiers

**Note:** Automatic routing (complexity-based) only works when Model Tier is set to "Auto (Smart Routing)". When you manually select a tier, that tier is always used.

| Tier | When Auto Uses It | Model | Cost |
|------|------------------|-------|------|
| 🏠 **Local** | Simple questions | llama3.2:3b | FREE |
| 🏫 **Lakeshore** | Medium questions | Qwen2.5-1.5B-Instruct | FREE |
| ☁️ **Cloud** | Complex questions | Claude Sonnet 4 | ~$0.01/query |

**Examples:**
- Simple: "What is Python?" → Local
- Medium: "Explain neural networks in detail" → Lakeshore
- Complex: "Write a research proposal comparing deep learning architectures" → Cloud

### Viewing Session Statistics

In the sidebar, you'll see:
- **Total Queries**: Number of questions asked
- **Cloud**: Number using cloud (paid) tier
- **Local**: Number using local (free) tier
- **Total Cost**: How much spent this session

**Example:**
```
Total Queries: 15
Cloud: 1
Local: 10
Total Cost: $0.0000
💡 Free
```

### Staying Free

**To avoid any costs:**

Keep Model Tier selection on "Local" or "Lakeshore" only. This ensures you never use paid cloud AI.

**Or:** Don't add API keys to `.env` file - STREAM will only use Local + Lakeshore (can't access cloud without keys).

---

## Troubleshooting

### General Issues

#### "Cannot connect to Docker daemon"

**Problem:** Docker isn't running

**Solution:**
1. Open Docker Desktop
2. Wait for it to fully start (whale icon in menu bar/system tray)
3. Try again

#### "Port 8501 is already in use"

**Problem:** Something else is using STREAM's port

**Solution:**
```bash
# Find what's using the port
lsof -i :8501   # Mac/Linux
netstat -ano | findstr :8501   # Windows

# Kill the process or restart STREAM
docker-compose down
docker-compose up -d
```

#### "Model not found"

**Problem:** AI models weren't downloaded

**Solution:**
```bash
# Re-download models
docker exec -it stream-ollama ollama pull llama3.2:3b
docker exec -it stream-ollama ollama pull llama3.2:1b

# Restart
docker-compose restart
```

---

### Lakeshore-Specific Issues

#### "Cannot connect to Lakeshore"

**Checklist:**
1. ✅ Is your VPN connected? (if off-campus)
2. ✅ Is your vLLM job running? Check: `squeue -u $USER`
3. ✅ Is port forwarding active? You should have one terminal with the SSH tunnel open
4. ✅ Test connection: `curl http://localhost:8000/health`

**If vLLM job not running:**
```bash
# On Lakeshore
cd ~/STREAM
sbatch deploy/vllm-ga001.sh

# Check it started
squeue -u $USER
```

#### "vLLM job keeps failing"

**Check the logs:**
```bash
# On Lakeshore
cd ~/STREAM
ls -lt logs/  # Find most recent log
cat logs/stream-vllm-ga-001-XXXXXX.log
```

**Common errors:**

**"Container not found":**
- Solution: Make sure you built the container in Step 4
  ```bash
  ls ~/STREAM/containers/vllm-openai_v0.13.0.sif
  # Should show the 8GB file
  ```

**"Out of memory":**
- Solution: The Qwen 1.5B model should work on 1g.10gb GPU
- If it still fails, request more memory in `deploy/vllm-ga001.sh`:
  ```bash
  #SBATCH --gres=gpu:2g.20gb:1
  ```

**"Job pending forever":**
- GPUs are busy
- Check queue: `squeue -p batch_gpu`
- Wait or try during off-peak hours (early morning/late evening)

#### "Port forwarding keeps disconnecting"

**Make it more stable:**
```bash
# Use ServerAliveInterval to keep connection alive
ssh -L 8000:ga-001:8000 YOUR_NETID@lakeshore.acer.uic.edu -N \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3
```

**Or use autossh (keeps reconnecting automatically):**
```bash
# Mac: Install autossh
brew install autossh

# Use autossh instead of ssh
autossh -M 0 -L 8000:ga-001:8000 YOUR_NETID@lakeshore.acer.uic.edu -N \
  -o ServerAliveInterval=60
```

---

### Complete Reset

**If everything is broken, start fresh:**

```bash
# On your computer
cd ~/Desktop/STREAM

# Stop and remove everything
docker-compose down -v

# Re-run setup
bash scripts/setup-stream.sh
```

**This deletes:**
- All Docker containers
- Downloaded models
- Database data

**This KEEPS:**
- Your `.env` file (API keys are safe!)
- STREAM source code

---

## Daily Workflow

### Starting STREAM (After First Setup)

**Morning routine:**

1. **Start Docker Desktop** (if not running)

2. **Start STREAM:**
   ```bash
   cd ~/Desktop/STREAM
   docker-compose up -d
   ```

3. **If using Lakeshore:**

   **Terminal 1 - On Lakeshore (via SSH):**
   ```bash
   ssh YOUR_NETID@lakeshore.acer.uic.edu
   cd STREAM
   sbatch deploy/vllm-ga001.sh
   # Wait 5 minutes for model to load, or check logs
   ```

   **Terminal 2 - On your computer (port forward):**
   ```bash
   ssh -L 8000:ga-001:8000 YOUR_NETID@lakeshore.acer.uic.edu -N
   ```

   *(Keep this terminal open all day)*

4. **Open STREAM:** http://localhost:8501

### Stopping STREAM

**End of day:**

```bash
# Stop STREAM
cd ~/Desktop/STREAM
docker-compose down

# Close SSH tunnel (Ctrl+C in the tunnel terminal)

# Stop Docker Desktop (optional - saves battery)
```

**On Lakeshore:** Your vLLM job will auto-stop after 30 minutes (or whatever time you set in the script)

---

## FAQ

### General Questions

**Q: Do I need to be on campus to use STREAM?**
A: No! Local tier works anywhere. Lakeshore tier needs VPN if off-campus. Cloud tier works from anywhere.

**Q: Will STREAM slow down my computer?**
A: Docker uses about 4GB RAM and some CPU when running. Close it when not using STREAM to save resources.

**Q: Can I use STREAM for my research paper?**
A: Yes! That's what it's for. Just remember:
- Cite AI assistance in your paper
- Verify AI-generated facts
- Be careful sharing proprietary data with cloud AI

**Q: Is my data private?**
A:
- Local tier: 100% private (never leaves your computer)
- Lakeshore tier: Stays on UIC servers (not sent to third parties)
- Cloud tier: Sent to Anthropic/OpenAI (read their privacy policies)

**Q: How much does it cost per month?**
A: Depends on usage:
- Only local/Lakeshore: **$0**
- Light cloud use (2-3 complex questions/day): **$2-5/month**
- Moderate cloud use (10 complex questions/day): **$10-15/month**
- Heavy cloud use (50+ complex questions/day): **$30-50/month**

### Technical Questions

**Q: Can I run STREAM on Windows?**
A: Yes! Follow the Windows instructions for Docker Desktop.

**Q: What if I don't have admin access to install Docker?**
A: Contact your IT department. Docker Desktop requires admin rights on first install, but not after.

**Q: Can I use STREAM offline?**
A: Local tier works offline. Lakeshore and Cloud tiers need internet.

**Q: How do I update STREAM?**
A:
```bash
cd ~/Desktop/STREAM
git pull
docker-compose down
bash scripts/setup-stream.sh
```

**Q: Can multiple people use the same Lakeshore vLLM instance?**
A: Yes! Share the SSH tunnel instructions or have everyone create their own. The vLLM server can handle multiple connections.

**Q: What model does Lakeshore use?**
A: Currently **Qwen2.5-1.5B-Instruct** - it's fast, lightweight, and provides good quality responses. This may be updated to larger models as GPU availability improves and/or in future STREAM versions.

---

## Model Information

### Local Models (Ollama)
- **llama3.2:1b** - 1.3 GB - Faster but lower quality
- **llama3.2:3b** - 2.0 GB - Used for routing decisions

### Lakeshore Models
- **Qwen2.5-1.5B-Instruct** - 1.5 GB - Currently deployed (fast, good quality)

### Cloud Models
- **Claude Sonnet 4** - Anthropic's latest balanced model (best quality)
- **GPT-4 / GPT-4 Turbo** - OpenAI's latest (if configured)

---

## Getting Help

### Self-Help Resources

1. **Check logs:**
   ```bash
   # STREAM logs
   docker-compose logs middleware
   docker-compose logs frontend

   # Lakeshore logs
   ssh YOUR_NETID@lakeshore.acer.uic.edu
   cat ~/STREAM/logs/stream-vllm-*.log
   ```

2. **Check service status:**
   ```bash
   docker-compose ps
   ```

3. **Test components:**
   ```bash
   # Test Ollama
   docker exec -it stream-ollama ollama list

   # Test middleware
   curl http://localhost:5000/health

   # Test Lakeshore (if configured)
   curl http://localhost:8000/health
   ```

### Getting Technical Support

**UIC Resources:**

- **ACER Support:** acer@uic.edu
  - Lakeshore access issues
  - HPC job problems
  - VPN issues

- **STREAM Support:** nassar@uic.edu
  - STREAM-specific questions
  - Bug reports
  - Feature requests

**GitHub Issues:**
- https://github.com/uicacer/STREAM/issues
- Check existing issues first
- Provide logs when reporting bugs

---

## Acknowledgments

- **Built for:** UIC Researchers
- **Developed by:** Anas Nassar, ACER Technology Solutions
- **Powered by:**
  - Ollama (local inference)
  - vLLM (campus inference)
  - LiteLLM (unified API)
  - Anthropic Claude (cloud AI)
  - UIC ACER (Lakeshore cluster)

---

**📧 Questions? Email nassar@uic.edu or acer@uic.edu**

**🎉 Happy researching with STREAM!**
