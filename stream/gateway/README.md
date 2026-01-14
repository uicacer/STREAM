# STREAM Gateway Configuration

## Hardware Requirements

### Minimum (16GB RAM)
- Can run small models (3B-7B parameters)
- Examples: llama3.2:3b, mistral:7b
- Set in `.env`: `OLLAMA_MEMORY=4G`

### Recommended (32GB RAM)
- Can run medium models (7B-13B parameters)
- Examples: llama3.1:8b, mixtral:8x7b
- Set in `.env`: `OLLAMA_MEMORY=8G`

### High-end (64GB+ RAM)
- Can run large models (30B-70B parameters)
- Examples: llama3.1:70b, wizardcoder:34b
- Set in `.env`: `OLLAMA_MEMORY=16G`

## Checking Your Hardware
```bash
# Check total RAM
free -h

# Check available RAM
docker stats

# Adjust limits in .env based on your hardware
```

## Model Size Guide

| Model | Parameters | RAM Needed | Speed |
|-------|-----------|------------|-------|
| llama3.2:1b | 1B | ~1GB | Very Fast |
| llama3.2:3b | 3B | ~2GB | Fast |
| llama3.1:8b | 8B | ~5GB | Medium |
| mixtral:8x7b | 47B | ~26GB | Slow |
| llama3.1:70b | 70B | ~40GB | Very Slow |

## No Limits Mode

If you have LOTS of RAM and want no restrictions:
```yaml
# Comment out or remove the deploy section in docker-compose.yml
# deploy:
#   resources:
#     limits: ...
```
