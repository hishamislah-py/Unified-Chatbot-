# Ollama LLM Deployment Guide

## For Server: ai.arttechgroup.com (192.168.15.17)

**GPU:** NVIDIA RTX 3060 (12GB VRAM)  
**RAM:** 32GB System Memory  
**New Port:** 4754 (Ollama Native) / 4755 (OpenAI Proxy)  
**Access URL:** https://ai.arttechgroup.com:7777/ollama/

---

## Current Port Map

| Port | Service | Status |
|------|---------|--------|
| 4751 | chatbot-backend | ✅ Running |
| 4752 | compliance-backend | ✅ Running |
| 4753 | tts-chatterbox | ✅ Running |
| **4754** | **ollama** | 🆕 NEW |
| **4755** | **ollama-proxy** | 🆕 NEW |
| 4761 | chatbot-frontend | ✅ Running |
| 4762 | compliance-frontend | ✅ Running |

---

## Recommended Model: qwen2.5:7b-instruct

**Why this model for your setup?**

| Feature | Value | Why It Fits |
|---------|-------|-------------|
| VRAM Usage | ~5-6GB | Leaves ~6GB for TTS + headroom |
| Tool Calling | ⭐⭐⭐⭐⭐ | Best-in-class function calling |
| Speed | ~30-50 tok/sec | Fast enough for voice agents |
| Context | 32K tokens | Long conversations supported |

**Alternative Models:**

| Model | VRAM | Best For |
|-------|------|----------|
| llama3.1:8b-instruct | ~6-7GB | General purpose |
| mistral:7b-instruct | ~5GB | Maximum speed |
| qwen2.5:14b-instruct-q4_K_M | ~9GB | More capability (when TTS idle) |

---

## Step 1: SSH into Server

```bash
ssh chatbotuser@192.168.15.17
```

---

## Step 2: Check Current GPU Usage

```bash
nvidia-smi
```

Expected: RTX 3060 with ~2-3GB used by Chatterbox TTS.

---

## Step 3: Install Ollama

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

---

## Step 4: Configure Ollama Service

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

Paste this content:

```ini
[Service]
Environment="OLLAMA_HOST=127.0.0.1:4754"
Environment="OLLAMA_ORIGINS=*"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=5m"
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

---

## Step 5: Start Ollama Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable ollama

# Start service
sudo systemctl start ollama

# Check status
sudo systemctl status ollama
```

---

## Step 6: Pull the Recommended Model

```bash
# Pull Qwen2.5 7B (Best for tool calling + voice agents)
ollama pull qwen2.5:7b-instruct
```

**First pull downloads ~4.7GB - wait 5-10 minutes.**

Verify model:

```bash
ollama list
```

---

## Step 7: Test Ollama Directly

```bash
# Simple test
curl http://localhost:4754/api/generate -d '{
  "model": "qwen2.5:7b-instruct",
  "prompt": "Hello, how are you?",
  "stream": false
}'

# Test with tool calling
curl http://localhost:4754/api/chat -d '{
  "model": "qwen2.5:7b-instruct",
  "messages": [{"role": "user", "content": "What is 25 * 17?"}],
  "stream": false,
  "tools": [{
    "type": "function",
    "function": {
      "name": "calculate",
      "description": "Perform math calculations",
      "parameters": {
        "type": "object",
        "properties": {
          "expression": {"type": "string"}
        },
        "required": ["expression"]
      }
    }
  }]
}'
```

---

## Step 8: Create OpenAI-Compatible Proxy Directory

```bash
# Create directory
cd /var/www/apps
mkdir -p ollama-proxy
cd ollama-proxy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn[standard] httpx
```

---

## Step 9: Create the Proxy Server File

```bash
nano /var/www/apps/ollama-proxy/server.py
```

**Paste this COMPLETE server code:**

```python
"""
Ollama OpenAI-Compatible Proxy Server
For Voice Agent Integration
Port: 4755
"""

import httpx
import json
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:4754"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

app = FastAPI(
    title="Ollama OpenAI Proxy",
    description="OpenAI-compatible API for Ollama LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai.arttechgroup.com:7777",
        "https://ai.arttechgroup.com",
        "http://localhost:*",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "Ollama OpenAI Proxy",
        "version": "1.0.0",
        "default_model": DEFAULT_MODEL,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            models = response.json().get("models", [])
            return {
                "status": "healthy",
                "ollama": "connected",
                "models_loaded": len(models),
                "default_model": DEFAULT_MODEL
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/v1/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            data = response.json()
        
        models = []
        for model in data.get("models", []):
            models.append({
                "id": model["name"],
                "object": "model",
                "created": 0,
                "owned_by": "ollama"
            })
        return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"List models failed: {e}")
        return {"object": "list", "data": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    
    model = body.get("model", DEFAULT_MODEL)
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    tools = body.get("tools", None)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 2048)
    
    ollama_request = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    if tools:
        ollama_request["tools"] = tools
    
    logger.info(f"Chat request: model={model}, messages={len(messages)}, stream={stream}")
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            if stream:
                async def generate():
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_URL}/api/chat",
                        json=ollama_request
                    ) as response:
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    content = data.get("message", {}).get("content", "")
                                    done = data.get("done", False)
                                    
                                    chunk = {
                                        "id": "chatcmpl-ollama",
                                        "object": "chat.completion.chunk",
                                        "created": 0,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": content} if content else {},
                                            "finish_reason": "stop" if done else None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                except json.JSONDecodeError:
                                    continue
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
            else:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=ollama_request
                )
                data = response.json()
                
                message = data.get("message", {})
                
                result = {
                    "id": "chatcmpl-ollama",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": (
                            data.get("prompt_eval_count", 0) + 
                            data.get("eval_count", 0)
                        )
                    }
                }
                
                return JSONResponse(result)
                
    except httpx.TimeoutException:
        logger.error("Ollama request timed out")
        return JSONResponse(
            {"error": "Request timed out"},
            status_code=504
        )
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    
    model = body.get("model", DEFAULT_MODEL)
    prompt = body.get("prompt", "")
    
    ollama_request = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": body.get("temperature", 0.7),
            "num_predict": body.get("max_tokens", 2048)
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=ollama_request
            )
            data = response.json()
            
            return {
                "id": "cmpl-ollama",
                "object": "text_completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "text": data.get("response", ""),
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (
                        data.get("prompt_eval_count", 0) + 
                        data.get("eval_count", 0)
                    )
                }
            }
    except Exception as e:
        logger.error(f"Completion error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4755)
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

---

## Step 10: Create Environment File

```bash
nano /var/www/apps/ollama-proxy/.env
```

Content:

```env
PORT=4755
OLLAMA_URL=http://127.0.0.1:4754
DEFAULT_MODEL=qwen2.5:7b-instruct
```

```bash
chmod 600 /var/www/apps/ollama-proxy/.env
```

---

## Step 11: Test Proxy Manually (Important!)

```bash
cd /var/www/apps/ollama-proxy
source venv/bin/activate
python server.py
```

In another SSH terminal, test:

```bash
# Health check
curl http://localhost:4755/health

# List models (OpenAI format)
curl http://localhost:4755/v1/models

# Chat completion (OpenAI format)
curl http://localhost:4755/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

If working, stop the test server: `Ctrl+C`

Deactivate venv:

```bash
deactivate
```

---

## Step 12: Update PM2 Ecosystem Config

```bash
nano /var/www/apps/ecosystem.config.js
```

**Replace the ENTIRE file with this COMPLETE configuration:**

```javascript
/**
 * PM2 Ecosystem Configuration
 * Port Assignments (Unique ports - 47xx range):
 *  - Chatbot Backend:      4751
 *  - Chatbot Frontend:     4761
 *  - Compliance Backend:   4752
 *  - Compliance Frontend:  4762
 *  - Chatterbox TTS:       4753
 *  - Ollama Proxy:         4755  <-- NEW
 */

module.exports = {
  apps: [
    // ==================== CHATBOT ====================
    {
      name: 'chatbot-backend',
      cwd: '/var/www/apps/customercare-chatbot/backend',
      script: 'venv/bin/uvicorn',
      args: 'fastapi_app:app --host 0.0.0.0 --port 4751 --workers 2',
      interpreter: 'none',
      env: { NODE_ENV: 'production' },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      error_file: '/var/www/apps/logs/chatbot-backend-error.log',
      out_file: '/var/www/apps/logs/chatbot-backend-out.log',
      log_file: '/var/www/apps/logs/chatbot-backend-combined.log',
      time: true,
    },
    {
      name: 'chatbot-frontend',
      cwd: '/var/www/apps/customercare-chatbot/frontend',
      script: 'npm',
      args: 'run preview -- --port 4761 --host 0.0.0.0',
      interpreter: 'none',
      env: { NODE_ENV: 'production' },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      error_file: '/var/www/apps/logs/chatbot-frontend-error.log',
      out_file: '/var/www/apps/logs/chatbot-frontend-out.log',
      log_file: '/var/www/apps/logs/chatbot-frontend-combined.log',
      time: true,
    },

    // ==================== COMPLIANCE ====================
    {
      name: 'compliance-backend',
      cwd: '/var/www/apps/compliance/backend',
      script: 'venv/bin/uvicorn',
      args: 'api:app --host 0.0.0.0 --port 4752 --workers 2',
      interpreter: 'none',
      env: { NODE_ENV: 'production' },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      error_file: '/var/www/apps/logs/compliance-backend-error.log',
      out_file: '/var/www/apps/logs/compliance-backend-out.log',
      log_file: '/var/www/apps/logs/compliance-backend-combined.log',
      time: true,
    },
    {
      name: 'compliance-frontend',
      cwd: '/var/www/apps/compliance/frontend',
      script: 'npm',
      args: 'run preview -- --port 4762 --host 0.0.0.0',
      interpreter: 'none',
      env: { NODE_ENV: 'production' },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      error_file: '/var/www/apps/logs/compliance-frontend-error.log',
      out_file: '/var/www/apps/logs/compliance-frontend-out.log',
      log_file: '/var/www/apps/logs/compliance-frontend-combined.log',
      time: true,
    },

    // ==================== CHATTERBOX TTS ====================
    {
      name: 'tts-chatterbox',
      cwd: '/var/www/apps/chatterbox-tts',
      script: 'venv/bin/uvicorn',
      args: 'server:app --host 0.0.0.0 --port 4753',
      interpreter: 'none',
      env: {
        NODE_ENV: 'production',
        CUDA_VISIBLE_DEVICES: '0'
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '6G',
      // Allow 2 minutes for model loading on startup
      kill_timeout: 120000,
      error_file: '/var/www/apps/logs/tts-chatterbox-error.log',
      out_file: '/var/www/apps/logs/tts-chatterbox-out.log',
      log_file: '/var/www/apps/logs/tts-chatterbox-combined.log',
      time: true,
    },

    // ==================== OLLAMA PROXY (NEW) ====================
    {
      name: 'ollama-proxy',
      cwd: '/var/www/apps/ollama-proxy',
      script: 'venv/bin/uvicorn',
      args: 'server:app --host 0.0.0.0 --port 4755',
      interpreter: 'none',
      env: {
        NODE_ENV: 'production',
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      error_file: '/var/www/apps/logs/ollama-proxy-error.log',
      out_file: '/var/www/apps/logs/ollama-proxy-out.log',
      log_file: '/var/www/apps/logs/ollama-proxy-combined.log',
      time: true,
    },
  ],
};
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

---

## Step 13: Start Ollama Proxy with PM2

```bash
cd /var/www/apps

# Create logs directory if needed
mkdir -p logs

# Start ONLY the new proxy service (won't affect others)
pm2 start ecosystem.config.js --only ollama-proxy

# Check all services are running
pm2 status

# Watch the proxy logs (wait for "Uvicorn running")
pm2 logs ollama-proxy --lines 30

# Save PM2 config (so it survives reboot)
pm2 save
```

**Expected pm2 status output:**

```
┌─────┬──────────────────────┬─────────────┬─────────┬─────────┬──────────┐
│ id  │ name                 │ status      │ cpu     │ memory  │          │
├─────┼──────────────────────┼─────────────┼─────────┼─────────┼──────────┤
│ 0   │ chatbot-backend      │ online      │ 0%      │ 200MB   │          │
│ 1   │ chatbot-frontend     │ online      │ 0%      │ 50MB    │          │
│ 2   │ compliance-backend   │ online      │ 0%      │ 150MB   │          │
│ 3   │ compliance-frontend  │ online      │ 0%      │ 50MB    │          │
│ 4   │ tts-chatterbox       │ online      │ 0%      │ 2.5GB   │          │
│ 5   │ ollama-proxy         │ online      │ 0%      │ 100MB   │          │
└─────┴──────────────────────┴─────────────┴─────────┴─────────┴──────────┘
```

---

## Step 14: Update Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/apps.conf
```

**Replace with this COMPLETE Nginx configuration:**

```nginx
upstream chatbot_backend {
    server 127.0.0.1:4751;
}

upstream compliance_backend {
    server 127.0.0.1:4752;
}

upstream tts_backend {
    server 127.0.0.1:4753;
}

upstream ollama_proxy {
    server 127.0.0.1:4755;
}

server {
    listen 7777 ssl http2;
    server_name ai.arttechgroup.com;

    ssl_certificate /etc/ssl/ai.arttechgroup.com/fullchain.crt;
    ssl_certificate_key /etc/ssl/ai.arttechgroup.com/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;

    client_max_body_size 50M;

    # Root redirect
    location = / {
        return 301 /chatbot/;
    }

    # Redirect without trailing slash
    location = /chatbot {
        return 301 /chatbot/;
    }
    location = /compliance {
        return 301 /compliance/;
    }
    location = /tts {
        return 301 /tts/;
    }
    location = /ollama {
        return 301 /ollama/;
    }

    # Catch root-level chatbot images
    location ~* ^/(logo\.png|logo2\.png|chat\.png|vite\.svg|bg3\.png|CC-bg.*|customerCareBG\.png|image\.png)$ {
        root /var/www/apps/customercare-chatbot/frontend/dist;
    }

    # Catch root-level compliance images
    location ~* ^/(PanasaLogo\.png|monogram-v2\.png|favicon\.svg)$ {
        root /var/www/apps/compliance/frontend/dist;
    }

    # ==================== CHATBOT ====================
    # Chatbot Frontend
    location /chatbot/ {
        alias /var/www/apps/customercare-chatbot/frontend/dist/;
        try_files $uri $uri/ /chatbot/index.html;
    }

    # Chatbot API
    location /chatbot/api/ {
        rewrite ^/chatbot/api/(.*) /$1 break;
        proxy_pass http://chatbot_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    # ==================== COMPLIANCE ====================
    # Compliance Frontend
    location /compliance/ {
        alias /var/www/apps/compliance/frontend/dist/;
        try_files $uri $uri/ /compliance/index.html;
    }

    # Compliance API
    location /compliance/api/ {
        rewrite ^/compliance/api/(.*) /api/$1 break;
        proxy_pass http://compliance_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    # ==================== CHATTERBOX TTS ====================
    # TTS API
    location /tts/ {
        rewrite ^/tts/(.*) /$1 break;
        proxy_pass http://tts_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;

        # Buffer for audio responses
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # TTS Health (direct access for monitoring)
    location = /tts/health {
        proxy_pass http://tts_backend/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    # ==================== OLLAMA LLM (NEW) ====================
    # Ollama OpenAI-compatible API
    location /ollama/ {
        rewrite ^/ollama/(.*) /$1 break;
        proxy_pass http://ollama_proxy;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Longer timeouts for LLM generation
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_connect_timeout 60s;
        
        # For streaming responses (SSE)
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        
        # SSE specific headers
        proxy_set_header Connection '';
    }

    # Ollama Health (direct access for monitoring)
    location = /ollama/health {
        proxy_pass http://ollama_proxy/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    # ==================== HEALTH CHECK ====================
    location /health {
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }
}
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

---

## Step 15: Test and Reload Nginx

```bash
# Test config syntax
sudo nginx -t

# If OK, reload
sudo systemctl reload nginx
```

**Expected output:**

```
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

---

## Step 16: Verify Everything Works

**Test Internal (localhost):**

```bash
# Ollama direct
curl http://localhost:4754/api/tags

# Proxy health
curl http://localhost:4755/health

# Proxy models
curl http://localhost:4755/v1/models

# Chat test
curl http://localhost:4755/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

**Test via Nginx (external):**

```bash
# Ollama Health check
curl.exe -k https://ai.arttechgroup.com:7777/ollama/health

# List models
curl.exe -k https://ai.arttechgroup.com:7777/ollama/v1/models

# Chat completion
curl.exe -k -X POST https://ai.arttechgroup.com:7777/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"Hello from ART Technology!"}]}'

# Check existing apps still work
curl.exe -k https://ai.arttechgroup.com:7777/chatbot/api/health
curl.exe -k https://ai.arttechgroup.com:7777/compliance/api/status
curl.exe -k https://ai.arttechgroup.com:7777/tts/health
```

---

## Step 17: Check All Services

```bash
# PM2 status
pm2 status

# Check all ports are listening
sudo netstat -tlnp | grep -E "4751|4752|4753|4754|4755|4761|4762"

# GPU usage (should show TTS + Ollama when active)
nvidia-smi

# Ollama service status
systemctl status ollama

# View Ollama logs
sudo journalctl -u ollama -f --lines 30
```

---

## Final URLs

| Service | URL |
|---------|-----|
| Chatbot | https://ai.arttechgroup.com:7777/chatbot/ |
| Chatbot API | https://ai.arttechgroup.com:7777/chatbot/api/ |
| Compliance | https://ai.arttechgroup.com:7777/compliance/ |
| Compliance API | https://ai.arttechgroup.com:7777/compliance/api/ |
| TTS API | https://ai.arttechgroup.com:7777/tts/ |
| TTS Docs | https://ai.arttechgroup.com:7777/tts/docs |
| TTS Health | https://ai.arttechgroup.com:7777/tts/health |


---

## API Usage Examples

**Basic Chat:**

```bash
curl.exe -k -X POST https://ai.arttechgroup.com:7777/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain machine learning in simple terms."}
    ],
    "temperature": 0.7
  }'
```

**With Tool Calling (for Voice Agents):**

```bash
curl.exe -k -X POST https://ai.arttechgroup.com:7777/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b-instruct",
    "messages": [{"role": "user", "content": "What time is it in London?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_current_time",
        "description": "Get current time for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

**Streaming Response:**

```bash
curl.exe -k -X POST https://ai.arttechgroup.com:7777/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b-instruct",
    "messages": [{"role": "user", "content": "Write a short poem about AI."}],
    "stream": true
  }'
```

**Python Client Example:**

```python
import openai

client = openai.OpenAI(
    base_url="https://ai.arttechgroup.com:7777/ollama/v1",
    api_key="not-needed"  # Ollama doesn't require API key
)

response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

**Voice Agent Integration (LiveKit):**

```python
# For LiveKit Voice Agent
from livekit.agents import llm

# Configure to use local Ollama
llm_plugin = llm.LLM.with_openai(
    base_url="https://ai.arttechgroup.com:7777/ollama/v1",
    api_key="not-needed",
    model="qwen2.5:7b-instruct"
)
```

---

## Useful PM2 Commands

```bash
# View all logs
pm2 logs

# View Ollama proxy logs
pm2 logs ollama-proxy --lines 50

# Restart Ollama proxy only
pm2 restart ollama-proxy

# Stop Ollama proxy only
pm2 stop ollama-proxy

# Restart all services
pm2 reload all

# Monitor all services
pm2 monit

# Show detailed status
pm2 show ollama-proxy
```

---

## Useful Ollama Commands

```bash
# View Ollama logs
sudo journalctl -u ollama -f

# Restart Ollama service
sudo systemctl restart ollama

# Stop Ollama service
sudo systemctl stop ollama

# List loaded models
ollama list

# Pull another model
ollama pull llama3.1:8b-instruct

# Remove a model
ollama rm model-name

# Run interactive chat
ollama run qwen2.5:7b-instruct
```

---

## Troubleshooting

**Ollama Not Starting:**

```bash
# Check logs
sudo journalctl -u ollama --lines 100

# Check if port is in use
sudo lsof -i :4754

# Restart service
sudo systemctl restart ollama
```

**Proxy Not Connecting to Ollama:**

```bash
# Check if Ollama is running
curl http://localhost:4754/api/tags

# Check proxy logs
pm2 logs ollama-proxy --lines 100

# Restart proxy
pm2 restart ollama-proxy
```

**GPU Out of Memory:**

```bash
# Check GPU usage
nvidia-smi

# Unload Ollama model (frees VRAM)
curl http://localhost:4754/api/generate -d '{"model":"qwen2.5:7b-instruct","keep_alive":0}'

# Or restart Ollama to clear
sudo systemctl restart ollama
```

**Port Already in Use:**

```bash
# Find what's using the port
sudo lsof -i :4754
sudo lsof -i :4755

# Kill the process
sudo kill -9 $(sudo lsof -t -i:4754)
sudo kill -9 $(sudo lsof -t -i:4755)

# Restart services
sudo systemctl restart ollama
pm2 restart ollama-proxy
```

**Model Not Found:**

```bash
# List available models
ollama list

# Pull model if missing
ollama pull qwen2.5:7b-instruct

# Check Ollama storage
du -sh ~/.ollama/models/
```

**Nginx 502 Bad Gateway:**

```bash
# Check if proxy is running
pm2 status

# Check nginx error logs
sudo tail -f /var/log/nginx/error.log

# Test proxy directly
curl http://localhost:4755/health
```

---

## Memory Optimization Tips

```bash
# Model auto-unloads after 5 minutes of inactivity (configured in Step 4)
# To manually unload and free VRAM:
curl http://localhost:4754/api/generate -d '{"model":"qwen2.5:7b-instruct","keep_alive":0}'

# Monitor GPU memory continuously
watch -n 2 nvidia-smi

# Check system RAM
free -h
```

---

## Summary

✅ Ollama installed and running on port **4754**  
✅ OpenAI-compatible proxy on port **4755**  
✅ Accessible at **https://ai.arttechgroup.com:7777/ollama/**  
✅ API Docs at **https://ai.arttechgroup.com:7777/ollama/docs**  
✅ Managed by **systemd** (Ollama) + **PM2** (proxy)  
✅ Model: **qwen2.5:7b-instruct** (~5-6GB VRAM)  
✅ Tool calling enabled for voice agents  
✅ Streaming responses supported  
✅ Existing apps unchanged and running  
✅ Optimized for RTX 3060 (12GB VRAM) + 32GB RAM  

---

## VRAM Usage Summary

| Service | VRAM Usage |
|---------|------------|
| Chatterbox TTS | ~2-3GB |
| Ollama (qwen2.5:7b) | ~5-6GB |
| **Total** | **~7-9GB** |
| **Available** | **~3-5GB headroom** |
