# Multi-Agent Chatbot System - Setup Guide

## Overview

This multi-agent chatbot system features:
- **Personal Assistant** - Conversational entry point, routes to specialists
- **HR Agent** - Handles HR and Leave policies
- **IT Support Agent** - Handles IT Security and Compliance policies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │  Voice Assistant │     │       Text Chat UI           │  │
│  │  (LiveKit React) │     │      (AIChatCard)            │  │
│  └────────┬─────────┘     └──────────────┬───────────────┘  │
└───────────┼──────────────────────────────┼──────────────────┘
            │ WebSocket                     │ HTTP/SSE
            ▼                               ▼
┌───────────────────────┐     ┌──────────────────────────────┐
│   LiveKit Server      │     │     FastAPI Server           │
│   (Voice Rooms)       │     │     (Port 8000)              │
└───────────┬───────────┘     └──────────────┬───────────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  Voice Agent Worker                          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐  │
│  │   VAD   │ → │   STT   │ → │Chat API │ → │    TTS      │  │
│  │ Silero  │   │  Groq   │   │  LLM    │   │  Edge TTS   │  │
│  └─────────┘   └─────────┘   └─────────┘   └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Multi-Agent System                    │
│    ┌──────────────────┐                                     │
│    │ Personal Assistant│                                    │
│    └────────┬─────────┘                                     │
│             │ Transfer                                       │
│    ┌────────┴────────┐                                      │
│    ▼                 ▼                                      │
│ ┌──────────┐   ┌──────────┐                                 │
│ │ HR Agent │   │ IT Agent │                                 │
│ │  (RAG)   │   │  (RAG)   │                                 │
│ └──────────┘   └──────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv env

   # Activate on Windows:
   env\Scripts\activate

   # Activate on Mac/Linux:
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create .env file** in `backend/env/` directory with your API keys:
   ```env
   # Required for LLM and STT
   GROQ_API_KEY=your_groq_api_key_here

   # Required for Voice Agent (LiveKit)
   LIVEKIT_URL=ws://localhost:7880
   LIVEKIT_API_KEY=devkey
   LIVEKIT_API_SECRET=secret
   ```

   > **Note:** For local development, use LiveKit's development server.
   > For production, get credentials from [LiveKit Cloud](https://cloud.livekit.io).

5. **Verify documents exist**
   Ensure these PDFs are in `backend/docs/`:
   - Leave Policy.pdf
   - HR_Policy_Art_Technology.pdf
   - IT_Security_Policy_AI_Usage.pdf
   - Compliance Handbook.pdf

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

## Running the System

### Terminal 1: Start Backend Server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Expected output:
```
======================================================================
STARTING MULTI-AGENT CHATBOT WITH VOICE SUPPORT
======================================================================

[1/3] Initializing RAG system...
[OK] RAG system initialized with HR and IT documents

[2/3] Setting RAG system for PolicyTools...
[OK] PolicyTools configured

[3/3] Building multi-agent LangGraph...
[OK] Multi-agent graph compiled

======================================================================
SERVER READY!
API Documentation: http://localhost:8000/docs
======================================================================
```

**Backend runs on:** `http://localhost:8000`

### Terminal 2: Start Voice Agent (Optional)

```bash
cd backend
python -m voice.voice_agent dev
```

Expected output:
```
============================================================
LiveKit Voice Agent
============================================================
LIVEKIT_URL: ws://localhost:7880
GROQ_API_KEY: set
============================================================

Starting Voice Agent Worker...
```

> **Note:** Voice agent requires LiveKit server running. For local dev:
> ```bash
> # Install LiveKit CLI, then:
> livekit-server --dev
> ```

### Terminal 3: Start Frontend

```bash
cd frontend
npm run dev
```

**Frontend runs on:** `http://localhost:5173`

### Quick Start (Text Chat Only)

If you only need text chat (no voice):
```bash
# Terminal 1
cd backend && uvicorn main:app --port 8000

# Terminal 2
cd frontend && npm run dev
```

## Testing Guide

### 1. Test Personal Assistant

**Test greeting:**
- User: "Hello"
- Expected: Greeting message with agent options

**Test general query:**
- User: "What is the company name?"
- Expected: Answer from Personal Assistant (no transfer)

**Test explicit transfer to HR:**
- User: "Connect me to HR" or "I need HR"
- Expected: Transfer message, switch to HR Agent tab

**Test explicit transfer to IT:**
- User: "Connect me to IT support" or "Talk to IT"
- Expected: Transfer message, switch to IT Agent tab

**Test no assumption (CRITICAL):**
- User: "What is the leave policy?"
- Expected: Personal Assistant suggests requesting transfer, does NOT auto-transfer

### 2. Test HR Agent

First, ask Personal Assistant: "Connect me to HR"

**Test policy query:**
- User: "What is the sick leave policy?"
- Expected: Answer with citations from Leave Policy.pdf
- Check: Sources displayed with page numbers

**Test clarification:**
- User: "Tell me about leave"
- Expected: Clarification question asking which leave type

**Test out-of-scope (CRITICAL):**
- User: "Who is the Indian president?"
- Expected: Polite decline, stays in HR Agent, suggests Personal Assistant

**Test validation retry:**
- User: "What is the xyz policy?" (intentionally vague)
- Expected: May trigger retry or fallback message

### 3. Test IT Support Agent

First, ask Personal Assistant: "Connect me to IT support"

**Test policy query:**
- User: "What is the password policy?"
- Expected: Answer with citations from IT Security Policy or Compliance Handbook
- Check: Sources displayed with page numbers

**Test clarification:**
- User: "Tell me about security"
- Expected: Clarification question asking for specific topic

**Test out-of-scope (CRITICAL):**
- User: "What's the weather today?"
- Expected: Polite decline, stays in IT Agent, suggests Personal Assistant

### 4. Test Agent Transfers

**Test proper routing:**
1. Start with Personal Assistant
2. Say "Connect me to HR"
3. Verify switch to HR Agent tab
4. Ask HR question
5. Manually switch to Personal Assistant tab
6. Say "Connect me to IT"
7. Verify switch to IT Agent tab

**Test conversation persistence:**
- Send multiple messages to one agent
- Switch tabs manually
- Return to previous agent
- Verify conversation history is preserved

### 5. Test Sources/Citations

**HR Agent:**
- Ask: "How many sick leave days do I get?"
- Verify: Response includes source citations like:
  ```
  Sources:
  [1] Leave Policy.pdf - Page 3
  ```

**IT Agent:**
- Ask: "What is the AI usage policy?"
- Verify: Response includes source citations

## API Documentation

Visit `http://localhost:8000/docs` when server is running for interactive API documentation.

### Key Endpoints

- **POST /api/sessions** - Create new chat session
- **GET /api/sessions/{session_id}** - Get session info
- **POST /api/chat** - Send chat message (non-streaming)
- **POST /api/chat/stream** - Send chat message (SSE streaming)
- **POST /api/livekit/token** - Generate LiveKit room token
- **GET /api/health** - Health check

## Troubleshooting

### Backend Issues

**Error: "RAG initialization failed"**
- Check that all 4 PDF files exist in `backend/docs/`
- Verify file names match exactly

**Error: "GROQ_API_KEY not found"**
- Create `.env` file in `backend/` directory
- Add your Groq API key

**Error: "Module not found"**
- Run `pip install -r requirements.txt` again
- Verify virtual environment is activated

### Frontend Issues

**Error: "Failed to connect to server"**
- Verify backend is running on port 8000
- Check console for CORS errors
- Try refreshing the page

**Sources not displaying:**
- Verify RAG is retrieving documents (check backend logs)
- Ask specific policy questions (e.g., "sick leave policy")

### Voice Agent Issues

**Error: "LIVEKIT_URL not set"**
- Create `.env` file in `backend/env/` directory
- Add `LIVEKIT_URL=ws://localhost:7880`

**Error: "Failed to connect to LiveKit"**
- Start LiveKit server: `livekit-server --dev`
- Verify LIVEKIT_URL matches server address

**Error: "streaming is not supported by this TTS"**
- Edge TTS uses chunked mode, not streaming
- This is expected behavior, use current configuration

**Voice response is slow**
- Check VAD settings (min_silence_duration should be 0.25s)
- Ensure backend has no artificial delays
- Check network latency to Groq API

**No audio output**
- Check browser microphone permissions
- Verify LiveKit room connection
- Check console for WebRTC errors

### Common Issues

**Agent not transferring:**
- Verify you used explicit keywords: "HR", "IT support", "connect to HR"
- Personal Assistant does NOT transfer on policy questions alone

**Out-of-scope questions getting answers:**
- Verify the question is truly out-of-scope
- Check workflow_path in response to see which nodes executed

**Session not found error:**
- Server may have restarted (sessions are in-memory)
- Refresh the page to create a new session
- Sessions auto-recreate on next request

## Development Tips

### Backend Development

**Enable verbose logging:**
```python
# In api/server.py startup event
rag_system.setup(verbose=True)  # Show RAG initialization details
```

**Test single endpoint:**
```bash
curl -X POST http://localhost:8000/api/sessions
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"Hello","agent":"personal"}'
```

### Frontend Development

**View API responses in console:**
- Open browser DevTools (F12)
- Check Console tab for API logs
- Check Network tab for request/response details

**Test without backend:**
- Comment out API calls in `handleSend()`
- Use mock responses for UI development

## Project Structure

```
backend/
├── agents/
│   ├── __init__.py
│   ├── personal_assistant.py      # Personal Assistant logic
│   ├── specialist_agents.py       # HR & IT agents
│   └── multi_agent_graph.py       # LangGraph orchestration
├── api/
│   ├── __init__.py
│   ├── models.py                  # Pydantic schemas
│   ├── session_manager.py         # Session handling
│   └── server.py                  # FastAPI app
├── voice/
│   ├── __init__.py
│   ├── voice_agent.py             # LiveKit voice agent
│   ├── chat_api_llm.py            # Custom LLM (routes to Chat API)
│   └── edge_tts_adapter.py        # Free TTS adapter
├── docs/
│   ├── Leave Policy.pdf
│   ├── HR_Policy_Art_Technology.pdf
│   ├── IT_Security_Policy_AI_Usage.pdf
│   └── Compliance Handbook.pdf
├── env/
│   └── .env                       # API keys (create this)
├── main.py                        # Main entry point
├── rag_node.py                    # RAG system
├── langGraph.py                   # PolicyTools
└── requirements.txt

frontend/
├── src/
│   ├── services/
│   │   └── api.ts                 # API client
│   └── components/
│       └── ui/
│           ├── ai-chat.tsx        # Text Chat UI
│           └── voice-assistant.tsx # Voice UI (LiveKit)
└── ...
```

## Success Criteria Checklist

### Text Chat
- [ ] Personal Assistant greets warmly
- [ ] Personal Assistant does NOT auto-transfer (only on explicit request)
- [ ] Transfer to HR works with "connect me to HR"
- [ ] Transfer to IT works with "connect me to IT support"
- [ ] HR Agent retrieves from HR documents only
- [ ] IT Agent retrieves from IT documents only
- [ ] Out-of-scope questions stay within agent
- [ ] Clarification works for vague questions
- [ ] Sources/citations appear in UI
- [ ] Agent tabs maintain conversation history
- [ ] Validation retries work
- [ ] Error handling works gracefully

### Voice Agent
- [ ] Voice agent connects to LiveKit room
- [ ] Speech-to-text transcribes user speech
- [ ] Responses use same multi-agent system as text
- [ ] Text-to-speech plays response audio
- [ ] Agent transfers work via voice ("connect to IT")
- [ ] Different voices for each agent (Personal/HR/IT)
- [ ] Response latency is acceptable (~2-4 seconds)

## Next Steps

### Adding More Agents

To add LMS Agent:
1. Add LMS policy PDFs to `backend/docs/`
2. Update `SimpleRAG` with `lms_documents` list
3. Create LMS agent nodes in `specialist_agents.py`
4. Add LMS routing to `multi_agent_graph.py`
5. Add LMS tab to `ai-chat.tsx`

### Production Deployment

- Replace in-memory sessions with Redis/PostgreSQL
- Add authentication/authorization
- Implement rate limiting
- Add logging and monitoring
- Use environment-specific configs
- Deploy backend to cloud (AWS, GCP, Azure)
- Deploy frontend to Vercel/Netlify

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review API docs at `http://localhost:8000/docs`
3. Check backend logs in terminal
4. Check frontend console in browser DevTools
