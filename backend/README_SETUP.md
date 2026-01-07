# Multi-Agent Chatbot System - Setup Guide

## Overview

This multi-agent chatbot system features:
- **Personal Assistant** - Conversational entry point, routes to specialists
- **HR Agent** - Handles HR and Leave policies
- **IT Support Agent** - Handles IT Security and Compliance policies

## Architecture

```
Frontend (React + TypeScript)
    ↓ HTTP
FastAPI Server (Python)
    ↓
LangGraph Multi-Agent System
    ├── Personal Assistant
    ├── HR Agent (RAG: Leave Policy, HR Policy)
    └── IT Agent (RAG: IT Security, Compliance)
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

4. **Create .env file** in `backend/` directory with your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

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
python api/server.py  |     uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
======================================================================
STARTING MULTI-AGENT CHATBOT SERVER
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

### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

**Frontend runs on:** `http://localhost:5173`

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
- **POST /api/chat** - Send chat message
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

### Common Issues

**Agent not transferring:**
- Verify you used explicit keywords: "HR", "IT support", "connect to HR"
- Personal Assistant does NOT transfer on policy questions alone

**Out-of-scope questions getting answers:**
- Verify the question is truly out-of-scope
- Check workflow_path in response to see which nodes executed

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
├── docs/
│   ├── Leave Policy.pdf
│   ├── HR_Policy_Art_Technology.pdf
│   ├── IT_Security_Policy_AI_Usage.pdf
│   └── Compliance Handbook.pdf
├── rag_node.py                    # RAG system
├── langGraph.py                   # PolicyTools (reused)
├── requirements.txt
└── .env                           # API keys (create this)

frontend/
├── src/
│   ├── services/
│   │   └── api.ts                # API client
│   └── components/
│       └── ui/
│           └── ai-chat.tsx       # Chat UI
└── ...
```

## Success Criteria Checklist

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
