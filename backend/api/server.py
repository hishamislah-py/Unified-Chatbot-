import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path(__file__).parent.parent / "env" / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json
import asyncio
import uuid
import os

from api.models import ChatRequest, ChatResponse, SessionInfo, HealthCheckResponse, Source
from api.session_manager import SessionManager
from agents.multi_agent_graph import create_multi_agent_graph, MultiAgentState, route_from_hr_entry, route_from_hr_validation, route_from_it_entry, route_from_it_validation
from agents.specialist_agents import (
    hr_agent_entry_node, hr_clarification_node, hr_rag_retrieval_node,
    hr_answer_generation_node, hr_answer_generation_node_stream, hr_validation_node, hr_out_of_scope_node,
    it_agent_entry_node, it_clarification_node, it_rag_retrieval_node,
    it_answer_generation_node, it_answer_generation_node_stream, it_validation_node, it_out_of_scope_node,
    it_troubleshooting_node, it_jira_offer_node, it_jira_create_node
)
from agents.personal_assistant import PersonalAssistantTools
from rag_node import SimpleRAG
from langGraph import PolicyTools
from langgraph.graph import StateGraph, END


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Multi-Agent Chatbot API",
    description="API for Personal Assistant, HR Agent, and IT Support chatbot system",
    version="1.0.0"
)

# =============================================================================
# CORS CONFIGURATION
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (DEVELOPMENT ONLY!)
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

session_manager = SessionManager()
rag_system = None
agent_graph = None

# Voice event queues for broadcasting transcriptions to chat UI
voice_event_queues: dict[str, asyncio.Queue] = {}


# =============================================================================
# VOICE EVENT BROADCASTING HELPER
# =============================================================================

async def broadcast_voice_event(session_id: str, event_type: str, data: dict):
    """
    Broadcast a voice event to subscribed clients.

    Args:
        session_id: The session ID to broadcast to
        event_type: Event type (user_message, ai_token, ai_complete)
        data: Event data to send
    """
    if session_id in voice_event_queues:
        try:
            voice_event_queues[session_id].put_nowait({"type": event_type, "data": data})
        except asyncio.QueueFull:
            pass  # Drop events if queue is full


# =============================================================================
# IT TROUBLESHOOTING FALLBACK HELPER
# =============================================================================

def get_generic_troubleshooting_steps(question: str) -> str:
    """Provide generic troubleshooting steps when LLM is unavailable"""
    question_lower = question.lower()

    if any(kw in question_lower for kw in ['mouse', 'touchpad', 'keyboard', 'trackpad']):
        return """Here are some basic troubleshooting steps for input device issues:

1. **Check physical connection** - If using a wired device, ensure the USB cable is securely connected
2. **Try a different USB port** - Sometimes ports can fail
3. **Restart your computer** - This often resolves driver issues
4. **Check device manager** - Look for any yellow warning icons
5. **Update drivers** - Download latest drivers from manufacturer website

If this doesn't resolve your issue, let me know and I can help create a JIRA ticket for further assistance."""

    elif any(kw in question_lower for kw in ['teams', 'outlook', 'email', 'office', 'word', 'excel', 'powerpoint']):
        return """Here are some basic troubleshooting steps:

1. **Restart the application** - Close and reopen the app
2. **Clear cache** - Go to Settings > Clear cache
3. **Sign out and sign back in** - This refreshes your session
4. **Check for updates** - Ensure you have the latest version
5. **Restart your computer** - Clears temporary issues

If this doesn't resolve your issue, let me know and I can help create a JIRA ticket for further assistance."""

    else:
        return """Here are some general troubleshooting steps:

1. **Restart the application** - Close and reopen it
2. **Restart your computer** - This resolves many issues
3. **Check your internet connection** - Run a speed test
4. **Check for updates** - Ensure software is up to date
5. **Clear temporary files** - Free up system resources

If this doesn't resolve your issue, let me know and I can help create a JIRA ticket for further assistance."""


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize RAG system and LangGraph on server startup
    """
    global rag_system, agent_graph

    print("\n" + "="*70)
    print("STARTING MULTI-AGENT CHATBOT SERVER")
    print("="*70)

    # Initialize RAG system
    print("\n[1/3] Initializing RAG system...")
    try:
        rag_system = SimpleRAG(docs_folder="./docs")
        rag_system.setup(verbose=False)
        print("[OK] RAG system initialized with HR and IT documents")
    except Exception as e:
        print(f"[ERROR] RAG initialization failed: {e}")
        raise

    # Set RAG system for PolicyTools (required for agents)
    print("\n[2/3] Setting RAG system for PolicyTools...")
    try:
        PolicyTools.set_rag_system(rag_system)
        print("[OK] PolicyTools configured")
    except Exception as e:
        print(f"[ERROR] PolicyTools configuration failed: {e}")
        raise

    # Build multi-agent graph
    print("\n[3/3] Building multi-agent LangGraph...")
    try:
        agent_graph = create_multi_agent_graph()
        print("[OK] Multi-agent graph compiled")
    except Exception as e:
        print(f"[ERROR] Graph compilation failed: {e}")
        raise

    print("\n" + "="*70)
    print("SERVER READY!")
    print("API Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - server info
    """
    return {
        "message": "Multi-Agent Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns server status and initialization state
    """
    return HealthCheckResponse(
        status="healthy" if (rag_system and agent_graph) else "unhealthy",
        rag_initialized=rag_system is not None,
        graph_initialized=agent_graph is not None
    )


# =============================================================================
# LIVEKIT VOICE ENDPOINTS
# =============================================================================

@app.post("/api/livekit/token", tags=["Voice"])
async def generate_livekit_token(request: dict = None):
    """
    Generate a LiveKit room token for voice connection

    Args:
        request: JSON body with optional session_id to associate with the voice room

    Returns:
        token: JWT token for LiveKit room access
        room_name: Name of the room to join
        livekit_url: URL of the LiveKit server
        session_id: The session ID associated with this voice room
    """
    try:
        from livekit import api

        # Get session_id from request body
        session_id = request.get("session_id") if request else None

        # Get LiveKit credentials from environment
        livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
        api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
        api_secret = os.getenv("LIVEKIT_API_SECRET", "devsecret")

        # Create unique room and participant names
        # Include session_id in room name for tracking
        room_name = f"voice-room-{session_id or str(uuid.uuid4())[:8]}"
        participant_name = f"user-{str(uuid.uuid4())[:8]}"

        # Create access token with session metadata
        token = api.AccessToken(api_key=api_key, api_secret=api_secret)
        token.with_identity(participant_name)
        token.with_name(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))

        # Include session_id in token metadata so voice agent can access it
        if session_id:
            token.with_metadata(json.dumps({"session_id": session_id}))

        return {
            "token": token.to_jwt(),
            "room_name": room_name,
            "participant_name": participant_name,
            "livekit_url": livekit_url,
            "session_id": session_id,
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LiveKit SDK not installed. Run: pip install livekit"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate LiveKit token: {str(e)}"
        )


@app.options("/api/livekit/token", tags=["Voice"])
async def options_livekit_token():
    """Handle CORS preflight for LiveKit token endpoint"""
    return {"message": "OK"}


@app.get("/api/voice/events/{session_id}", tags=["Voice"])
async def voice_events_stream(session_id: str):
    """
    SSE endpoint for streaming voice conversation events to the chat UI.

    Events:
    - user_message: User's transcribed speech (final)
    - ai_token: Individual AI response token
    - ai_complete: AI response complete with metadata

    Args:
        session_id: The session ID to subscribe to

    Returns:
        StreamingResponse: SSE stream of voice events
    """
    queue = asyncio.Queue(maxsize=100)
    voice_event_queues[session_id] = queue

    async def event_generator():
        try:
            while True:
                try:
                    # Wait for events with timeout to detect disconnection
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: {event['type']}\n"
                    yield f"data: {json.dumps(event['data'])}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"event: keepalive\ndata: {{}}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected - exit gracefully
            pass
        except Exception as e:
            # Handle unexpected errors (including connection errors)
            if "closed" not in str(e).lower():
                print(f"[SSE] Voice events stream error: {e}")
        finally:
            voice_event_queues.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.options("/api/voice/events/{session_id}", tags=["Voice"])
async def options_voice_events(session_id: str):
    """Handle CORS preflight for voice events endpoint"""
    return {"message": "OK"}


@app.options("/api/sessions", tags=["Sessions"])
async def options_create_session():
    """Handle CORS preflight for POST /api/sessions"""
    return {"message": "OK"}

@app.post("/api/sessions", response_model=SessionInfo, tags=["Sessions"])
async def create_session():
    """
    Create a new chat session

    Returns:
        SessionInfo: New session metadata
    """
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)

    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        message_count=0,
        current_agent="personal"
    )


@app.options("/api/sessions/{session_id}", tags=["Sessions"])
async def options_get_session(session_id: str):
    """Handle CORS preflight for GET /api/sessions/{session_id}"""
    return {"message": "OK"}

@app.get("/api/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(session_id: str):
    """
    Get session information

    Args:
        session_id: Session identifier

    Returns:
        SessionInfo: Session metadata

    Raises:
        HTTPException: If session not found
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        message_count=len(session["messages"]),
        current_agent=session["current_agent"]
    )


@app.options("/api/chat", tags=["Chat"])
async def options_chat():
    """Handle CORS preflight for POST /api/chat"""
    return {"message": "OK"}

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user messages through multi-agent system

    Flow:
    1. Validate session exists (auto-create if not found)
    2. Prepare initial state for LangGraph
    3. Execute graph (routes through Personal Assistant → Specialists)
    4. Save messages to session history
    5. Return AI response with sources

    Args:
        request: ChatRequest with session_id, message, and current agent

    Returns:
        ChatResponse: AI response with sources and metadata

    Raises:
        HTTPException: If processing error
    """
    # Validate session - auto-create if not found (handles server restarts gracefully)
    session = session_manager.get_session(request.session_id)
    if not session:
        # Auto-create session with the provided ID
        session_manager.sessions[request.session_id] = {
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "current_agent": request.agent,
            "agent_state": {"original_issue": "", "awaiting_jira_confirmation": False}
        }
        session = session_manager.get_session(request.session_id)
        print(f"[INFO] Auto-created session: {request.session_id}")

    # Check if systems are initialized
    if not rag_system or not agent_graph:
        raise HTTPException(
            status_code=503,
            detail="Server not ready. RAG or Graph system not initialized."
        )

    try:
        # Determine which agent to use based on current context
        # If request.agent is not 'personal', use that agent
        entry_agent = request.agent

        # Get previous state from session if available (for JIRA confirmation flow)
        previous_state = session.get("agent_state", {})

        # Prepare initial state for LangGraph
        initial_state = {
            "current_message": request.message,
            "answer": "",
            "current_agent": entry_agent,
            "transfer_requested": False,
            "target_agent": "",
            "intent": "",
            "specialist_intent": "",
            "category": "",
            "retrieved_chunks": [],
            "sources": [],
            "needs_clarification": False,
            "is_valid": False,
            "retry_count": 0,
            "validation_reason": "",
            "session_id": request.session_id,
            "workflow_path": [],
            # JIRA ticket creation fields - preserve from previous state
            "original_issue": previous_state.get("original_issue", ""),
            "jira_ticket_id": "",
            "jira_ticket_url": "",
            "awaiting_jira_confirmation": previous_state.get("awaiting_jira_confirmation", False)
        }

        # Execute multi-agent graph with dynamic entry point
        config = {"configurable": {"thread_id": request.session_id}}

        # If we're in HR or IT agent, start from that agent's entry node
        if entry_agent == "hr":
            # Skip personal assistant, go directly to HR agent
            initial_state['workflow_path'] = []
            state_after_entry = hr_agent_entry_node(initial_state)

            # Route within HR agent
            next_node = route_from_hr_entry(state_after_entry)

            # Create a sub-graph execution for HR agent
            workflow = StateGraph(MultiAgentState)
            workflow.add_node("hr_clarification", hr_clarification_node)
            workflow.add_node("hr_rag_retrieval", hr_rag_retrieval_node)
            workflow.add_node("hr_answer_generation", hr_answer_generation_node)
            workflow.add_node("hr_validation", hr_validation_node)
            workflow.add_node("hr_out_of_scope", hr_out_of_scope_node)

            workflow.set_entry_point(next_node)

            if next_node == "hr_rag_retrieval":
                workflow.add_edge("hr_rag_retrieval", "hr_answer_generation")
                workflow.add_edge("hr_answer_generation", "hr_validation")
                workflow.add_conditional_edges("hr_validation", route_from_hr_validation, {
                    "hr_rag_retrieval": "hr_rag_retrieval",
                    "end": END
                })
            else:
                workflow.add_edge(next_node, END)

            hr_graph = workflow.compile()
            final_state = hr_graph.invoke(state_after_entry, config)

        elif entry_agent == "it":
            # Skip personal assistant, go directly to IT agent
            initial_state['workflow_path'] = []
            state_after_entry = it_agent_entry_node(initial_state)

            # Route within IT agent
            next_node = route_from_it_entry(state_after_entry)

            # Create a sub-graph execution for IT agent
            workflow = StateGraph(MultiAgentState)
            workflow.add_node("it_clarification", it_clarification_node)
            workflow.add_node("it_rag_retrieval", it_rag_retrieval_node)
            workflow.add_node("it_answer_generation", it_answer_generation_node)
            workflow.add_node("it_validation", it_validation_node)
            workflow.add_node("it_out_of_scope", it_out_of_scope_node)
            workflow.add_node("it_troubleshooting", it_troubleshooting_node)
            workflow.add_node("it_jira_offer", it_jira_offer_node)
            workflow.add_node("it_jira_create", it_jira_create_node)

            workflow.set_entry_point(next_node)

            if next_node == "it_rag_retrieval":
                workflow.add_edge("it_rag_retrieval", "it_answer_generation")
                workflow.add_edge("it_answer_generation", "it_validation")
                workflow.add_conditional_edges("it_validation", route_from_it_validation, {
                    "it_rag_retrieval": "it_rag_retrieval",
                    "end": END
                })
            else:
                workflow.add_edge(next_node, END)

            it_graph = workflow.compile()

            # Handle async node (it_jira_create_node is async)
            if next_node == "it_jira_create":
                # Run async node directly
                final_state = await it_jira_create_node(state_after_entry)
            else:
                final_state = it_graph.invoke(state_after_entry, config)

        else:
            # Use personal assistant (default entry point)
            final_state = agent_graph.invoke(initial_state, config)

        # Save user message to session
        session_manager.add_message(request.session_id, {
            "sender": "user",
            "text": request.message,
            "agent": request.agent,
            "timestamp": datetime.now().isoformat()
        })

        # Save AI response to session
        session_manager.add_message(request.session_id, {
            "sender": "ai",
            "text": final_state['answer'],
            "agent": final_state['current_agent'],
            "timestamp": datetime.now().isoformat()
        })

        # Update current agent in session
        session_manager.update_current_agent(
            request.session_id,
            final_state['current_agent']
        )

        # Save agent state for JIRA flow persistence
        session_manager.sessions[request.session_id]["agent_state"] = {
            "original_issue": final_state.get("original_issue", ""),
            "awaiting_jira_confirmation": final_state.get("awaiting_jira_confirmation", False)
        }

        # Convert sources to Source models
        sources = [
            Source(
                source=s['source'],
                page=s['page'],
                rank=s['rank'],
                preview=s['preview']
            )
            for s in final_state.get('sources', [])
        ]

        # Return response
        return ChatResponse(
            session_id=request.session_id,
            message=final_state['answer'],
            agent=final_state['current_agent'],
            sources=sources,
            needs_clarification=final_state.get('needs_clarification', False),
            workflow_path=final_state.get('workflow_path', [])
        )

    except Exception as e:
        print(f"[ERROR] Chat processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.options("/api/chat/stream", tags=["Chat"])
async def options_chat_stream():
    """Handle CORS preflight for POST /api/chat/stream"""
    return {"message": "OK"}

@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - sends tokens via Server-Sent Events (SSE)

    Flow:
    1. Validate session exists (auto-create if not found)
    2. Determine which agent to use
    3. Stream tokens as they're generated from LLM
    4. Send completion event with metadata (sources, agent, workflow_path)

    Args:
        request: ChatRequest with session_id, message, and current agent

    Returns:
        StreamingResponse: SSE stream with token and completion events

    Raises:
        HTTPException: If processing error
    """
    # Validate session - auto-create if not found (handles server restarts gracefully)
    session = session_manager.get_session(request.session_id)
    if not session:
        # Auto-create session with the provided ID to handle server restarts
        session_manager.sessions[request.session_id] = {
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "current_agent": request.agent,
            "agent_state": {"original_issue": "", "awaiting_jira_confirmation": False}
        }
        session = session_manager.get_session(request.session_id)
        print(f"[INFO] Auto-created session: {request.session_id}")

    # Check if systems are initialized
    if not rag_system or not agent_graph:
        raise HTTPException(
            status_code=503,
            detail="Server not ready. RAG or Graph system not initialized."
        )

    async def generate_stream():
        """Generator that yields SSE-formatted chunks"""
        client_disconnected = False

        def check_disconnected():
            return client_disconnected

        try:
            entry_agent = request.agent
            is_voice = request.source == "voice"

            # Save user message to session
            session_manager.add_message(request.session_id, {
                "sender": "user",
                "text": request.message,
                "agent": request.agent,
                "timestamp": datetime.now().isoformat(),
                "source": request.source
            })

            # Broadcast user message to chat UI if this is from voice
            if is_voice:
                await broadcast_voice_event(request.session_id, "user_message", {
                    "text": request.message,
                    "agent": request.agent
                })

            # Prepare initial state
            initial_state = {
                "current_message": request.message,
                "answer": "",
                "current_agent": entry_agent,
                "transfer_requested": False,
                "target_agent": "",
                "intent": "",
                "specialist_intent": "",
                "category": "",
                "retrieved_chunks": [],
                "sources": [],
                "needs_clarification": False,
                "is_valid": False,
                "retry_count": 0,
                "validation_reason": "",
                "session_id": request.session_id,
                "workflow_path": []
            }

            accumulated_answer = ""
            final_sources = []
            final_agent = entry_agent
            workflow_path = []

            # Route based on agent type
            if entry_agent == "personal":
                # Personal Assistant - handle intent classification and possible transfer
                tools = PersonalAssistantTools()
                classification = tools.classify_intent(request.message)

                if classification['intent'] == "transfer_request":
                    # Handle transfer
                    target = classification['target_agent']
                    if target == 'hr':
                        final_agent = 'hr'
                        response_text = "Connecting you to our HR specialist now. How can they help you today?"
                    elif target == 'it':
                        final_agent = 'it'
                        response_text = "Connecting you to our IT Support specialist now. How can they help you today?"
                    elif target == 'personal':
                        # Already with Personal Assistant
                        response_text = "You're already with me, the Personal Assistant! How can I help you today?"
                    else:
                        response_text = "I'd be happy to connect you to the right specialist. Could you specify if you need HR or IT support?"

                    # Stream the transfer message
                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                        # Removed delay for faster streaming

                elif classification['intent'] == "greeting":
                    response_text = (
                        "Hello! 👋 I'm your Personal Assistant. I'm here to help with general questions "
                        "or connect you to our specialists:\n\n"
                        "• **HR Agent** - for HR policies, leave requests, and employee benefits\n"
                        "• **IT Support** - for technical issues, security policies, and IT systems\n\n"
                        "How can I assist you today?"
                    )
                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                elif classification['intent'] == "general_query":
                    # Stream general query answer
                    async for token in tools.answer_general_query_stream(request.message):
                        accumulated_answer += token
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"

                elif classification['intent'] == "out_of_scope":
                    response_text = (
                        "I can help with company-related questions or connect you to our HR or IT specialists. "
                        "Your question seems to be outside my area. Could you ask about company policies or services instead?"
                    )
                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

            elif entry_agent == "hr":
                # HR Agent - first check for transfer requests
                pa_tools = PersonalAssistantTools()
                transfer_check = pa_tools.classify_intent(request.message)

                if transfer_check['intent'] == 'transfer_request':
                    # Handle transfer from HR to another agent
                    target = transfer_check['target_agent']
                    if target == 'it':
                        final_agent = 'it'
                        response_text = "[HR Agent] Connecting you to our IT Support specialist now. How can they help you today?"
                    elif target == 'hr':
                        # Already in HR, just acknowledge
                        response_text = "[HR Agent] You're already connected to HR. How can I help you?"
                    else:
                        # Transfer to personal assistant
                        final_agent = 'personal'
                        response_text = "[HR Agent] Transferring you back to the Personal Assistant. How can they help you today?"

                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                else:
                    # Continue with HR Agent processing
                    policy_tools = PolicyTools()

                    # Intent classification for HR
                    classification = policy_tools.classify_intent(request.message)
                    specialist_intent = classification['intent']
                    category = classification['category']

                    if specialist_intent == "ambiguous":
                        # Clarification needed
                        clarification = policy_tools.generate_clarification(
                            request.message,
                            "Your question about HR policies needs more detail"
                        )
                        response_text = f"[HR Agent] {clarification}"

                        for char in response_text:
                            accumulated_answer += char
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
    
                    elif specialist_intent == "policy_query":
                        # RAG retrieval and answer generation with streaming
                        if category not in ["HR", "Leave"]:
                            category = "HR"

                        # Retrieve relevant chunks
                        chunks = policy_tools.retrieve_policy(request.message, category, num_chunks=4)

                        # Stream the answer
                        prefix = "[HR Agent] "
                        accumulated_answer = prefix

                        # Send prefix first
                        for char in prefix:
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                        # Stream answer tokens with error handling and fallback
                        try:
                            async for token in policy_tools.generate_answer_with_citations_stream(request.message, chunks):
                                accumulated_answer += token
                                yield f"event: token\n"
                                yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"
                        except Exception as hr_error:
                            print(f"[HR Stream] LLM error: {hr_error}")
                            # Fallback: Extract and format RAG content directly
                            if chunks:
                                fallback_response = "Based on our HR documentation:\n\n"
                                accumulated_answer += fallback_response
                                for char in fallback_response:
                                    yield f"event: token\n"
                                    yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                                for i, chunk in enumerate(chunks[:3], 1):
                                    content = chunk.get('content', '').strip()
                                    if content:
                                        step_text = f"**{i}. From {chunk.get('source', 'HR Policy')}:**\n{content}\n\n"
                                        accumulated_answer += step_text
                                        for char in step_text:
                                            yield f"event: token\n"
                                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                            else:
                                error_msg = "I'm having trouble generating a response. Please try again."
                                accumulated_answer += error_msg
                                yield f"event: token\n"
                                yield f"data: {json.dumps({'content': error_msg, 'type': 'token'})}\n\n"

                        # Extract sources
                        final_sources = [
                            {
                                "source": chunk['source'],
                                "page": chunk['page'],
                                "rank": chunk['rank'],
                                "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                            }
                            for chunk in chunks
                        ]

                    else:  # out_of_scope
                        response_text = (
                            "[HR Agent] I specialize in HR and Leave policies (hiring, termination, probation, "
                            "annual leave, sick leave, maternity leave, etc.). "
                            "Your question seems outside my area of expertise.\n\n"
                            "If you need IT support or have technical questions, please ask the Personal Assistant "
                            "to connect you to IT Support."
                        )

                        for char in response_text:
                            accumulated_answer += char
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
    
            elif entry_agent == "it":
                # IT Agent - use EXPLICIT keyword detection for transfers (not LLM-based)
                # This prevents technical issues like "Teams not working" from being misclassified
                message_lower = request.message.lower()

                # Explicit transfer keywords - only these trigger a transfer
                hr_transfer_keywords = [
                    "connect me to hr", "talk to hr", "transfer to hr",
                    "connect to hr", "speak to hr", "hr agent"
                ]
                personal_transfer_keywords = [
                    "connect me to personal", "talk to personal assistant",
                    "transfer to personal", "go back", "transfer back",
                    "main menu", "back to main"
                ]

                is_hr_transfer = any(kw in message_lower for kw in hr_transfer_keywords)
                is_personal_transfer = any(kw in message_lower for kw in personal_transfer_keywords)

                if is_hr_transfer:
                    # Transfer to HR
                    final_agent = 'hr'
                    response_text = "[IT Support] Connecting you to our HR specialist now. How can they help you today?"

                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                elif is_personal_transfer:
                    # Transfer to Personal Assistant
                    final_agent = 'personal'
                    response_text = "[IT Support] Transferring you back to the Personal Assistant. How can they help you today?"

                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                else:
                    # Continue with IT Agent processing (troubleshooting, policy queries, etc.)
                    print(f"[IT Stream] Processing message: {request.message[:100]}")
                    policy_tools = PolicyTools()

                    # Use IT-specific intent classification (supports troubleshooting)
                    print(f"[IT Stream] Starting intent classification...")
                    try:
                        classification = policy_tools.classify_it_intent(request.message)
                        specialist_intent = classification['intent']
                        category = classification['category']
                        print(f"[IT Stream] Intent classified: {specialist_intent}, Category: {category}")
                    except Exception as classify_error:
                        print(f"[IT Stream] Intent classification FAILED: {classify_error}")
                        specialist_intent = "troubleshooting"  # Default to troubleshooting
                        category = "IT"

                    print(f"[IT Stream] Message: {request.message}")
                    print(f"[IT Stream] Classified intent: {specialist_intent}")
                    print(f"[IT Stream] Category: {category}")

                    if specialist_intent == "ambiguous":
                        # Clarification needed
                        clarification = policy_tools.generate_clarification(
                            request.message,
                            "Your question about IT policies needs more detail"
                        )
                        response_text = f"[IT Support] {clarification}"

                        for char in response_text:
                            accumulated_answer += char
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
    
                    elif specialist_intent == "policy_query":
                        # RAG retrieval for IT policies
                        if category not in ["IT", "Compliance"]:
                            category = "IT"

                        # Retrieve relevant chunks
                        chunks = policy_tools.retrieve_policy(request.message, category, num_chunks=4)

                        # Stream the answer
                        prefix = "[IT Support] "
                        accumulated_answer = prefix

                        # Send prefix first
                        for char in prefix:
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                        # Stream answer tokens with error handling and fallback
                        try:
                            async for token in policy_tools.generate_answer_with_citations_stream(request.message, chunks):
                                accumulated_answer += token
                                yield f"event: token\n"
                                yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"
                        except Exception as it_error:
                            print(f"[IT Stream] LLM error: {it_error}")
                            # Fallback: Extract and format RAG content directly
                            if chunks:
                                fallback_response = "Based on our IT documentation:\n\n"
                                accumulated_answer += fallback_response
                                for char in fallback_response:
                                    yield f"event: token\n"
                                    yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                                for i, chunk in enumerate(chunks[:3], 1):
                                    content = chunk.get('content', '').strip()
                                    if content:
                                        step_text = f"**{i}. From {chunk.get('source', 'IT Policy')}:**\n{content}\n\n"
                                        accumulated_answer += step_text
                                        for char in step_text:
                                            yield f"event: token\n"
                                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                            else:
                                error_msg = "I'm having trouble generating a response. Please try again."
                                accumulated_answer += error_msg
                                yield f"event: token\n"
                                yield f"data: {json.dumps({'content': error_msg, 'type': 'token'})}\n\n"

                        # Extract sources
                        final_sources = [
                            {
                                "source": chunk['source'],
                                "page": chunk['page'],
                                "rank": chunk['rank'],
                                "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                            }
                            for chunk in chunks
                        ]

                    elif specialist_intent == "troubleshooting":
                        # Troubleshooting - FIRST check RAG, then fall back to LLM knowledge
                        from langchain_core.prompts import ChatPromptTemplate
                        from langchain_core.output_parsers import StrOutputParser

                        # Store original issue for potential JIRA ticket creation
                        session_manager.sessions[request.session_id]["agent_state"] = {
                            "original_issue": request.message,
                            "awaiting_jira_confirmation": False
                        }

                        # =================================================================
                        # STEP 1: Check RAG first for relevant IT Support documents
                        # =================================================================
                        print(f"[IT Troubleshooting Stream] Checking RAG first for: {request.message}")

                        # Force category to IT for RAG search
                        rag_chunks = policy_tools.retrieve_policy(
                            request.message,
                            "IT",
                            num_chunks=4
                        )

                        # =================================================================
                        # STEP 2: Check SEMANTIC RELEVANCE of retrieved chunks
                        # =================================================================
                        has_relevant_rag_results = False

                        if rag_chunks and len(rag_chunks) > 0:
                            # Use semantic relevance check instead of just length check
                            relevance_result = policy_tools.check_context_relevance(request.message, rag_chunks)
                            print(f"[IT Troubleshooting Stream] Relevance check: {relevance_result}")

                            if relevance_result['is_relevant']:
                                has_relevant_rag_results = True
                            else:
                                # Also check for explicit keyword matches in source documents
                                question_lower = request.message.lower()
                                for chunk in rag_chunks:
                                    source = chunk.get('source', '').lower()
                                    # Direct source match check
                                    if 'teams' in question_lower and 'teams' in source:
                                        has_relevant_rag_results = True
                                        print(f"[IT Troubleshooting Stream] Source match found: {source}")
                                        break
                                    elif 'url' in question_lower and 'url' in source:
                                        has_relevant_rag_results = True
                                        break
                                    elif 'outlook' in question_lower and 'outlook' in source:
                                        has_relevant_rag_results = True
                                        break
                                    elif 'onedrive' in question_lower and 'onedrive' in source:
                                        has_relevant_rag_results = True
                                        break
                                    elif 'sharepoint' in question_lower and 'sharepoint' in source:
                                        has_relevant_rag_results = True
                                        break
                                    elif ('mouse' in question_lower or 'keyboard' in question_lower or 'touchpad' in question_lower) and 'hardware' in source:
                                        has_relevant_rag_results = True
                                        break
                                    elif ('camera' in question_lower or 'mic' in question_lower or 'headset' in question_lower) and ('camera' in source or 'mic' in source or 'headset' in source):
                                        has_relevant_rag_results = True
                                        break
                                    elif ('freeze' in question_lower or 'freezing' in question_lower) and 'freezing' in source:
                                        has_relevant_rag_results = True
                                        break
                                    elif 'screenshare' in question_lower and 'screenshare' in source:
                                        has_relevant_rag_results = True
                                        break
                                    elif 'vm' in question_lower and 'vm' in source:
                                        has_relevant_rag_results = True
                                        break

                        print(f"[IT Troubleshooting Stream] Final relevance decision: {has_relevant_rag_results}")

                        # Stream the answer
                        prefix = "[IT Support] "
                        accumulated_answer = prefix

                        # Send prefix first
                        for char in prefix:
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                        # =================================================================
                        # STEP 2: If RAG has results, use them; otherwise use LLM knowledge
                        # =================================================================
                        if has_relevant_rag_results:
                            # Use RAG results - stream answer with citations
                            print("[IT Troubleshooting Stream] Using RAG-based answer")
                            llm_succeeded = False
                            try:
                                async for token in policy_tools.generate_answer_with_citations_stream(request.message, rag_chunks):
                                    llm_succeeded = True
                                    accumulated_answer += token
                                    yield f"event: token\n"
                                    yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"
                            except Exception as rag_error:
                                print(f"[IT Troubleshooting RAG Stream] LLM error: {rag_error}")
                                # Fallback: Extract and format RAG content directly
                                print("[IT Troubleshooting Stream] Falling back to direct RAG content")
                                fallback_response = "Based on our documentation, here are some troubleshooting steps:\n\n"
                                accumulated_answer += fallback_response
                                for char in fallback_response:
                                    yield f"event: token\n"
                                    yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                                # Format RAG chunks as numbered steps
                                for i, chunk in enumerate(rag_chunks[:3], 1):  # Top 3 chunks
                                    content = chunk.get('content', '').strip()
                                    if content:
                                        step_text = f"**{i}. From {chunk.get('source', 'Documentation')}:**\n{content}\n\n"
                                        accumulated_answer += step_text
                                        for char in step_text:
                                            yield f"event: token\n"
                                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                                llm_succeeded = True  # Mark as handled

                            # Add JIRA offer at the end
                            jira_offer = "\n\nIf this doesn't resolve your issue, let me know and I can help create a JIRA ticket for further assistance."
                            accumulated_answer += jira_offer
                            for char in jira_offer:
                                yield f"event: token\n"
                                yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                            # Extract sources
                            final_sources = [
                                {
                                    "source": chunk['source'],
                                    "page": chunk['page'],
                                    "rank": chunk['rank'],
                                    "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                                }
                                for chunk in rag_chunks
                            ]
                        else:
                            # Fall back to LLM knowledge
                            print("[IT Troubleshooting Stream] No RAG results, using LLM knowledge")
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", """You are an IT Support specialist. Provide helpful troubleshooting steps for the user's technical issue.

RULES:
1. Give practical solutions the user can try immediately
2. Format your response with clear numbered steps
3. Start with the simplest solutions first
4. Be concise but thorough
5. End with: "\n\nIf this doesn't resolve your issue, let me know and I can help create a JIRA ticket for further assistance."
"""),
                                ("user", "{question}")
                            ])

                            # Stream troubleshooting answer from LLM with error handling
                            try:
                                stream_started = False
                                async for chunk in (prompt | policy_tools.llm).astream({"question": request.message}):
                                    stream_started = True
                                    if hasattr(chunk, 'content') and chunk.content:
                                        accumulated_answer += chunk.content
                                        yield f"event: token\n"
                                        yield f"data: {json.dumps({'content': chunk.content, 'type': 'token'})}\n\n"

                                # If no content was streamed, provide fallback
                                if not stream_started or accumulated_answer == prefix:
                                    fallback_msg = "I'm having trouble generating a response. Please try again."
                                    accumulated_answer += fallback_msg
                                    yield f"event: token\n"
                                    yield f"data: {json.dumps({'content': fallback_msg, 'type': 'token'})}\n\n"
                            except Exception as llm_error:
                                print(f"[IT Troubleshooting Stream] LLM error: {llm_error}")
                                # Provide helpful fallback instead of error message
                                fallback_steps = get_generic_troubleshooting_steps(request.message)
                                accumulated_answer += fallback_steps
                                for char in fallback_steps:
                                    yield f"event: token\n"
                                    yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                    elif specialist_intent == "follow_up_issue":
                        # User says previous solution didn't work - offer JIRA ticket
                        response_text = (
                            "[IT Support] I'm sorry the previous solutions didn't resolve your issue. "
                            "Would you like me to create a JIRA ticket for further assistance? "
                            "An IT specialist will review your case and get back to you.\n\n"
                            "Just say **'yes'** or **'create ticket'** to proceed."
                        )

                        for char in response_text:
                            accumulated_answer += char
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                        # Set flag for JIRA confirmation
                        session_manager.sessions[request.session_id]["agent_state"] = {
                            "original_issue": session_manager.sessions[request.session_id].get("agent_state", {}).get("original_issue", request.message),
                            "awaiting_jira_confirmation": True
                        }

                    elif specialist_intent in ["jira_confirmation", "jira_create_direct"]:
                        # User confirmed ticket creation or directly requested it
                        from mcp_client import mcp_client

                        # Get original issue from session state
                        agent_state = session_manager.sessions[request.session_id].get("agent_state", {})
                        original_issue = agent_state.get("original_issue", "")
                        awaiting_confirmation = agent_state.get("awaiting_jira_confirmation", False)

                        # For jira_confirmation, require awaiting_confirmation to be True
                        if specialist_intent == "jira_confirmation" and not awaiting_confirmation:
                            response_text = (
                                "[IT Support] I'm not sure what you'd like me to confirm. "
                                "If you're having a technical issue, please describe it and I'll help troubleshoot."
                            )
                            for char in response_text:
                                accumulated_answer += char
                                yield f"event: token\n"
                                yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                        else:
                            # Create JIRA ticket
                            if not original_issue:
                                original_issue = request.message

                            summary = f"IT Support: {original_issue[:100]}"
                            description = f"**Issue Reported:** {original_issue}\n\n---\n*Auto-generated by IT Support Chatbot*"

                            try:
                                result = await mcp_client.create_jira_issue(
                                    summary=summary,
                                    description=description,
                                    issue_type="Task",
                                    project_key="KAN"
                                )

                                if result.success:
                                    response_text = (
                                        f"[IT Support] I've created a JIRA ticket for your issue.\n\n"
                                        f"**Ticket ID:** {result.ticket_id}\n\n"
                                        f"Our IT team will review your case and get back to you soon."
                                    )
                                else:
                                    response_text = (
                                        f"[IT Support] I apologize, but I encountered an error creating the ticket: "
                                        f"{result.error}\n\n"
                                        f"Please try again later or contact IT support directly."
                                    )
                            except Exception as e:
                                response_text = (
                                    f"[IT Support] I apologize, but there was an unexpected error creating your ticket. "
                                    f"Please try again or contact IT support directly.\n\n"
                                    f"Error: {str(e)}"
                                )

                            for char in response_text:
                                accumulated_answer += char
                                yield f"event: token\n"
                                yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"

                            # Reset JIRA confirmation flag
                            session_manager.sessions[request.session_id]["agent_state"] = {
                                "original_issue": "",
                                "awaiting_jira_confirmation": False
                            }

                    else:  # out_of_scope
                        response_text = (
                            "[IT Support] I specialize in IT Security and Compliance policies (device security, "
                            "passwords, VPN, data privacy, code of conduct, etc.). "
                            "Your question seems outside my area of expertise.\n\n"
                            "If you need HR assistance or have questions about employee policies, please ask the "
                            "Personal Assistant to connect you to the HR Agent."
                        )

                        for char in response_text:
                            accumulated_answer += char
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
    
            # Save AI response to session
            session_manager.add_message(request.session_id, {
                "sender": "ai",
                "text": accumulated_answer,
                "agent": final_agent,
                "timestamp": datetime.now().isoformat(),
                "source": request.source
            })

            # Update current agent in session
            session_manager.update_current_agent(request.session_id, final_agent)

            # Broadcast AI response completion to chat UI if this is from voice
            if is_voice:
                await broadcast_voice_event(request.session_id, "ai_complete", {
                    "text": accumulated_answer,
                    "agent": final_agent,
                    "sources": final_sources
                })

            # Send completion event with metadata
            yield f"event: complete\n"
            yield f"data: {json.dumps({
                'agent': final_agent,
                'sources': final_sources,
                'workflow_path': workflow_path
            })}\n\n"

        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected - exit gracefully without logging error
            client_disconnected = True
            return
        except Exception as e:
            # Only log and send error if client is still connected
            if not client_disconnected:
                error_msg = str(e)
                # Don't log connection-related errors as they're expected when client disconnects
                if "closed" not in error_msg.lower() and "peer" not in error_msg.lower():
                    print(f"[ERROR] Streaming chat failed: {e}")
                try:
                    yield f"event: error\n"
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                except (GeneratorExit, Exception):
                    # Client disconnected while sending error
                    pass

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )
