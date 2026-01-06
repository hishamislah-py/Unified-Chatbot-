import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json
import asyncio

from api.models import ChatRequest, ChatResponse, SessionInfo, HealthCheckResponse, Source
from api.session_manager import SessionManager
from agents.multi_agent_graph import create_multi_agent_graph, MultiAgentState, route_from_hr_entry, route_from_hr_validation, route_from_it_entry, route_from_it_validation
from agents.specialist_agents import (
    hr_agent_entry_node, hr_clarification_node, hr_rag_retrieval_node,
    hr_answer_generation_node, hr_answer_generation_node_stream, hr_validation_node, hr_out_of_scope_node,
    it_agent_entry_node, it_clarification_node, it_rag_retrieval_node,
    it_answer_generation_node, it_answer_generation_node_stream, it_validation_node, it_out_of_scope_node
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
    1. Validate session exists
    2. Prepare initial state for LangGraph
    3. Execute graph (routes through Personal Assistant → Specialists)
    4. Save messages to session history
    5. Return AI response with sources

    Args:
        request: ChatRequest with session_id, message, and current agent

    Returns:
        ChatResponse: AI response with sources and metadata

    Raises:
        HTTPException: If session not found or processing error
    """
    # Validate session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

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
            "workflow_path": []
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
    1. Validate session exists
    2. Determine which agent to use
    3. Stream tokens as they're generated from LLM
    4. Send completion event with metadata (sources, agent, workflow_path)

    Args:
        request: ChatRequest with session_id, message, and current agent

    Returns:
        StreamingResponse: SSE stream with token and completion events

    Raises:
        HTTPException: If session not found or processing error
    """
    # Validate session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if systems are initialized
    if not rag_system or not agent_graph:
        raise HTTPException(
            status_code=503,
            detail="Server not ready. RAG or Graph system not initialized."
        )

    async def generate_stream():
        """Generator that yields SSE-formatted chunks"""
        try:
            entry_agent = request.agent

            # Save user message to session
            session_manager.add_message(request.session_id, {
                "sender": "user",
                "text": request.message,
                "agent": request.agent,
                "timestamp": datetime.now().isoformat()
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
                    else:
                        response_text = "I'd be happy to connect you to the right specialist. Could you specify if you need HR or IT support?"

                    # Stream the transfer message
                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for smooth streaming

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
                        await asyncio.sleep(0.01)

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
                        await asyncio.sleep(0.01)

            elif entry_agent == "hr":
                # HR Agent - use RAG pipeline with streaming
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
                        await asyncio.sleep(0.01)

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
                        await asyncio.sleep(0.01)

                    # Stream answer tokens
                    async for token in policy_tools.generate_answer_with_citations_stream(request.message, chunks):
                        accumulated_answer += token
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"

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
                        await asyncio.sleep(0.01)

            elif entry_agent == "it":
                # IT Agent - use IT-specific intent classifier with troubleshooting support
                policy_tools = PolicyTools()

                # Use IT-specific intent classification (supports troubleshooting)
                classification = policy_tools.classify_it_intent(request.message)
                specialist_intent = classification['intent']
                category = classification['category']

                print(f"[IT Stream] Message: {request.message}")
                print(f"[IT Stream] Classified intent: {specialist_intent}")

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
                        await asyncio.sleep(0.01)

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
                        await asyncio.sleep(0.01)

                    # Stream answer tokens
                    async for token in policy_tools.generate_answer_with_citations_stream(request.message, chunks):
                        accumulated_answer += token
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"

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
                    # Troubleshooting - use LLM knowledge (NOT RAG)
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser

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

                    chain = prompt | policy_tools.llm | StrOutputParser()

                    # Stream the answer
                    prefix = "[IT Support] "
                    accumulated_answer = prefix

                    # Send prefix first
                    for char in prefix:
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                        await asyncio.sleep(0.01)

                    # Stream troubleshooting answer
                    async for chunk in (prompt | policy_tools.llm).astream({"question": request.message}):
                        if hasattr(chunk, 'content') and chunk.content:
                            accumulated_answer += chunk.content
                            yield f"event: token\n"
                            yield f"data: {json.dumps({'content': chunk.content, 'type': 'token'})}\n\n"

                elif specialist_intent == "follow_up_issue":
                    # User says previous solution didn't work - offer JIRA ticket
                    response_text = (
                        "[IT Support] I'm sorry the previous solutions didn't resolve your issue. "
                        "Would you like me to create a JIRA ticket for further assistance? "
                        "An IT support technician will review your case and get back to you.\n\n"
                        "(JIRA integration coming soon - for now, please contact IT helpdesk directly)"
                    )

                    for char in response_text:
                        accumulated_answer += char
                        yield f"event: token\n"
                        yield f"data: {json.dumps({'content': char, 'type': 'token'})}\n\n"
                        await asyncio.sleep(0.01)

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
                        await asyncio.sleep(0.01)

            # Save AI response to session
            session_manager.add_message(request.session_id, {
                "sender": "ai",
                "text": accumulated_answer,
                "agent": final_agent,
                "timestamp": datetime.now().isoformat()
            })

            # Update current agent in session
            session_manager.update_current_agent(request.session_id, final_agent)

            # Send completion event with metadata
            yield f"event: complete\n"
            yield f"data: {json.dumps({
                'agent': final_agent,
                'sources': final_sources,
                'workflow_path': workflow_path
            })}\n\n"

        except Exception as e:
            print(f"[ERROR] Streaming chat failed: {e}")
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
