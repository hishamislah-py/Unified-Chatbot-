"""
Combined entry point for Multi-Agent Chatbot with Voice Support

Usage with uvicorn:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

This will start:
1. FastAPI server (HTTP API)
2. LiveKit Voice Agent Worker (in background thread)
"""
import sys
import os
import threading
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
env_path = Path(__file__).parent / "env" / ".env"
load_dotenv(env_path)

# Import the FastAPI app
from api.server import app as fastapi_app

# Voice agent process
voice_agent_process = None


def start_voice_agent_worker():
    """Start LiveKit Voice Agent as a subprocess"""
    global voice_agent_process

    try:
        # Prepare environment with loaded variables
        env = os.environ.copy()

        # Start voice agent as a subprocess (inherit stdout/stderr for debugging)
        voice_agent_process = subprocess.Popen(
            [sys.executable, "-m", "voice.voice_agent", "dev"],
            cwd=str(Path(__file__).parent),
            env=env,
            # Show output for debugging
            stdout=None,
            stderr=None,
        )
        print("[OK] Voice Agent Worker started (PID: {})".format(voice_agent_process.pid))
    except Exception as e:
        print(f"[WARNING] Failed to start Voice Agent: {e}")
        print("[INFO] Voice features will not be available")


def stop_voice_agent_worker():
    """Stop the Voice Agent subprocess"""
    global voice_agent_process

    if voice_agent_process:
        print("[INFO] Stopping Voice Agent Worker...")
        voice_agent_process.terminate()
        voice_agent_process.wait(timeout=5)
        print("[OK] Voice Agent Worker stopped")


# Create lifespan context manager
@asynccontextmanager
async def lifespan(app):
    """
    Lifespan event handler - initializes RAG/Graph systems on startup
    """
    import api.server as server_module
    from rag_node import SimpleRAG
    from langGraph import PolicyTools
    from agents.multi_agent_graph import create_multi_agent_graph

    print("\n" + "="*70)
    print("STARTING MULTI-AGENT CHATBOT WITH VOICE SUPPORT")
    print("="*70)

    # Initialize RAG system
    print("\n[1/3] Initializing RAG system...")
    try:
        server_module.rag_system = SimpleRAG(docs_folder="./docs")
        server_module.rag_system.setup(verbose=False)
        print("[OK] RAG system initialized with HR and IT documents")
    except Exception as e:
        print(f"[ERROR] RAG initialization failed: {e}")
        raise

    # Set RAG system for PolicyTools
    print("\n[2/3] Setting RAG system for PolicyTools...")
    try:
        PolicyTools.set_rag_system(server_module.rag_system)
        print("[OK] PolicyTools configured")
    except Exception as e:
        print(f"[ERROR] PolicyTools configuration failed: {e}")
        raise

    # Build multi-agent graph
    print("\n[3/3] Building multi-agent LangGraph...")
    try:
        server_module.agent_graph = create_multi_agent_graph()
        print("[OK] Multi-agent graph compiled")
    except Exception as e:
        print(f"[ERROR] Graph compilation failed: {e}")
        raise

    print("\n[INFO] To enable voice, run in a SEPARATE terminal:")
    print("       cd backend && python -m voice.voice_agent dev")

    print("\n" + "="*70)
    print("SERVER READY!")
    print("API Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")

    yield  # Server is running

    # Shutdown
    print("[INFO] Server shutting down...")


# Override the lifespan on the imported app
fastapi_app.router.lifespan_context = lifespan

# Export the app for uvicorn
app = fastapi_app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
