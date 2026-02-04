"""
LiveKit Voice Agent for Multi-Agent Chatbot
Uses ART Technology APIs for STT/TTS and Ollama for LLM

Pipeline: Voice -> VAD -> STT (Whisper Edge) -> LLM (Chat API) -> TTS (Chatterbox) -> Voice
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load .env from backend/env/.env
env_path = Path(__file__).parent.parent / "env" / ".env"
load_dotenv(env_path)

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins.silero import VAD

# TTS - use Chatterbox
from voice.chatterbox_tts_adapter import ChatterboxTTS, AGENT_CONFIG
from voice.chat_api_llm import ChatAPILLM

# STT - choose between Whisper Edge (self-hosted) or Groq (cloud)
USE_GROQ_STT = os.getenv("USE_GROQ_STT", "false").lower() == "true"

if USE_GROQ_STT:
    from livekit.plugins.groq import STT as GroqSTT
else:
    from voice.whisper_edge_stt import WhisperEdgeSTT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")


async def entrypoint(ctx: JobContext):
    """Main entry point for LiveKit voice agent"""
    logger.info(f"Voice agent starting for room: {ctx.room.name}")

    # Connect to room first
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to LiveKit room")

    # Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Extract session_id from participant metadata (set during token generation)
    import json
    session_id = None
    if participant.metadata:
        try:
            metadata = json.loads(participant.metadata)
            session_id = metadata.get("session_id")
            logger.info(f"Session ID from participant metadata: {session_id}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse participant metadata")

    # Create the voice agent with ART Technology APIs
    logger.info("Creating Voice Agent with ART Technology APIs...")

    # Create agent with instructions
    agent = Agent(
        instructions="""You are a voice interface for the enterprise assistant.
The backend handles all responses through the multi-agent system (Personal Assistant, HR, IT).
Keep your acknowledgments brief as responses come from the chat API.""",
    )

    # Create TTS with initial emotion (Personal Assistant - friendly)
    tts = ChatterboxTTS(
        voice=AGENT_CONFIG["personal"]["voice"],
        emotion=AGENT_CONFIG["personal"]["emotion"],
    )

    # Create STT - use Groq (cloud) or Whisper Edge (self-hosted)
    if USE_GROQ_STT:
        stt = GroqSTT(model="whisper-large-v3")
        logger.info("Using Groq Whisper STT (cloud)")
    else:
        stt = WhisperEdgeSTT()
        logger.info("Using Whisper Edge STT (self-hosted)")

    # Create LLM with Chat API (pass session_id for unified session)
    chat_llm = ChatAPILLM(api_base="http://localhost:8000", session_id=session_id)

    # Set up voice/emotion switching callback - when agent changes, switch TTS
    def on_agent_change(new_agent: str):
        config = AGENT_CONFIG.get(new_agent, AGENT_CONFIG["personal"])
        tts.update_options(voice=config["voice"], emotion=config["emotion"])
        logger.info(f"Switched to {new_agent} - voice: {config['voice']}, emotion: {config['emotion']}")

    chat_llm.set_agent_change_callback(on_agent_change)

    # Create session with Whisper Edge STT, Chat API LLM, and Chatterbox TTS
    # Optimized VAD settings for faster response
    session = AgentSession(
        stt=stt,  # Self-hosted Whisper Edge STT
        llm=chat_llm,  # Uses Chat API with RAG + Multi-Agent
        tts=tts,  # Self-hosted Chatterbox TTS with emotion switching
        vad=VAD.load(
            min_silence_duration=0.25,  # 250ms (was 550ms) - faster end detection
            min_speech_duration=0.05,   # 50ms - minimum speech to register
            prefix_padding_duration=0.3, # 300ms (was 500ms) - less audio prefix
        ),
    )
    logger.info("Using Chat API LLM (RAG + Multi-Agent)")
    logger.info(f"STT: Whisper Edge at {os.getenv('WHISPER_EDGE_URL', 'default')}")
    logger.info(f"TTS: Chatterbox at {os.getenv('CHATTERBOX_TTS_URL', 'default')}")
    logger.info(f"Agent configs: Personal={AGENT_CONFIG['personal']}, HR={AGENT_CONFIG['hr']}, IT={AGENT_CONFIG['it']}")

    # Register cleanup callback for when the job shuts down
    async def cleanup():
        logger.info("Cleaning up resources...")
        await chat_llm.aclose()
        logger.info("Cleanup complete")

    ctx.add_shutdown_callback(cleanup)

    # Start the session (must await)
    await session.start(agent=agent, room=ctx.room)

    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user and offer your assistance as an enterprise assistant for HR policies and IT support."
    )


if __name__ == "__main__":
    logger.info("Starting LiveKit Voice Agent Worker")
    logger.info(f"LIVEKIT_URL: {os.getenv('LIVEKIT_URL', 'not set')}")
    logger.info(f"USE_GROQ_STT: {USE_GROQ_STT}")
    if USE_GROQ_STT:
        logger.info("STT: Groq Whisper (cloud)")
    else:
        logger.info(f"STT: Whisper Edge at {os.getenv('WHISPER_EDGE_URL', 'not set')}")
    logger.info(f"TTS: Chatterbox at {os.getenv('CHATTERBOX_TTS_URL', 'not set')}")
    logger.info(f"LLM: Ollama at {os.getenv('OLLAMA_BASE_URL', 'not set')}")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
