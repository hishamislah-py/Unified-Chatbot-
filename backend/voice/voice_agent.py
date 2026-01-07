"""
LiveKit Voice Agent for Multi-Agent Chatbot
Uses livekit-agents 1.x API with Groq for STT/TTS/LLM

Pipeline: Voice -> VAD -> STT (Groq Whisper) -> LLM (Groq) -> TTS (Groq) -> Voice
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
from livekit.plugins.groq import STT
from voice.edge_tts_adapter import EdgeTTS as TTS_Edge
from voice.chat_api_llm import ChatAPILLM, AGENT_VOICES

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

    # Create the voice agent with Groq services
    logger.info("Creating Voice Agent with Groq...")

    # Create agent with instructions
    agent = Agent(
        instructions="""You are a voice interface for the enterprise assistant.
The backend handles all responses through the multi-agent system (Personal Assistant, HR, IT).
Keep your acknowledgments brief as responses come from the chat API.""",
    )

    # Create TTS with initial voice (Personal Assistant)
    tts = TTS_Edge(voice=AGENT_VOICES["personal"])

    # Create LLM with Chat API
    chat_llm = ChatAPILLM(api_base="http://localhost:8000")

    # Set up voice switching callback - when agent changes, switch TTS voice
    def on_agent_change(new_agent: str):
        new_voice = AGENT_VOICES.get(new_agent, AGENT_VOICES["personal"])
        tts.update_options(voice=new_voice)
        logger.info(f"Switched to {new_agent} voice: {new_voice}")

    chat_llm.set_agent_change_callback(on_agent_change)

    # Create session with Groq STT, Chat API LLM (RAG + Multi-Agent), and Edge TTS
    # Optimized VAD settings for faster response
    session = AgentSession(
        stt=STT(model="whisper-large-v3"),
        llm=chat_llm,  # Uses Chat API with RAG + Multi-Agent
        tts=tts,  # Free Microsoft Edge TTS with voice switching
        vad=VAD.load(
            min_silence_duration=0.25,  # 250ms (was 550ms) - faster end detection
            min_speech_duration=0.05,   # 50ms - minimum speech to register
            prefix_padding_duration=0.3, # 300ms (was 500ms) - less audio prefix
        ),
    )
    logger.info("Using Chat API LLM (RAG + Multi-Agent)")
    logger.info(f"Agent voices: Personal={AGENT_VOICES['personal']}, HR={AGENT_VOICES['hr']}, IT={AGENT_VOICES['it']}")

    # Start the session (must await)
    await session.start(agent=agent, room=ctx.room)

    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user and offer your assistance as an enterprise assistant for HR policies and IT support."
    )


if __name__ == "__main__":
    logger.info("Starting LiveKit Voice Agent Worker")
    logger.info(f"LIVEKIT_URL: {os.getenv('LIVEKIT_URL', 'not set')}")
    logger.info(f"GROQ_API_KEY: {'set' if os.getenv('GROQ_API_KEY') else 'not set'}")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
