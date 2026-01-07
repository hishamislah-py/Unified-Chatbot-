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
from livekit.plugins.groq import STT, LLM
from voice.edge_tts_adapter import EdgeTTS as TTS_Edge

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
        instructions="""You are a helpful enterprise assistant for voice conversations.
You can answer general questions about company policies, HR matters, and IT support.
Be concise and professional. Keep responses brief and natural for voice output.
When users ask about:
- HR topics (leave, benefits, policies): Provide helpful HR guidance
- IT topics (technical issues, security): Provide IT support guidance
- General questions: Answer helpfully and offer to connect to specialists if needed
Limit responses to 2-3 sentences for natural conversation flow.""",
    )

    # Create session with Groq STT/LLM and Edge TTS (free)
    session = AgentSession(
        stt=STT(model="whisper-large-v3"),
        llm=LLM(model="llama-3.3-70b-versatile"),
        tts=TTS_Edge(voice="en-US-AriaNeural"),  # Free Microsoft Edge TTS
        vad=VAD.load(),
    )

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
